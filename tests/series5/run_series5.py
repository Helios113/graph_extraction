"""Series 5: time and GPU-memory use of ONNX (KV-cache / cacheless) vs torch (HF), GPT-2, fp32, TF32 off.

Two operations, both for one (sample, position) group of 768 candidate last tokens after a shared prefix of length L:
  fwd  residual stream after the last block (boundary 24)
  pb   JacobianExtractor pullback of the attention sublayer of layer 1 (boundary 2 -> 3) with one-hot masks, i.e. the full
       768x768 last-token Jacobian in one batch.  Torch = autograd on the HF module, same one-hot formulation.
Variants: kv_onnx (prefix pass once + batched length-1 continuation), cacheless_onnx (full (768, L+1) pass),
torch_cache (HF DynamicCache), torch (HF, no cache).  ONNX inputs/outputs are device-resident (IOBinding), as in torch.

Every (op, variant, L) runs in its own subprocess so the ORT arena / torch caching allocator of one run cannot leak into
another.  Time = median of ITERS runs after WARMUP.  Memory = device-wide used memory (torch.cuda.mem_get_info polled
every 2 ms, one GPU is allocated to the job) minus the bare-CUDA-context baseline: `load_mb` after model/session/inputs
exist, `peak_mb` the maximum over warm-up and timed runs.  `pb` outputs of every variant are compared in the driver.
Usage: run_series5.py [driver]   |   run_series5.py worker <op> <variant> <L>
"""
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tests" / "kv_tests"))
import conftest  # noqa: E402,F401  (sets NVIDIA_TF32_OVERRIDE=0 before CUDA init)

RESULTS = ROOT / "results" / "series5"
WORK = HERE / "work"
LENGTHS = tuple(int(x) for x in os.environ.get("S5_LENGTHS", "0,8,32,128").split(","))
OPS = {"fwd": ("kv_onnx", "cacheless_onnx", "torch_cache", "torch"), "pb": ("kv_onnx", "cacheless_onnx", "torch_cache", "torch")}
WARMUP, ITERS, N = 3, 10, 768
PB_J = 2  # attention sublayer of layer 1: boundary 2 -> 3


def used_mb():
    import torch
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2**20


class Sampler:
    def __init__(self):
        self.peak, self._stop = 0.0, threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, used_mb())
            time.sleep(0.002)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self.t.join()
        self.peak = max(self.peak, used_mb())


def inputs(L):
    """Shared by all variants and written once by the driver: prefix, candidates, boundary-2 activations of prefix+cand."""
    d = np.load(WORK / f"inputs_L{L}.npz")
    return d["prefix"], d["cand"], d["hidden"]


def make_inputs(L):
    import torch
    from kv_helpers import candidates, hf_all_boundaries, hf_model, random_tokens
    prefix = random_tokens(1, L, seed=900 + L)
    cand = candidates()
    hf = hf_model()
    seq = torch.from_numpy(np.concatenate([prefix, cand[:1][None]], axis=1)).cuda()
    with torch.no_grad():
        b, _ = hf_all_boundaries(hf, seq)
    hidden = b[PB_J][0].cpu().numpy()  # (L+1, d): its last row is the "token under optimisation" input
    np.savez(WORK / f"inputs_L{L}.npz", prefix=prefix, cand=cand, hidden=hidden)


# ------------------------------------------------------------------ builders: return (run_fn, post_load_cleanup)
def build(op, variant, L):
    import torch
    from kv_helpers import (D_MODEL, N_HEADS, HEAD_DIM, N_LAYERS, cacheless_cfg, cacheless_sublayer_path, gpt2_cfg, hf_model, kv_boundaries_session,
                            kv_sublayer_feeds)
    from test_kv_timing import GpuRunner, empty_past_gpu
    from construct.config import LossType
    from construct.construct import ensure_base_model, ensure_extra_outputs, ensure_subgraph_pullback
    from construct.kv_cache_subgraph import KV_SUBLAYER_IN, KV_SUBLAYER_OUT, ensure_kv_cache_extra_outputs, ensure_kv_cache_sublayer_pullback

    cfg = gpt2_cfg()
    prefix_np, cand_np, hidden_np = inputs(L)
    prefix = torch.from_numpy(prefix_np).cuda()
    cand = torch.from_numpy(cand_np).cuda()
    seqs = torch.cat([prefix.expand(N, -1), cand[:, None]], dim=1)
    ones = lambda n, m: torch.ones((n, m), dtype=torch.int64, device="cuda")  # noqa: E731
    onehot = torch.eye(D_MODEL, device="cuda")[:, None, :]  # (768, 1, d): row i reads dim i

    # ---- build / cache the ONNX artefacts first (CPU work; not part of the measured load)
    if variant == "kv_onnx":
        kv_boundaries_session(cfg)
        if op == "fwd":
            path, out_name, _ = ensure_kv_cache_extra_outputs(cfg, 11, "mlp_out")
        else:
            sub, pb_path = ensure_kv_cache_sublayer_pullback(cfg, PB_J, LossType.JacobianExtractorLoss)
            path, out_name, _ = ensure_kv_cache_extra_outputs(cfg, 11, "mlp_out")  # full model for the prefix pass
    elif variant == "cacheless_onnx":
        ccfg = cacheless_cfg(cfg, seq=L + 1, tag="_last" if op == "fwd" else "")
        ensure_base_model(ccfg, verbose=False)
        if op == "fwd":
            path = ensure_extra_outputs(ccfg, [cfg.sublayers[-1].output])
        else:
            fwd, sub = cacheless_sublayer_path(ccfg, PB_J + 1)
            import dataclasses
            sub = dataclasses.replace(sub, loss_type=LossType.JacobianExtractorLoss)
            [pb_path] = ensure_subgraph_pullback([(fwd, sub)], force=False)

    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    base = used_mb()

    # ---- load (measured)
    if variant == "kv_onnx":
        runner = GpuRunner(path)
        if op == "fwd":
            from test_kv_timing import kv_pipeline
            run = kv_pipeline(runner, out_name, prefix, cand)
        else:
            pb = GpuRunner(pb_path)
            hid = torch.from_numpy(hidden_np[-1]).cuda().expand(N, 1, D_MODEL).contiguous()

            def run():
                if L:
                    pre = runner({"input_ids": prefix, "attention_mask": ones(1, L), "position_ids": torch.arange(L, device="cuda")[None], **empty_past_gpu(1)})
                    past = {f"past_key_values.{n.split('.', 1)[1]}": pre[n].expand(N, -1, -1, -1) for n in pre if n.startswith("present.")}
                else:
                    past = empty_past_gpu(N)
                feeds = {KV_SUBLAYER_IN: hid, "mask": onehot}
                for name in sub.extra_inputs:
                    feeds[name] = (torch.zeros((N, 1), dtype=torch.int64, device="cuda") if name == "input_ids" else ones(N, L + 1)
                                   if name == "attention_mask" else past[name])
                return pb(feeds)[f"{KV_SUBLAYER_IN}_grad"][:, 0]
    elif variant == "cacheless_onnx":
        if op == "fwd":
            cl = GpuRunner(path)
            run = lambda: next(iter(cl({"input_ids": seqs}).values()))[:, -1]  # noqa: E731
        else:
            pb = GpuRunner(pb_path)
            x = torch.from_numpy(hidden_np).cuda().expand(N, -1, -1).contiguous()
            mask = torch.zeros((N, L + 1, D_MODEL), device="cuda")
            mask[torch.arange(D_MODEL), L, torch.arange(D_MODEL)] = 1.0
            run = lambda: pb({sub.input: x, "mask": mask})[f"{sub.input}_grad"][:, L]  # noqa: E731
    else:
        hf = hf_model()
        for p in hf.parameters():
            p.requires_grad_(False)
        cached = variant == "torch_cache"
        if op == "fwd":
            def run():
                with torch.no_grad():
                    if cached and L:
                        past = hf.transformer(prefix, use_cache=True).past_key_values
                        past.batch_repeat_interleave(N)
                        return hf.transformer(cand[:, None], past_key_values=past, use_cache=True).last_hidden_state[:, -1]
                    if cached:
                        return hf.transformer(cand[:, None], use_cache=True).last_hidden_state[:, -1]
                    return hf.transformer(seqs, use_cache=False).last_hidden_state[:, -1]
        else:
            blk = hf.transformer.h[PB_J // 2]
            hid_last = torch.from_numpy(hidden_np[-1]).cuda().expand(N, 1, D_MODEL)
            hid_full = torch.from_numpy(hidden_np).cuda().expand(N, -1, -1)

            def run():
                if cached:
                    from transformers import DynamicCache
                    if L:
                        with torch.no_grad():
                            full = hf.transformer(prefix, use_cache=True).past_key_values
                        past = DynamicCache()
                        k, v = full.layers[PB_J // 2].keys, full.layers[PB_J // 2].values
                        past.update(k.expand(N, -1, -1, -1), v.expand(N, -1, -1, -1), PB_J // 2)
                    else:
                        past = DynamicCache()
                    leaf = hid_last.clone().requires_grad_(True)
                    a = blk.attn(blk.ln_1(leaf), past_key_values=past, cache_position=torch.tensor([L], device="cuda"), use_cache=True)[0]
                    out = (leaf + a)[:, 0]
                    return torch.autograd.grad((out * onehot[:, 0]).sum(), leaf)[0][:, 0]
                leaf = hid_full.clone().requires_grad_(True)
                out = (leaf + blk.attn(blk.ln_1(leaf), use_cache=False)[0])[:, L]
                return torch.autograd.grad((out * onehot[:, 0]).sum(), leaf)[0][:, L]
    torch.cuda.synchronize()
    return run, base


def worker(op, variant, L):
    import torch
    run, base = build(op, variant, L)
    load = used_mb() - base
    torch.cuda.reset_peak_memory_stats()
    with Sampler() as s:
        for _ in range(WARMUP):
            out = run()
        torch.cuda.synchronize()
        times = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            out = run()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
    np.save(WORK / f"out_{op}_{variant}_L{L}.npy", out.detach().float().cpu().numpy())
    print("RESULT " + json.dumps({"op": op, "variant": variant, "L": L, "ms": statistics.median(times), "ms_min": min(times),
                                  "load_mb": load, "peak_mb": s.peak - base, "torch_peak_alloc_mb": torch.cuda.max_memory_allocated() / 2**20}), flush=True)


def driver():
    WORK.mkdir(exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    for L in LENGTHS:
        subprocess.run([sys.executable, __file__, "inputs", str(L)], check=True)
    rows, errors = [], {}
    for op, variants in OPS.items():
        for L in LENGTHS:
            for v in variants:
                print(f"=== {op} {v} L={L}", flush=True)
                p = subprocess.run([sys.executable, __file__, "worker", op, v, str(L)], capture_output=True, text=True)
                line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
                if p.returncode or not line:
                    errors[f"{op}|{v}|{L}"] = (p.stderr.strip().splitlines() or ["?"])[-1][:300]
                    print("  FAILED", errors[f"{op}|{v}|{L}"], flush=True)
                    continue
                rows.append(json.loads(line[0][7:]))
                print("  ", rows[-1], flush=True)
    # agreement of the pb outputs (same inputs, different implementations)
    agree = {}
    for L in LENGTHS:
        ref = WORK / f"out_pb_torch_L{L}.npy"
        if not ref.exists():
            continue
        r = np.load(ref)
        for v in OPS["pb"]:
            f = WORK / f"out_pb_{v}_L{L}.npy"
            if f.exists() and v != "torch":
                agree[f"{v}|{L}"] = float(np.abs(np.load(f) - r).max() / (np.abs(r).max() + 1e-12))
    for L in LENGTHS:
        ref = WORK / f"out_fwd_torch_L{L}.npy"
        if ref.exists():  # fwd: ONNX returns the pre-ln_f stream, torch the post-ln_f one -> compare only the torch pair
            c = WORK / f"out_fwd_torch_cache_L{L}.npy"
            k, cl = WORK / f"out_fwd_kv_onnx_L{L}.npy", WORK / f"out_fwd_cacheless_onnx_L{L}.npy"
            if k.exists() and cl.exists():
                agree[f"fwd_kv_vs_cacheless|{L}"] = float(np.abs(np.load(k) - np.load(cl)).max() / (np.abs(np.load(cl)).max() + 1e-12))
            if c.exists():
                agree[f"fwd_torch_cache|{L}"] = float(np.abs(np.load(c) - np.load(ref)).max() / (np.abs(np.load(ref)).max() + 1e-12))
    (RESULTS / "results.json").write_text(json.dumps({"rows": rows, "agreement_rel_max_err": agree, "errors": errors}, indent=1))
    print("done", len(rows), "rows", len(errors), "errors", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "driver"
    if mode == "inputs":
        make_inputs(int(sys.argv[2]))
    elif mode == "worker":
        worker(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        driver()
