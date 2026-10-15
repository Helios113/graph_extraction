"""Wall-clock comparison for one (sample, position) group of the perturbed sweep: 768 candidate last tokens
after a shared prefix of length L (0, 1, 8, 32), computing the residual stream after the final block (boundary 24).

  kv_onnx        prefix pass once (batch 1) + one batched length-1 continuation reading the broadcast prefix K/V
                 (what scripts/helper/generate_perturbed_sweep_kvcache.py does)
  cacheless_onnx full (768, L+1) pass of the static-shape cacheless export (what generate_perturbed_sweep.py does)
  hf             torch HF GPT2 transformer, full (768, L+1) pass
  hf_cache       torch HF GPT2 with its own DynamicCache: prefix once, cache repeated 768x, one-token continuation

All run on GPU with inputs/outputs device-resident, fp32, TF32 off. Results are written to
tests/kv_tests/results/timing.{md,json}. The only timing assertion is the robust one: at L=32 the KV pipeline
is faster than the cacheless one. Correctness of every variant is checked against HF in the same test.
"""

import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from construct.construct import ensure_base_model, ensure_extra_outputs
from construct.kv_cache_subgraph import ensure_kv_cache_extra_outputs, past_kv_input_names
from kv_helpers import (
    N_CAND, N_HEADS, HEAD_DIM, N_LAYERS, PREFIX_LENGTHS, VOCAB, cacheless_cfg, candidates, gpt2_cfg, hf_model,
    log, random_tokens, session,
)
from kv_helpers import kv_boundaries_session  # noqa: F401  (ensures the KV export exists)

RESULTS = Path(__file__).parent / "results"
WARMUP, ITERS = 3, 10
NP = {torch.float32: np.float32, torch.int64: np.int64}


def timed(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times), out


class GpuRunner:
    """ORT session with device-resident torch inputs/outputs (IOBinding + dlpack)."""

    def __init__(self, path):
        self.sess = session(path)
        self.io = self.sess.io_binding()
        self.out_names = [o.name for o in self.sess.get_outputs()]

    def __call__(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        io = self.io
        io.clear_binding_inputs()
        io.clear_binding_outputs()
        torch.cuda.synchronize()
        keep = []
        for name, t in tensors.items():
            t = t.contiguous()
            keep.append(t)
            if t.numel() == 0:                       # zero-length past: no device buffer to point at
                io.bind_cpu_input(name, np.zeros(tuple(t.shape), NP[t.dtype]))
            else:
                io.bind_input(name, "cuda", 0, NP[t.dtype], tuple(t.shape), t.data_ptr())
        for name in self.out_names:
            io.bind_output(name, "cuda", 0)
        self.sess.run_with_iobinding(io)
        io.synchronize_outputs()
        return {n: torch.utils.dlpack.from_dlpack(o._ortvalue.to_dlpack()) for n, o in zip(self.out_names, io.get_outputs())}


@pytest.fixture(scope="module")
def cfg():
    return gpt2_cfg()


@pytest.fixture(scope="module")
def hf():
    return hf_model()


@pytest.fixture(scope="module")
def kv_runner(cfg):
    kv_boundaries_session(cfg)
    path, out_name, n_layers = ensure_kv_cache_extra_outputs(cfg, 11, "mlp_out")  # boundary 24
    return GpuRunner(path), out_name


def empty_past_gpu(n):
    z = torch.zeros((n, N_HEADS, 0, HEAD_DIM), device="cuda")
    return {k: z for k in past_kv_input_names(N_LAYERS)}


def kv_pipeline(runner, out_name, prefix: torch.Tensor, cand: torch.Tensor):
    L = prefix.shape[1]

    def run():
        if L:
            pre = runner({
                "input_ids": prefix, "attention_mask": torch.ones((1, L), dtype=torch.int64, device="cuda"),
                "position_ids": torch.arange(L, device="cuda")[None], **empty_past_gpu(1),
            })
            past = {
                f"past_key_values.{n.split('.', 1)[1]}": pre[n].expand(N_CAND, -1, -1, -1)
                for n in pre if n.startswith("present.")
            }
        else:
            past = empty_past_gpu(N_CAND)
        return runner({
            "input_ids": cand[:, None], "attention_mask": torch.ones((N_CAND, L + 1), dtype=torch.int64, device="cuda"),
            "position_ids": torch.full((N_CAND, 1), L, device="cuda"), **past,
        })[out_name][:, 0]
    return run


def test_timing_kv_vs_cacheless_vs_hf(cfg, hf, kv_runner):
    runner, out_name = kv_runner
    rows = []
    for L in PREFIX_LENGTHS:
        prefix_np = random_tokens(1, L, seed=500 + L)
        cand_np = candidates()
        prefix = torch.from_numpy(prefix_np).cuda()
        cand = torch.from_numpy(cand_np).cuda()
        seqs = torch.cat([prefix.expand(N_CAND, -1), cand[:, None]], dim=1)

        # cacheless ONNX, only the final boundary as output (own dir: ensure_extra_outputs keeps one file per cfg)
        ccfg = cacheless_cfg(cfg, seq=L + 1, tag="_last")
        ensure_base_model(ccfg, verbose=False)
        cl = GpuRunner(ensure_extra_outputs(ccfg, [cfg.sublayers[-1].output]))

        t_kv, out_kv = timed(kv_pipeline(runner, out_name, prefix, cand))
        t_kv_pre, _ = timed(lambda: runner({
            "input_ids": prefix, "attention_mask": torch.ones((1, L), dtype=torch.int64, device="cuda"),
            "position_ids": torch.arange(L, device="cuda")[None], **empty_past_gpu(1),
        })) if L else (0.0, None)
        t_cl, out_cl = timed(lambda: next(iter(cl({"input_ids": seqs}).values()))[:, -1])

        with torch.no_grad():
            t_hf, out_hf = timed(lambda: hf.transformer(seqs, use_cache=False).last_hidden_state[:, -1])

            def hf_cached():
                if L:
                    past = hf.transformer(prefix, use_cache=True).past_key_values
                    past.batch_repeat_interleave(N_CAND)
                else:
                    past = None
                return hf.transformer(cand[:, None], past_key_values=past, use_cache=True).last_hidden_state[:, -1]
            t_hfc, out_hfc = timed(hf_cached)

        # correctness: the ONNX variants return the pre-ln_f stream, HF's last_hidden_state is post-ln_f
        ref = hf.transformer.ln_f
        for name, out in (("kv", out_kv), ("cacheless", out_cl)):
            assert torch.allclose(ref(out), out_hf, atol=2e-3, rtol=1e-4), f"{name} (L={L}) != hf"
        assert torch.allclose(out_hfc, out_hf, atol=2e-3, rtol=1e-4), f"hf_cache (L={L}) != hf"

        groups = VOCAB / N_CAND  # a full-vocabulary group is this many 768-candidate batches
        rows.append({
            "prefix_len": L, "n_candidates": N_CAND,
            "kv_onnx_ms": t_kv, "kv_prefix_ms": t_kv_pre, "cacheless_onnx_ms": t_cl, "hf_ms": t_hf, "hf_cache_ms": t_hfc,
            "kv_vs_cacheless_speedup": t_cl / t_kv, "kv_vs_hf_speedup": t_hf / t_kv,
            "full_vocab_group_s": {
                "kv_onnx": (t_kv_pre + groups * (t_kv - t_kv_pre)) / 1e3,
                "cacheless_onnx": groups * t_cl / 1e3,
                "hf": groups * t_hf / 1e3,
            },
        })
        log(f"L={L}: kv {t_kv:.2f} ms (prefix {t_kv_pre:.2f}) | cacheless {t_cl:.2f} | hf {t_hf:.2f} | hf+cache {t_hfc:.2f}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "timing.json").write_text(json.dumps(rows, indent=2))
    lines = [
        "# KV-cache vs cacheless vs HF timing (GPT-2, 768 candidate last tokens, boundary 24, fp32, 1x A40)",
        "",
        f"Median of {ITERS} runs after {WARMUP} warm-ups, ms per 768-candidate group. `kv_onnx` = prefix pass + continuation.",
        "",
        "| prefix | kv_onnx | (prefix only) | cacheless_onnx | hf | hf_cache | kv vs cacheless | kv vs hf | full-vocab group: kv / cacheless / hf (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        g = r["full_vocab_group_s"]
        lines.append(
            f"| {r['prefix_len']} | {r['kv_onnx_ms']:.2f} | {r['kv_prefix_ms']:.2f} | {r['cacheless_onnx_ms']:.2f} | {r['hf_ms']:.2f} | "
            f"{r['hf_cache_ms']:.2f} | {r['kv_vs_cacheless_speedup']:.1f}x | {r['kv_vs_hf_speedup']:.1f}x | "
            f"{g['kv_onnx']:.2f} / {g['cacheless_onnx']:.2f} / {g['hf']:.2f} |",
        )
    (RESULTS / "timing.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))

    by_len = {r["prefix_len"]: r for r in rows}
    assert by_len[32]["kv_onnx_ms"] < by_len[32]["cacheless_onnx_ms"]
