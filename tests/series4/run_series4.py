"""Series 4: the KV-cache pipeline (optimum export -> last-token sublayer subgraph -> pullback -> optimisation) on Hugging
Face transformers. Configs are the series 3 ones (tests/series3/configs/*.toml). Baseline is the HF model itself in torch:
hooks for boundary values / logits, autograd through an injected boundary for gradients and optimisation (full sequence
prefix + candidate, only the last position carries the optimised variable). One shared prefix, N candidate last tokens."""
import gc
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "series1"))
sys.path.insert(0, str(HERE))
import run_series1 as S1  # noqa: E402  (torch_loss, Adam, step_sgd)
import series4_plots as P  # noqa: E402
from construct import config as cconfig  # noqa: E402
from construct.config import LossType, load_config  # noqa: E402
from construct.kv_cache_export import ensure_kv_cache_model  # noqa: E402
from construct.kv_cache_subgraph import (  # noqa: E402
    KV_SUBLAYER_IN, KV_SUBLAYER_OUT, ensure_kv_cache_boundaries, ensure_kv_cache_sublayer_pullback, past_kv_input_names, present_kv_names,
)
from construct.make_config import find_attention, find_layers, first_tensor, residual_stream  # noqa: E402

RESULTS = ROOT / "results" / "series4"
WORK = HERE / "work"
cconfig.ONNX_DIR = WORK
PROVIDERS = [("CUDAExecutionProvider", {"use_tf32": "0"})]
PREFIX_LENGTHS = (0, 1, 8, 32)
OPT_LENGTHS = (0, 8, 32)
N_FWD, B_OPT, STEPS = 16, 8, 30
OPTS = {
    "SGD": (lambda: S1.step_sgd(0.01), lambda p: torch.optim.SGD([p], lr=0.01)),
    "AdamW": (lambda: S1.Adam(lr=0.002), lambda p: torch.optim.AdamW([p], lr=0.002, weight_decay=1e-2)),
}
ADV_LOSSES = [l for l in LossType if l != LossType.JacobianExtractorLoss]  # JacobianExtractor is the Jacobian extractor, not an objective


class Stop(Exception):
    pass


def session(path):
    assert "CUDAExecutionProvider" in ort.get_available_providers(), "series 4 needs a GPU node"
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return ort.InferenceSession(str(path), sess_options=so, providers=PROVIDERS)


def rel(a, b):
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))


# ------------------------------------------------------------------ KV side ----

def kv_dims(model_path):
    """(n_layers, n_kv_heads, head_dim) from the past_key_values inputs of the KV graph."""
    g = onnx.load(str(model_path), load_external_data=False).graph
    ins = {i.name: i for i in g.input}
    n = sum(1 for k in ins if k.startswith("past_key_values.") and k.endswith(".key"))
    dims = [d.dim_value for d in ins["past_key_values.0.key"].type.tensor_type.shape.dim]
    return n, dims[1], dims[3]


def kv_run(sess, ids, past, offset):
    names = [o.name for o in sess.get_outputs()]
    batch, n = ids.shape
    feeds = {"input_ids": ids.astype(np.int64), "attention_mask": np.ones((batch, offset + n), np.int64),
             "position_ids": np.broadcast_to(np.arange(offset, offset + n, dtype=np.int64), (batch, n)).copy(), **past}
    have = {i.name for i in sess.get_inputs()}
    return dict(zip(names, sess.run(None, {k: v for k, v in feeds.items() if k in have})))


def bcast(past, n):
    return {k: np.ascontiguousarray(np.broadcast_to(v, (n, *v.shape[1:]))) for k, v in past.items()}


def sublayer_feeds(sub, hidden, past_b, L):
    n = hidden.shape[0]
    feeds = {KV_SUBLAYER_IN: np.ascontiguousarray(hidden, dtype=np.float32)}
    for name in sub.extra_inputs:
        if name == "input_ids":
            feeds[name] = np.zeros((n, 1), np.int64)  # only its shape is read
        elif name == "attention_mask":
            feeds[name] = np.ones((n, L + 1), np.int64)
        elif name == "position_ids":
            feeds[name] = np.full((n, 1), L, np.int64)
        elif name.startswith("past_key_values."):
            feeds[name] = past_b[name]
        else:
            raise KeyError(f"don't know how to feed sublayer input {name!r}")
    return feeds


# ----------------------------------------------------------------- torch side ----

def sublayer_fn(model, layers, j_in, ids):
    """f(y) -> boundary j_in + 1, for y the (B, S, d) boundary j_in, by injecting into the real HF model.
    Even j_in (attention): inject at the layer input, read y + attention output. Odd j_in (MLP): the mid-layer residual
    is not a module boundary, so the attention output is replaced by y - layer_input (the residual add then yields y
    and the gradient reaches only y), and the layer output is read."""
    l = j_in // 2

    def f(y):
        box = {}

        def pre(m, a, kw):
            box["in"] = (a[0] if a else kw["hidden_states"]).detach()
            if j_in % 2 == 0:
                if a:
                    return (y,) + tuple(a[1:]), kw
                return a, {**kw, "hidden_states": y}

        def attn_post(m, a, o):
            if j_in % 2 == 0:
                box["o"] = y + first_tensor(o)
                raise Stop
            new = y - box["in"]
            return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new

        def layer_post(m, a, o):
            box["o"] = first_tensor(o)
            raise Stop

        hs = [layers[l].register_forward_pre_hook(pre, with_kwargs=True), find_attention(layers[l]).register_forward_hook(attn_post)]
        if j_in % 2 == 1:
            hs.append(layers[l].register_forward_hook(layer_post))
        try:
            model(ids, use_cache=False)
        except Stop:
            pass
        finally:
            for h in hs:
                h.remove()
        return box["o"]
    return f


def hf_boundaries(model, layers, ids):
    """Boundary tensors 0..2n of model(ids), each (B, S, d), and the logits."""
    cap, hooks = residual_stream(model, layers)
    with torch.no_grad():
        logits = model(ids, use_cache=False).logits
    for h in hooks:
        h.remove()
    vals = [cap["emb"]]
    for i in range(len(layers)):
        vals += [cap[("in", i)] + cap[("attn", i)], cap[("out", i)]]
    return vals, logits


# ----------------------------------------------------------------------- model ----

def run_model(cfg_path):
    cfg = load_config(str(cfg_path))
    name = cfg.name
    rd = RESULTS / name
    rd.mkdir(parents=True, exist_ok=True)
    nb = len(cfg.sublayers)  # boundaries 0..2n
    d, vocab = cfg.d_model, cfg.vocab_size
    kv_path = ensure_kv_cache_model(cfg.input_name, cfg.model, cfg.dir)
    n_layers, n_kvh, head_dim = kv_dims(kv_path)
    ops = sorted({nd.op_type for nd in onnx.load(str(kv_path), load_external_data=False).graph.node})
    bpath, bnames, _ = ensure_kv_cache_boundaries(cfg)
    kv_full, kv_bnd = session(kv_path), session(bpath)
    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch.float32).cuda().eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    _, layers = find_layers(model, n_layers)
    assert nb == 2 * n_layers + 1, (nb, n_layers)
    report = {"model": name, "hf_id": cfg.model, "n_layers": n_layers, "kv_heads": n_kvh, "head_dim": head_dim, "d_model": d, "ops": ops,
              "control_flow": bool({"If", "Loop", "Scan"} & set(ops)), "prefix_lengths": list(PREFIX_LENGTHS), "n_candidates": N_FWD,
              "forward": {}, "logits": {}, "jacobian": {}, "optimisation": {}, "errors": {}}

    def fail(kind, tag, e):
        report["errors"][f"{kind}|{tag}"] = (str(e).strip().splitlines() or [type(e).__name__])[-1][:300]
        print(f"  FAILED {kind} {tag}: {report['errors'][f'{kind}|{tag}']}", flush=True)

    def empty_past(batch):
        return {k: np.zeros((batch, n_kvh, 0, head_dim), np.float32) for k in past_kv_input_names(n_layers)}

    def prefix_past(prefix):
        """Batch-1 past K/V of `prefix` (1, L); L == 0 -> zero-length past (the graph cannot run an empty input_ids)."""
        if prefix.shape[1] == 0:
            return empty_past(1)
        out = kv_run(kv_bnd, prefix, empty_past(1), 0)
        return {f"past_key_values.{n.split('.', 1)[1]}": out[n] for n in present_kv_names(n_layers)}

    def tokens(seed, shape):
        return np.random.default_rng(seed).integers(0, vocab, size=shape, dtype=np.int64)

    # ---- forward: boundaries + logits of the last token, KV (prefix past + candidate) vs HF on the full sequence
    for L in PREFIX_LENGTHS:
        try:
            prefix = tokens(40 + L, (1, L))
            cand = np.sort(np.random.default_rng(7).choice(vocab, size=N_FWD, replace=False)).astype(np.int64)
            ids = np.concatenate([np.broadcast_to(prefix, (N_FWD, L)), cand[:, None]], axis=1)
            bv, hl = hf_boundaries(model, layers, torch.from_numpy(ids).cuda())
            past_b = bcast(prefix_past(prefix), N_FWD)
            out = kv_run(kv_bnd, cand[:, None], past_b, L)
            report["forward"][str(L)] = {"boundaries": [rel(out[bn].reshape(N_FWD, -1, d)[:, -1], bv[j][:, -1].cpu().numpy()) for j, bn in enumerate(bnames)]}
            lg = kv_run(kv_full, cand[:, None], past_b, L)["logits"][:, 0]
            report["logits"][str(L)] = rel(lg, hl[:, -1].cpu().numpy())
        except Exception as e:  # noqa: BLE001
            fail("forward", f"L{L}", e)

    # ---- Jacobian: full d x d last-token Jacobian of single sublayers, KV JacobianExtractor pullback vs torch autograd rows
    j_list = sorted({0, 1, 2, 3, 2 * n_layers - 2, 2 * n_layers - 1})
    for j in j_list:
        try:
            sub, pb = ensure_kv_cache_sublayer_pullback(cfg, j, LossType.JacobianExtractorLoss)
            ps = session(pb)
        except Exception as e:  # noqa: BLE001
            fail("jacobian-build", f"j{j}", e)
            continue
        for L in PREFIX_LENGTHS:
            try:
                seq = tokens(300 + L, (1, L + 1))
                bv, _ = hf_boundaries(model, layers, torch.from_numpy(seq).cuda())
                hidden = bv[j][0, -1].cpu().numpy()
                feeds = sublayer_feeds(sub, np.broadcast_to(hidden, (d, 1, d)), bcast(prefix_past(seq[:, :L]), d), L)
                feeds["mask"] = np.eye(d, dtype=np.float32)[:, None, :]
                grad, outk = ps.run([f"{KV_SUBLAYER_IN}_grad", KV_SUBLAYER_OUT], feeds)
                J_kv = grad[:, 0, :]
                y = bv[j].clone().requires_grad_(True)
                with torch.enable_grad():
                    out = sublayer_fn(model, layers, j, torch.from_numpy(seq).cuda())(y)[0, -1]
                    J_t = torch.stack([torch.autograd.grad(out[i], y, retain_graph=True)[0][0, -1] for i in range(d)]).cpu().numpy()
                assert np.abs(J_t).max() > 0 and np.isfinite(J_kv).all()
                report["jacobian"][f"{j}|{L}"] = {"rel_fro": float(np.linalg.norm(J_kv - J_t) / np.linalg.norm(J_t)),
                                                  "max_rel": rel(J_kv, J_t), "fwd_rel": rel(outk[0, 0], bv[j + 1][0, -1].cpu().numpy())}
            except Exception as e:  # noqa: BLE001
                fail("jacobian", f"j{j}|L{L}", e)

    # ---- optimisation: B candidates after a shared prefix, 30 steps of SGD / AdamW on the sublayer input of the last token
    for kind, j in (("attention", 2), ("mlp", 3)):
        for loss in ADV_LOSSES:
            try:
                sub, pb = ensure_kv_cache_sublayer_pullback(cfg, j, loss)
                ps = session(pb)
            except Exception as e:  # noqa: BLE001
                fail("optimisation-build", f"{kind}|{loss.value}", e)
                continue
            in_names = {i.name for i in ps.get_inputs()}
            outs = [o.name for o in ps.get_outputs()]
            loss_name = next(n for n in outs if n not in (KV_SUBLAYER_OUT, f"{KV_SUBLAYER_IN}_grad"))
            for L in OPT_LENGTHS:
                try:
                    rng = np.random.default_rng(100 + L)
                    prefix = tokens(40 + L, (1, L))
                    cand = rng.choice(vocab, size=B_OPT, replace=False).astype(np.int64)
                    ids = torch.from_numpy(np.concatenate([np.broadcast_to(prefix, (B_OPT, L)), cand[:, None]], axis=1)).cuda()
                    bv, _ = hf_boundaries(model, layers, ids)
                    y_in = bv[j][:, -1:].cpu().numpy()
                    pre_const = bv[j][:, :L]
                    sf = sublayer_fn(model, layers, j, ids)
                    f = lambda y: sf(torch.cat([pre_const, y], 1))[:, -1:]  # noqa: E731
                    y0 = (y_in + 0.02 * rng.standard_normal(y_in.shape)).astype(np.float32)
                    x = (y0 + 0.05).astype(np.float32)
                    with torch.no_grad():
                        target = f(torch.from_numpy(y_in + 0.1 * rng.standard_normal(y_in.shape).astype(np.float32)).cuda()).cpu().numpy()
                    mask = rng.standard_normal(y0.shape).astype(np.float32)
                    past_b = bcast(prefix_past(prefix), B_OPT)

                    def ort_step(y):
                        feeds = sublayer_feeds(sub, y, past_b, L)
                        feeds.update({k: v for k, v in {"x": x, "target": target, "mask": mask}.items() if k in in_names})
                        res = dict(zip(outs, ps.run(outs, feeds)))
                        return float(np.mean(res[loss_name])), res[f"{KV_SUBLAYER_IN}_grad"]

                    xt, tt, mt = (torch.from_numpy(a).cuda() for a in (x, target, mask))
                    for oname, (mk_np, mk_t) in OPTS.items():
                        step, y, ho = mk_np(), y0.copy(), []
                        for _ in range(STEPS):
                            l, g = ort_step(y)
                            ho.append(l); y = step(y, g, None)
                        yt = torch.tensor(y0.copy(), device="cuda", requires_grad=True)
                        topt, ht = mk_t(yt), []
                        for _ in range(STEPS):
                            topt.zero_grad()
                            with torch.enable_grad():
                                lt = S1.torch_loss(loss, yt, xt, tt, f(yt), mt)
                                ht.append(float(lt.mean())); lt.sum().backward()
                            topt.step()
                        ho, ht = np.array(ho), np.array(ht)
                        report["optimisation"][f"{kind}|L{L}|{oname}|{loss.value}"] = {
                            "ort": ho.tolist(), "torch": ht.tolist(), "traj_diff": float(np.abs(ho - ht).max() / (np.abs(ht).max() + 1e-12)),
                            "loss_decreased": bool(ho[-1] < ho[0] and ht[-1] < ht[0]), "finite": bool(np.isfinite(ho).all() and np.isfinite(ht).all())}
                except Exception as e:  # noqa: BLE001
                    fail("optimisation", f"{kind}|L{L}|{loss.value}", e)
    json.dump(report, open(rd / "results.json", "w"), indent=1)
    P.plot_model(rd, report)
    del model, kv_full, kv_bnd
    gc.collect(); torch.cuda.empty_cache()
    return report


if __name__ == "__main__":
    cfgs = sorted((HERE.parent / "series3" / "configs").glob("*.toml"))
    want = sys.argv[1:]
    for c in cfgs:
        if want and c.stem not in want:
            continue
        print("=== ", c.stem, flush=True)
        try:
            r = P.summarise(run_model(c))
            print(r, flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            print(f"MODEL FAILED {c.stem}: {type(e).__name__}: {str(e)[:300]}", flush=True)
    P.rebuild(RESULTS)
