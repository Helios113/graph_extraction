"""Series 3: the export -> subgraph extraction -> pullback -> optimisation pipeline on Hugging Face transformers.
Configs come from construct/make_config.py (tests/series3/configs/*.toml). Baselines are torch hooks / autograd on the
HF model itself: boundary values and the logits for the forward checks, and gradients through the injected layer input."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModelForCausalLM

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "series1"))
import run_series1 as S1  # noqa: E402  (torch_loss, Adam, step_sgd, LOSSES)
import series1_plots as P  # noqa: E402
from construct import config as cconfig  # noqa: E402
from construct.config import SublayerConfig, load_config  # noqa: E402
from construct.construct import ensure_base_model, ensure_subgraph_pullback, ensure_subgraphs  # noqa: E402
from construct.make_config import find_attention, find_layers  # noqa: E402

RESULTS = ROOT / "results" / "series3"
WORK = HERE / "work"
cconfig.ONNX_DIR = WORK
B, SEQ, STEPS = 2, 8, 30
P.TOL.update(static=1e-4, traj=1e-3)  # deep fp32 transformers: 1e-4 (the single-layer-MLP series used 1e-5)


class Stop(Exception):
    pass


def layer_fn(model, layers, l0, l1, ids):
    """torch function h_in(layer l0 input) -> output of layer l1-1, by injecting into the real model."""
    def f(y):
        box = {}

        def pre(m, a, kw):
            if a:
                return (y,) + tuple(a[1:]), kw
            kw = dict(kw); kw["hidden_states"] = y
            return a, kw

        def post(m, a, o):
            box["o"] = o[0] if isinstance(o, tuple) else o
            raise Stop

        h1 = layers[l0].register_forward_pre_hook(pre, with_kwargs=True)
        h2 = layers[l1 - 1].register_forward_hook(post)
        try:
            model(ids)
        except Stop:
            pass
        finally:
            h1.remove(); h2.remove()
        return box["o"]
    return f


def boundary_values(model, layers, ids):
    from construct.make_config import residual_stream
    cap, hooks = residual_stream(model, layers)
    with torch.no_grad():
        logits = model(ids).logits
    for h in hooks:
        h.remove()
    vals = [cap["emb"]]
    for i in range(len(layers)):
        vals += [cap[("in", i)] + cap[("attn", i)], cap[("out", i)]]
    return [v.numpy() for v in vals], logits.numpy()


def run_model(cfg_path):
    cfg0 = load_config(str(cfg_path))
    name = cfg0.name
    rd = RESULTS / name
    rd.mkdir(parents=True, exist_ok=True)
    n = len(cfg0.sublayers) // 2
    bnames = [s.output for s in cfg0.sublayers]  # boundary k = bnames[k]
    cfg = replace(cfg0, name=name, batch=B, seq=SEQ, batch_by_node={}, random_init=False)
    ensure_base_model(cfg)
    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch.float32).eval()
    model.config.use_cache = False
    _, layers = find_layers(model, n)
    rng = np.random.default_rng(0)
    ids = torch.tensor(rng.integers(0, cfg.vocab_size, (B, SEQ)))
    bvals, logits = boundary_values(model, layers, ids)
    base = onnx.load(str(cfg.base_model_path), load_external_data=False)
    ops = sorted({nd.op_type for nd in base.graph.node})
    sess = ort.InferenceSession(str(cfg.base_model_path), providers=S1.PROVIDERS)
    got = sess.run(None, {"input_ids": ids.numpy()})[0]
    report = {"model": name, "hf_id": cfg.model, "ops": ops, "stages": n, "shape": [B, SEQ, cfg.d_model], "control_flow": bool({"If", "Loop", "Scan"} & set(ops)),
              "full_model_logits_max_abs_err": float(np.abs(got - logits).max()), "logits_scale": float(np.abs(logits).max()),
              "extraction_label": "max |ORT − torch| / max |torch|", "extraction": {}, "pullback": {}, "optimisation": {}, "jacobian": {}, "errors": {}}
    # boundary k: 0 = embedding, 2l+1 = post-attn(l), 2l+2 = post-MLP(l). layer l spans boundaries (2l, 2l+2).
    spans = [(0, 1), (1, 2), (0, 2), (2, 4) if n > 1 else None, (2 * n - 2, 2 * n), (0, 2 * min(3, n))]
    spans = list(dict.fromkeys(s for s in spans if s))
    shape = [B, SEQ, cfg.d_model]

    def fail(kind, tag, e):
        report["errors"][f"{kind}|{tag}"] = (str(e).strip().splitlines() or [type(e).__name__])[-1][:300]
        print(f"  FAILED {kind} {tag}: {report['errors'][f'{kind}|{tag}']}", flush=True)

    last = max(spans, key=lambda s: (s[1] - s[0], -s[0]))  # optimised span: the multi-layer one
    for loss in S1.LOSSES:
        for a, b in spans:
            tag = f"h{a}->h{b}"
            layer_aligned = a % 2 == 0 and b % 2 == 0
            sub = SublayerConfig(loss_type=loss, input_shape=shape, input=bnames[a], output=bnames[b])
            c = replace(cfg, name=f"{name}/{loss.value}/h{a}h{b}", sublayers=[sub])
            c.dir.mkdir(parents=True, exist_ok=True)
            c.base_model_path.write_bytes(cfg.base_model_path.read_bytes())
            for ext in cfg.base_model_path.parent.glob("*.data"):
                (c.dir / ext.name).write_bytes(ext.read_bytes()) if not (c.dir / ext.name).exists() else None
            try:
                sp = ensure_subgraphs(c, force=True)
                feeds_extra = {}
                s = ort.InferenceSession(str(sp[0][0]), providers=S1.PROVIDERS)
                for i in s.get_inputs():
                    if i.name == bnames[a]:
                        continue
                    if i.name == "input_ids":
                        feeds_extra[i.name] = ids.numpy()
                    else:
                        raise RuntimeError(f"extracted subgraph needs unsupported extra input {i.name}")
                if loss == S1.LOSSES[0]:
                    out = s.run(None, {bnames[a]: bvals[a], **feeds_extra})[0]
                    report["extraction"][tag] = float(np.abs(out - bvals[b]).max() / (np.abs(bvals[b]).max() + 1e-12))
            except Exception as e:  # noqa: BLE001
                fail("extraction", tag, e)
                continue
            try:
                pb = ensure_subgraph_pullback(sp, force=True)[0]
                ps = ort.InferenceSession(str(pb), providers=S1.PROVIDERS)
            except Exception as e:  # noqa: BLE001
                fail(f"pullback:{loss.value}", tag, e)
                continue
            if not layer_aligned:
                continue  # no injectable torch boundary mid-layer: forward-only + pullback construction
            outs = [o.name for o in ps.get_outputs()]
            in_names = {i.name for i in ps.get_inputs()}
            f = layer_fn(model, layers, a // 2, b // 2, ids)
            r = np.random.default_rng(1)
            y = (bvals[a] + 0.1 * r.standard_normal(bvals[a].shape)).astype(np.float32)
            xx, tgt, mask = (r.standard_normal(y.shape).astype(np.float32) for _ in range(3))
            feed = {k: v for k, v in {bnames[a]: y, "x": xx, "target": tgt, "mask": mask, **feeds_extra}.items() if k in in_names}
            lo, _, gr = ps.run(outs, feed)
            yt = torch.tensor(y, requires_grad=True)
            lt = S1.torch_loss(loss, yt, torch.tensor(xx), torch.tensor(tgt), f(yt), torch.tensor(mask))
            lt.sum().backward()
            gref = yt.grad.numpy()
            report["pullback"][f"{loss.value}|{tag}"] = {
                "grad_rel_err": float(np.abs(gr - gref).max() / (np.abs(gref).max() + 1e-12)),
                "loss_rel_err": float(np.abs(np.asarray(lo).reshape(lt.shape) - lt.detach().numpy()).max() / (S1.loss_scale(loss, f, yt, mask, lt) + 1e-12))}
            if loss.value == "JacobianExtractorLoss":
                # JacobianExtractor extracts the Jacobian (not an optimisation objective): full last-token Jacobian
                # d out[last token, :] / d input (d_model rows) against torch autograd rows
                idx = range((SEQ - 1) * cfg.d_model, SEQ * cfg.d_model)
                report["jacobian"][tag] = S1.jacobian_check(ps, outs, in_names, bnames[a], f, y, idx, feeds_extra)
                continue
            if (a, b) != last:
                continue
            with torch.no_grad():
                tgt2 = f(torch.tensor(bvals[a] + 0.1 * r.standard_normal(y.shape).astype(np.float32))).numpy()  # reachable target
            y0 = (bvals[a] + 0.02 * r.standard_normal(y.shape)).astype(np.float32)
            for oname, mk_np, mk_t in [("SGD", lambda: S1.step_sgd(0.01), lambda p: torch.optim.SGD([p], lr=0.01)),
                                       ("AdamW", lambda: S1.Adam(lr=0.002), lambda p: torch.optim.AdamW([p], lr=0.002, weight_decay=1e-2))]:
                fc = {"x": (y0 + 0.05).astype(np.float32), "target": tgt2, "mask": mask}
                step, yo, ho = mk_np(), y0.copy(), []
                for _ in range(STEPS):
                    lo, _, gr = ps.run(outs, {k: v for k, v in {bnames[a]: yo, **fc, **feeds_extra}.items() if k in in_names})
                    ho.append(float(np.mean(lo))); yo = step(yo, gr, None)
                yt = torch.tensor(y0.copy(), requires_grad=True); opt = mk_t(yt); ht = []
                for _ in range(STEPS):
                    opt.zero_grad()
                    l = S1.torch_loss(loss, yt, torch.tensor(fc["x"]), torch.tensor(tgt2), f(yt), torch.tensor(mask))
                    ht.append(float(l.mean())); l.sum().backward(); opt.step()
                report["optimisation"][f"{loss.value}|{oname}"] = {"ort": ho, "torch": ht, "max_rel_traj_diff": float(np.max(np.abs(np.array(ho) - np.array(ht)) / (np.abs(ht) + 1e-8)))}
    json.dump(report, open(rd / "results.json", "w"), indent=1)
    P.plot_model(rd, report)
    return report


if __name__ == "__main__":
    cfgs = sorted((HERE / "configs").glob("*.toml"))
    want = sys.argv[1:]
    for c in cfgs:
        if want and c.stem not in want:
            continue
        print("=== ", c.stem, flush=True)
        try:
            r = run_model(c)
            print({k: r[k] for k in ("full_model_logits_max_abs_err", "logits_scale")}, "extraction", max(r["extraction"].values(), default=None),
                  "grad", max((v["grad_rel_err"] for v in r["pullback"].values()), default=None), "errors", len(r["errors"]), flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            print(f"MODEL FAILED {c.stem}: {type(e).__name__}: {str(e)[:300]}", flush=True)
    P.rebuild(RESULTS)
