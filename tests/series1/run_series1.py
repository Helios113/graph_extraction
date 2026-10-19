"""Series 1: validity of export -> subgraph extraction -> pullback -> optimisation on simple
control-flow-free networks, against torch baselines. One results/series1/<model>/ per model."""
import os
import json
import sys
from pathlib import Path




import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from construct import config as cconfig  # noqa: E402
from construct.config import LossType, ModelConfig, SublayerConfig  # noqa: E402
from construct.construct import ensure_subgraph_pullback, ensure_subgraphs  # noqa: E402

RESULTS = ROOT / "results" / "series1"
WORK = ROOT / "tests" / "series1" / "work"
cconfig.ONNX_DIR = WORK  # all artifacts of this series live under tests/series1/work
B, D, STEPS = 8, 16, 100
LOSSES = list(LossType)
# ORT execution provider for the pipeline under test: CPU by default (series 1), SERIES_EP=cuda for GPU runs (TF32 off).
PROVIDERS = [("CUDAExecutionProvider", {"use_tf32": "0"})] if os.environ.get("SERIES_EP") == "cuda" else ["CPUExecutionProvider"]
torch.manual_seed(0)


def mlp_stage(act):
    return nn.Sequential(nn.Linear(D, D), act())


class Res(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.n = nn.LayerNorm(D) if norm else nn.Identity()
        self.f = nn.Sequential(nn.Linear(D, 2 * D), nn.SiLU(), nn.Linear(2 * D, D))

    def forward(self, x):
        return x + self.f(self.n(x))


def conv_stage():
    return nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.ReLU())


MODELS = {
    "mlp_tanh": ([mlp_stage(nn.Tanh) for _ in range(3)], (B, D)),
    "mlp_gelu_deep": ([mlp_stage(nn.GELU) for _ in range(6)], (B, D)),
    "resmlp": ([Res(False) for _ in range(3)], (B, D)),
    "resmlp_layernorm": ([Res(True) for _ in range(3)], (B, D)),
    "conv_relu": ([conv_stage() for _ in range(3)], (B, 4, 8, 8)),
    "mlp_softmax_mixer": ([nn.Sequential(nn.Linear(D, D), nn.Softmax(-1)) for _ in range(3)], (B, D)),
}


def torch_loss(kind, y, x, target, fy, mask):
    """Per-row losses mirroring construct/loss.py (sum over trailing axis only)."""
    if kind == LossType.JacobianExtractorLoss:
        return (fy * mask).sum()
    out = ((target - fy) ** 2).sum(-1)
    if kind == LossType.AdversarialLoss_SquaredDiffLoss:
        return out
    return out / ((x - y) ** 2).sum(-1)


def loss_scale(kind, mod, y, mask, lt):
    """Scale for the loss-value relative error. JacobianExtractor is a sum of signed terms that can cancel to ~0, so |loss|
    is a poor denominator: use sum|f * mask| (the condition scale of the sum). Other losses are positive per row."""
    if kind == LossType.JacobianExtractorLoss:
        with torch.no_grad():
            return float((mod(y) * torch.tensor(mask)).abs().sum())
    return float(np.abs(lt.detach().numpy()).max())


def jacobian_check(ps, outs, in_names, in_key, f, y, out_idx, extra_feed=None):
    """JacobianExtractorLoss is the Jacobian extractor: with a one-hot mask at flat output element j (same j in every batch row;
    rows are independent) the pullback gradient is row j of the Jacobian d out[j] / d input. Collect those rows for every
    j in out_idx and compare with torch autograd rows of the same Jacobian. Returns max|J_ort - J_torch| / max|J_torch|."""
    yt = torch.tensor(y, requires_grad=True)
    out = f(yt)
    bsz = out.shape[0]
    flat = out.reshape(bsz, -1)
    err = ref_max = 0.0
    for j in out_idx:
        mask = np.zeros(flat.shape, np.float32)
        mask[:, j] = 1.0
        feed = {k: v for k, v in {in_key: y, "mask": mask.reshape(out.shape), **(extra_feed or {})}.items() if k in in_names}
        gr = ps.run(outs, feed)[-1]
        gref = torch.autograd.grad(flat[:, j].sum(), yt, retain_graph=True)[0].numpy()
        err, ref_max = max(err, float(np.abs(gr - gref).max())), max(ref_max, float(np.abs(gref).max()))
    return err / (ref_max + 1e-12)


def step_sgd(lr):
    return lambda y, g, s: (y - lr * g)


class Adam:
    def __init__(self, lr=1e-2, b=(0.9, 0.999), eps=1e-8, wd=1e-2):
        self.lr, self.b, self.eps, self.wd, self.m, self.v, self.t = lr, b, eps, wd, 0, 0, 0

    def __call__(self, y, g, _=None):
        self.t += 1
        y = y - self.lr * self.wd * y
        self.m = self.b[0] * self.m + (1 - self.b[0]) * g
        self.v = self.b[1] * self.v + (1 - self.b[1]) * g * g
        mh, vh = self.m / (1 - self.b[0] ** self.t), self.v / (1 - self.b[1] ** self.t)
        return y - self.lr * mh / (np.sqrt(vh) + self.eps)


def sanitize_and_name(path, boundary_outputs):
    """Rename the input to h0 and the boundary tensors to h1..hN (final = hN); sanitise '/'.
    Recurses into If/Loop/Scan bodies, whose nodes reference outer-scope tensors by name."""
    m = onnx.load(str(path))
    ren = {m.graph.input[0].name: "h0"}
    for i, n in enumerate(boundary_outputs, 1):
        ren[n] = f"h{i}"

    def r(n):
        return ren.get(n, n.replace("/", "_").strip("_"))

    def rename_graph(g, top):
        for n in g.node:
            n.input[:] = [r(i) if i else i for i in n.input]
            n.output[:] = [r(o) for o in n.output]
            n.name = n.name.replace("/", "_")
            for a in n.attribute:
                if a.type == onnx.AttributeProto.GRAPH:
                    rename_graph(a.g, False)
                elif a.type == onnx.AttributeProto.GRAPHS:
                    for sg in a.graphs:
                        rename_graph(sg, False)
        if not top:  # body graphs: their own inputs/outputs are named too
            for v in list(g.input) + list(g.output):
                v.name = r(v.name)

    rename_graph(m.graph, True)
    m.graph.input[0].name = "h0"
    m.graph.output[0].name = r(m.graph.output[0].name)
    onnx.save(m, str(path))


def _pb_child(sub_paths, conn):
    try:
        conn.send(("ok", [str(p) for p in ensure_subgraph_pullback(sub_paths, force=True)]))
    except Exception as e:  # noqa: BLE001
        conn.send(("err", str(e)))


def pullback_isolated(sub_paths):
    """ensure_subgraph_pullback in a child process: ORT's gradient builder can segfault on some graphs (e.g. If), which
    would otherwise kill the whole run. A crash is raised as RuntimeError naming the exit code."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    pc, cc = ctx.Pipe()
    proc = ctx.Process(target=_pb_child, args=(sub_paths, cc))
    proc.start()
    proc.join()
    if pc.poll():
        kind, val = pc.recv()
        if kind == "ok":
            return [Path(v) for v in val]
        raise RuntimeError(val)
    raise RuntimeError(f"pullback generation crashed the process (exit code {proc.exitcode}; -11 = SIGSEGV)")


def run_model(name, stages, shape, series="series1", allow_cf=False, script=False, shift=0.0):
    """allow_cf: the exported graph may contain If/Loop/Scan. script: export torch.jit.script(net) (needed
    to keep Python control flow). shift: added to every input tensor (selects which branch data-dependent
    control flow takes). A failing extraction / pullback is recorded under report["errors"], not raised."""
    global RESULTS, WORK
    RESULTS, WORK = ROOT / "results" / series, ROOT / "tests" / series / "work"
    cconfig.ONNX_DIR = WORK
    rd = RESULTS / name
    rd.mkdir(parents=True, exist_ok=True)
    net = nn.Sequential(*stages).eval()
    N = len(stages)
    x0 = torch.randn(*shape) + shift
    wd = WORK / name
    wd.mkdir(parents=True, exist_ok=True)
    raw = wd / "raw.onnx"
    export_net = torch.jit.script(net) if script else net
    torch.onnx.export(export_net, (x0,), str(raw), input_names=["x"], output_names=["output"], dynamo=False)
    m = onnx.load(str(raw))
    ops = sorted({n.op_type for n in m.graph.node})
    if not allow_cf:
        assert not {"If", "Loop", "Scan"} & set(ops)
    with torch.no_grad():
        bvals, h = [], x0
        for s in stages:
            h = s(h)
            bvals.append(h.numpy())
    all_out = [o for n in m.graph.node for o in n.output]
    mm = onnx.load(str(raw))
    mm.graph.output.extend([onnx.ValueInfoProto(name=o) for o in all_out if o != "output"])
    sess = ort.InferenceSession(mm.SerializeToString(), providers=PROVIDERS)
    names = [o.name for o in sess.get_outputs()]
    vals = dict(zip(names, sess.run(names, {"x": x0.numpy()})))
    bnames = []
    for k, bv in enumerate(bvals):
        cands = [o for o in all_out if o in vals and vals[o].shape == bv.shape and np.allclose(vals[o], bv, atol=1e-5)]
        assert cands, f"no tensor matches stage {k}"
        bnames.append(cands[-1] if k < N - 1 else "output")
    sanitize_and_name(raw, bnames)

    spans = [(i, i + 1) for i in range(N)] + [(0, N)]
    report = {"model": name, "ops": ops, "stages": N, "shape": list(shape), "extraction": {}, "pullback": {}, "optimisation": {}, "jacobian": {}, "errors": {}}
    h_t = [x0] + [torch.tensor(v) for v in bvals]
    rng = np.random.default_rng(1)
    sh = np.float32(shift)

    def sub_module(a, b):
        return nn.Sequential(*stages[a:b])

    def fail(kind, tag, e):
        report["errors"][f"{kind}|{tag}"] = (str(e).strip().splitlines() or [type(e).__name__])[-1][:300]
        print(f"  FAILED {kind} {tag}: {report['errors'][f'{kind}|{tag}']}", flush=True)

    for loss in LOSSES:
        for a, b in spans:
            tag = f"h{a}->h{b}"
            mod = sub_module(a, b)
            x_in = h_t[a].numpy()
            cfg = ModelConfig(name=f"{name}/{loss.value}/h{a}h{b}", model="", vocab_size=0, batch=B, seq=1, d_model=D, input_name="h0",
                              sublayers=[SublayerConfig(loss_type=loss, input_shape=list(shape), input=f"h{a}", output=f"h{b}")])
            cfg.dir.mkdir(parents=True, exist_ok=True)
            cfg.base_model_path.write_bytes(raw.read_bytes())
            try:
                sub_paths = ensure_subgraphs(cfg, force=True)
                sp = sub_paths[0][0]
                if loss == LOSSES[0]:
                    s = ort.InferenceSession(str(sp), providers=PROVIDERS)
                    got = s.run(None, {f"h{a}": x_in})[0]
                    with torch.no_grad():
                        ref = mod(h_t[a]).numpy()
                    report["extraction"][tag] = float(np.abs(got - ref).max())
            except Exception as e:  # noqa: BLE001
                fail("extraction", tag, e)
                if (a, b) != (0, N):
                    continue
                # whole-network span: the exported graph already is h0 -> hN, so use it directly to still exercise
                # pullback generation / gradients through the control flow (extraction itself stays recorded as failed)
                sp_path = cfg.sub_block_path / f"h{a}.h{b}.subgraph.forward.onnx"
                sp_path.write_bytes(raw.read_bytes())
                sub_paths = [(sp_path, cfg.sublayers[0])]
                report["errors"]["note|" + tag] = "extraction failed; pullback built from the full exported graph instead"
            try:
                pbp = (pullback_isolated(sub_paths) if allow_cf else ensure_subgraph_pullback(sub_paths, force=True))[0]
                ps = ort.InferenceSession(str(pbp), providers=PROVIDERS)
            except Exception as e:  # noqa: BLE001
                fail(f"pullback:{loss.value}", tag, e)
                continue
            outs = [o.name for o in ps.get_outputs()]
            assert outs[0] != f"h{a}_grad" and outs[-1] == f"h{a}_grad", outs
            y = (rng.standard_normal(x_in.shape) + 0).astype(np.float32) + (sh if a == 0 else 0)
            xx = rng.standard_normal(x_in.shape).astype(np.float32)
            tgt = rng.standard_normal(x_in.shape).astype(np.float32)
            mask = rng.standard_normal(x_in.shape).astype(np.float32)
            in_names = {i.name for i in ps.get_inputs()}
            feed = {k: v for k, v in {f"h{a}": y, "x": xx, "target": tgt, "mask": mask}.items() if k in in_names}
            lo, _, gr = ps.run(outs, feed)
            yt = torch.tensor(y, requires_grad=True)
            lt = torch_loss(loss, yt, torch.tensor(xx), torch.tensor(tgt), mod(yt), torch.tensor(mask))
            lt.sum().backward()
            gref = yt.grad.numpy()
            rel = float(np.abs(gr - gref).max() / (np.abs(gref).max() + 1e-12))
            lerr = float(np.abs(np.asarray(lo).reshape(lt.shape) - lt.detach().numpy()).max() / (loss_scale(loss, mod, yt, mask, lt) + 1e-12))
            report["pullback"][f"{loss.value}|{tag}"] = {"grad_rel_err": rel, "loss_rel_err": lerr}

            if loss == LossType.JacobianExtractorLoss:
                # JacobianExtractor extracts the full Jacobian; it is not an optimisation objective (linear, no minimum)
                with torch.no_grad():
                    n_out = int(np.prod(mod(torch.tensor(y)).shape[1:]))
                report.setdefault("jacobian", {})[tag] = jacobian_check(ps, outs, in_names, f"h{a}", mod, y, range(n_out))
                continue
            if (a, b) != (0, N):
                continue
            ystar = rng.standard_normal(x_in.shape).astype(np.float32) + sh
            with torch.no_grad():
                tgt = mod(torch.tensor(ystar)).numpy()  # reachable target
            y0 = (ystar + 0.5 * rng.standard_normal(x_in.shape)).astype(np.float32)
            for oname, mk_np, mk_t in [("SGD", lambda: step_sgd(0.05), lambda p: torch.optim.SGD([p], lr=0.05)),
                                       ("AdamW", lambda: Adam(lr=0.02), lambda p: torch.optim.AdamW([p], lr=0.02, weight_decay=1e-2))]:
                feed_c = {"x": (y0 + 0.3).astype(np.float32), "target": tgt, "mask": mask}
                step = mk_np()
                yo, hist_o = y0.copy(), []
                for _ in range(STEPS):
                    feed = {k: v for k, v in {f"h{a}": yo, **feed_c}.items() if k in in_names}
                    lo, _, gr = ps.run(outs, feed)
                    hist_o.append(float(np.mean(lo)))
                    yo = step(yo, gr, None)
                yt = torch.tensor(y0.copy(), requires_grad=True)
                opt = mk_t(yt)
                hist_t = []
                for _ in range(STEPS):
                    opt.zero_grad()
                    l = torch_loss(loss, yt, torch.tensor(feed_c["x"]), torch.tensor(tgt), mod(yt), torch.tensor(mask))
                    hist_t.append(float(l.mean()))
                    l.sum().backward()
                    opt.step()
                report["optimisation"][f"{loss.value}|{oname}"] = {
                    "ort": hist_o, "torch": hist_t,
                    "max_rel_traj_diff": float(np.max(np.abs(np.array(hist_o) - np.array(hist_t)) / (np.abs(hist_t) + 1e-8))),
                    "final_y_max_abs_diff": float(np.abs(yo - yt.detach().numpy()).max()),
                }
    json.dump(report, open(rd / "results.json", "w"), indent=1)
    plot(rd, report)
    return report


from series1_plots import plot_model as plot  # noqa: E402


if __name__ == "__main__":
    for n in sys.argv[1:] or list(MODELS):
        print("=== ", n, flush=True)
        run_model(n, *MODELS[n])
    # summary.json + summary figure are rebuilt from every results.json present
    import series1_plots
    series1_plots.rebuild(ROOT / "results" / "series1")
