"""Optimisation through the last-token KV pullback: a batch of B candidate last tokens after a shared prefix, each row's
sublayer input optimised for 30 steps of SGD / AdamW, for all four loss types and prefix lengths 0/8/32.
Baseline: torch autograd + torch.optim on the HF sublayer, with HF's own cached prefix K/V (the prefix positions are
constants, so batching rows is the same as optimising them independently).

  attention  boundary 2 -> 3 (layer 1)    x + attn(ln_1(x)) reading the prefix K/V
  mlp        boundary 3 -> 4 (layer 1)    x + mlp(ln_2(x))

Checked: ORT loss trajectories track torch's (max |L_ort - L_torch| / max |L_torch| < 1e-3), and for the three
adversarial losses with a reachable target the SGD/AdamW loss goes down. Also writes the trajectories to
results/series4/optimisation.json for the findings write-up."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import DynamicCache

from construct.config import LossType
from construct.kv_cache_subgraph import KV_SUBLAYER_IN, KV_SUBLAYER_OUT, ensure_kv_cache_sublayer_pullback
from kv_helpers import (
    D_MODEL, broadcast_past, gpt2_cfg, hf_all_boundaries, hf_model, kv_boundaries_session, kv_prefix_past, kv_sublayer_feeds,
    random_tokens, session,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "series1"))
import run_series1 as S1  # noqa: E402  (torch_loss, Adam, step_sgd)

B, STEPS, TOL = 8, 30, 1e-3
J_IN = {"attention": 2, "mlp": 3}
OPTS = {
    "SGD": (lambda: S1.step_sgd(0.01), lambda p: torch.optim.SGD([p], lr=0.01)),
    "AdamW": (lambda: S1.Adam(lr=0.002), lambda p: torch.optim.AdamW([p], lr=0.002, weight_decay=1e-2)),
}
RESULTS = Path(__file__).resolve().parents[2] / "results" / "series4"
_traj: dict = {}


@pytest.fixture(scope="module")
def cfg():
    return gpt2_cfg()


@pytest.fixture(scope="module")
def hf():
    return hf_model()


@pytest.fixture(scope="module")
def kv(cfg):
    return kv_boundaries_session(cfg)[0]


def torch_sublayer(hf, kind, past_kv, L):
    """f(y: (B,1,d)) -> sublayer output for the last token; past_kv: (k, v) of layer 1, batch-1, or None for L == 0."""
    blk = hf.transformer.h[1]
    if kind == "mlp":
        return lambda y: y + blk.mlp(blk.ln_2(y))

    def f(y):
        cache = DynamicCache()
        if past_kv is not None:
            cache.update(past_kv[0].expand(y.shape[0], -1, -1, -1), past_kv[1].expand(y.shape[0], -1, -1, -1), 1)
        a = blk.attn(blk.ln_1(y), past_key_values=cache, cache_position=torch.tensor([L], device=y.device), use_cache=True)[0]
        return y + a
    return f


# JacobianExtractorLoss is the Jacobian extractor (see test_kv_pullback_jacobian.py), not an optimisation objective
@pytest.mark.parametrize("loss", [l for l in LossType if l != LossType.JacobianExtractorLoss], ids=lambda l: l.value)
@pytest.mark.parametrize("opt", OPTS)
@pytest.mark.parametrize("kind", J_IN)
@pytest.mark.parametrize("L", [0, 8, 32])
def test_kv_optimisation_matches_torch(cfg, hf, kv, kind, L, opt, loss):
    j = J_IN[kind]
    rng = np.random.default_rng(100 + L)
    prefix = random_tokens(1, L, seed=40 + L)
    cand = rng.choice(50000, size=B, replace=False).astype(np.int64)
    ids = torch.from_numpy(np.concatenate([np.broadcast_to(prefix, (B, L)), cand[:, None]], axis=1)).cuda()
    with torch.no_grad():
        bnd, _ = hf_all_boundaries(hf, ids)
        y_in = bnd[j][:, -1:].cpu().numpy()                                    # (B, 1, d) last-token inputs
        full = hf.transformer(torch.from_numpy(prefix).cuda(), use_cache=True).past_key_values if L else None
    past_kv = (full.layers[1].keys, full.layers[1].values) if L else None
    f = torch_sublayer(hf, kind, past_kv, L)

    y0 = (y_in + 0.02 * rng.standard_normal(y_in.shape)).astype(np.float32)
    x = (y0 + 0.05).astype(np.float32)
    with torch.no_grad():
        target = f(torch.from_numpy(y_in + 0.1 * rng.standard_normal(y_in.shape).astype(np.float32)).cuda()).cpu().numpy()
    mask = rng.standard_normal(y0.shape).astype(np.float32)

    sub, pb_path = ensure_kv_cache_sublayer_pullback(cfg, j, B, loss)
    sess = session(pb_path)
    in_names = {i.name for i in sess.get_inputs()}
    outs = [o.name for o in sess.get_outputs()]
    loss_name = next(n for n in outs if n not in (KV_SUBLAYER_OUT, f"{KV_SUBLAYER_IN}_grad"))
    past_b = broadcast_past(kv_prefix_past(kv, prefix), B)

    def ort_step(y):
        feeds = kv_sublayer_feeds(sub, y, past_b, L)
        feeds.update({k: v for k, v in {"x": x, "target": target, "mask": mask}.items() if k in in_names})
        res = dict(zip(outs, sess.run(outs, feeds)))
        return float(np.mean(res[loss_name])), res[f"{KV_SUBLAYER_IN}_grad"]

    mk_np, mk_t = OPTS[opt]
    step, y, ho = mk_np(), y0.copy(), []
    for _ in range(STEPS):
        l, g = ort_step(y)
        ho.append(l)
        y = step(y, g, None)

    yt = torch.tensor(y0.copy(), device="cuda", requires_grad=True)
    topt, ht = mk_t(yt), []
    xt, tt, mt = (torch.from_numpy(a).cuda() for a in (x, target, mask))
    for _ in range(STEPS):
        topt.zero_grad()
        lt = S1.torch_loss(loss, yt, xt, tt, f(yt), mt)
        ht.append(float(lt.mean()))
        lt.sum().backward()
        topt.step()

    ho, ht = np.array(ho), np.array(ht)
    diff = float(np.abs(ho - ht).max() / np.abs(ht).max())
    _traj[f"{kind}|L{L}|{opt}|{loss.value}"] = {"ort": ho.tolist(), "torch": ht.tolist(), "traj_diff": diff}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "optimisation.json").write_text(json.dumps(_traj, indent=1))
    assert diff < TOL, f"trajectory diff {diff:.2e}"
    assert ho[-1] < ho[0], f"loss did not decrease: {ho[0]:.4g} -> {ho[-1]:.4g}"
