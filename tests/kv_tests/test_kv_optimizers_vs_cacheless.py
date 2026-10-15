"""Adversarial-sample optimisers (execute.optimization_loop: SGD, AdamW, GaussNewton, NewtonRaphson) on the
three adversarial losses, KV-cache last-token pullback vs the cacheless full-sequence pullback.

Problem: given a sequence prefix + last token with sublayer input x (and target = f(x)), find y for the last
token minimising the loss. KV: y is (B, 1, d), prefix K/V constant. Cacheless: y is (B, S, d) with the prefix
positions frozen at x (LastTokenOnly), so both optimise the same variable under the same loss. The loss
history and the final y must agree.
"""

import numpy as np
import pytest
import torch

from config import LossType
from construct.construct import ensure_subgraph_pullback
from construct.kv_cache_subgraph import KV_SUBLAYER_IN, ensure_kv_cache_sublayer_pullback
from execute.optimization_loop import GaussNewton, NewtonRaphson, SGD, AdamW, optimization_loop
from kv_helpers import (
    D_MODEL, LastTokenOnly, broadcast_past, cacheless_cfg, cacheless_sublayer_path, gpt2_cfg, hf_all_boundaries, hf_model,
    kv_boundaries_session, kv_prefix_past, kv_sublayer_feeds, random_tokens, session,
)
import dataclasses

B = 4
STEPS = 10
LOSSES = [
    LossType.AdversarialLoss_SquaredDiffLoss,
    LossType.AdversarialLoss_NormalizedSquaredDiffLoss,
    LossType.AdversarialLoss_EnvelopeDiffLoss,
]
f32 = np.float32
CPU = ["CPUExecutionProvider"]


def grad_providers(loss):
    """All losses run on the GPU. (The Div-based losses used to need the CPU provider because the generated
    gradient graph seeded a [B,1] Div output with a [1] initializer, which the CUDA DivGrad does not broadcast;
    construct.generate_artifact now expands the seed -- see test_cuda_divgrad_seed_regression.)"""
    return None
OPTIMIZERS = {
    "SGD": lambda: SGD(lr=1e-3, dtype=f32),
    "AdamW": lambda: AdamW(lr=1e-3, dtype=f32),
    "GaussNewton": lambda: GaussNewton(lr=1e-2, dtype=f32),
    "NewtonRaphson": lambda: NewtonRaphson(lr=1e-2, dtype=f32),
}
PREFIX_LENGTHS = (0, 8, 32)
J_IN = {"attention": 2, "mlp": 3}  # h.1 attention, h.1 MLP

# KV and cacheless graphs compute the same function but fuse/reduce in different orders (~1e-6 relative);
# the optimisers re-feed their own outputs for STEPS steps.
RTOL, ATOL = 5e-3, 1e-4


@pytest.fixture(scope="module")
def cfg():
    return gpt2_cfg()


@pytest.fixture(scope="module")
def hf():
    return hf_model()


@pytest.fixture(scope="module")
def kv(cfg):
    return kv_boundaries_session(cfg)


def _problem(cfg, hf, kv_sess, j_in, prefix_len, loss):
    """Everything both pipelines need, for B candidates after a shared prefix."""
    prefix = random_tokens(1, prefix_len, seed=400 + prefix_len)
    cand = np.array([11, 4096, 20000, 50000], dtype=np.int64)
    seqs = np.concatenate([np.broadcast_to(prefix, (B, prefix_len)), cand[:, None]], axis=1)
    x_full = hf_all_boundaries(hf, torch.from_numpy(seqs).cuda())[0][j_in].detach().cpu().numpy()  # (B, S, d)
    rng = np.random.default_rng(5)
    y_last0 = (x_full[:, -1] + 0.05 * rng.standard_normal((B, D_MODEL))).astype(f32)

    # --- KV
    sub, pb_path = ensure_kv_cache_sublayer_pullback(cfg, j_in, B, loss)
    past_b = broadcast_past(kv_prefix_past(kv_sess, prefix), B)
    feeds = kv_sublayer_feeds(sub, x_full[:, -1:], past_b, prefix_len)
    extra = {k: v for k, v in feeds.items() if k != KV_SUBLAYER_IN}
    fwd = session(sub.path)
    target_kv = fwd.run(None, feeds)[0]
    kv_args = dict(
        gradient_session=session(pb_path, grad_providers(loss)), x=x_full[:, -1:].astype(f32), target=target_kv,
        y0=y_last0[None, :, None, :], input_name=KV_SUBLAYER_IN, extra_feeds=extra,
    )

    # --- cacheless (own directory per loss: pullback file names don't encode the loss)
    ccfg = cacheless_cfg(cfg, seq=prefix_len + 1, batch=B, tag=f"_{loss.value}")
    cl_path, cl_sub = cacheless_sublayer_path(ccfg, j_in + 1)
    [cl_pb] = ensure_subgraph_pullback([(cl_path, dataclasses.replace(cl_sub, loss_type=loss))], force=False)
    target_cl = session(cl_path).run([cl_sub.output], {cl_sub.input: np.ascontiguousarray(x_full, dtype=f32)})[0]
    y0_full = x_full.astype(f32).copy()
    y0_full[:, -1] = y_last0
    # `x` only enters the normalized losses' denominator ||x - y||^2. With y == x on the (frozen) prefix
    # positions that is 0/0 = NaN there, and the NaN gradient rows leak into the last position through the
    # attention backward matmuls (NaN * 0). Shift x on the prefix so the denominator is nonzero while the
    # numerator ||f(y) - target||^2 stays exactly the prefix's own zero; the last position is untouched.
    x_cl = x_full.astype(f32).copy()
    x_cl[:, :-1] += 1.0
    cl_args = dict(
        gradient_session=session(cl_pb, grad_providers(loss)), x=x_cl, target=target_cl,
        y0=y0_full[None], input_name=cl_sub.input,
    )
    return kv_args, cl_args, x_full.astype(f32)


@pytest.mark.parametrize("prefix_len", PREFIX_LENGTHS)
@pytest.mark.parametrize("kind", J_IN)
@pytest.mark.parametrize("loss", LOSSES, ids=lambda l: l.value.replace("AdversarialLoss_", ""))
@pytest.mark.parametrize("opt_name", OPTIMIZERS)
def test_optimizer_kv_matches_cacheless(cfg, hf, kv, opt_name, loss, kind, prefix_len):
    kv_args, cl_args, x_full = _problem(cfg, hf, kv[0], J_IN[kind], prefix_len, loss)

    y_kv, loss_kv = optimization_loop(steps=STEPS, optimizer=OPTIMIZERS[opt_name](), **kv_args)
    y_cl, loss_cl = optimization_loop(steps=STEPS, optimizer=LastTokenOnly(OPTIMIZERS[opt_name]()), **cl_args)

    # cacheless: prefix of y untouched, i.e. same problem as the KV one
    np.testing.assert_array_equal(y_cl[0, :, :-1], cl_args["y0"][0, :, :-1])

    lk = loss_kv[:, 0, :, 0]    # (steps, B)
    lc = loss_cl[:, 0, :, -1]
    assert np.isfinite(lk).all() and np.isfinite(lc).all()
    yk, yc = y_kv[0, :, 0], y_cl[0, :, -1]

    if opt_name in ("SGD", "AdamW"):
        np.testing.assert_allclose(lk, lc, rtol=RTOL, atol=ATOL, err_msg="loss history kv vs cacheless")
        np.testing.assert_allclose(yk, yc, rtol=RTOL, atol=RTOL * np.abs(yc).max(), err_msg="final y")
    else:
        # Secant-based steps divide by (grad - grad_prev) / (y - y_prev) with a tiny y - y_prev, which amplifies the
        # ~1e-6 floating-point difference between the two graphs' gradients. So agreement is measured against the
        # method's own noise sensitivity: perturb the cacheless start point by float noise (1e-6 relative) and
        # require KV to deviate from cacheless by no more than a small multiple of that.
        # steps 0 and 1 happen before the first secant step, which is where the amplification starts
        np.testing.assert_allclose(lk[:2], lc[:2], rtol=RTOL, atol=ATOL, err_msg="first steps kv vs cacheless")
        cl_noisy = dict(cl_args)
        y0n = cl_args["y0"].copy()
        y0n[..., -1, :] *= 1.0 + 1e-6
        cl_noisy["y0"] = y0n
        y_n, loss_n = optimization_loop(steps=STEPS, optimizer=LastTokenOnly(OPTIMIZERS[opt_name]()), **cl_noisy)
        sens_loss = np.abs(loss_n[:, 0, :, -1] - lc).max()
        dev_loss = np.abs(lk - lc).max()
        sens_y = np.abs(y_n[0, :, -1] - yc).max()
        dev_y = np.abs(yk - yc).max()
        assert dev_loss <= 20 * sens_loss + ATOL, f"loss deviation {dev_loss:.3e} vs noise sensitivity {sens_loss:.3e}"
        assert dev_y <= 20 * sens_y + ATOL, f"y deviation {dev_y:.3e} vs noise sensitivity {sens_y:.3e}"

    # sanity: the first-order methods reduce the plain squared-difference loss (the Newton-style secant
    # steps and the normalized losses are not monotone at these step counts -- only agreement is tested)
    if opt_name in ("SGD", "AdamW") and loss == LossType.AdversarialLoss_SquaredDiffLoss:
        assert (lk[-1] < lk[0]).all(), f"{opt_name} did not reduce the loss: {lk[0]} -> {lk[-1]}"


@pytest.mark.parametrize("prefix_len", PREFIX_LENGTHS)
@pytest.mark.parametrize("kind", J_IN)
@pytest.mark.parametrize("loss", LOSSES, ids=lambda l: l.value.replace("AdversarialLoss_", ""))
def test_loss_and_gradient_match_at_fixed_y(cfg, hf, kv, loss, kind, prefix_len):
    """Optimiser-independent core claim: at the same y (only the last position differs from x), the
    last-position loss and d loss / d y_last are the same through the KV and the cacheless pullbacks."""
    kv_args, cl_args, _ = _problem(cfg, hf, kv[0], J_IN[kind], prefix_len, loss)
    rng = np.random.default_rng(9)
    for trial in range(3):
        step = (0.05 * rng.standard_normal((B, D_MODEL))).astype(f32) * (trial + 1)
        y_kv = kv_args["y0"][0].copy()          # (B, 1, d)
        y_cl = cl_args["y0"][0].copy()          # (B, S, d)
        y_kv[:, 0] += step
        y_cl[:, -1] += step

        def run(args, y):
            sess = args["gradient_session"]
            feed = {args["input_name"]: y, "target": args["target"], "x": args["x"], **args.get("extra_feeds", {})}
            names = {i.name for i in sess.get_inputs()}
            outs = [o.name for o in sess.get_outputs()]
            res = dict(zip(outs, sess.run(outs, {k: v for k, v in feed.items() if k in names})))
            return res[outs[0]], res[f"{args['input_name']}_grad"]

        l_kv, g_kv = run(kv_args, y_kv)
        l_cl, g_cl = run(cl_args, y_cl)
        assert np.isfinite(g_cl[:, -1]).all() and np.isfinite(g_kv).all()
        np.testing.assert_allclose(l_kv[:, 0], l_cl[:, -1], rtol=1e-4, atol=1e-6, err_msg="loss")
        np.testing.assert_allclose(
            g_kv[:, 0], g_cl[:, -1], rtol=1e-3, atol=1e-4 * np.abs(g_cl[:, -1]).max(), err_msg="gradient wrt y_last",
        )


@pytest.mark.parametrize("loss", [LossType.AdversarialLoss_NormalizedSquaredDiffLoss,
                                  LossType.AdversarialLoss_EnvelopeDiffLoss])
@pytest.mark.parametrize("kind", J_IN)
def test_cuda_divgrad_seed_regression(cfg, hf, kv, kind, loss):
    """CUDA and CPU gradients of the Div-terminated losses agree for every row (regression for the [1]-shaped
    loss-gradient seed that the CUDA DivGrad failed to broadcast, zeroing rows 1..)."""
    kv_args, _, _ = _problem(cfg, hf, kv[0], J_IN[kind], 8, loss)
    cuda_sess = kv_args["gradient_session"]
    cpu_sess = session(cuda_sess._model_path, CPU)
    y = kv_args["y0"][0].copy()
    y[:, 0] += 0.05

    def grad(sess):
        names = {i.name for i in sess.get_inputs()}
        feed = {kv_args["input_name"]: y, "target": kv_args["target"], "x": kv_args["x"], **kv_args["extra_feeds"]}
        return sess.run([f"{kv_args['input_name']}_grad"], {k: v for k, v in feed.items() if k in names})[0]

    g_cuda, g_cpu = grad(cuda_sess), grad(cpu_sess)
    assert np.abs(g_cpu[1:]).max() > 0
    np.testing.assert_allclose(g_cuda, g_cpu, rtol=1e-3, atol=1e-4)
