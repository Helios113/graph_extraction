"""Construction of the last-token KV pullbacks (construct.kv_cache_subgraph.ensure_kv_cache_sublayer_pullback)
for every loss type the project has: graph I/O contract, caching, and loss values vs a numpy
re-computation from the forward pass."""

import numpy as np
import pytest

from config import LossType
from construct.kv_cache_subgraph import (
    KV_SUBLAYER_IN, KV_SUBLAYER_OUT, ensure_kv_cache_sublayer, ensure_kv_cache_sublayer_pullback,
)
from kv_helpers import D_MODEL, broadcast_past, gpt2_cfg, kv_boundaries_session, kv_prefix_past, kv_sublayer_feeds, random_tokens, session

B = 4
LOSSES = list(LossType)
J_IN = {"attention": 2, "mlp": 3}  # h.1 attention (needs past K/V), h.1 MLP


@pytest.fixture(scope="module")
def cfg():
    return gpt2_cfg()


@pytest.mark.parametrize("loss_type", LOSSES, ids=lambda l: l.value)
@pytest.mark.parametrize("kind", J_IN)
def test_pullback_io_contract(cfg, kind, loss_type):
    sub, pb_path = ensure_kv_cache_sublayer_pullback(cfg, J_IN[kind], B, loss_type)
    sess = session(pb_path)
    inputs = {i.name: i.shape for i in sess.get_inputs()}
    outputs = [o.name for o in sess.get_outputs()]

    assert KV_SUBLAYER_IN in inputs and set(sub.extra_inputs) <= set(inputs)
    extra_loss_inputs = {
        LossType.MaskedSumLoss: {"mask"},
        LossType.AdversarialLoss_SquaredDiffLoss: {"target"},
        LossType.AdversarialLoss_NormalizedSquaredDiffLoss: {"target", "x"},
        LossType.AdversarialLoss_EnvelopeDiffLoss: {"target", "x"},
    }[loss_type]
    for name in extra_loss_inputs:
        assert inputs[name] == [B, 1, D_MODEL], name   # last token only
    assert KV_SUBLAYER_OUT in outputs and f"{KV_SUBLAYER_IN}_grad" in outputs
    assert len(outputs) == 3  # loss, forward output, input gradient


def test_pullback_is_cached(cfg):
    _, p1 = ensure_kv_cache_sublayer_pullback(cfg, 3, B, LossType.MaskedSumLoss)
    mtime = p1.stat().st_mtime_ns
    _, p2 = ensure_kv_cache_sublayer_pullback(cfg, 3, B, LossType.MaskedSumLoss)
    assert p1 == p2 and p2.stat().st_mtime_ns == mtime


def test_pullbacks_do_not_collide_across_batch_and_loss(cfg):
    paths = {
        ensure_kv_cache_sublayer_pullback(cfg, 3, b, l)[1]
        for b in (B, 2 * B) for l in (LossType.MaskedSumLoss, LossType.AdversarialLoss_SquaredDiffLoss)
    }
    assert len(paths) == 4


@pytest.mark.parametrize("kind", J_IN)
def test_squared_diff_loss_matches_numpy(cfg, kind):
    """loss = sum_d (target - f(y))^2 per row, with f the forward output the pullback also returns."""
    kv_sess, _ = kv_boundaries_session(cfg)
    j_in = J_IN[kind]
    sub, pb_path = ensure_kv_cache_sublayer_pullback(cfg, j_in, B, LossType.AdversarialLoss_SquaredDiffLoss)
    prefix = random_tokens(1, 5, seed=3)
    past_b = broadcast_past(kv_prefix_past(kv_sess, prefix), B)
    rng = np.random.default_rng(0)
    y = rng.standard_normal((B, 1, D_MODEL)).astype(np.float32)
    target = rng.standard_normal((B, 1, D_MODEL)).astype(np.float32)
    feeds = kv_sublayer_feeds(sub, y, past_b, 5)
    feeds["target"] = target
    sess = session(pb_path)
    names = [o.name for o in sess.get_outputs()]
    res = dict(zip(names, sess.run(names, feeds)))
    expected = ((target - res[KV_SUBLAYER_OUT]) ** 2).sum(-1)
    loss_name = next(n for n in names if n not in (KV_SUBLAYER_OUT, f"{KV_SUBLAYER_IN}_grad"))
    np.testing.assert_allclose(res[loss_name].reshape(expected.shape), expected, rtol=1e-4)
    assert res[f"{KV_SUBLAYER_IN}_grad"].shape == (B, 1, D_MODEL)
    assert np.isfinite(res[f"{KV_SUBLAYER_IN}_grad"]).all()
