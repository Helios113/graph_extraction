"""Single-sublayer last-token subgraphs (construct.kv_cache_subgraph.ensure_kv_cache_sublayer) vs
HF and vs the cacheless sublayer subgraphs, for prefix lengths 0/1/8/32.

Feeding the exact HF boundary-j input of the last token (plus the prefix K/V) must reproduce HF's
boundary-(j+1) output of that token. Sublayers: attention (0->1, 2->3, 22->23) and MLP (1->2, 3->4,
23->24), first / second / last layers.
"""

import numpy as np
import pytest
import torch

from construct.kv_cache_subgraph import KV_SUBLAYER_OUT, ensure_kv_cache_sublayer
from kv_helpers import (
    D_MODEL, N_CAND, PREFIX_LENGTHS, cacheless_cfg, cacheless_sublayer_path, candidates, gpt2_cfg,
    hf_all_boundaries, hf_model, kv_boundaries_session, kv_prefix_past, kv_sublayer_feeds, broadcast_past,
    random_tokens, session,
)

ATOL, RTOL = 1e-3, 1e-4
J_IN = [0, 1, 2, 3, 22, 23]


@pytest.fixture(scope="module")
def cfg():
    return gpt2_cfg()


@pytest.fixture(scope="module")
def hf():
    return hf_model()


@pytest.fixture(scope="module")
def kv(cfg):
    return kv_boundaries_session(cfg)


def test_sublayer_io_signature(cfg, kv):
    mlp = ensure_kv_cache_sublayer(cfg, 1)
    attn = ensure_kv_cache_sublayer(cfg, 2)
    assert mlp.layer_idx is None and mlp.extra_inputs == ()
    assert attn.layer_idx == 1
    assert {"past_key_values.1.key", "past_key_values.1.value"} <= set(attn.extra_inputs)
    assert not any(n.startswith("past_key_values.2") for n in attn.extra_inputs)  # other layers pruned


@pytest.mark.parametrize("prefix_len", PREFIX_LENGTHS)
@pytest.mark.parametrize("j_in", J_IN)
def test_sublayer_matches_hf_and_cacheless(cfg, hf, kv, prefix_len, j_in):
    kv_sess, _ = kv
    prefix = random_tokens(1, prefix_len, seed=200 + prefix_len)
    cand = candidates()
    seqs = np.concatenate([np.broadcast_to(prefix, (N_CAND, prefix_len)), cand[:, None]], axis=1)

    hf_b, _ = hf_all_boundaries(hf, torch.from_numpy(seqs).cuda())
    hf_in = hf_b[j_in].detach().cpu().numpy()    # (n, S, d)
    hf_out = hf_b[j_in + 1][:, -1].detach().cpu().numpy()

    # KV sublayer: last token only, prefix K/V as constants
    sub = ensure_kv_cache_sublayer(cfg, j_in)
    past_b = broadcast_past(kv_prefix_past(kv_sess, prefix), N_CAND)
    kv_out = session(sub.path).run(
        [KV_SUBLAYER_OUT], kv_sublayer_feeds(sub, hf_in[:, -1:], past_b, prefix_len),
    )[0]
    assert kv_out.shape == (N_CAND, 1, D_MODEL)
    np.testing.assert_allclose(kv_out[:, 0], hf_out, atol=ATOL, rtol=RTOL, err_msg="kv sublayer vs hf")

    # cacheless sublayer: whole sequence in, last position out
    ccfg = cacheless_cfg(cfg, seq=prefix_len + 1)
    cl_path, cl_sub = cacheless_sublayer_path(ccfg, j_in + 1)
    cl_out = session(cl_path).run([cl_sub.output], {cl_sub.input: np.ascontiguousarray(hf_in)})[0]
    np.testing.assert_allclose(cl_out[:, -1], hf_out, atol=ATOL, rtol=RTOL, err_msg="cacheless sublayer vs hf")
    np.testing.assert_allclose(kv_out[:, 0], cl_out[:, -1], atol=ATOL, rtol=RTOL, err_msg="kv vs cacheless sublayer")
