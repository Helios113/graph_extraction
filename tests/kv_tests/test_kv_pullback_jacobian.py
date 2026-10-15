"""Last-token Jacobian d out[pos]/d in[pos] of a sublayer through the KV pullback (prefix K/V as
constants, MaskedSum loss with one-hot rows) vs the cacheless pullback (full sequence in, last position
read) vs torch autograd on the HF model, for prefix lengths 0/1/8/32.

Sublayers: attention 0->1, 2->3, 22->23 (the Jacobian includes the token's own K/V path), MLP 1->2, 3->4, 23->24.
"""

import numpy as np
import pytest
import torch

from kv_helpers import (
    N_CAND, PREFIX_LENGTHS, cacheless_cfg, cacheless_jacobian, gpt2_cfg, hf_all_boundaries, hf_jacobian, hf_model,
    kv_boundaries_session, kv_jacobian, random_tokens,
)

J_IN = [0, 1, 2, 3, 22, 23]
RTOL = 2e-3
ATOL_FRAC = 2e-4  # atol as a fraction of max|J| (Jacobian entries span orders of magnitude across layers)


@pytest.fixture(scope="module")
def cfg():
    return gpt2_cfg()


@pytest.fixture(scope="module")
def hf():
    return hf_model()


@pytest.fixture(scope="module")
def kv(cfg):
    return kv_boundaries_session(cfg)


def close(a, b, what):
    np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL_FRAC * np.abs(b).max(), err_msg=what)


@pytest.mark.parametrize("prefix_len", PREFIX_LENGTHS)
@pytest.mark.parametrize("j_in", J_IN)
def test_last_token_jacobian(cfg, hf, kv, prefix_len, j_in):
    kv_sess, _ = kv
    rng_seed = 300 + prefix_len
    seq = random_tokens(1, prefix_len + 1, seed=rng_seed)  # prefix + last token
    prefix = seq[:, :prefix_len]

    hf_b, _ = hf_all_boundaries(hf, torch.from_numpy(seq).cuda())
    in_act = hf_b[j_in][0].detach().cpu().numpy()  # (S, d): full-sequence input at the sublayer boundary

    J_kv, out_kv = kv_jacobian(cfg, kv_sess, j_in, in_act[-1], prefix)
    ccfg = cacheless_cfg(cfg, seq=prefix_len + 1)
    J_cl, out_cl = cacheless_jacobian(ccfg, j_in + 1, in_act)
    J_hf = hf_jacobian(hf, torch.from_numpy(seq).cuda(), j_in)

    out_hf = hf_b[j_in + 1][0, -1].detach().cpu().numpy()
    close(out_kv, out_hf, "forward output kv vs hf")
    close(out_cl, out_hf, "forward output cacheless vs hf")

    assert J_kv.shape == J_cl.shape == J_hf.shape == (768, 768)
    assert np.abs(J_hf).max() > 0.1 and np.isfinite(J_kv).all()  # non-degenerate
    for other, name in ((J_cl, "cacheless"), (J_hf, "hf")):
        rel = np.linalg.norm(J_kv - other) / np.linalg.norm(other)
        assert rel < 1e-3, f"kv vs {name}: relative Frobenius error {rel:.2e}"
    close(J_kv, J_cl, f"kv vs cacheless Jacobian (j_in={j_in}, prefix={prefix_len})")
    close(J_kv, J_hf, f"kv vs hf autograd Jacobian (j_in={j_in}, prefix={prefix_len})")
    close(J_cl, J_hf, f"cacheless vs hf autograd Jacobian (j_in={j_in}, prefix={prefix_len})")
