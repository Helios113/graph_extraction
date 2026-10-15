"""Last-token activations at all 25 residual boundaries, for prefix lengths 0/1/8/32:
KV-cache model (prefix pass once + batched length-1 continuation) vs the HF torch model vs the
cacheless static-shape ONNX export. 768 candidate last tokens per prefix.
"""

import numpy as np
import pytest
import torch

from kv_helpers import (
    D_MODEL, N_CAND, PREFIX_LENGTHS, candidates, cacheless_boundaries_session, gpt2_cfg, hf_all_boundaries,
    hf_model, kv_boundaries_session, kv_last_token_boundaries, random_tokens,
)

ATOL, RTOL = 1e-3, 1e-4  # massive-activation dims reach |x| ~ 400


@pytest.fixture(scope="module")
def cfg():
    return gpt2_cfg()


@pytest.fixture(scope="module")
def hf():
    return hf_model()


@pytest.fixture(scope="module")
def kv(cfg):
    return kv_boundaries_session(cfg)


@pytest.mark.parametrize("prefix_len", PREFIX_LENGTHS)
def test_all_boundaries_match(cfg, hf, kv, prefix_len):
    kv_sess, kv_names = kv
    prefix = random_tokens(1, prefix_len, seed=100 + prefix_len)
    cand = candidates()
    seqs = np.concatenate([np.broadcast_to(prefix, (N_CAND, prefix_len)), cand[:, None]], axis=1)

    kv_out = kv_last_token_boundaries(kv_sess, kv_names, prefix, cand)

    cl_sess, ccfg = cacheless_boundaries_session(cfg, seq=prefix_len + 1)
    cl_raw = cl_sess.run(None, {"input_ids": seqs})
    cl_out = {j: cl_raw[[o.name for o in cl_sess.get_outputs()].index(s.output)][:, -1]
              for j, s in enumerate(ccfg.sublayers)}

    hf_raw, _ = hf_all_boundaries(hf, torch.from_numpy(seqs).cuda())
    hf_out = {j: t[:, -1].detach().cpu().numpy() for j, t in hf_raw.items()}

    assert sorted(kv_out) == sorted(cl_out) == sorted(hf_out) == list(range(25))
    for j in range(25):
        assert kv_out[j].shape == (N_CAND, D_MODEL)
        np.testing.assert_allclose(kv_out[j], hf_out[j], atol=ATOL, rtol=RTOL, err_msg=f"kv vs hf, boundary {j}")
        np.testing.assert_allclose(cl_out[j], hf_out[j], atol=ATOL, rtol=RTOL, err_msg=f"cacheless vs hf, boundary {j}")
        np.testing.assert_allclose(kv_out[j], cl_out[j], atol=ATOL, rtol=RTOL, err_msg=f"kv vs cacheless, boundary {j}")
