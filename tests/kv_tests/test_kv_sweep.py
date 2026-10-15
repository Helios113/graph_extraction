"""End-to-end test of scripts/helper/generate_perturbed_sweep_kvcache.py on GPU:
sweeps a few (sample, pos, vocab_id) rows through the real script (prefix once,
batched continuation, ShardedLayerWriter output) and compares every row to the
HF torch model run on the explicitly substituted sequence.

TF32 is disabled process-wide by tests/conftest.py (NVIDIA_TF32_OVERRIDE=0); with it on,
layer-11 massive-activation dims (|x| ~ 360) deviate by ~0.3%.
"""

import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from transformers import AutoModelForCausalLM

from config import load_config
from execute.sharded_writer import load_shard_rows
from scripts.helper.generate_perturbed_sweep_kvcache import run_perturbed_sweep_kvcache

N_SAMPLES = 2
N_VOCAB_SAMPLE = 50
CANDIDATE_BATCH = 16  # not a divisor of 50 -> exercises ragged last chunk
ATOL, RTOL = 1e-2, 1e-4

CASES = [(0, "embed"), (0, "attn_out"), (3, "mlp_out"), (11, "mlp_out")]


@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/gpt2.toml")


@pytest.fixture(scope="module")
def hf_model():
    assert "CUDAExecutionProvider" in ort.get_available_providers(), "tests require a GPU node"
    m = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").cuda().eval()
    return m


@pytest.fixture(scope="module")
def base_seqs(cfg):
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, cfg.vocab_size, (N_SAMPLES, cfg.seq), generator=g).numpy()


def hf_activation(hf_model, ids: torch.Tensor, layer: int, boundary: str) -> torch.Tensor:
    """(n, seq, d_model) residual-stream tensor at the given boundary."""
    got = {}
    block = hf_model.transformer.h[layer]
    if boundary == "embed":
        h = hf_model.transformer.h[0].register_forward_pre_hook(
            lambda m, args, kwargs: got.__setitem__("x", (args[0] if args else kwargs["hidden_states"]).detach()),
            with_kwargs=True,
        )
    elif boundary == "attn_out":
        h = block.ln_2.register_forward_pre_hook(lambda m, args: got.__setitem__("x", args[0].detach()))
    else:
        h = block.register_forward_hook(
            lambda m, args, out: got.__setitem__("x", (out[0] if isinstance(out, tuple) else out).detach()),
        )
    with torch.no_grad():
        hf_model(ids)
    h.remove()
    return got["x"]


@pytest.mark.parametrize(("layer", "boundary"), CASES)
def test_sweep_matches_hf(cfg, hf_model, base_seqs, layer, boundary):
    out_dir = f"onnx_models/gpt2_kv_sweep_test/{boundary}_{layer}"  # project-relative (ShardedLayerWriter layer_key)
    shutil.rmtree(out_dir, ignore_errors=True)

    run_perturbed_sweep_kvcache(
        cfg, base_seqs, out_dir,
        n_samples=N_SAMPLES, layer_idx=layer, boundary=boundary,
        flush_every=2, shard_rows=300,  # several shards
        candidate_batch=CANDIDATE_BATCH, no_push=True,
        n_vocab_sample=N_VOCAB_SAMPLE, vocab_sample_seed=0,
    )

    layer_key = next(Path(out_dir).glob("*.manifest.json")).name.removesuffix(".manifest.json")
    rows = load_shard_rows(".", f"{out_dir}/{layer_key}")
    n_rows = N_SAMPLES * cfg.seq * N_VOCAB_SAMPLE
    assert rows["vec"].shape == (n_rows, cfg.d_model)

    # row order: unravel (sample, pos, vocab_idx); vocab_id is the substituted token
    s_idx, pos, vocab_idx = np.unravel_index(np.arange(n_rows), (N_SAMPLES, cfg.seq, N_VOCAB_SAMPLE))
    np.testing.assert_array_equal(rows["sample_idx"], s_idx)
    np.testing.assert_array_equal(rows["pos"], pos)
    vocab_ids = np.sort(np.random.default_rng(0).choice(cfg.vocab_size, N_VOCAB_SAMPLE, replace=False))
    np.testing.assert_array_equal(rows["vocab_id"], vocab_ids[vocab_idx])

    # reference: substitute the token explicitly and run the full torch model
    subst = base_seqs[s_idx].copy()
    subst[np.arange(n_rows), pos] = rows["vocab_id"]
    ref = np.empty_like(rows["vec"])
    for start in range(0, n_rows, 200):
        sl = slice(start, start + 200)
        act = hf_activation(hf_model, torch.from_numpy(subst[sl]).cuda(), layer, boundary)
        ref[sl] = act[torch.arange(act.shape[0]), torch.from_numpy(pos[sl])].cpu().numpy()

    np.testing.assert_allclose(rows["vec"], ref, atol=ATOL, rtol=RTOL)
    assert (pos == 0).any() and np.abs(rows["vec"][pos == 0]).max() > 0  # pos==0 rows really computed


def test_sweep_resume_is_idempotent(cfg, base_seqs):
    """Re-running a completed sweep skips everything and leaves the output unchanged."""
    out_dir = "onnx_models/gpt2_kv_sweep_test/resume"
    shutil.rmtree(out_dir, ignore_errors=True)
    kwargs = dict(
        n_samples=N_SAMPLES, layer_idx=0, boundary="attn_out", flush_every=2, shard_rows=300,
        candidate_batch=CANDIDATE_BATCH, no_push=True, n_vocab_sample=N_VOCAB_SAMPLE, vocab_sample_seed=0,
    )
    run_perturbed_sweep_kvcache(cfg, base_seqs, out_dir, **kwargs)
    layer_key = next(Path(out_dir).glob("*.manifest.json")).name.removesuffix(".manifest.json")
    before = load_shard_rows(".", f"{out_dir}/{layer_key}")["vec"]
    run_perturbed_sweep_kvcache(cfg, base_seqs, out_dir, **kwargs)
    after = load_shard_rows(".", f"{out_dir}/{layer_key}")["vec"]
    np.testing.assert_array_equal(before, after)
