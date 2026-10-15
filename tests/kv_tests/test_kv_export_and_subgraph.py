"""Verifies the GPT-2 KV-cache export (construct/kv_cache_export.py, via HF
optimum) and the sublayer subgraph extraction (construct/kv_cache_subgraph.py)
against the plain HF torch model as ground truth. Runs on GPU (CUDA execution provider, TF32 off so tolerances stay tight).

The first run downloads/exports GPT-2 (~30s); the export is cached under
onnx_models/gpt2/... and reused afterwards.
"""

import dataclasses

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from transformers import AutoModelForCausalLM

from config import load_config
from construct.kv_cache_export import ensure_kv_cache_model
from construct.kv_cache_subgraph import (
    boundary_for_index,
    check_block_prefix,
    ensure_kv_cache_extra_outputs,
    past_kv_input_names,
    present_kv_names,
    sublayer_tensor_name,
)

PROVIDERS = [("CUDAExecutionProvider", {"use_tf32": "0"})]  # fail loudly if CUDA EP is unavailable
N_LAYERS, N_HEADS, HEAD_DIM = 12, 12, 64
SEQ = 6
# fp32 ORT vs torch: observed ~1e-3 on logits (|logits| ~ 100), ~1e-5 on residuals.
LOGITS_ATOL = 5e-3
ACT_ATOL = 1e-3
ACT_RTOL = 1e-4  # GPT-2 has massive-activation dims (|x| ~ 400)


@pytest.fixture(scope="session")
def cfg():
    return load_config("configs/gpt2.toml")


@pytest.fixture(scope="session")
def model_path(cfg):
    return ensure_kv_cache_model(cfg)


@pytest.fixture(scope="session")
def sess(model_path):
    assert "CUDAExecutionProvider" in ort.get_available_providers(), "tests require a GPU node"
    return ort.InferenceSession(str(model_path), providers=PROVIDERS)


@pytest.fixture(scope="session")
def hf_model():
    m = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    m.eval()
    return m


@pytest.fixture(scope="session")
def tokens():
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, 50257, (3, SEQ), generator=g)


def empty_past(batch):
    return {
        name: np.zeros((batch, N_HEADS, 0, HEAD_DIM), dtype=np.float32)
        for name in past_kv_input_names(N_LAYERS)
    }


def run(sess, ids, past, offset=0):
    """Runs ORT on ids (batch, n) given a `past` dict whose length is `offset`."""
    batch, n = ids.shape
    feeds = {
        "input_ids": ids.astype(np.int64),
        "attention_mask": np.ones((batch, offset + n), dtype=np.int64),
        "position_ids": np.broadcast_to(np.arange(offset, offset + n, dtype=np.int64), (batch, n)).copy(),
        **past,
    }
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(None, feeds)))


def as_past(outputs):
    return {f"past_key_values.{n.split('.', 1)[1]}": outputs[n] for n in present_kv_names(N_LAYERS)}


def hf_boundaries(hf_model, ids):
    """Ground-truth residual-stream tensors from the torch model:
    embed, and per layer attn_out (input of ln_2) / mlp_out (block output)."""
    got = {}
    handles = [hf_model.transformer.h[0].register_forward_pre_hook(
        lambda mod, args, kwargs: got.__setitem__((0, "embed"), (args[0] if args else kwargs["hidden_states"]).detach()),
        with_kwargs=True,
    )]
    for i, block in enumerate(hf_model.transformer.h):
        handles.append(block.ln_2.register_forward_pre_hook(
            lambda mod, args, i=i: got.__setitem__((i, "attn_out"), args[0].detach()),
        ))
        handles.append(block.register_forward_hook(
            lambda mod, args, out, i=i: got.__setitem__((i, "mlp_out"), (out[0] if isinstance(out, tuple) else out).detach()),
        ))
    with torch.no_grad():
        hf_model(ids)
    for h in handles:
        h.remove()
    return got


# ---------------------------------------------------------------- export ----

def test_export_io_signature(sess):
    inputs = {i.name: i.shape for i in sess.get_inputs()}
    outputs = {o.name: o.shape for o in sess.get_outputs()}
    assert {"input_ids", "attention_mask", "position_ids"} <= set(inputs)
    for name in past_kv_input_names(N_LAYERS):
        assert inputs[name][1:2] == [N_HEADS] and inputs[name][3] == HEAD_DIM
        assert isinstance(inputs[name][0], str) and isinstance(inputs[name][2], str)  # dynamic batch, past len
    assert "logits" in outputs
    assert [n for n in outputs if n.startswith("present.")] == present_kv_names(N_LAYERS)
    assert len(inputs) == 3 + 2 * N_LAYERS


def test_full_pass_logits_match_hf(sess, hf_model, tokens):
    out = run(sess, tokens.numpy(), empty_past(tokens.shape[0]))
    with torch.no_grad():
        ref = hf_model(tokens).logits.numpy()
    np.testing.assert_allclose(out["logits"], ref, atol=LOGITS_ATOL)


def test_present_kv_matches_hf(sess, hf_model, tokens):
    out = run(sess, tokens.numpy(), empty_past(tokens.shape[0]))
    with torch.no_grad():
        ref = hf_model(tokens, use_cache=True).past_key_values
    for i in range(N_LAYERS):
        k, v = ref[i] if not hasattr(ref, "layers") else (ref.layers[i].keys, ref.layers[i].values)
        np.testing.assert_allclose(out[f"present.{i}.key"], k.numpy(), atol=ACT_ATOL)
        np.testing.assert_allclose(out[f"present.{i}.value"], v.numpy(), atol=ACT_ATOL)


# ------------------------------------------------------------ KV semantics ----

@pytest.mark.parametrize("pos", [1, 3, SEQ - 1])
def test_prefix_then_continuation_equals_full_pass(sess, tokens, pos):
    ids = tokens.numpy()
    full = run(sess, ids, empty_past(ids.shape[0]))
    prefix = run(sess, ids[:, :pos], empty_past(ids.shape[0]))
    cont = run(sess, ids[:, pos:], as_past(prefix), offset=pos)
    np.testing.assert_allclose(cont["logits"], full["logits"][:, pos:], atol=LOGITS_ATOL)
    # the continuation's present is the full-length cache again
    for name in present_kv_names(N_LAYERS):
        assert cont[name].shape[2] == SEQ
        np.testing.assert_allclose(cont[name], full[name], atol=ACT_ATOL)


def test_broadcast_prefix_batched_candidates(sess, tokens):
    """The sweep's pattern: one batch-1 prefix, many independent length-1 candidates."""
    pos = 4
    prefix_ids = tokens.numpy()[:1, :pos]
    prefix = run(sess, prefix_ids, empty_past(1))
    candidates = np.array([0, 1, 7, 50256, 1234, 42, 9999], dtype=np.int64)
    n = len(candidates)
    past = {k: np.broadcast_to(v, (n, *v.shape[1:])).copy() for k, v in as_past(prefix).items()}
    cont = run(sess, candidates[:, None], past, offset=pos)
    for j, c in enumerate(candidates):
        full = run(sess, np.concatenate([prefix_ids, [[c]]], axis=1), empty_past(1))
        np.testing.assert_allclose(cont["logits"][j, 0], full["logits"][0, -1], atol=LOGITS_ATOL)


def test_empty_input_ids_unsupported(sess):
    """Documents a limitation of the exported graph: a zero-length input_ids
    (prefix at pos == 0) fails in ORT's Reshape (-1 is ambiguous at 0 elements)."""
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.RuntimeException, match="Reshape"):
        run(sess, np.zeros((1, 0), dtype=np.int64), empty_past(1))


def test_pos0_workaround_zero_length_past(sess, tokens):
    """pos == 0 must skip the prefix call and feed candidates directly with a
    zero-length past -- equivalent to a plain 1-token forward pass."""
    candidates = np.array([[3], [50256], [17]], dtype=np.int64)
    cont = run(sess, candidates, empty_past(3), offset=0)
    for j in range(3):
        full = run(sess, candidates[j : j + 1], empty_past(1))
        np.testing.assert_allclose(cont["logits"][j], full["logits"][0], atol=LOGITS_ATOL)
    assert cont["present.0.key"].shape == (3, N_HEADS, 1, HEAD_DIM)


# --------------------------------------------------------------- subgraph ----

def test_boundary_for_index():
    assert boundary_for_index(0) == (0, "embed")
    assert boundary_for_index(1) == (0, "attn_out")
    assert boundary_for_index(2) == (0, "mlp_out")
    assert boundary_for_index(3) == (1, "attn_out")
    assert boundary_for_index(24) == (11, "mlp_out")
    with pytest.raises(ValueError):
        boundary_for_index(-1)


def test_config_sublayers_chain_over_all_boundaries(cfg):
    assert len(cfg.sublayers) == 2 * N_LAYERS + 1
    for a, b in zip(cfg.sublayers, cfg.sublayers[1:]):
        assert a.output == b.input


def test_block_prefix_from_config(cfg, model_path):
    model = onnx.load(str(model_path), load_external_data=False)
    assert cfg.block_prefix == "/transformer/h"
    assert check_block_prefix(model, cfg.block_prefix) == "/transformer/h"
    assert sublayer_tensor_name(3, "mlp_out", cfg.block_prefix) == "/transformer/h.3/Add_1_output_0"
    with pytest.raises(ValueError):
        check_block_prefix(model, "/model/layers")
    with pytest.raises(ValueError):
        check_block_prefix(model, None)


SUBLAYER_CASES = [(0, "embed"), (0, "attn_out"), (0, "mlp_out"), (5, "attn_out"), (11, "mlp_out")]


@pytest.mark.parametrize(("layer", "boundary"), SUBLAYER_CASES)
def test_extracted_subgraph_matches_hf(cfg, sess, hf_model, tokens, layer, boundary):
    path, out_name, n_layers = ensure_kv_cache_extra_outputs(cfg, layer, boundary, force=True)
    assert n_layers == N_LAYERS

    sub = ort.InferenceSession(str(path), providers=PROVIDERS)
    assert [o.name for o in sub.get_outputs()] == [out_name] + present_kv_names(N_LAYERS)
    assert "logits" not in {o.name for o in sub.get_outputs()}

    ids = tokens.numpy()
    batch = ids.shape[0]
    feeds = {
        "input_ids": ids,
        "attention_mask": np.ones_like(ids),
        "position_ids": np.broadcast_to(np.arange(SEQ), ids.shape).copy(),
        **empty_past(batch),
    }
    names = [o.name for o in sub.get_outputs()]
    res = dict(zip(names, sub.run(None, {k: v for k, v in feeds.items() if k in {i.name for i in sub.get_inputs()}})))

    ref = hf_boundaries(hf_model, tokens)[(layer, boundary)].numpy()
    np.testing.assert_allclose(res[out_name], ref, atol=ACT_ATOL, rtol=ACT_RTOL)

    # extracted graph's KV equals the full model's KV (extraction did not alter the KV path)
    full = run(sess, ids, empty_past(batch))
    for name in present_kv_names(N_LAYERS):
        np.testing.assert_allclose(res[name], full[name], atol=1e-5)


def test_subgraph_with_prefix_cache_matches_full_pass(cfg, sess, tokens):
    """The sweep's use: sublayer activation at the new position from prefix KV
    equals the same activation from a full from-scratch pass."""
    layer, boundary, pos = 5, "attn_out", 3
    path, out_name, _ = ensure_kv_cache_extra_outputs(cfg, layer, boundary)
    sub = ort.InferenceSession(str(path), providers=PROVIDERS)
    in_names = {i.name for i in sub.get_inputs()}

    def run_sub(ids, past, offset):
        b, n = ids.shape
        feeds = {
            "input_ids": ids, "attention_mask": np.ones((b, offset + n), dtype=np.int64),
            "position_ids": np.broadcast_to(np.arange(offset, offset + n), (b, n)).copy(), **past,
        }
        return dict(zip([o.name for o in sub.get_outputs()],
                        sub.run(None, {k: v for k, v in feeds.items() if k in in_names})))

    ids = tokens.numpy()
    full = run_sub(ids, empty_past(ids.shape[0]), 0)
    prefix = run_sub(ids[:, :pos], empty_past(ids.shape[0]), 0)
    cont = run_sub(ids[:, pos:], as_past(prefix), pos)
    np.testing.assert_allclose(cont[out_name], full[out_name][:, pos:], atol=ACT_ATOL, rtol=ACT_RTOL)


def test_kv_subgraph_matches_cacheless_add_names(cfg, tokens):
    """Config sublayer names (add, add_2, add_5, ...) from the cacheless
    torch.onnx export map to the KV-cache graph's tensors via boundary_for_index."""
    from construct.construct import ensure_base_model, ensure_extra_outputs

    small = dataclasses.replace(cfg, name="gpt2_kv_test_cacheless", batch=3, batch_by_node={})
    ensure_base_model(small, verbose=False)
    names = [s.output for s in cfg.sublayers]  # names[j] is residual boundary j
    check = [0, 1, 2, 3, 12, 24]  # embed, h0 attn/mlp, h1 attn, h5 mlp, final mlp
    extra = ensure_extra_outputs(small, [names[j] for j in check], force=True)
    ids = tokens.numpy()[:, : cfg.seq]  # cacheless export has a static (batch, cfg.seq) shape
    ref_sess = ort.InferenceSession(str(extra), providers=PROVIDERS)
    ref = dict(zip([o.name for o in ref_sess.get_outputs()], ref_sess.run(None, {"input_ids": ids})))

    for j in check:
        layer, boundary = boundary_for_index(j)
        path, out_name, _ = ensure_kv_cache_extra_outputs(cfg, layer, boundary)
        sub = ort.InferenceSession(str(path), providers=PROVIDERS)
        in_names = {i.name for i in sub.get_inputs()}
        feeds = {"input_ids": ids, "attention_mask": np.ones_like(ids),
                 "position_ids": np.broadcast_to(np.arange(cfg.seq), ids.shape).copy(), **empty_past(ids.shape[0])}
        got = sub.run([out_name], {k: v for k, v in feeds.items() if k in in_names})[0]
        np.testing.assert_allclose(got, ref[names[j]], atol=ACT_ATOL, rtol=ACT_RTOL, err_msg=f"{names[j]} vs {layer}/{boundary}")
