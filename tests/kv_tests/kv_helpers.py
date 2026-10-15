"""Shared helpers for tests/kv_tests: GPU ORT sessions, cacheless (static-shape) GPT-2
exports at a chosen sequence length, KV-cache prefix handling, and HF ground truth."""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from config import ModelConfig, load_config
from construct.construct import ensure_base_model, ensure_extra_outputs
from construct.fs_utils import exclusive_lock
from construct.kv_cache_export import ensure_kv_cache_model
from construct.kv_cache_subgraph import (
    ensure_kv_cache_boundaries,
    past_kv_input_names,
    present_kv_names,
)
from construct.subgraph_extract import extract_subgraph

# TF32 off: fp32-tight comparisons (tests/conftest.py also sets NVIDIA_TF32_OVERRIDE=0).
PROVIDERS = [("CUDAExecutionProvider", {"use_tf32": "0"})]
N_LAYERS, N_HEADS, HEAD_DIM, D_MODEL, VOCAB = 12, 12, 64, 768, 50257
PREFIX_LENGTHS = (0, 1, 8, 32)
N_CAND = 768  # candidate last tokens per prefix == d_model, so one batch is one full Jacobian


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def gpt2_cfg() -> ModelConfig:
    return load_config("configs/gpt2.toml")


def session(path, providers=None) -> ort.InferenceSession:
    assert "CUDAExecutionProvider" in ort.get_available_providers(), "tests require a GPU node"
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return ort.InferenceSession(str(path), sess_options=so, providers=providers or PROVIDERS)


def random_tokens(n: int, length: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, VOCAB, size=(n, length), dtype=np.int64)


def candidates(seed: int = 7, n: int = N_CAND) -> np.ndarray:
    """n distinct candidate last tokens."""
    return np.sort(np.random.default_rng(seed).choice(VOCAB, size=n, replace=False)).astype(np.int64)


# ------------------------------------------------------------- cacheless ----

def cacheless_cfg(cfg: ModelConfig, seq: int, batch: int = N_CAND, tag: str = "") -> ModelConfig:
    """cfg re-targeted at a static (batch, seq) cacheless export, in its own directory. `tag` gives a
    separate directory (the cacheless pullback file name doesn't encode its loss type)."""
    sublayers = [dataclasses.replace(s, input_shape=[batch, seq, cfg.d_model]) for s in cfg.sublayers]
    return dataclasses.replace(
        cfg, name=f"gpt2_kvtest_cacheless_s{seq}_b{batch}{tag}", seq=seq, batch=batch,
        batch_by_node={}, sublayers=sublayers,
    )


def cacheless_boundaries_session(cfg: ModelConfig, seq: int, batch: int = N_CAND):
    """(session, ccfg) for the cacheless graph outputting all 25 residual boundaries
    (names == [s.output for s in cfg.sublayers]) for input_ids (batch, seq)."""
    ccfg = cacheless_cfg(cfg, seq, batch)
    ensure_base_model(ccfg, verbose=False)
    path = ensure_extra_outputs(ccfg, [s.output for s in ccfg.sublayers])
    return session(path), ccfg


def cacheless_sublayer_path(ccfg: ModelConfig, k: int) -> tuple[Path, "object"]:
    """Forward subgraph for config sublayer k (boundary k-1 -> boundary k), k >= 1."""
    sub = ccfg.sublayers[k]
    ccfg.sub_block_path.mkdir(parents=True, exist_ok=True)
    path = ccfg.sub_block_path / f"{sub.input}.{sub.output}.subgraph.forward.onnx"
    with exclusive_lock(path.with_name(path.name + ".lock")):
        if not path.exists():
            ensure_base_model(ccfg, verbose=False)
            extract_subgraph(ccfg.base_model_path, path, sub.input, sub.output)
    return path, sub


# ------------------------------------------------------------ KV prefixes ----

def kv_boundaries_session(cfg: ModelConfig):
    """(session over the all-boundaries KV subgraph, boundary tensor names)."""
    ensure_kv_cache_model(cfg)
    path, names, _ = ensure_kv_cache_boundaries(cfg)
    return session(path), names


def empty_past(batch: int) -> dict[str, np.ndarray]:
    return {n: np.zeros((batch, N_HEADS, 0, HEAD_DIM), np.float32) for n in past_kv_input_names(N_LAYERS)}


def kv_feeds(ids: np.ndarray, past: dict[str, np.ndarray], offset: int, sess_inputs) -> dict:
    batch, n = ids.shape
    feeds = {
        "input_ids": ids.astype(np.int64),
        "attention_mask": np.ones((batch, offset + n), np.int64),
        "position_ids": np.broadcast_to(np.arange(offset, offset + n, dtype=np.int64), (batch, n)).copy(),
        **past,
    }
    return {k: v for k, v in feeds.items() if k in sess_inputs}


def kv_run(sess, ids, past, offset) -> dict[str, np.ndarray]:
    names = [o.name for o in sess.get_outputs()]
    feeds = kv_feeds(ids, past, offset, {i.name for i in sess.get_inputs()})
    return dict(zip(names, sess.run(None, feeds)))


def kv_prefix_past(sess, prefix: np.ndarray) -> dict[str, np.ndarray]:
    """Batch-1 past K/V (as past_key_values.* feeds) for `prefix` (1, L); L == 0 -> empty past
    (the exported graph cannot run a zero-length input_ids, see test_empty_input_ids_unsupported)."""
    if prefix.shape[1] == 0:
        return empty_past(1)
    out = kv_run(sess, prefix, empty_past(1), 0)
    return {f"past_key_values.{n.split('.', 1)[1]}": out[n] for n in present_kv_names(N_LAYERS)}


def broadcast_past(past: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    return {k: np.ascontiguousarray(np.broadcast_to(v, (n, *v.shape[1:]))) for k, v in past.items()}


def kv_last_token_boundaries(sess, names, prefix: np.ndarray, cand: np.ndarray, past=None):
    """{j: (n, d_model)} boundary activations of each candidate as the token after `prefix`."""
    past = past if past is not None else kv_prefix_past(sess, prefix)
    L = prefix.shape[1]
    out = kv_run(sess, cand[:, None], broadcast_past(past, len(cand)), L)
    return {j: out[name][:, 0] for j, name in enumerate(names)}


# -------------------------------------------------------------------- HF ----

def hf_model():
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained("openai-community/gpt2").cuda().eval()


def hf_all_boundaries(model, ids: torch.Tensor, grad_at: int | None = None):
    """Residual boundaries 0..24 of `model(ids)`, each (n, seq, d). With `grad_at=j`, boundary j is
    replaced by a detached leaf requiring grad (returned too) so callers can differentiate later
    boundaries w.r.t. it -- the leaf's prefix positions are constants, so the gradient at the last
    position is exactly the diagonal Jacobian block."""
    got, leaf, handles = {}, [None], []

    def maybe_leaf(j, t):
        got[j] = t
        if grad_at == j:
            leaf[0] = t.detach().clone().requires_grad_(True)
            got[j] = leaf[0]
            return leaf[0]
        return None

    handles.append(model.transformer.h[0].register_forward_pre_hook(
        lambda m, args, kwargs: _embed_hook(m, args, kwargs, maybe_leaf), with_kwargs=True,
    ))
    for i, block in enumerate(model.transformer.h):
        handles.append(block.ln_2.register_forward_pre_hook(
            lambda m, args, j=2 * i + 1: _pre_hook(args, j, maybe_leaf),
        ))
        handles.append(block.register_forward_hook(
            lambda m, args, out, j=2 * i + 2: _block_hook(out, j, maybe_leaf),
        ))
    try:
        with torch.set_grad_enabled(grad_at is not None):
            model(ids)
    finally:
        for h in handles:
            h.remove()
    return got, leaf[0]


def _embed_hook(m, args, kwargs, maybe_leaf):
    hidden = args[0] if args else kwargs["hidden_states"]
    new = maybe_leaf(0, hidden)
    if new is not None:
        return ((new,) + tuple(args[1:]), kwargs) if args else ((), {**kwargs, "hidden_states": new})
    return None


def _pre_hook(args, j, maybe_leaf):
    new = maybe_leaf(j, args[0])
    return (new,) + tuple(args[1:]) if new is not None else None


def _block_hook(out, j, maybe_leaf):
    hidden = out[0] if isinstance(out, tuple) else out
    new = maybe_leaf(j, hidden)
    if new is None:
        return None
    return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new


# ------------------------------------------------------- KV sublayers ----

def kv_sublayer_feeds(sub, hidden: np.ndarray, past_b: dict[str, np.ndarray], prefix_len: int) -> dict:
    """Feeds for a construct.kv_cache_subgraph.KVSublayer. hidden: (n, 1, d_model) last-token input;
    past_b: batch-broadcast past K/V of the prefix (all layers)."""
    from construct.kv_cache_subgraph import KV_SUBLAYER_IN

    n = hidden.shape[0]
    feeds = {KV_SUBLAYER_IN: np.ascontiguousarray(hidden, dtype=np.float32)}
    for name in sub.extra_inputs:
        if name == "input_ids":
            feeds[name] = np.zeros((n, 1), np.int64)  # only its shape is read
        elif name == "attention_mask":
            feeds[name] = np.ones((n, prefix_len + 1), np.int64)
        elif name.startswith("past_key_values."):
            feeds[name] = past_b[name]
        else:
            raise KeyError(f"don't know how to feed sublayer input {name!r}")
    return feeds


# ------------------------------------------------------------ pullbacks ----

def kv_jacobian(cfg, kv_sess, j_in: int, hidden: np.ndarray, prefix: np.ndarray, past=None):
    """Last-token Jacobian J[i, j] = d out_i / d in_j of sublayer j_in -> j_in + 1 via the KV
    MaskedSum pullback: batch N_CAND == d_model rows, row i has a one-hot mask at dim i.
    hidden: (d_model,) the last token's input. Returns (J (d, d), out (d,))."""
    from config import LossType
    from construct.kv_cache_subgraph import KV_SUBLAYER_IN, KV_SUBLAYER_OUT, ensure_kv_cache_sublayer_pullback

    L = prefix.shape[1]
    sub, pb_path = ensure_kv_cache_sublayer_pullback(cfg, j_in, N_CAND, LossType.MaskedSumLoss)
    past = past if past is not None else kv_prefix_past(kv_sess, prefix)
    feeds = kv_sublayer_feeds(sub, np.broadcast_to(hidden, (N_CAND, 1, D_MODEL)), broadcast_past(past, N_CAND), L)
    feeds["mask"] = np.eye(D_MODEL, dtype=np.float32)[:, None, :]
    grad, out = session(pb_path).run([f"{KV_SUBLAYER_IN}_grad", KV_SUBLAYER_OUT], feeds)
    return grad[:, 0, :], out[0, 0]


def cacheless_jacobian(ccfg, k: int, in_act: np.ndarray):
    """Same quantity through the cacheless MaskedSum pullback of config sublayer k (boundary k-1 -> k):
    in_act (S, d_model) is the full sequence input; every batch row gets a copy, with a one-hot mask
    at (row i, last position, dim i). Returns (J (d, d), out_last (d,))."""
    from config import LossType
    from construct.construct import ensure_subgraph_pullback

    fwd, sub = cacheless_sublayer_path(ccfg, k)
    sub = dataclasses.replace(sub, loss_type=LossType.MaskedSumLoss)
    [pb_path] = ensure_subgraph_pullback([(fwd, sub)], force=False)
    S = in_act.shape[0]
    mask = np.zeros((N_CAND, S, D_MODEL), np.float32)
    mask[np.arange(D_MODEL), S - 1, np.arange(D_MODEL)] = 1.0
    x = np.ascontiguousarray(np.broadcast_to(in_act, (N_CAND, S, D_MODEL)), dtype=np.float32)
    grad, out = session(pb_path).run([f"{sub.input}_grad", sub.output], {sub.input: x, "mask": mask})
    return grad[:, S - 1, :], out[0, S - 1]


def hf_jacobian(model, seq_ids: torch.Tensor, j_in: int) -> np.ndarray:
    """Last-token Jacobian d out_i / d in_j of sublayer j_in -> j_in + 1 by torch autograd on the HF model
    for one sequence seq_ids (1, S). Even j_in (attention sublayers): the boundary is replaced by a leaf
    whose prefix positions are constants. Odd j_in (MLP sublayers): the sublayer is a per-token function
    x + mlp(ln_2(x)), differentiated directly (an ln_2 hook can't replace the residual path)."""
    layer = j_in // 2
    if j_in % 2 == 1:
        block = model.transformer.h[layer]
        x = hf_all_boundaries(model, seq_ids)[0][j_in][0, -1].detach().clone()
        f = lambda v: v + block.mlp(block.ln_2(v))
        return torch.autograd.functional.jacobian(f, x, vectorize=True).cpu().numpy()
    got, leaf = hf_all_boundaries(model, seq_ids, grad_at=j_in)
    out = got[j_in + 1][0, -1]
    rows = [
        torch.autograd.grad(out[i], leaf, retain_graph=True)[0][0, -1] for i in range(D_MODEL)
    ]
    return torch.stack(rows).detach().cpu().numpy()


class LastTokenOnly:
    """Wraps an execute.optimization_loop optimizer so only the last sequence position of y is optimised:
    the gradient is zeroed elsewhere (also discarding the 0/0 NaNs the normalized losses produce at
    positions where y == x) and prefix positions are reset to their previous value after each step.
    Makes a full-sequence cacheless pullback solve the same problem as the last-token KV pullback."""

    def __init__(self, inner):
        self.inner = inner

    def step(self, y, grad):
        g = np.zeros_like(grad)
        g[..., -1, :] = grad[..., -1, :]
        y_next = self.inner.step(y, g)
        y_next[..., :-1, :] = y[..., :-1, :]
        return y_next
