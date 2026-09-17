"""ONNX-backed replacement for gmanifold's original HF-forward-hook transformer
adapter. Sublayer boundaries are not auto-discovered from module structure --
per this repo's convention (see CLAUDE.md, config.py) they're literal ONNX
tensor names declared once per model in a `ModelConfig` toml
(configs/gmanifold_*.toml). The manifold code itself (manifold.py) never
imports this; only forward inference is needed here (no Jacobian of a
Transformer sublayer is required by gmanifold -- only GlobalManifold.decode's
Jacobian is, and that stays plain PyTorch/torch.func), so no gradient/pullback
graph is built here -- see construct.construct + execute.jacobian for that
machinery if it's ever needed for a Transformer sublayer's own Jacobian.

Scope note on `make_map`: a `{src}.{dst}` sublayer is extracted (via
construct.subgraph_extract) as an isolated ONNX subgraph whose only input is
the `src` tensor itself. That's exactly correct for a *tokenwise* sublayer
(e.g. an FFN block, add_6->add_8 below) whose output at each sequence
position depends only on the input at that same position -- the extracted
subgraph is self-contained. It is NOT correct for a sublayer that crosses an
attention block (output at one position depends on other positions' hidden
states, which live outside this subgraph's inputs) -- there is no ONNX
analogue here of gmanifold.transformer's forward-hook "patch position pos of
a fixed context" trick, since ORT has no mechanism to splice an external
value into the middle of a running graph. `make_map` therefore only supports
tokenwise sublayers; use `collect_states` to read real states at any
location (including across attention) instead.
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort
import torch

from config import ModelConfig, SublayerConfig
from construct.construct import ensure_base_model, ensure_extra_outputs, ensure_subgraphs

_SESSION_PROVIDERS = ["CUDAExecutionProvider"]


def _session_options() -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.intra_op_num_threads = 8
    return opts


def _make_session(path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), sess_options=_session_options(), providers=_SESSION_PROVIDERS)


def collect_states(
    cfg: ModelConfig,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    locs: list[str],
    batch_size: int | None = None,
) -> dict[str, torch.Tensor]:
    """Real residual states {loc: (N, D)} at `positions` (N,) of the
    sequences `input_ids` (N, cfg.seq), read from the ONNX graph's named
    intermediate tensors `locs` (e.g. "embedding", "add_6") -- the ONNX
    analogue of gmanifold.transformer.collect_states's forward hooks.

    Each chunk sent to the session is padded up to the model's fixed
    exported batch dim (`cfg.batch_for_node()`); `batch_size` overrides the
    number of real rows per chunk (must be <= the fixed batch dim).
    """
    ensure_base_model(cfg, force=False, verbose=False)
    extra_outputs_path = ensure_extra_outputs(cfg, sorted(set(locs)), force=False)
    session = _make_session(extra_outputs_path)
    input_name = session.get_inputs()[0].name

    fixed_batch = cfg.batch_for_node()
    chunk = min(batch_size or fixed_batch, fixed_batch)
    N = len(input_ids)
    out = {loc: torch.empty(N, cfg.d_model) for loc in locs}

    ids_np = input_ids.cpu().numpy().astype(np.int64)
    pos_np = positions.cpu().numpy()

    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        n = e - s
        padded = np.zeros((fixed_batch, cfg.seq), dtype=np.int64)
        padded[:n] = ids_np[s:e]
        outputs = session.run(locs, {input_name: padded})
        for loc, arr in zip(locs, outputs):
            rows = arr[np.arange(n), pos_np[s:e]]
            out[loc][s:e] = torch.from_numpy(np.asarray(rows))
    return out


def vocab_states(cfg: ModelConfig, embedding_output_name: str = "embedding") -> tuple[torch.Tensor, list[int]]:
   
    ensure_base_model(cfg, force=False, verbose=False)
    extra_outputs_path = ensure_extra_outputs(cfg, [embedding_output_name], force=False)
    session = _make_session(extra_outputs_path)
    input_name = session.get_inputs()[0].name

    fixed_batch = cfg.batch_for_node()
    ids = list(range(cfg.vocab_size))
    out = torch.empty(cfg.vocab_size, cfg.d_model)

    for s in range(0, cfg.vocab_size, fixed_batch):
        e = min(s + fixed_batch, cfg.vocab_size)
        n = e - s
        padded = np.zeros((fixed_batch, cfg.seq), dtype=np.int64)
        padded[:n, 0] = np.arange(s, e)
        (arr,) = session.run([embedding_output_name], {input_name: padded})
        out[s:e] = torch.from_numpy(np.asarray(arr[:n, 0]))
    return out, ids


def _find_sublayer(cfg: ModelConfig, src: str, dst: str) -> SublayerConfig:
    for s in cfg.sublayers:
        if s.input == src and s.output == dst:
            return s
    raise ValueError(
        f"no sublayer {src!r} -> {dst!r} in cfg.sublayers for {cfg.name!r} -- "
        "declare it as a [[sublayers]] entry in the config toml first "
        "(make_map only supports tokenwise sublayers already present in the config)",
    )


def make_map(cfg: ModelConfig, src: str, dst: str, batch_size: int | None = None):
    """map_fn(x) = the real tokenwise sublayer `src -> dst` applied to
    arbitrary states x (N, D) -- the ONNX analogue of
    gmanifold.transformer.make_map's 'Lk.attn' -> 'Lk.ffn' fast path (see
    this module's docstring for why only tokenwise sublayers are supported).
    `src`/`dst` must match a `[[sublayers]]` entry already declared in
    `cfg` (e.g. configs/simple_stories.toml's "add_6" -> "add_8" is
    SimpleStories-V2-11M block 0's FFN sub-layer).
    """
    sublayer = _find_sublayer(cfg, src, dst)
    # sublayer.input_shape's batch dim is a Jacobian-sweep chunk size (e.g.
    # 32, tuned for the one-hot Jacobian sweep's GPU throughput -- see
    # config.py/CLAUDE.md), unrelated to the fixed batch dim the ONNX graph
    # was actually exported/extracted with (cfg.batch_for_node(), e.g.
    # 32000). The extracted subgraph's real input shape always has the
    # latter as its batch dim, so that -- not sublayer.input_shape[0] -- is
    # what every padded session.run call below must match.
    fixed_batch = cfg.batch_for_node()
    _, seq, d_model = sublayer.input_shape
    paths = ensure_subgraphs(cfg, force=False)
    [(path, _)] = [p for p in paths if p[1].input == src and p[1].output == dst]

    session = _make_session(path)
    input_name = session.get_inputs()[0].name
    chunk = min(batch_size or fixed_batch, fixed_batch)

    @torch.no_grad()
    def map_fn(x: torch.Tensor) -> torch.Tensor:
        """x: (N, D) states at `src`, taken to be at sequence position 0 of
        a dummy all-zero sequence (the sublayer is tokenwise, so the other
        positions -- and their content -- don't affect position 0's output)."""
        x_np = x.float().cpu().numpy()
        out = torch.empty(len(x), d_model)
        for s in range(0, len(x), chunk):
            e = min(s + chunk, len(x))
            n = e - s
            padded = np.zeros((fixed_batch, seq, d_model), dtype=np.float32)
            padded[:n, 0] = x_np[s:e]
            (arr,) = session.run([dst], {input_name: padded})
            out[s:e] = torch.from_numpy(np.asarray(arr[:n, 0]))
        return out

    return map_fn
