"""Extracts a sublayer-activation-plus-KV-cache subgraph from a KV-cache
model.onnx (see construct/kv_cache_export.py), for use by
scripts/helper/generate_perturbed_sweep_kvcache.py.

Sublayer tensors in the KV-cache export are named positionally and
predictably by decoder layer index -- verified against the old cacheless
export's opaque add_N names for SimpleStories-V2-11M's layer 0
(add_6 == /model/layers.0/Add_output_0, add_8 == /model/layers.0/Add_1_output_0,
both equal to fp32 rounding noise):

  /model/layers.{i}/Add_output_0    -- post-attention residual (sublayer "input"
                                        boundary in the old add_N scheme)
  /model/layers.{i}/Add_1_output_0  -- post-MLP residual (sublayer "output"
                                        boundary, and also the residual stream
                                        entering layer i+1)

This module resolves a layer index to those two tensor names and extracts a
subgraph whose outputs are exactly [sublayer_output_tensor] + every
present.{layer}.{key,value} tensor -- the sublayer activation the sweep
records, plus every layer's KV so a prefix pass's cache can be fed into a
continuation pass. `logits` (and the lm_head matmul that produces it, the
single largest per-row tensor in the graph) is dropped, same rationale as
ensure_extra_outputs in construct/construct.py.
"""

from dataclasses import dataclass
from pathlib import Path

import onnx

from config import ModelConfig
from construct.fs_utils import exclusive_lock
from construct.kv_cache_export import kv_cache_model_dir, kv_cache_model_path


# Decoder-block names in optimum's KV-cache export are "{cfg.block_prefix}.{i}/Add"
# (post-attention residual) and ".../Add_1" (post-MLP residual), e.g.
#   SimpleStories (Llama-style):  /model/layers.{i}/Add_output_0
#   GPT-2:                        /transformer/h.{i}/Add_output_0


def check_block_prefix(model: onnx.ModelProto, block_prefix: str | None) -> str:
    """Validates cfg.block_prefix against the graph (layer 0's residual Adds
    must exist) and returns it."""
    if not block_prefix:
        raise ValueError("config has no block_prefix -- set it in the model's toml")
    node_names = {n.name for n in model.graph.node}
    if f"{block_prefix}.0/Add" not in node_names or f"{block_prefix}.0/Add_1" not in node_names:
        raise ValueError(f"block_prefix {block_prefix!r} does not match this graph's decoder blocks")
    return block_prefix


def sublayer_tensor_name(
    layer_idx: int,
    boundary: str,
    block_prefix: str,
    model: onnx.ModelProto | None = None,
) -> str:
    """`boundary`: "attn_out" (post-attention residual, i.e. the old add_N
    scheme's sublayer "input" boundary), "mlp_out" (post-MLP residual,
    the "output" boundary and also the next layer's "input"), or "embed"
    (the residual stream entering layer 0, i.e. token+position embeddings --
    layer_idx is ignored; needs `model` to find it).

    `block_prefix` is cfg.block_prefix."""
    if boundary == "attn_out":
        return f"{block_prefix}.{layer_idx}/Add_output_0"
    if boundary == "mlp_out":
        return f"{block_prefix}.{layer_idx}/Add_1_output_0"
    if boundary == "embed":
        if model is None:
            raise ValueError("boundary 'embed' needs `model` to locate the embedding tensor")
        return _embedding_tensor_name(model, block_prefix)
    raise ValueError(f"boundary must be 'attn_out', 'mlp_out' or 'embed', got {boundary!r}")


def _embedding_tensor_name(model: onnx.ModelProto, block_prefix: str) -> str:
    """The residual stream entering layer 0: the input of layer 0's
    post-attention `Add` that is not produced inside layer 0 (its other
    input is the attention output, which is)."""
    layer0_nodes = [n for n in model.graph.node if n.name.startswith(f"{block_prefix}.0/")]
    produced_in_layer0 = {o for n in layer0_nodes for o in n.output}
    add = next(n for n in layer0_nodes if n.name == f"{block_prefix}.0/Add")
    outside = [i for i in add.input if i not in produced_in_layer0]
    if len(outside) != 1:
        raise ValueError(f"could not identify the embedding input of {add.name}: {list(add.input)}")
    return outside[0]


def boundary_for_index(j: int) -> tuple[int, str]:
    """Maps the j-th residual-stream boundary (j=0 is the embedding, then
    alternating post-attention / post-MLP residuals) to (layer_idx, boundary)
    for sublayer_tensor_name. In a sublayers config, sublayer k's *output* is
    boundary k (sublayer 0 is input_ids -> embedding), e.g. GPT-2's
    add, add_2, add_5, ... = embed, h.0 attn_out, h.0 mlp_out, ... and the
    final add_49 is boundary 24 = h.11 mlp_out."""
    if j < 0:
        raise ValueError(f"boundary index must be >= 0, got {j}")
    if j == 0:
        return 0, "embed"
    if j % 2 == 1:
        return (j - 1) // 2, "attn_out"
    return j // 2 - 1, "mlp_out"


def present_kv_names(n_layers: int) -> list[str]:
    names = []
    for i in range(n_layers):
        names.append(f"present.{i}.key")
        names.append(f"present.{i}.value")
    return names


def past_kv_input_names(n_layers: int) -> list[str]:
    names = []
    for i in range(n_layers):
        names.append(f"past_key_values.{i}.key")
        names.append(f"past_key_values.{i}.value")
    return names


def kv_cache_extra_outputs_path(cfg: ModelConfig, layer_idx: int, boundary: str) -> Path:
    return kv_cache_model_dir(cfg) / f"extra_outputs.layers_{layer_idx}_{boundary}.onnx"


def ensure_kv_cache_extra_outputs(
    cfg: ModelConfig,
    layer_idx: int,
    boundary: str,
    force: bool = False,
) -> tuple[Path, str, int]:
    """`boundary` as in sublayer_tensor_name (including "embed"). Extracts (and caches) a subgraph of cfg's KV-cache model whose outputs
    are the requested sublayer tensor plus every layer's present KV, dropping
    logits. Returns (subgraph_path, sublayer_output_name, n_layers)."""
    model_path = kv_cache_model_path(cfg)
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} does not exist -- call ensure_kv_cache_model(cfg) first",
        )

    output_path = kv_cache_extra_outputs_path(cfg, layer_idx, boundary)
    lock_path = output_path.with_suffix(output_path.suffix + ".lock")

    with exclusive_lock(lock_path):
        base_model = onnx.load(str(model_path), load_external_data=False)
        input_names = [i.name for i in base_model.graph.input]
        n_layers = sum(
            1 for name in input_names if name.startswith("past_key_values.") and name.endswith(".key")
        )
        sublayer_output = sublayer_tensor_name(
            layer_idx, boundary, check_block_prefix(base_model, cfg.block_prefix), base_model,
        )

        if not force and output_path.exists():
            print(f"KV-cache extra outputs already exist, skipping extraction -> {output_path}")
            return output_path, sublayer_output, n_layers

        output_dir = output_path.parent
        model = onnx.load(str(model_path), load_external_data=False)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.load_external_data_for_model(model, str(model_path.parent))

        from onnx.utils import Extractor

        extractor = Extractor(model)
        outputs = [sublayer_output] + present_kv_names(n_layers)
        extracted = extractor.extract_model(input_names, outputs)

        onnx.save(
            extracted, str(output_path),
            save_as_external_data=True, location=output_path.name + ".data",
        )
        onnx.checker.check_model(str(output_path))
        print(f"KV-cache extra outputs extracted -> {output_path}")

    return output_path, sublayer_output, n_layers


# --------------------------------------------------------------------------
# Single-sublayer subgraphs for the *last token*, with prefix K/V as inputs.
#
# A sublayer is one step between adjacent residual-stream boundaries j -> j+1
# (see boundary_for_index): odd j+1 is an attention sublayer of layer
# (j+1-1)//2 (needs that layer's past K/V), even j+1 is an MLP sublayer. The
# subgraph takes the last token's (batch, 1, d_model) residual `hidden_in` and
# returns `hidden_out`; its input-output Jacobian is the diagonal block
# d out[pos] / d in[pos] of the cacheless sublayer (prefix K/V are constants:
# by causality they don't depend on the last token's input).
# ---------------------------------------------------------------------------

KV_SUBLAYER_IN = "hidden_in"
KV_SUBLAYER_OUT = "hidden_out"


@dataclass(frozen=True)
class KVSublayer:
    path: Path
    j_in: int
    layer_idx: int | None  # attention layer whose past K/V is consumed; None for MLP sublayers
    extra_inputs: tuple[str, ...]
    """Every graph input besides KV_SUBLAYER_IN: the layer's past K/V plus the
    shape-only inputs the exported attention-mask computation reads
    (input_ids, attention_mask, past_key_values.0.key). Feed them as for the
    full KV model, with input_ids (batch, 1) and attention_mask (batch, past + 1)."""


def _rename_tensor(model: onnx.ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for k, name in enumerate(node.input):
            if name == old:
                node.input[k] = new
        for k, name in enumerate(node.output):
            if name == old:
                node.output[k] = new
    for coll in (model.graph.input, model.graph.output, model.graph.value_info):
        for vi in coll:
            if vi.name == old:
                vi.name = new


def kv_cache_sublayer_path(cfg: ModelConfig, j_in: int, out_dir: Path | None = None) -> Path:
    return (out_dir or kv_cache_model_dir(cfg)) / f"sublayer.{j_in}.{j_in + 1}.forward.onnx"


def ensure_kv_cache_sublayer(
    cfg: ModelConfig,
    j_in: int,
    out_dir: Path | None = None,
    force: bool = False,
) -> KVSublayer:
    """Extracts (and caches) the last-token subgraph for the sublayer between
    residual boundaries j_in and j_in + 1 from cfg's KV-cache model. Input
    `hidden_in` (batch, 1, d_model) -> output `hidden_out`; see the section
    comment above for the other inputs. `out_dir` defaults to the KV-cache
    export dir (pullback construction passes its own per-config dir so the
    pullback artifacts don't collide)."""
    model_path = kv_cache_model_path(cfg)
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} does not exist -- call ensure_kv_cache_model(cfg) first")

    j_out = j_in + 1
    layer_idx = (j_out - 1) // 2 if j_out % 2 == 1 else None
    output_path = kv_cache_sublayer_path(cfg, j_in, out_dir)
    lock_path = output_path.with_suffix(output_path.suffix + ".lock")

    with exclusive_lock(lock_path):
        if not force and output_path.exists():
            cached = onnx.load(str(output_path), load_external_data=False)
            extra = tuple(i.name for i in cached.graph.input if i.name != KV_SUBLAYER_IN)
            return KVSublayer(output_path, j_in, layer_idx, extra)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        model = onnx.load(str(model_path), load_external_data=False)
        prefix = check_block_prefix(model, cfg.block_prefix)
        in_name = sublayer_tensor_name(*boundary_for_index(j_in), prefix, model)
        out_name = sublayer_tensor_name(*boundary_for_index(j_out), prefix, model)
        original_inputs = [i.name for i in model.graph.input]

        model = onnx.shape_inference.infer_shapes(model)
        onnx.load_external_data_for_model(model, str(model_path.parent))
        from onnx.utils import Extractor

        # Cut at the boundary tensor, keeping every original graph input available so the
        # extractor can resolve whatever the attention mask needs; unused ones are pruned below.
        extracted = Extractor(model).extract_model([in_name] + original_inputs, [out_name])
        used = {name for node in extracted.graph.node for name in node.input}
        kept = [i for i in extracted.graph.input if i.name == in_name or i.name in used]
        del extracted.graph.input[:]
        extracted.graph.input.extend(kept)

        _rename_tensor(extracted, in_name, KV_SUBLAYER_IN)
        _rename_tensor(extracted, out_name, KV_SUBLAYER_OUT)

        onnx.save(
            extracted, str(output_path),
            save_as_external_data=True, location=output_path.name + ".data",
        )
        onnx.checker.check_model(str(output_path))

    extra = tuple(i.name for i in kept if i.name != in_name)
    return KVSublayer(output_path, j_in, layer_idx, extra)


def ensure_kv_cache_sublayer_pullback(
    cfg: ModelConfig,
    j_in: int,
    batch: int,
    loss_type,
    force: bool = False,
) -> tuple[KVSublayer, Path]:
    """Builds (and caches) the pullback of the last-token sublayer subgraph for a fixed
    `batch` (the loss's mask / target / x inputs are (batch, 1, d_model)), via the same
    generate_subgraph_pullback the cacheless pipeline uses. Returns (sublayer, pullback_path).
    The pullback graph's gradient output is f"{KV_SUBLAYER_IN}_grad" and its extra
    output is KV_SUBLAYER_OUT, as in the cacheless pullbacks."""
    from construct.fs_utils import _strip_last_quantifier
    from construct.pullback import generate_subgraph_pullback

    work_dir = kv_cache_model_dir(cfg) / f"pullback_{loss_type.value}_b{batch}"
    sub = ensure_kv_cache_sublayer(cfg, j_in, out_dir=work_dir, force=force)
    pullback_path = sub.path.with_name(_strip_last_quantifier(sub.path.stem) + ".pullback.onnx")
    lock_path = pullback_path.with_suffix(".onnx.lock")
    with exclusive_lock(lock_path):
        if force or not pullback_path.exists():
            generate_subgraph_pullback(
                sub.path, loss_type, [batch, 1, cfg.d_model], KV_SUBLAYER_IN, KV_SUBLAYER_OUT,
            )
    return sub, pullback_path


def ensure_kv_cache_boundaries(
    cfg: ModelConfig,
    force: bool = False,
) -> tuple[Path, list[str], int]:
    """Like ensure_kv_cache_extra_outputs, but one subgraph whose outputs are *every* residual
    boundary of the config (index j == boundary_for_index(j), so the j-th name lines up with
    cfg.sublayers[j].output) followed by every layer's present K/V. Returns
    (path, boundary_tensor_names, n_layers)."""
    model_path = kv_cache_model_path(cfg)
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} does not exist -- call ensure_kv_cache_model(cfg) first")
    output_path = kv_cache_model_dir(cfg) / "extra_outputs.all_boundaries.onnx"
    lock_path = output_path.with_suffix(output_path.suffix + ".lock")

    with exclusive_lock(lock_path):
        model = onnx.load(str(model_path), load_external_data=False)
        prefix = check_block_prefix(model, cfg.block_prefix)
        input_names = [i.name for i in model.graph.input]
        n_layers = sum(1 for n in input_names if n.startswith("past_key_values.") and n.endswith(".key"))
        names = [
            sublayer_tensor_name(*boundary_for_index(j), prefix, model)
            for j in range(len(cfg.sublayers))
        ]
        if not force and output_path.exists():
            return output_path, names, n_layers

        model = onnx.shape_inference.infer_shapes(model)
        onnx.load_external_data_for_model(model, str(model_path.parent))
        from onnx.utils import Extractor

        extracted = Extractor(model).extract_model(input_names, names + present_kv_names(n_layers))
        onnx.save(
            extracted, str(output_path),
            save_as_external_data=True, location=output_path.name + ".data",
        )
        onnx.checker.check_model(str(output_path))
    return output_path, names, n_layers
