"""Extracts a sublayer-activation-plus-KV-cache subgraph from a KV-cache
model.onnx (see construct/kv_cache_export.py), for use by
scripts/generate_perturbed_sweep_kvcache.py.

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

from pathlib import Path

import onnx

from config import ModelConfig
from construct.fs_utils import exclusive_lock
from construct.kv_cache_export import kv_cache_model_dir, kv_cache_model_path


def sublayer_tensor_name(layer_idx: int, boundary: str) -> str:
    """`boundary`: "attn_out" (post-attention residual, i.e. the old add_N
    scheme's sublayer "input" boundary) or "mlp_out" (post-MLP residual,
    the "output" boundary and also the next layer's "input")."""
    if boundary == "attn_out":
        return f"/model/layers.{layer_idx}/Add_output_0"
    if boundary == "mlp_out":
        return f"/model/layers.{layer_idx}/Add_1_output_0"
    raise ValueError(f"boundary must be 'attn_out' or 'mlp_out', got {boundary!r}")


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
    """Extracts (and caches) a subgraph of cfg's KV-cache model whose outputs
    are the requested sublayer tensor plus every layer's present KV, dropping
    logits. Returns (subgraph_path, sublayer_output_name, n_layers)."""
    model_path = kv_cache_model_path(cfg)
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} does not exist -- call ensure_kv_cache_model(cfg) first",
        )

    sublayer_output = sublayer_tensor_name(layer_idx, boundary)
    output_path = kv_cache_extra_outputs_path(cfg, layer_idx, boundary)
    lock_path = output_path.with_suffix(output_path.suffix + ".lock")

    with exclusive_lock(lock_path):
        base_model = onnx.load(str(model_path), load_external_data=False)
        input_names = [i.name for i in base_model.graph.input]
        n_layers = sum(
            1 for name in input_names if name.startswith("past_key_values.") and name.endswith(".key")
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
