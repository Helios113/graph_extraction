"""Builds a "twin" ORT-training graph for one sublayer: two copies of the same x + g(x)
subgraph (copy A keeps the original tensor names, copy B is prefixed "y_"), joined by a
squared-difference loss between the two copies' outputs, with an AdamW optimizer graph
that updates copy B's input ("y", a trainable initializer) towards colliding with copy A's
output ("x", a plain input).

Usage: from collision_opt.build_twin_graph import build_twin_training_model
"""
import sys
from pathlib import Path

import numpy as np
import onnx
import onnx.compose
import onnxruntime.training.artifacts as artifacts
import onnxruntime.training.onnxblock as onnxblock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ModelConfig, SublayerConfig
from new_code.construct_utils import _add_layer_norm_stats_outputs, chdir, exclusive_lock


class SquaredDiffLoss(onnxblock.blocks.Block):
    """loss = sum((f(x) - f(y))**2). Unlike MaskedSumLoss, no `mask` input -- every element of
    both copies' outputs contributes to the loss directly."""

    def __init__(self):
        super().__init__()
        self._cast_a = onnxblock.blocks.Cast(onnx.TensorProto.FLOAT)
        self._cast_b = onnxblock.blocks.Cast(onnx.TensorProto.FLOAT)
        self._sub = onnxblock.blocks.Sub()
        self._pow = onnxblock.blocks.Pow(2.0)
        self._reduce_sum = onnxblock.blocks.ReduceSum(keepdims=False)

    def build(self, a_name, b_name):
        diff = self._sub(self._cast_a(a_name), self._cast_b(b_name))
        return self._reduce_sum(self._pow(diff))


class NormalizedSquaredDiffLoss(onnxblock.blocks.Block):
    """loss = sum((f(x) - f(y))**2) / sum((x - y)**2) -- output squared-distance normalized
    by input squared-distance, so shrinking ||x - y|| doesn't trivially shrink the loss on
    its own; the optimizer is rewarded only for closing the output gap faster than the input
    gap grows."""

    def __init__(self):
        super().__init__()
        self._cast_x = onnxblock.blocks.Cast(onnx.TensorProto.FLOAT)
        self._cast_y = onnxblock.blocks.Cast(onnx.TensorProto.FLOAT)
        self._cast_a = onnxblock.blocks.Cast(onnx.TensorProto.FLOAT)
        self._cast_b = onnxblock.blocks.Cast(onnx.TensorProto.FLOAT)
        self._sub_inputs = onnxblock.blocks.Sub()
        self._sub_outputs = onnxblock.blocks.Sub()
        self._pow_inputs = onnxblock.blocks.Pow(2.0)
        self._pow_outputs = onnxblock.blocks.Pow(2.0)
        self._reduce_sum_inputs = onnxblock.blocks.ReduceSum(keepdims=False)
        self._reduce_sum_outputs = onnxblock.blocks.ReduceSum(keepdims=False)
        self._div = onnxblock.blocks.Div()

    def build(self, x_name, y_name, a_name, b_name):
        input_diff = self._sub_inputs(self._cast_x(x_name), self._cast_y(y_name))
        input_sq_dist = self._reduce_sum_inputs(self._pow_inputs(input_diff))

        output_diff = self._sub_outputs(self._cast_a(a_name), self._cast_b(b_name))
        output_sq_dist = self._reduce_sum_outputs(self._pow_outputs(output_diff))

        return self._div(output_sq_dist, input_sq_dist)


def _prepare_base_subgraph(cfg: ModelConfig, sublayer: SublayerConfig, sub_graph_path: Path) -> onnx.ModelProto:
    """Loads the already-extracted sublayer subgraph (see collisions/search_graphs.py's
    ensure_sub_graphs) and applies the LayerNorm-stats patch if this model needs it --
    ORT training's gradient-graph building crashes on any LayerNormalization node missing
    Mean/InvStdDev outputs (see convert_model.py's _add_layer_norm_stats_outputs docstring),
    same requirement as the jacobians pipeline's severable model."""
    model = onnx.load(str(sub_graph_path))
    if cfg.needs_layernorm_patch:
        model = _add_layer_norm_stats_outputs(model)
    return model


def _build_twin_graph(cfg: ModelConfig, sublayer: SublayerConfig, base_model: onnx.ModelProto) -> onnx.ModelProto:
    """Concatenates a copy of base_model (unprefixed, feeding `sublayer.input`) with a
    "y_"-prefixed copy (feeding "y_" + sublayer.input) into one graph with both copies'
    nodes/initializers/value_info -- zero name collisions by construction, so no dedup/union
    logic is needed (unlike union_per_pair_artifacts.py, which unions graphs that are meant
    to share their forward pass)."""
    copy_a = onnx.ModelProto()
    copy_a.CopyFrom(base_model)

    copy_b = onnx.compose.add_prefix(base_model, prefix="y_", inplace=False)

    merged = onnx.ModelProto()
    merged.CopyFrom(copy_a)
    merged.graph.ClearField("name")
    merged.graph.name = "twin_graph"
    merged.graph.node.extend(copy_b.graph.node)
    merged.graph.initializer.extend(copy_b.graph.initializer)
    merged.graph.value_info.extend(copy_b.graph.value_info)
    merged.graph.input.extend(copy_b.graph.input)
    merged.graph.output.extend(copy_b.graph.output)
    return merged


def _clear_value_info(graph: onnx.GraphProto, name: str) -> None:
    for vi in list(graph.value_info):
        if vi.name == name:
            graph.value_info.remove(vi)


def _rewrite_bf16_input_as_float(graph: onnx.GraphProto, name: str, shape: list[int]) -> None:
    """Rewrites graph input `name` (bf16) so it's exposed as a float32 graph input feeding
    an internal Cast-to-bf16 -- same boundary-cast idea as collisions/max_search.py's
    _wrap_bfloat16_io_with_float32 (ORT's Python bindings can't accept bf16 numpy arrays),
    needed here because training.api.Module.__call__ feeds user inputs as plain numpy."""
    inp = next(i for i in graph.input if i.name == name)
    bf16_name = f"{name}_bf16"
    for node in graph.node:
        node.input[:] = [bf16_name if x == name else x for x in node.input]
    inp.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    _clear_value_info(graph, name)
    graph.node.insert(0, onnx.helper.make_node("Cast", [name], [bf16_name], to=onnx.TensorProto.BFLOAT16))
    graph.value_info.append(onnx.helper.make_tensor_value_info(bf16_name, onnx.TensorProto.BFLOAT16, shape))


def _make_y_a_trainable_initializer(graph: onnx.GraphProto, name: str, shape: list[int]) -> None:
    """Converts graph input `name` (y's bf16 input tensor) into a float32 initializer feeding
    an internal Cast-to-bf16, matching _rewrite_bf16_input_as_float's boundary-cast pattern.
    generate_artifacts' AdamW optimizer only ever optimizes graph.initializer entries (see
    get_model_parameters, which iterates initializers only) -- requires_grad on a plain graph
    input is silently dropped from the trainable-parameter bucketing, so `y` must become an
    initializer for the optimizer artifact to actually update it. Initialized to zeros here;
    optimize.py overwrites it with the real y0 per (x, y) restart via CheckpointState."""
    inp = next(i for i in graph.input if i.name == name)
    bf16_name = f"{name}_bf16"
    for node in graph.node:
        node.input[:] = [bf16_name if x == name else x for x in node.input]
    graph.input.remove(inp)
    _clear_value_info(graph, name)
    zeros = np.zeros(int(np.prod(shape)), dtype=np.float32).tobytes()
    graph.initializer.append(onnx.helper.make_tensor(name, onnx.TensorProto.FLOAT, shape, zeros, raw=True))
    graph.node.insert(0, onnx.helper.make_node("Cast", [name], [bf16_name], to=onnx.TensorProto.BFLOAT16))
    graph.value_info.append(onnx.helper.make_tensor_value_info(bf16_name, onnx.TensorProto.BFLOAT16, shape))


def _artifact_dir(cfg: ModelConfig, sublayer_idx: int, tag: str) -> Path:
    return cfg.collision_opt_graph_dir / f"{tag}_artifacts"


def training_model_path(cfg: ModelConfig, sublayer_idx: int) -> Path:
    tag = _tag(cfg, sublayer_idx)
    return _artifact_dir(cfg, sublayer_idx, tag) / f"{tag}_training_model.onnx"


def eval_model_path(cfg: ModelConfig, sublayer_idx: int) -> Path:
    tag = _tag(cfg, sublayer_idx)
    return _artifact_dir(cfg, sublayer_idx, tag) / f"{tag}_eval_model.onnx"


def optimizer_model_path(cfg: ModelConfig, sublayer_idx: int) -> Path:
    tag = _tag(cfg, sublayer_idx)
    return _artifact_dir(cfg, sublayer_idx, tag) / f"{tag}_optimizer_model.onnx"


def checkpoint_path(cfg: ModelConfig, sublayer_idx: int) -> Path:
    tag = _tag(cfg, sublayer_idx)
    return _artifact_dir(cfg, sublayer_idx, tag) / f"{tag}_checkpoint"


def _tag(cfg: ModelConfig, sublayer_idx: int) -> str:
    sublayer = cfg.sublayers[sublayer_idx]
    return f"sublayer{sublayer_idx}_{sublayer.input}_{sublayer.output}"


def y_input_name(sublayer: SublayerConfig) -> str:
    return f"y_{sublayer.input}"


def y_output_name(sublayer: SublayerConfig) -> str:
    return f"y_{sublayer.output}"


def build_twin_training_model(cfg: ModelConfig, sublayer_idx: int, force: bool = False) -> Path:
    """Builds sublayer_idx's twin ORT-training artifacts (training/eval/optimizer models +
    checkpoint) via generate_artifacts: forward pass computes both sublayer.output (from
    sublayer.input, "x") and y_output_name (from y_input_name, "y"), backward pass is seeded
    from SquaredDiffLoss(sublayer.output, y_output_name) with requires_grad=[y_input_name]
    and optimizer=AdamW -- gradients and optimizer updates flow into y only, x is a plain
    (non-trainable) graph input. Returns the training model's path.

    Two build-environment fixes applied before generate_artifacts runs (both worked around
    at the graph level, not by changing what's computed):

    1. y_input_name(sublayer) is rewritten from a graph input to a trainable float32
       initializer (_make_y_a_trainable_initializer) -- generate_artifacts' AdamW optimizer
       only ever optimizes graph.initializer entries, so a plain input can't be a trainable
       "parameter" as far as the optimizer artifact is concerned.
    2. Both x and y's boundary tensors are exposed as float32 (Cast to bf16 immediately
       inside the graph) -- training.api.Module.__call__ feeds/reads plain numpy, which can't
       hold bf16 (ml_dtypes' bfloat16 isn't accepted), same reasoning as
       collisions/max_search.py's _wrap_bfloat16_io_with_float32.

    Guarded by an exclusive_lock + exists-check, same reasoning as
    convert_model.py's build_gradient_graph: two jobs sharing this config's `name` must not
    race on the same output file.
    """
    out_path = training_model_path(cfg, sublayer_idx)
    lock_path = out_path.with_suffix(out_path.suffix + ".lock")
    with exclusive_lock(lock_path):
        if not force and out_path.exists():
            print(f"Twin training graph already exists, skipping build -> {out_path}")
            return out_path

        artifact_dir = _artifact_dir(cfg, sublayer_idx, _tag(cfg, sublayer_idx))
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for stale in artifact_dir.glob("temp*.onnx*"):
            stale.unlink()

        sublayer = cfg.sublayers[sublayer_idx]
        tag = _tag(cfg, sublayer_idx)
        sub_graph_path = cfg.per_pair_dir / f"sublayer{sublayer_idx}_{sublayer.input}_{sublayer.output}_subgraph_eval.onnx"

        # base_model = _prepare_base_subgraph(cfg, sublayer, sub_graph_path)
      
      
        base_model = model = onnx.load(str(sub_graph_path))
      
        base_scratch_path = artifact_dir / f"sublayer{sublayer_idx}_base.onnx"
        
        onnx.save(base_model, str(base_scratch_path))
        onnx.checker.check_model(str(base_scratch_path))

        twin_model = _build_twin_graph(cfg, sublayer, onnx.load(str(base_scratch_path)))
        
        
        
        shape = [cfg.batch, cfg.seq, cfg.d_model]
        _rewrite_bf16_input_as_float(twin_model.graph, sublayer.input, shape)
        _make_y_a_trainable_initializer(twin_model.graph, y_input_name(sublayer), shape)
        twin_path = (artifact_dir / f"sublayer{sublayer_idx}_twin.onnx").resolve()
        onnx.save(twin_model, str(twin_path))
        onnx.checker.check_model(str(twin_path))

        if cfg.collision_opt_loss == "normalized":
            loss = NormalizedSquaredDiffLoss()
            loss_input_names = [
                sublayer.input, y_input_name(sublayer),
                sublayer.output, y_output_name(sublayer),
            ]
        else:
            loss = SquaredDiffLoss()
            loss_input_names = [sublayer.output, y_output_name(sublayer)]

        artifact_dir_abs = artifact_dir.resolve()
        with chdir(artifact_dir_abs):
            artifacts.generate_artifacts(
                onnx.load(str(twin_path)),
                optimizer=artifacts.OptimType.SGD,
                requires_grad=[y_input_name(sublayer)],
                loss_input_names=loss_input_names,
                prefix=f"{tag}_",
                loss=loss,
                artifact_directory=str(artifact_dir_abs),
            )

        # # This ORT build's onnx-package IR version (13) exceeds the training runtime's max
        # # supported IR version (10) -- generate_artifacts saves the optimizer model at the
        # # installed onnx package's default IR version, but (for reasons not fully understood)
        # # only the optimizer model actually lands on 13 while the training/eval models land
        # # on 10, so only the optimizer model needs downgrading before training.api.Optimizer
        # # can load it.
        opt_path = optimizer_model_path(cfg, sublayer_idx)
        opt_model = onnx.load(str(opt_path))
        opt_model.ir_version = 10
        onnx.save(opt_model, str(opt_path))

        # print(f"Saved twin training artifacts -> {artifact_dir}")
        return out_path
