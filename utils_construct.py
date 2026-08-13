"""
Usage: uv run python3 convert_model.py configs/gpt2.toml
"""

import contextlib
import importlib
import os
import shutil
import time
from pathlib import Path

import onnx
import onnxruntime.training.artifacts as artifacts
import onnxruntime.training.onnxblock as onnxblock
import torch
from matplotlib.pylab import full
from onnx import helper
from onnx.utils import Extractor
from PIL.ImImagePlugin import i
from transformers.models.auto import AutoModelForCausalLM

from config import LossType, ModelConfig, SublayerConfig, grad_names
from loss import MaskedSumLoss, NormalizedSquaredDiffLoss, SquaredDiffLoss
from union_per_pair_artifacts import build_union_artifacts


def _dummy_input(cfg: ModelConfig) -> torch.Tensor:
    if cfg.input_name == "input_ids":
        return torch.randint(0, cfg.vocab_size, (cfg.batch, cfg.seq))
    assert cfg.input_dim is not None, (
        f"{cfg.name}: input_dim required in config for non-input_ids models"
    )
    return torch.randn(cfg.batch, cfg.seq, cfg.input_dim)


def _needs_embeds_kwarg(cfg: ModelConfig) -> bool:
    """`inputs_embeds` isn't the first positional arg in HF forward signatures (input_ids
    is), so it must be exported as a kwarg rather than a positional tensor -- otherwise it
    silently lands in e.g. `past_key_values` instead."""
    return cfg.input_name == "inputs_embeds" and ":" not in cfg.model


def _load_model(spec: str) -> torch.nn.Module:

    model = AutoModelForCausalLM.from_pretrained(spec)
    model.eval()
    model.config.use_cache = False
    return model


def _random_init(model: torch.nn.Module, seed: int) -> None:
    """Reinitialize every parameter in-place with fresh random weights (matching each
    tensor's existing shape/dtype), instead of the pretrained/factory-built values
    _load_model produced -- works for any module (HF or custom factory) since it doesn't
    rely on architecture-specific init_weights()/reset_parameters() methods."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(
                torch.randn(param.shape, generator=generator, dtype=param.dtype),
            )


@contextlib.contextmanager
def exclusive_lock(lock_path: Path, poll_seconds: float = 2.0):
    """Waits until `lock_path` can be created exclusively (os.O_CREAT|O_EXCL is atomic even
    on network filesystems, unlike flock -- this repo's onnx_models/ can be reached from
    several cluster nodes at once), then holds it for the block, deleting it on exit (even
    on error, so a crash doesn't wedge future runs forever)."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            time.sleep(poll_seconds)
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


@contextlib.contextmanager
def chdir(path: Path):
    """Temporarily changes the process cwd. Every path passed to the wrapped block must be
    absolute -- callers are responsible for resolving any relative path first."""
    prev = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(prev)


def _add_layer_norm_stats_outputs(
    model: onnx.ModelProto,
) -> onnx.ModelProto:
    model = onnx.shape_inference.infer_shapes(model)
    shapes = {
        vi.name: vi
        for vi in list(model.graph.value_info)
        + list(model.graph.input)
        + list(model.graph.output)
    }
    # dtype = _weight_dtype(model)

    for node in model.graph.node:
        if node.op_type != "LayerNormalization" or len(node.output) > 1:
            continue
        x_vi = shapes.get(node.input[0])
        if x_vi is None or not x_vi.type.tensor_type.HasField("shape"):
            raise RuntimeError(
                f"No static shape for {node.input[0]!r} feeding {node.name}",
            )
        x_shape = [d.dim_value for d in x_vi.type.tensor_type.shape.dim]
        axis = next((a.i for a in node.attribute if a.name == "axis"), -1)
        if axis < 0:
            axis += len(x_shape)
        stats_shape = x_shape[:axis] + [1] * (len(x_shape) - axis)

        mean_name, invstd_name = node.name + "_Mean", node.name + "_InvStdDev"
        node.output.extend([mean_name, invstd_name])
        for out_name in (mean_name, invstd_name):
            model.graph.value_info.append(
                helper.make_tensor_value_info(out_name, dtype, stats_shape),
            )

    return model


def _add_layer_value_outputs(
    model: onnx.ModelProto,
    layer_names: list[str],
    cfg: ModelConfig,
) -> onnx.ModelProto:
    """Expose each severed layer's own activations as a graph output (not just its
    gradient), so callers can read e.g. `add_1` alongside `add_1_grad`. generate_artifacts
    keeps the full forward pass regardless of which slice has requires_grad, so every layer
    name is already a node output in every per-pair graph -- it just isn't declared yet.
    """
    existing = {o.name for o in model.graph.output}
    dtype = _weight_dtype(model)
    for name in layer_names:
        if name not in existing:
            model.graph.output.append(
                helper.make_tensor_value_info(
                    name,
                    dtype,
                    [cfg.batch, cfg.seq, cfg.d_model],
                ),
            )
    return model

def export_base_model(
    cfg: ModelConfig,
    verbose,
):
    cfg.dir.mkdir(parents=True, exist_ok=True)
    model = _load_model(cfg.model)
    if cfg.random_init:
        _random_init(model, cfg.random_init_seed)
    dummy_input = _dummy_input(cfg)
    if _needs_embeds_kwarg(cfg):
        args = ({"inputs_embeds": dummy_input},)
    else:
        args = (dummy_input,)
    torch.onnx.export(
        model,
        args,
        str(cfg.base_model_path),
        input_names=[cfg.input_name],
        output_names=["output"],
        verbose=verbose,
    )


def extract_subgraph(
    input_path: Path,
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
    check_model: bool = True,
    infer_shapes: bool = True,
) -> None:
    if not os.path.exists(input_path):
        raise ValueError(f"Invalid input model path: {input_path}")
    if not output_path:
        raise ValueError("Output model path shall not be empty!")
    if not input_names:
        raise ValueError("Input tensor names shall not be empty!")
    if not output_names:
        raise ValueError("Output tensor names shall not be empty!")

    if len(input_names) != len(set(input_names)):
        raise ValueError("Duplicate names found in the input tensor names.")
    if len(output_names) != len(set(output_names)):
        raise ValueError("Duplicate names found in the output tensor names.")

    if check_model:
        onnx.checker.check_model(input_path)

    if infer_shapes and os.path.getsize(input_path) > onnx.checker.MAXIMUM_PROTOBUF:
        onnx.shape_inference.infer_shapes_path(input_path, output_path)
        model = onnx.load(output_path)
    elif infer_shapes:
        model = onnx.load(input_path, load_external_data=False)
        model = onnx.shape_inference.infer_shapes(model)
        base_dir = os.path.dirname(input_path)
        onnx.load_external_data_for_model(model, base_dir)
    else:
        model = onnx.load(input_path)

    e = Extractor(model)
    extracted = e.extract_model(input_names, output_names)

    location = os.path.basename(output_path) + ".data"
    onnx.save(extracted, output_path, save_as_external_data=True, location=location)

    if check_model:
        onnx.checker.check_model(output_path)


def ensure_base_model(
    cfg: ModelConfig,
    force: bool = False,
    verbose: bool = False,
):
    # All ensure methods return a Path
    """Exports cfg.base_model_path if it doesn't already exist (or always, if force), guarded
    by a lockfile"""
    lock_path = cfg.base_model_path.with_suffix(cfg.base_model_path.suffix + ".lock")
    with exclusive_lock(lock_path):
        if not force and cfg.base_model_path.exists():
            print(
                f"Base model already exists, skipping export -> {cfg.base_model_path}",
            )
        else:
            export_base_model(cfg, verbose)
            print(f"Base model exported -> {cfg.base_model_path}")
            if cfg.needs_layernorm_patch:
                model = onnx.load(str(cfg.base_model_path), load_external_data=False)
                model = _add_layer_norm_stats_outputs(model)
                onnx.save(model, str(cfg.base_model_path))
                print(f"LayerNorm stats outputs added -> {cfg.base_model_path}")


def ensure_subgraphs(
    cfg: ModelConfig,
    force: bool = False,
) -> list[tuple[Path, SublayerConfig]]:
    """Extracts cfg.per_pair_dir's subgraphs if they don't already all exist (or always, if
    force), guarded by a lockfile."""

    cfg.sub_block_path.mkdir(parents=True, exist_ok=True)
    lock_path = cfg.sub_block_path.with_name(cfg.sub_block_path.name + ".lock")
    paths = [
        (
            cfg.sub_block_path
            / f"sublayer{cnt}_{sublayer_conf.input}_{sublayer_conf.output}_subgraph_forward.onnx",
            sublayer_conf,
        )
        for cnt, sublayer_conf in enumerate(cfg.sublayers)
    ]
    with exclusive_lock(lock_path):
        for path, sublayer_conf in paths:
            if not path.exists() or force:
                extract_subgraph(
                    input_path=cfg.base_model_path,
                    output_path=path,
                    input_names=sublayer_conf.input,
                    output_names=sublayer_conf.output,
                )
                print(f"Extracted subgraph -> {path}")
            else:
                print(f"Subgraph already exist, skipping extraction -> {path}")

    return paths


def ensure_subgraph_pullback(
    paths: list[tuple[Path, SublayerConfig]],
    cfg: ModelConfig,
    force: bool = False,
    full_jacobian: bool = False,
):
    # Wait on this
    #
    # Let's do the rest and see how the ensures will work
    # Maybe we don't want this path object
    for path, sublayer in paths:
        lock_path = path.with_name(path.name + ".lock")
        with exclusive_lock(lock_path):
            if not force and lock_path.exists():
                # This is wrong the path name here is not accurate
                print(f"Sub-graphs already exist, skipping extraction -> {path}")
            else:
                generate_subgraph_pullback(
                    path,
                    sublayer.loss_type,
                    sublayer.input_shape,
                    sublayer.input,
                    sublayer.output,
                )


def generate_loss(
    sub_graph_path: Path,
    loss_type: LossType,
    input_shape: list[int] | None,
) -> tuple[Path, onnxblock.blocks.Block]:
    mod_path = sub_graph_path.parent / (sub_graph_path.stem + "_mod.onnx")
    tag = sub_graph_path.stem + "_pullback"
    loss = onnxblock.blocks.Block()
    match loss_type:
        case LossType.MaskedSumLoss:
            assert input_shape is not None
            sub_graph_path = _jacobian_modification(
                sub_graph_path,
                mod_path,
                input_shape,
            )
            loss = MaskedSumLoss(tag, mod_path.parent)
        case LossType.SquaredDiffLoss:
            loss = SquaredDiffLoss()
        case LossType.NormalizedSquaredDiffLoss:
            loss = NormalizedSquaredDiffLoss()
        case _:
            raise ValueError(f"Invalid loss type: {loss_type}")

    return sub_graph_path, loss


def _jacobian_modification(
    path: Path,
    save_path: Path,
    input_shape: list[int],
) -> Path:
    base_model = onnx.load(str(path))
    base_model.graph.input.append(
        helper.make_tensor_value_info(
            "mask",
            onnx.TensorProto.FLOAT,
            input_shape,
        ),
    )
    onnx.save(
        base_model,
        str(save_path),
        save_as_external_data=True,
        location=os.path.basename(str(save_path)) + ".data",
    )

    return save_path


def generate_subgraph_pullback(
    sub_graph_path: Path,
    loss_type: LossType,
    input_shape: list[int] | None,
    input: list[str],
    output: list[str],
):
    # This will always override -- make sure file protection is elsewhere

    """Builds one sublayer's own independent gradient graph"""
    # ONNX Save removes base_dir info so when saving multiple tensors, we end up with errors
    # Using the absolute path remedies this directly.
    sub_graph_path = sub_graph_path.resolve()
    tag = "_pullback"

    sub_graph_path, loss = generate_loss(
        sub_graph_path,
        loss_type,
        input_shape,
    )

    with chdir(sub_graph_path.parent):
        artifacts.generate_artifacts(
            str(sub_graph_path.name),
            requires_grad=input,
            loss_input_names=output,
            loss=loss,
            artifact_directory=str(sub_graph_path.parent),
            prefix=sub_graph_path.stem + tag,
        )
    print(f"generated artifacts -> {sub_graph_path}")
