from huggingface_hub.hf_api import PaperAuthor
import contextlib
import importlib
import os
import shutil
import time
from pathlib import Path
import tempfile
import onnx
import onnxruntime.training.onnxblock as onnxblock
import torch
from matplotlib.pylab import full
from onnx import helper
from onnx.utils import Extractor
from PIL.ImImagePlugin import i
from transformers.models.auto import AutoModelForCausalLM
from generate_artifact import generate_artifacts
from config import LossType, ModelConfig, SublayerConfig
from loss import (
    MaskedSumLoss,
    AdversarialLoss_EnvelopeDiffLoss,
    AdversarialLoss_SquaredDiffLoss,
    AdversarialLoss_NormalizedSquaredDiffLoss,
)
import hashlib

def _strip_last_quantifier(path:str) -> str:
    return ".".join(path.split(".")[:-1])
    
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
    input_names: str,
    output_names: str,
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

    # if len(input_names) != len(set(input_names)):
    #     print(input_names, flush=True)
    #     print(set(input_names), flush=True)
    #     print(len(input_names), flush=True)
    #     print(len(set(input_names)), flush=True)

    # raise ValueError("Duplicate names found in the input tensor names.")
    # if len(output_names) != len(set(output_names)):
    #     raise ValueError("Duplicate names found in the output tensor names.")

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
    extracted = e.extract_model([input_names], [output_names])

    location = os.path.basename(output_path) + ".data"
    onnx.save(extracted, output_path, save_as_external_data=True, location=location)

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


def ensure_subgraphs(
    cfg: ModelConfig, force: bool = False, do_not_materialise_subgraphs: bool = False,
) -> list[tuple[Path, SublayerConfig]]:
    """Extracts cfg.per_pair_dir's subgraphs if they don't already all exist (or always, if
    force), guarded by a lockfile."""

    cfg.sub_block_path.mkdir(parents=True, exist_ok=True)

    paths = [
        (
            cfg.sub_block_path
            / f"{sublayer_conf.input}.{sublayer_conf.output}.subgraph.forward.onnx",
            sublayer_conf,
        )
        for cnt, sublayer_conf in enumerate(cfg.sublayers)
    ]
    if do_not_materialise_subgraphs:
        return paths
    for path, sublayer_conf in paths:
        lock_path = path.with_name(path.name + ".lock")
        with exclusive_lock(lock_path):
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
    force: bool = False,
    base_model_path: Path | None = None,
) -> dict[str, list[Path]]:
    generated_paths = {}

    for path, sublayer_conf in paths:
        pullback_path = path.with_name(_strip_last_quantifier(path.stem) + ".pullback.onnx")
        lock_path = path.with_name(pullback_path.name + ".lock")
        with exclusive_lock(lock_path):
            if not pullback_path.exists() or force:
                res = generate_subgraph_pullback(
                    path,
                    sublayer_conf.loss_type,
                    sublayer_conf.input_shape,
                    sublayer_conf.input,
                    sublayer_conf.output,
                    base_model_path=base_model_path,
                )
                for k, v in res.items():
                    generated_paths.setdefault(k, []).append(v)
                print(f"Extracted subgraph -> {path}")
            else:
                generated_paths.setdefault("gradient", []).append(pullback_path)
                print(f"Subgraph pullbac already exist, skipping extraction -> {path}")
                print(
                    f"Subgraph pullbac already exist, have not checked for optimizer -> {path}",
                )

    return generated_paths


def generate_loss(
    sub_graph_path: Path,
    loss_type: LossType,
    input_shape: list[int] | None,
    output_dtype: int,
    temp_dir: Path | None = None,
) -> tuple[Path, onnxblock.blocks.Block]:
    match loss_type:
        case LossType.MaskedSumLoss:
            assert input_shape is not None
            assert temp_dir is not None, "temp_dir required for MaskedSumLoss"

            target_path = _jacobian_modification(
                sub_graph_path,
                temp_dir,
                input_shape,
                output_dtype,
            )
            tag = f"{sub_graph_path.stem}_pullback"
            loss = MaskedSumLoss(tag, target_path.parent)

        case LossType.AdversarialLoss_SquaredDiffLoss:
            target_path = sub_graph_path
            loss = AdversarialLoss_SquaredDiffLoss()

        case LossType.AdversarialLoss_NormalizedSquaredDiffLoss:
            target_path = sub_graph_path
            loss = AdversarialLoss_NormalizedSquaredDiffLoss()

        case LossType.AdversarialLoss_EnvelopeDiffLoss:
            target_path = sub_graph_path
            loss = AdversarialLoss_EnvelopeDiffLoss()

        case _:
            raise ValueError(f"Invalid loss type: {loss_type}")

    return target_path, loss


def _jacobian_modification(
    path: Path,
    save_dir: Path,
    input_shape: list[int],
    output_dtype: int,
) -> Path:
    save_path = save_dir / f"{path.stem}_mod.onnx"
    data_filename = f"{save_path.name}.data"

    base_model = onnx.load(str(path))
    base_model.graph.input.append(
        helper.make_tensor_value_info(
            "mask",
            output_dtype,
            input_shape,
        ),
    )

    onnx.save(
        base_model,
        str(save_path),
        save_as_external_data=True,
        location=data_filename,
    )
    return save_path


def generate_subgraph_pullback(
    sub_graph_path: Path,
    loss_type: LossType,
    input_shape: list[int] | None,
    input_name: str,
    output_name: str,
    base_model_path: Path | None = None,
) -> dict[str, Path]:
    # This will always override -- make sure file protection is elsewhere

    """Builds one sublayer's own independent gradient graph"""
    # ONNX Save removes base_dir info so when saving multiple tensors, we end up with errors
    # Using the absolute path remedies this directly.
    sub_graph_path = sub_graph_path.resolve()
    tag = ".pullback"

    # get the dtype of the output.
    # we load the model
    model_graph = onnx.load(base_model_path or sub_graph_path, load_external_data=False)
    # get the node called output_name
    output_node = next(
        vi for vi in model_graph.graph.value_info if vi.name == output_name
    )
    output_dtype = output_node.type.tensor_type.elem_type
    del model_graph

    with tempfile.TemporaryDirectory() as tmp:
        target_path, loss = generate_loss(
            base_model_path or sub_graph_path,
            loss_type,
            input_shape,
            output_dtype,
            temp_dir=Path(tmp),
        )

        prefix = _strip_last_quantifier(sub_graph_path.stem) + tag
        generated_paths = generate_artifacts(
            target_path,
            save_directory=sub_graph_path.parent,
            requires_grad=[input_name],
            loss_input_names=[output_name],
            loss=loss,
            gradient_model_name=f"{prefix}.onnx",
            save_loss_model=False,
            optimizer_model_name=f"{prefix}_optimizer.onnx",
        )

    return generated_paths


def generate_union_of_subgraphs(
    paths: list[Path],
):
    merged_path = "-".join([".".join(path.stem.split(".")[0:2]) for path in paths])
    hash = hashlib.md5(string=merged_path.encode("utf-8")).hexdigest()[:8]
    final_name =  "-".join([".".join(path.stem.split(".")[0:2]) for path in [paths[0], paths[-1]]])+"_"+hash
    base = onnx.load(paths[0])
    merged_graph = onnx.GraphProto()
    merged_graph.CopyFrom(base.graph)

    node_by_name = {n.name: n for n in merged_graph.node}
    init_by_name = {i.name: i for i in merged_graph.initializer}
    vi_by_name = {v.name: v for v in merged_graph.value_info}
    output_names = {o.name for o in merged_graph.output}

    for pair_idx, path in enumerate(paths[1:], start=1):
        g = onnx.load(path).graph

        for i in g.input:
            if i.name not in {inp.name for inp in merged_graph.input}:
                merged_graph.input.append(i)

        for n in g.node:
            existing = node_by_name.get(n.name)
            if existing is not None:
                if existing.SerializeToString() != n.SerializeToString():
                    raise ValueError(
                        f"node {n.name!r} differs between graphs; cannot union",
                    )
                continue
            merged_graph.node.append(n)
            node_by_name[n.name] = n

        for i in g.initializer:
            existing = init_by_name.get(i.name)
            if existing is not None:
                if existing.raw_data != i.raw_data or list(existing.dims) != list(
                    i.dims,
                ):
                    raise ValueError(
                        f"initializer {i.name!r} differs between graphs; cannot union",
                    )
                continue
            merged_graph.initializer.append(i)
            init_by_name[i.name] = i

        for v in g.value_info:
            if v.name not in vi_by_name:
                merged_graph.value_info.append(v)
                vi_by_name[v.name] = v

        for o in g.output:
            if o.name not in output_names:
                merged_graph.output.append(o)
                output_names.add(o.name)

    merged_model = onnx.ModelProto()
    merged_model.CopyFrom(base)
    merged_model.graph.CopyFrom(merged_graph)
    
    

    save_path = paths[0].with_name(final_name + ".union.onnx")
    location = os.path.basename(save_path) + ".data"
    onnx.save(
        merged_model,
        save_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=location,
    )
    return save_path
