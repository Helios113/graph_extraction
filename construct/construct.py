import os
from pathlib import Path

import onnx

from config import ModelConfig, SublayerConfig
from construct.fs_utils import _strip_last_quantifier, exclusive_lock
from construct.model_export import export_base_model
from construct.pullback import find_output_node, generate_subgraph_pullback
from construct.subgraph_extract import extract_subgraph


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
    cfg: ModelConfig,
    force: bool = False,
    do_not_materialise_subgraphs: bool = False,
) -> list[tuple[Path, SublayerConfig]]:
    """Extract subgraphs from base model,
    do_not_materialise_subgraphs is set true, the subgraphs are not materialised as onnx files but kept logically. This is used to generate a jacobians insitu
    """

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


def ensure_extra_outputs(
    cfg: ModelConfig,
    extra_outputs: list[str],
    force: bool = False,
) -> Path:
    """Adds extra_outputs (names of intermediate tensors) as additional outputs on a copy of
    the base model, without going through the sub-block extraction pipeline."""
    output_path = cfg.dir / "base_model_extra_outputs.onnx"
    lock_path = output_path.with_suffix(output_path.suffix + ".lock")
    with exclusive_lock(lock_path):
        if not force and output_path.exists():
            print(
                f"Extra outputs model already exists, skipping generation -> {output_path}",
            )
        else:
            input_path = cfg.base_model_path
            if os.path.getsize(input_path) > onnx.checker.MAXIMUM_PROTOBUF:
                onnx.shape_inference.infer_shapes_path(
                    str(input_path), str(output_path)
                )
                model = onnx.load(output_path)
            else:
                model = onnx.load(input_path, load_external_data=False)
                model = onnx.shape_inference.infer_shapes(model)
                base_dir = os.path.dirname(input_path)
                onnx.load_external_data_for_model(model, base_dir)

            existing_outputs = {vi.name for vi in model.graph.output}
            for name in extra_outputs:
                if name in existing_outputs:
                    continue
                vi = find_output_node(model, name)
                new_output = vi.__class__()
                new_output.CopyFrom(vi)
                model.graph.output.append(new_output)

            location = os.path.basename(output_path) + ".data"
            onnx.save(model, output_path, save_as_external_data=True, location=location)
            onnx.checker.check_model(output_path)
            print(f"Extra outputs model generated -> {output_path}")

    return output_path


def ensure_subgraph_pullback(
    paths: list[tuple[Path, SublayerConfig]],
    force: bool = False,
    base_model_path: Path | None = None,
) -> list[Path]:
    """
    base_model_path -- if we want to extract jacobians from a subgraph without materialising it
    """

    generated_paths = []

    for path, sublayer_conf in paths:
        pullback_path = path.with_name(
            _strip_last_quantifier(path.stem) + ".pullback.onnx"
        )
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
                    optimizer_type=sublayer_conf.optimizer_type,
                )
                generated_paths.append(res)
            else:
                # generated_paths.setdefault("gradient", []).append(pullback_path)
                generated_paths.append(pullback_path)

                print(f"Subgraph pullbac already exist, skipping extraction -> {path}")
                print(
                    f"Subgraph pullbac already exist, have not checked for optimizer -> {path}",
                )

    return generated_paths
