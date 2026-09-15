import os
from pathlib import Path

import onnx

from config import ModelConfig, SublayerConfig
from construct.fs_utils import _strip_last_quantifier, exclusive_lock
from construct.model_export import export_base_model
from construct.pullback import find_output_node, generate_subgraph_pullback
from construct.subgraph_extract import extract_subgraph


def exported_batch_size(model_path: Path) -> int | None:
    """The static batch dim (graph.input[0]'s dim 0) that `model_path` was
    exported with, or None if the file doesn't exist yet or that dim isn't a
    concrete int (e.g. a dynamic_axes export, which this codebase's own
    export_base_model never produces, but a model dropped in by hand
    might). Loaded with load_external_data=False since only the input shape
    -- not the weights -- is needed here."""
    if not model_path.exists():
        return None
    model = onnx.load(str(model_path), load_external_data=False)
    dim = model.graph.input[0].type.tensor_type.shape.dim[0]
    return dim.dim_value if dim.HasField("dim_value") else None


def ensure_base_model(
    cfg: ModelConfig,
    force: bool = False,
    verbose: bool = False,
    model=None,
):
    # All ensure methods return a Path
    """Exports cfg.base_model_path if it doesn't already exist (or always, if force), guarded
    by a lockfile. `model`: see export_base_model -- reuse an already-loaded model instead of
    loading cfg.model from HF fresh (e.g. when re-exporting the same model at many candidate
    batch sizes)."""
    lock_path = cfg.base_model_path.with_suffix(cfg.base_model_path.suffix + ".lock")
    with exclusive_lock(lock_path):
        if not force and cfg.base_model_path.exists():
            print(
                f"Base model already exists, skipping export -> {cfg.base_model_path}",
            )
        else:
            export_base_model(cfg, verbose, model=model)
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
    """Extracts a subgraph of the base model whose outputs are exactly
    `extra_outputs` (intermediate tensor names) -- unlike the base model's own
    "output" (final logits, e.g. through lm_head), which every current caller
    (generate_perturbed_sweep.py, generate_base_activations.py) discards
    without ever reading it. Every node not on some extra_outputs tensor's
    dependency path -- including the final logits projection, whose output
    is `(batch, seq, vocab_size)` and so is the single largest per-row
    allocation in the whole graph -- is dropped from the extracted graph, so
    ORT never computes or allocates it. This is why the batch size that fits
    the plain base model (or find_max_batch_size.py's torch-only probe)
    doesn't necessarily fit this graph before truncation: retaining a large,
    unused logits tensor alongside every requested sublayer activation could
    push memory well past what fits at the same batch size."""
    output_path = cfg.dir / "base_model_extra_outputs.onnx"
    lock_path = output_path.with_suffix(output_path.suffix + ".lock")
    with exclusive_lock(lock_path):
        if not force and output_path.exists():
            print(
                f"Extra outputs model already exists, skipping generation -> {output_path}",
            )
        else:
            input_path = cfg.base_model_path

            # Every requested tensor must be a known value_info/output/input
            # in the (shape-inferred) graph before extraction -- this also
            # gives a clear error immediately if a sublayer name is wrong,
            # rather than a confusing failure from inside Extractor.
            if os.path.getsize(input_path) > onnx.checker.MAXIMUM_PROTOBUF:
                onnx.shape_inference.infer_shapes_path(str(input_path), str(output_path))
                model = onnx.load(output_path)
            else:
                model = onnx.load(input_path, load_external_data=False)
                model = onnx.shape_inference.infer_shapes(model)
                base_dir = os.path.dirname(input_path)
                onnx.load_external_data_for_model(model, base_dir)
            for name in extra_outputs:
                find_output_node(model, name)

            extract_subgraph(
                input_path=input_path,
                output_path=output_path,
                input_names=cfg.input_name,
                output_names=list(extra_outputs),
            )
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
