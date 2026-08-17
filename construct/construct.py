from pathlib import Path

from config import ModelConfig, SublayerConfig
from construct.fs_utils import _strip_last_quantifier, exclusive_lock
from construct.model_export import export_base_model
from construct.pullback import generate_subgraph_pullback
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
