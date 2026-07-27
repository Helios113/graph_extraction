"""Full jacobians pipeline, driven by a single config: export base model -> build gradient
graph -> compute diagonal-block Jacobians -> save (raw, SVD, or slogdet-only streamed
incrementally to disk, per cfg.svd/cfg.slogdet_only).

Usage: uv run python3 jacobians/run_pipeline.py configs/qwen_block0_jacobians.toml
"""
import argparse
import sys
from pathlib import Path
import shutil
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_config
from construct_utils import ensure_base_model, build_per_pair_artifacts, build_union_artifacts, exclusive_lock, clear_stale_temp_files, _add_grad_outputs


from post_utils import (
    compute_diagonal_jacobians, sample_diagonal_jacobians, sample_diagonal_jacobians_slogdet,
    save_diagonal_jacobians, save_diagonal_jacobians_svd,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to a TOML config with pipeline = \"jacobians\"")
    parser.add_argument("--force", action="store_true",
                         help="rebuild the base model / gradient graph even if they already exist")
    args = parser.parse_args()
    cfg = load_config(args.config)
    assert cfg.pipeline == "jacobians", f"{cfg.name}: expected pipeline = \"jacobians\", got {cfg.pipeline!r}"

    clear_stale_temp_files(cfg)
    
    ensure_base_model(cfg, force=args.force)
   
    lock_path = cfg.gradient_path.with_suffix(cfg.gradient_path.suffix + ".lock")
    with exclusive_lock(lock_path):
        if not args.force and cfg.gradient_path.exists():
            print(f"Gradient graph already exists, skipping build -> {cfg.gradient_path}")
            return

        print("Building per-pair artifacts...")
        per_pair_training_models = build_per_pair_artifacts(cfg)
            
        build_union_artifacts([str(p) for p in per_pair_training_models], str(cfg.gradient_path))
            
        print("Adding grad outputs (+ float32 casts for bf16 outputs)...")
        _add_grad_outputs(cfg, cfg.gradient_path)

        print(f"Removing per-pair artifacts (folded into the union graph) -> {cfg.union_artifact_dir}")
        shutil.rmtree(cfg.union_artifact_dir)
        

    if cfg.slogdet_only:
        sample_diagonal_jacobians_slogdet(cfg)
    elif cfg.samples is not None:
        diagonal_blocks, activations = sample_diagonal_jacobians(cfg)
    else:
        diagonal_blocks, activations = compute_diagonal_jacobians(cfg)

    if not cfg.slogdet_only:
        if cfg.svd:
            save_diagonal_jacobians_svd(cfg, diagonal_blocks, activations)
        else:
            save_diagonal_jacobians(cfg, diagonal_blocks, activations)


if __name__ == "__main__":
    main()
