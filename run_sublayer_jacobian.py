"""Per-sublayer jacobians pipeline, driven by a single config: export base model -> for
each [[sublayers]] entry, build that pair's own independent gradient graph (never unioned
with any other pair's, unlike jacobians/run_pipeline.py) -> compute its diagonal-block
Jacobian -> save.

Usage: uv run python3 sublayer_jacobians/run_pipeline.py configs/x.toml
"""
import argparse

from config import load_config
from utils_compute import compute_sublayer_jacobian
from utils_post import save_sublayer_jacobian
from utils_construct import ensure_base_model, clear_stale_temp_files, ensure_sub_graphs, build_single_pair_gradient_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to a TOML config with pipeline = \"sublayer_jacobians\"")
    parser.add_argument("--force", action="store_true",
                         help="rebuild the base model / gradient graphs even if they already exist")
    args = parser.parse_args()
    cfg = load_config(args.config)
    assert cfg.pipeline == "sublayer_jacobians", f"{cfg.name}: expected pipeline = \"sublayer_jacobians\", got {cfg.pipeline!r}"

    clear_stale_temp_files(cfg)
    ensure_base_model(cfg, force=args.force)
    ensure_sub_graphs(cfg, force=args.force)
    
    for sublayer_idx in range(len(cfg.sublayers)):
        sublayer = cfg.sublayers[sublayer_idx]
        print(f"[{sublayer_idx}] {sublayer.input} -> {sublayer.output}")
        build_single_pair_gradient_graph(cfg, sublayer_idx, force=args.force)
        diagonal_block, activations = compute_sublayer_jacobian(cfg, sublayer_idx)
        save_sublayer_jacobian(cfg, sublayer_idx, diagonal_block, activations)


if __name__ == "__main__":
    main()
