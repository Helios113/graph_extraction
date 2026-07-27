"""Full collision_opt pipeline, driven by a single config: export base model -> extract
per-sublayer eval subgraphs -> auto-fill missing activation stats -> build each sublayer's
twin gradient graph -> gradient-descend y towards a collision with a fixed x.

Usage: uv run python3 collision_opt/run_pipeline.py configs/x.toml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_config
from construct_utils import ensure_base_model, clear_stale_temp_files, ensure_sub_graphs
# from collisions.activation_stats import ensure_activation_stats
from build_twin_graph import build_twin_training_model
from optimize import run_collision_opt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to a TOML config with pipeline = \"collision_opt\"")
    parser.add_argument("--force", action="store_true",
                         help="rebuild the base model / subgraphs / twin graphs even if they already exist")
    args = parser.parse_args()
    cfg = load_config(args.config)
    assert cfg.pipeline == "collision_opt", f"{cfg.name}: expected pipeline = \"collision_opt\", got {cfg.pipeline!r}"

    clear_stale_temp_files(cfg)
    ensure_base_model(cfg, force=args.force)
    ensure_sub_graphs(cfg, force=args.force)

    print("Building twin gradient graphs...")
    for sublayer_idx in range(len(cfg.sublayers)):
        build_twin_training_model(cfg, sublayer_idx, force=args.force)

    print("Optimizing for collisions...")
    run_collision_opt(cfg)


if __name__ == "__main__":
    main()
