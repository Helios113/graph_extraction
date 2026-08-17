"""Full collisions pipeline, driven by a single config: export base model -> extract
per-sublayer eval subgraphs -> auto-fill missing activation stats -> search for max output
norm + collision-distance plots.

Usage: uv run python3 collisions/run_pipeline.py configs/qwen_block0_collisions.toml
"""
import argparse

from config import load_config
from construct.utils_construct import ensure_base_model, clear_stale_temp_files, ensure_sub_graphs
from optimise_activations import compute_max_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to a TOML config with pipeline = \"collisions\"")
    parser.add_argument("--force", action="store_true",
                         help="rebuild the base model / subgraphs / activation stats even if they already exist")
    args = parser.parse_args()
    cfg = load_config(args.config)
    assert cfg.pipeline == "collisions", f"{cfg.name}: expected pipeline = \"collisions\", got {cfg.pipeline!r}"
    clear_stale_temp_files(cfg)
    ensure_base_model(cfg, force=args.force)
    ensure_sub_graphs(cfg, force=args.force)

    print("Searching for max sublayer output norm + collision distances...")
    compute_max_norm(cfg)


if __name__ == "__main__":
    main()
