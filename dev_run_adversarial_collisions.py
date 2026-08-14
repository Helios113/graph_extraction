# %%

from pathlib import Path

from build_twin_graph import build_twin_training_model
from config import SublayerConfig, load_config
from utils_construct import (
    ensure_base_model,
    ensure_subgraph_pullback,
    ensure_subgraphs,
    generate_subgraph_pullback,
    generate_union_of_subgraphs,
)

# from optimize import run_collision_opt

# %%

cfg = load_config("configs/qwen_jac_sublayer.toml")
# assert cfg.pipeline == "collision_opt", f"{cfg.name}: expected pipeline = \"collision_opt\", got {cfg.pipeline!r}"
force = False


# %%

# clear_stale_temp_files(cfg)
ensure_base_model(cfg, force=force, verbose=False)
sub_graphs = ensure_subgraphs(cfg, force=force, do_not_materialise_subgraphs=True)
# output = ensure_subgraph_pullback(sub_graphs, force=force)
# print(output)


output = ensure_subgraph_pullback(
    sub_graphs,
    force=force,
    base_model_path=Path(
        "/nfs-share/pa511/code_bases/new_jac/onnx_models/qwen_jac_trained_sublayers/base_model.onnx"
    ),
)
print(output)

generate_union_of_subgraphs(output["gradient"])
