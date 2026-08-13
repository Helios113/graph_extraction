# %%

from utils_construct import ensure_base_model, ensure_subgraphs, ensure_subgraph_pullback, generate_subgraph_pullback

from config import load_config
from build_twin_graph import build_twin_training_model
# from optimize import run_collision_opt

# %%

cfg = load_config("configs/qwen_jac_sublayer.toml")
# assert cfg.pipeline == "collision_opt", f"{cfg.name}: expected pipeline = \"collision_opt\", got {cfg.pipeline!r}"
force = False


# %%

# clear_stale_temp_files(cfg)
ensure_base_model(cfg, force=force, verbose = False)
sub_graphs = ensure_subgraphs(cfg, force=force)
# ensure_subgraph_pullback(sub_graphs, cfg, force=force)
