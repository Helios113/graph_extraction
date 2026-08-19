# %%

from pathlib import Path
from pathlib import Path
import onnx
import onnxruntime as ort
import numpy as np
from config import SublayerConfig, load_config
from construct.construct import (
    ensure_base_model,
    ensure_subgraph_pullback,
    ensure_subgraphs,
)
from construct.subgraph_extract import generate_union_of_subgraphs

cfg = load_config("configs/qwen_jac_sublayer.toml")
# assert cfg.pipeline == "collision_opt", f"{cfg.name}: expected pipeline = \"collision_opt\", got {cfg.pipeline!r}"
force = False




# makes sure that our base model is loaded
ensure_base_model(cfg, force=force, verbose=False)

force = True
# makes sure that out subgraphs are present
sub_graphs = ensure_subgraphs(cfg, force=force)

# makes sure that we have the jacobains
output = ensure_subgraph_pullback(
    sub_graphs,
    force=force,
)





# Load model
graph_path = output["gradient"][0]
model = onnx.load(graph_path)

print(graph_path)

# Data


# Run

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 8   # pick a value <= your usable core count
# sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

sess = ort.InferenceSession(str(graph_path),sess_options=sess_options, providers=["CUDAExecutionProvider"])
