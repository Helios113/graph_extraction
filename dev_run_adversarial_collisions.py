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
    generate_subgraph_pullback,
)
from construct.subgraph_extract import generate_union_of_subgraphs

# from optimize import run_collision_opt

# %%

cfg = load_config("configs/qwen_jac_sublayer.toml")
# assert cfg.pipeline == "collision_opt", f"{cfg.name}: expected pipeline = \"collision_opt\", got {cfg.pipeline!r}"
force = False


# %%

# clear_stale_temp_files(cfg)
#
#
# Document
ensure_base_model(cfg, force=force, verbose=False)


sub_graphs = ensure_subgraphs(cfg, force=force)
# output = ensure_subgraph_pullback(sub_graphs, force=force)
# print(output)


output = ensure_subgraph_pullback(
    sub_graphs,
    force=force,
)




# Adjust to the graph you want to inspect
graph_path = output["gradient"][0]
# graph_path = Path("onnx_models/qwen_jac_trained_sublayers/sub_blocks/add_6.add_8.subgraph.pullback.onnx")

# --- Load and print I/O signature ---
model = onnx.load(graph_path)  # external data (.onnx.data) is picked up automatically

def describe(value_infos, label):
    print(f"\n{label}:")
    for v in value_infos:
        t = v.type.tensor_type
        dims = [
            (d.dim_param if d.dim_param else d.dim_value)
            for d in t.shape.dim
        ]
        dtype = onnx.TensorProto.DataType.Name(t.elem_type)
        print(f"  {v.name}: shape={dims} dtype={dtype}")

describe(model.graph.input, "Inputs")
describe(model.graph.output, "Outputs")

# --- Run it ---

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 8   # pick a value <= your usable core count
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

sess = ort.InferenceSession(str(graph_path),sess_options=sess_options, providers=["CUDAExecutionProvider"])


print("\nSession inputs:")
for i in sess.get_inputs():
    print(f"  {i.name}: shape={i.shape} dtype={i.type}")

print("\nSession outputs:")
for o in sess.get_outputs():
    print(f"  {o.name}: shape={o.shape} dtype={o.type}")

import ml_dtypes

# Build dummy feed: random data matching each input's static shape
# (replace non-static/dynamic dims with a concrete value, e.g. 1)
feed = {}
for inp in sess.get_inputs():
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    if "bfloat16" in inp.type:
        feed[inp.name] = np.random.randn(*shape).astype(np.float32).astype(ml_dtypes.bfloat16)
    elif "float" in inp.type:
        feed[inp.name] = np.random.randn(*shape).astype(np.float32)
    elif "bool" in inp.type:
        feed[inp.name] = np.zeros(shape, dtype=np.bool_)
    else:
        feed[inp.name] = np.zeros(shape, dtype=np.int64)

outputs = sess.run(None, feed)
for o, val in zip(sess.get_outputs(), outputs):
    print(f"\n{o.name}: shape={val.shape} dtype={val.dtype}")
