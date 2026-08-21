import onnx
import onnxruntime as ort
from config import load_config
from construct.construct import (
    ensure_base_model,
    ensure_subgraph_pullback,
    ensure_subgraphs,
)
import ml_dtypes

from execute.optimization_loop import optimization_loop, NewtonRaphson
import numpy as np
cfg = load_config("configs/qwen_jac_sublayer.toml")
# assert cfg.pipeline == "collision_opt", f"{cfg.name}: expected pipeline = \"collision_opt\", got {cfg.pipeline!r}"
force = False




# makes sure that our base model is loaded
ensure_base_model(cfg, force=force, verbose=False)

force = True
# makes sure that out subgraphs are present
sub_graphs = ensure_subgraphs(cfg, force=force)

# makes sure that we have the jacobains
sub_graphs_pullbacks = ensure_subgraph_pullback(
    sub_graphs,
    force=force,
)





# Load model
graph_path = sub_graphs_pullbacks[0]
model = onnx.load(graph_path)

# Data


# Define output session
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
sess_options.intra_op_num_threads = 8   # pick a value <= your usable core count
out_sess = ort.InferenceSession(str(sub_graphs[0][0]),sess_options=sess_options, providers=["CUDAExecutionProvider"])
output_names = {o.name for o in out_sess.get_outputs()}
input_names = {o.name for o in out_sess.get_inputs()}
print(output_names, input_names)
input_names = [i for i in input_names]
# feed = {}

# where input_data is your numpy array
x = np.random.rand(32, 8, 896).astype(ml_dtypes.bfloat16)
y0 = np.random.rand(32, 8, 896).astype(ml_dtypes.bfloat16)


out = out_sess.run(
            None, {input_names[0]: x},
        )
print(out)
# # Define gradient session
grad_sess = ort.InferenceSession(str(graph_path),sess_options=sess_options, providers=["CUDAExecutionProvider"])

optim = NewtonRaphson(1e-10, dtype=ml_dtypes.bfloat16)


y_final, loss_history = optimization_loop(
    grad_sess,
    x = x,
    target=out[0],
    y0=y0,
    input_name="add_6",
    steps=100,
    optimizer=optim,
)

print(loss_history)