# Import all basic functions
import onnx
import onnxruntime as ort
import ml_dtypes
import numpy as np
import h5py
import torch

# Import config
from config import load_config

# data generation
from generate_data.datagen import (
    sample_ambient_points,
    sample_around_points,
    repeat_points_to_match,
)

# model generation
from construct.construct import (
    ensure_base_model,
    ensure_subgraph_pullback,
    ensure_subgraphs,
)


# execution code
from execute.optimization_loop import optimization_loop, NewtonRaphson


# Load config
cfg = load_config("configs/qwen_adversarial_sublayer.toml")

# Re-use all existing models
force = False


# makes sure that our base model is loaded
ensure_base_model(cfg, force=force, verbose=False)

force = False
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

# Very Important --- will cause errors if missing
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
sess_options.intra_op_num_threads = 8  # pick a value <= your usable core count
out_sess = ort.InferenceSession(
    str(sub_graphs[0][0]),
    sess_options=sess_options,
    providers=["CUDAExecutionProvider"],
)


# Get all output and input names
output_names = {o.name for o in out_sess.get_outputs()}
input_names = {o.name for o in out_sess.get_inputs()}
input_names = [i for i in input_names]


# Get the correct data for this approach
# Sample lots of points in a wide range of spaces

x = sample_ambient_points(
    batch= cfg.batch,
    seq= cfg.seq,
    dim=cfg.d_model,
    low=-1000,
    high=1000,
).astype(ml_dtypes.bfloat16)

out = out_sess.run(
    None,
    {input_names[0]: x},
)
y0 = sample_around_points(x, n_samples=1000).astype(ml_dtypes.bfloat16)

# Sample around all of these points


# Define gradient session
grad_sess = ort.InferenceSession(
    str(graph_path), sess_options=sess_options, providers=["CUDAExecutionProvider"]
)

optim = NewtonRaphson(1e-10, dtype=ml_dtypes.bfloat16)


y_final, loss_history = optimization_loop(
    grad_sess,
    x=x,
    target=out[0],
    y0=y0,
    input_name="add_6",
    steps=1000,
    optimizer=optim,
)
