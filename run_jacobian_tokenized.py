# Import all basic functions
import onnxruntime as ort
import ml_dtypes
import numpy as np
import h5py
import torch

# Import config
from config import load_config

# data generation
from generate_data.datagen import get_token_ids_array

# model generation
from construct.construct import (
    ensure_base_model,
    ensure_subgraph_pullback,
    ensure_subgraphs,
)

# execution code
from execute.jacobian import compute_jacobian


# Load config
cfg = load_config("configs/qwen_jac_sublayer_tokens.toml")
# Re-use all existing models
force = False


# makes sure that our base model is loaded
ensure_base_model(cfg, force=force, verbose=False)

# makes sure that out subgraphs are present
sub_graphs = ensure_subgraphs(cfg, force=force, do_not_materialise_subgraphs=True)

# makes sure that we have the jacobains
sub_graphs_pullbacks = ensure_subgraph_pullback(
    sub_graphs,
    force=force,
    base_model_path=cfg.base_model_path,
)

# Get inputs
x = get_token_ids_array(cfg.input_source, seq_len=cfg.seq)


# Define output session
sess_options = ort.SessionOptions()

# Very Important --- will cause errors if missing
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
sess_options.intra_op_num_threads = 8  # pick a value <= your usable core count


out_sess = ort.InferenceSession(
    sub_graphs_pullbacks[0],
    sess_options=sess_options,
    providers=["CUDAExecutionProvider"],
)



# Compute jacobians
jacobians = compute_jacobian(
    out_sess,
    input_data=x,
    input_name="input_ids",
    mask_name="mask",
    mask_dtype=ml_dtypes.bfloat16,
    gradient_output_name="add_6_grad",
    output_shape=[32, 8, 896],
    mode="diagonal",
)


# Save everything

with h5py.File("jacobians.h5", "w") as f:
    f.create_dataset("jacobians", data=jacobians)

jac_tensor = torch.from_numpy(jacobians.astype(np.float32)).cuda()

# 2. Compute singular values and save directly
with h5py.File("singular_values.h5", "w") as f:
    f.create_dataset(
        "singular_values", data=torch.linalg.svdvals(jac_tensor).cpu().numpy(),
    )

# 3. Compute signs, log-determinants and save to separate files
signs, log_abs_dets = torch.linalg.slogdet(jac_tensor)

with h5py.File("signs.h5", "w") as f:
    f.create_dataset("signs", data=signs.cpu().numpy())

with h5py.File("log_determinants.h5", "w") as f:
    f.create_dataset("log_determinants", data=log_abs_dets.cpu().numpy())

# 4. Clean up allocated GPU and CPU memory
del jac_tensor, signs, log_abs_dets
torch.cuda.empty_cache()
