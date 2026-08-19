"""Config-driven Jacobian discovery.

Jacobians are only ever discovered via MaskedSumLoss (loss = sum(output * mask)):
one-hot masks against that graph's gradient output give exact Jacobian rows
d(output_i)/d(input), see execute.jacobian.compute_jacobian. This script wires
that up end to end for every sublayer in a config: ensure the base model and
extracted subgraphs exist, force a MaskedSumLoss pullback for each pair
(regardless of whatever loss_type the config's toml declares -- that field is
for the optimization-loop pipeline, not Jacobian discovery), run the one-hot
sweep, and save the resulting Jacobians to disk.
"""

import argparse
import dataclasses
from pathlib import Path

import h5py
import numpy as np
import onnxruntime as ort

from config import LossType, ModelConfig, SublayerConfig, load_config
from construct.construct import ensure_base_model, ensure_subgraph_pullback, ensure_subgraphs
from execute.jacobian import compute_jacobian




def find_jacobians(
    sub_graphs: list[Path],
    mode: str = "diagonal",
) -> dict[str, np.ndarray]:
    """Computes one Jacobian per (path, sublayer_conf) pair in sub_graphs (e.g.
    construct.construct.ensure_subgraphs(cfg)'s return value) -- cfg is only used
    for shared settings (ensuring the base model exists, force, ...), not for
    discovering which sublayers to run; that comes from sub_graphs."""

    jacobians: dict[str, np.ndarray] = {}

    for path in sub_graphs:

        session = ort.InferenceSession(str(path))

        jacobian = compute_jacobian(
            session,
            input_data=input_data,
            input_name=sublayer_conf.input,
            mask_name="mask",
            gradient_output_name=f"{sublayer_conf.input}_grad",
            output_shape=sublayer_conf.input_shape,
            mode=mode,
        )
        jacobians[f"{sublayer_conf.input}.{sublayer_conf.output}"] = jacobian

    return jacobians


def save_jacobians(jacobians: dict[str, np.ndarray], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        for name, jac in jacobians.items():
            f.create_dataset(name, data=jac)

