"""Collect activations at the original (unperturbed) sequences.

For every sublayer in a reduced Jacobian config (e.g. `configs/smoll_jac.toml`),
run the base sequences from the tokenized sequences h5 (see
`scripts/tokenize_to_h5.py`) through the extra-outputs model and save each
sublayer input's activation (plus the raw embeddings) to an h5 file.
`compute_base_jacobian_stats.py` loads that h5 and computes the Jacobian
singular-value statistics -- kept as a separate step since the activations
are cheap to produce but the Jacobians are the expensive part.

Usage:
    uv run python -m scripts.generate_base_activations configs/smoll_jac.toml
"""

import argparse
import os

import h5py
import ml_dtypes
import numpy as np
import onnxruntime as ort
import torch
import torch.utils.dlpack
import tqdm

from config import load_config
from construct.construct import ensure_base_model, ensure_extra_outputs
from execute.io_onnx import run_iobinding_cpu_input_gpu_output

DEVICE = torch.device("cuda")


def bf16_torch_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Bit-reinterpret a torch bfloat16 tensor as an ml_dtypes.bfloat16 numpy
    array (torch tensors can't go through .numpy() directly in this bf16
    dtype -- see the plain uint16 view/roundtrip this relies on)."""
    return t.contiguous().view(torch.uint16).cpu().numpy().view(ml_dtypes.bfloat16)


def collect_activations(
    cfg,
    base_seqs: np.ndarray,
) -> dict[str, np.ndarray]:
    """Returns activations (including "embedding"), bf16 numpy arrays viewed
    as uint16 (so they round-trip through h5py -- see save_activations)."""
    n_samples = base_seqs.shape[0]

    ensure_base_model(cfg, force=False, verbose=False)

    # "embedding" is a valid intermediate tensor of the base graph (the
    # lookup output feeding the first sublayer), so it can be read off the
    # extra-outputs model in the same forward pass as every other
    # activation -- no separate embedding-only session/subgraph needed.
    extra_outputs = sorted(
        {s.input for s in cfg.sublayers} | {s.output for s in cfg.sublayers}
    )

    extra_outputs_model_path = ensure_extra_outputs(cfg, extra_outputs, force=False)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.intra_op_num_threads = 8

    extra_outputs_sess = ort.InferenceSession(
        str(extra_outputs_model_path),
        sess_options=sess_options,
        providers=["CUDAExecutionProvider"],
    )
    extra_output_names = [o.name for o in extra_outputs_sess.get_outputs()]
    extra_input_name = extra_outputs_sess.get_inputs()[0].name

    # Run all base sequences through the session, chunked by cfg.batch (the
    # graph has a fixed batch dim of cfg.batch). Pad the last chunk by
    # repeating the final row, then trim the padding rows back off after.
    pad = (-n_samples) % cfg.batch
    padded_seqs = base_seqs
    if pad:
        padded_seqs = np.concatenate(
            [base_seqs, np.repeat(base_seqs[-1:], pad, axis=0)], axis=0
        )
    n_padded = padded_seqs.shape[0]

    activations = {
        name: torch.empty(
            (n_padded, cfg.seq, cfg.d_model), dtype=torch.bfloat16, device=DEVICE
        )
        for name in extra_output_names
        if name != "output"
    }

    for start in tqdm.tqdm(range(0, n_padded, cfg.batch)):
        chunk = padded_seqs[start : start + cfg.batch]

        fx = run_iobinding_cpu_input_gpu_output(
            extra_outputs_sess, extra_input_name, chunk, extra_output_names
        )
        for name, val in zip(extra_output_names, fx):
            if name == "output":
                continue
            activations[name][start : start + cfg.batch] = val

    # Trim padding rows back off.
    activations = {name: val[:n_samples] for name, val in activations.items()}

    activations_np = {
        name: bf16_torch_to_numpy(val).view(np.uint16)
        for name, val in activations.items()
    }
    return activations_np


def save_activations(
    out_path: str,
    base_seqs: np.ndarray,
    activations: dict[str, np.ndarray],
) -> None:
    """Activations are bf16 but stored as raw uint16 (h5py has no native
    bfloat16 type) -- view back to ml_dtypes.bfloat16 after loading, same as
    compute_base_jacobian_stats.py does."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as h5f:
        h5f.create_dataset("base_seqs", data=base_seqs)
        activations_grp = h5f.create_group("activations")
        for name, val in activations.items():
            activations_grp.create_dataset(name, data=val)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str, help="Path to a toml ModelConfig")
    parser.add_argument(
        "--base-seqs-h5",
        type=str,
        default="data/smoll_sequences.h5",
        help="h5 file (from scripts/tokenize_to_h5.py) containing an input_ids dataset of (n_samples, seq) token ids",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="onnx_models/smoll/base_activations.h5",
        help="Output h5 path for the collected embeddings/activations",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    with h5py.File(args.base_seqs_h5, "r") as h5f:
        base_seqs = h5f["input_ids"][:]
    print(f"base_seqs: {base_seqs.shape}")

    activations = collect_activations(cfg, base_seqs)
    save_activations(args.out, base_seqs, activations)
    print(f"saved base_seqs + {len(activations)} activations to {args.out}")


if __name__ == "__main__":
    main()
