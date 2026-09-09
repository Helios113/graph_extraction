"""Compute Jacobian singular-value statistics at the original (unperturbed)
sequences.

Loads the activations saved by `collect_base_activations.py` and, for each
sublayer in a reduced Jacobian config (e.g. `configs/smoll_jac.toml`),
computes the exact Jacobian of that sublayer's output w.r.t. its input, at
every sequence position. From each d_model x d_model Jacobian we keep only
the smallest singular value and the condition number, and save those to disk
(not the full Jacobians, which would be far larger).

Each sublayer is written to its own h5 file under --out-dir, and within a
sublayer, samples are processed in --chunk-size chunks -- each chunk is
appended and flushed to disk as soon as it's ready, so a partial/interrupted
run still leaves usable results, and progress is visible on disk as it
happens.

Usage:
    uv run python -m scripts.compute_base_jacobian_stats configs/smoll_jac.toml
"""
import argparse
import os

import h5py
import ml_dtypes
import numpy as np
import onnxruntime as ort
import torch
import tqdm
from config import load_config
from construct.construct import ensure_base_model, ensure_subgraph_pullback, ensure_subgraphs
from execute.jacobian import compute_jacobian

DEVICE = torch.device("cuda")


def load_activations(path: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Inverse of collect_base_activations.save_activations: activations were
    stored as raw uint16 (h5py has no native bfloat16 type), so view them
    back to ml_dtypes.bfloat16 here."""
    with h5py.File(path, "r") as h5f:
        base_seqs = h5f["base_seqs"][:]
        activations = {
            name: dset[:].view(ml_dtypes.bfloat16) for name, dset in h5f["activations"].items()
        }
    return base_seqs, activations


def _open_layer_h5(path: str, base_seqs: np.ndarray, seq: int, d_model: int) -> h5py.File:
    """Creates the per-layer output file with resizable, chunk-appendable
    smallest_singular_value / condition_number datasets."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h5f = h5py.File(path, "w")
    h5f.attrs["n_samples"] = base_seqs.shape[0]
    h5f.attrs["seq"] = seq
    h5f.attrs["d_model"] = d_model
    h5f.create_dataset("base_seqs", data=base_seqs)
    h5f.create_dataset(
        "smallest_singular_value",
        shape=(0, seq),
        maxshape=(base_seqs.shape[0], seq),
        chunks=True,
        dtype=np.float32,
    )
    h5f.create_dataset(
        "condition_number",
        shape=(0, seq),
        maxshape=(base_seqs.shape[0], seq),
        chunks=True,
        dtype=np.float32,
    )
    return h5f


def _append_chunk(h5f: h5py.File, smallest_sv: np.ndarray, condition_number: np.ndarray) -> None:
    for name, chunk in (
        ("smallest_singular_value", smallest_sv),
        ("condition_number", condition_number),
    ):
        dset = h5f[name]
        start = dset.shape[0]
        dset.resize(start + chunk.shape[0], axis=0)
        dset[start : start + chunk.shape[0]] = chunk
    h5f.flush()


def compute_and_save_jacobian_stats(
    cfg,
    activations: dict[str, np.ndarray],
    base_seqs: np.ndarray,
    out_dir: str,
    chunk_size: int,
) -> None:
    """For each sublayer, computes the diagonal Jacobian (one d_model x
    d_model block per sequence position per sample) in chunks of chunk_size
    samples, then reduces each block to its smallest singular value and
    condition number. The full Jacobians (chunk_size, seq, d_model, d_model)
    are kept only long enough to run svdvals on, then discarded -- only the
    two scalars per (sample, position) are kept.

    Each chunk is appended to that sublayer's own h5 file under out_dir and
    flushed to disk immediately, rather than waiting for the whole sublayer
    (or all sublayers) to finish."""
    ensure_base_model(cfg, force=False, verbose=False)
    sub_graphs = ensure_subgraphs(cfg, force=False)
    sub_graph_pullbacks = ensure_subgraph_pullback(sub_graphs, force=False)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.intra_op_num_threads = 8

    n_samples = base_seqs.shape[0]

    for (path, sublayer_conf), pullback_path in tqdm.tqdm(
        zip(sub_graphs, sub_graph_pullbacks), total=len(sub_graphs), desc="sublayers"
    ):
        layer_name = f"{sublayer_conf.input}.{sublayer_conf.output}"
        jac_sess = ort.InferenceSession(
            str(pullback_path),
            sess_options=sess_options,
            providers=["CUDAExecutionProvider"],
        )

        input_data = activations[sublayer_conf.input]

        layer_h5_path = os.path.join(out_dir, f"{layer_name}.h5")
        h5f = _open_layer_h5(
            layer_h5_path, base_seqs, sublayer_conf.input_shape[1], sublayer_conf.input_shape[2]
        )

        sv_min, sv_max = np.inf, -np.inf
        cond_min, cond_max = np.inf, -np.inf

        for start in tqdm.trange(0, n_samples, chunk_size, desc=layer_name, leave=False):
            chunk_input = input_data[start : start + chunk_size]

            jacobian = compute_jacobian(
                jac_sess,
                input_data=chunk_input,
                input_name=sublayer_conf.input,
                mask_name="mask",
                mask_dtype=ml_dtypes.bfloat16,
                gradient_output_name=[f"{sublayer_conf.input}_grad"],
                output_shape=sublayer_conf.input_shape,
                mode="diagonal",
            )  # (chunk, 1, seq, d_model, d_model)
            jacobian = jacobian[:, 0]  # (chunk, seq, d_model, d_model)

            jacobian_t = torch.from_numpy(jacobian.astype(np.float32)).to(DEVICE)
            singular_values = torch.linalg.svdvals(jacobian_t)  # (chunk, seq, d_model), descending

            smallest_sv = singular_values[..., -1].cpu().numpy()
            condition_number = (singular_values[..., 0] / singular_values[..., -1]).cpu().numpy()

            del jacobian, jacobian_t, singular_values
            torch.cuda.empty_cache()

            _append_chunk(h5f, smallest_sv, condition_number)

            sv_min, sv_max = min(sv_min, smallest_sv.min()), max(sv_max, smallest_sv.max())
            cond_min = min(cond_min, condition_number.min())
            cond_max = max(cond_max, condition_number.max())

        h5f.close()

        print(
            f"{layer_name}: smallest_sv range [{sv_min:.4g}, {sv_max:.4g}], "
            f"condition_number range [{cond_min:.4g}, {cond_max:.4g}] -> {layer_h5_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str, help="Path to a toml ModelConfig")
    parser.add_argument(
        "--activations-h5",
        type=str,
        default="onnx_models/smoll/base_activations.h5",
        help="h5 file produced by collect_base_activations.py",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="onnx_models/smoll/jacobian_stats",
        help="Output directory -- one h5 file per sublayer is written here",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="Number of samples to process (and flush to disk) at a time",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    base_seqs, activations = load_activations(args.activations_h5)
    print(f"base_seqs: {base_seqs.shape}, activations: {sorted(activations)}")

    compute_and_save_jacobian_stats(cfg, activations, base_seqs, args.out_dir, args.chunk_size)
    print(f"saved per-layer singular value / condition number summaries to {args.out_dir}")


if __name__ == "__main__":
    main()
