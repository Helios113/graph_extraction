"""Compute Jacobian singular-value statistics at the original (unperturbed)
sequences.

Loads the activations saved by `scripts/generate_base_activations.py` and,
for each sublayer in a reduced Jacobian config (e.g. `configs/smoll_jac.toml`),
computes the exact Jacobian of that sublayer's output w.r.t. its input, at
every sequence position. From each d_model x d_model Jacobian we keep only
the smallest singular value and the condition number, and save those to disk
(not the full Jacobians, which would be far larger).

Each sublayer is written via a ShardedLayerWriter under --out-dir: rows
(samples) are processed in --chunk-size chunks, sharded to local scratch
disk, pushed to the remote store and deleted locally as soon as a shard
fills, with progress checkpointed on NFS -- so a crash partway through only
loses the in-progress shard, and re-running the same command resumes rather
than restarting.

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
from execute.sharded_writer import ShardedLayerWriter, load_activations

DEVICE = torch.device("cuda")


def _save_base_seqs_once(out_dir: str, base_seqs: np.ndarray) -> None:
    """base_seqs is tiny (n_samples rows) relative to the per-layer sweep
    outputs, and shared across all layers -- kept as one small NFS-side file
    per run rather than duplicated into every layer's shards."""
    path = os.path.join(out_dir, "base_seqs.h5")
    if os.path.exists(path):
        return
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(path, "w") as h5f:
        h5f.attrs["n_samples"] = base_seqs.shape[0]
        h5f.create_dataset("base_seqs", data=base_seqs)


def compute_and_save_jacobian_stats(
    cfg,
    activations: dict[str, np.ndarray],
    base_seqs: np.ndarray,
    out_dir: str,
    chunk_size: int,
    shard_rows: int,
    flush_every: int,
    no_push: bool,
) -> None:
    """For each sublayer, computes the diagonal Jacobian (one d_model x
    d_model block per sequence position per sample) in chunks of chunk_size
    samples, then reduces each block to its smallest singular value and
    condition number. The full Jacobians (chunk_size, seq, d_model, d_model)
    are kept only long enough to run svdvals on, then discarded -- only the
    two scalars per (sample, position) are kept.

    Each chunk is written to that sublayer's ShardedLayerWriter (rows =
    samples), which shards to local scratch disk, uploads+deletes each shard
    as soon as it fills, and checkpoints progress on NFS so a crashed run
    resumes from the last completed chunk instead of restarting."""
    ensure_base_model(cfg, force=False, verbose=False)
    sub_graphs = ensure_subgraphs(cfg, force=False)
    sub_graph_pullbacks = ensure_subgraph_pullback(sub_graphs, force=False)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.intra_op_num_threads = 8

    n_samples = base_seqs.shape[0]
    _save_base_seqs_once(out_dir, base_seqs)

    for (path, sublayer_conf), pullback_path in tqdm.tqdm(
        zip(sub_graphs, sub_graph_pullbacks), total=len(sub_graphs), desc="sublayers"
    ):
        layer_name = f"{sublayer_conf.input}.{sublayer_conf.output}"
        seq = sublayer_conf.input_shape[1]

        writer = ShardedLayerWriter(
            layer_key=os.path.join(out_dir, layer_name),
            n_rows=n_samples,
            schema={
                "smallest_singular_value": ((seq,), "float32"),
                "condition_number": ((seq,), "float32"),
            },
            shard_rows=shard_rows,
            flush_every=flush_every,
            attrs={"n_samples": n_samples, "seq": seq, "d_model": sublayer_conf.input_shape[2]},
            no_push=no_push,
            project_root=".",
        )
        if writer.next_row >= n_samples:
            continue  # already complete from a prior run

        jac_sess = ort.InferenceSession(
            str(pullback_path),
            sess_options=sess_options,
            providers=["CUDAExecutionProvider"],
        )

        input_data = activations[sublayer_conf.input]

        sv_min, sv_max = np.inf, -np.inf
        cond_min, cond_max = np.inf, -np.inf

        for start in tqdm.trange(
            writer.next_row, n_samples, chunk_size, desc=layer_name, leave=False
        ):
            end = min(start + chunk_size, n_samples)
            chunk_input = input_data[start:end]

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

            writer.write_rows(
                start, end,
                smallest_singular_value=smallest_sv,
                condition_number=condition_number,
            )

            sv_min, sv_max = min(sv_min, smallest_sv.min()), max(sv_max, smallest_sv.max())
            cond_min = min(cond_min, condition_number.min())
            cond_max = max(cond_max, condition_number.max())

        writer.close()

        print(
            f"{layer_name}: this run's smallest_sv range [{sv_min:.4g}, {sv_max:.4g}], "
            f"condition_number range [{cond_min:.4g}, {cond_max:.4g}] -> {writer.layer_key}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str, help="Path to a toml ModelConfig")
    parser.add_argument(
        "--activations-dir",
        type=str,
        default="onnx_models/smoll/base_activations",
        help="Output directory produced by scripts.generate_base_activations",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="onnx_models/smoll/jacobian_stats",
        help="Output directory -- one sharded output per sublayer is written here",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="Number of samples to process (and flush to local scratch disk) at a time",
    )
    parser.add_argument(
        "--shard-rows",
        type=int,
        default=200_000,
        help="Rows (samples) per shard -- each shard is written to local scratch disk, "
        "then pushed to the remote store and deleted locally as soon as it's full",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=20,
        help="Flush the current shard's h5 file to local scratch disk every this many chunks",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Keep finished shards on local scratch disk instead of pushing to the remote store",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    base_seqs, activations = load_activations(args.activations_dir)
    print(f"base_seqs: {base_seqs.shape}, activations: {sorted(activations)}")

    compute_and_save_jacobian_stats(
        cfg, activations, base_seqs, args.out_dir, args.chunk_size,
        shard_rows=args.shard_rows, flush_every=args.flush_every, no_push=args.no_push,
    )
    print(f"saved per-layer singular value / condition number summaries to {args.out_dir}")


if __name__ == "__main__":
    main()
