import argparse
import os

import h5py
import numpy as np
import onnxruntime as ort
import torch
import torch.utils.dlpack
import tqdm

from config import load_config
from construct.construct import ensure_base_model, ensure_extra_outputs
from execute.io_onnx import run_iobinding_cpu_input_gpu_output
from execute.remote_storage import ensure_local, push

DEVICE = torch.device("cuda")


def build_perturbed_inputs(
    cfg,
    base_seqs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """For every (sample, position) pair, substitute every vocab id into
    that position, holding the other cfg.seq - 1 positions fixed.

    Returns (perturbed_inputs, row_sample_idx, row_pos, row_vocab_id, n_rows)
    where perturbed_inputs has shape (n_chunks, cfg.batch, cfg.seq) (padded
    up to a multiple of cfg.batch by repeating the last row) and the row_*
    arrays are indexed by unpadded row (length n_rows).
    """
    n_samples = base_seqs.shape[0]
    vocab_ids = np.arange(cfg.vocab_size)  # (vocab_size,)

    # perturbed[sample_idx, pos] has shape (vocab_size, cfg.seq):
    #   column `pos` = 0..vocab_size-1, all other columns = base_seqs[sample_idx]
    perturbed = np.broadcast_to(
        base_seqs[:, None, None, :],
        (n_samples, cfg.seq, cfg.vocab_size, cfg.seq),
    ).copy()
    for pos in range(cfg.seq):
        perturbed[:, pos, :, pos] = vocab_ids  # broadcasts over n_samples

    perturbed_flat = perturbed.reshape(-1, cfg.seq)
    n_rows = perturbed_flat.shape[0]

    pad = (-n_rows) % cfg.batch
    if pad:
        perturbed_flat = np.concatenate(
            [perturbed_flat, np.repeat(perturbed_flat[-1:], pad, axis=0)], axis=0
        )
    perturbed_inputs = perturbed_flat.reshape(-1, cfg.batch, cfg.seq)

    row_sample_idx, row_pos, row_vocab_id = np.unravel_index(
        np.arange(n_rows), (n_samples, cfg.seq, cfg.vocab_size)
    )

    return perturbed_inputs, row_sample_idx, row_pos, row_vocab_id, n_rows


def run_perturbed_sweep(
    cfg,
    base_seqs: np.ndarray,
    out_path: str,
    n_samples: int,
    vocab_sample_size: int,
    flush_every: int,
    seed: int,
    no_push: bool = False,
) -> None:
    base_seqs = base_seqs[:n_samples]

    rng = np.random.default_rng(seed)
    perturbed_inputs, row_sample_idx, row_pos, row_vocab_id, n_rows = (
        build_perturbed_inputs(cfg, base_seqs)
    )
    print(perturbed_inputs.shape, "n_rows (unpadded):", n_rows)

    ensure_base_model(cfg, force=False, verbose=False)
    extra_outputs = [sublayer.output for sublayer in cfg.sublayers]
    model_path = ensure_extra_outputs(cfg, extra_outputs, force=False)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.intra_op_num_threads = 8

    out_sess = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=["CUDAExecutionProvider"],
    )
    out_output_names = [o.name for o in out_sess.get_outputs()]
    out_input_names = [o.name for o in out_sess.get_inputs()]
    print(out_output_names, out_input_names)

    n_chunks = perturbed_inputs.shape[0]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as h5f:
        h5f.attrs["n_samples"] = n_samples
        h5f.attrs["vocab_sample_size"] = vocab_sample_size
        h5f.attrs["seq"] = cfg.seq
        h5f.attrs["d_model"] = cfg.d_model

        # original (unperturbed) sequences, indexed by sample_idx -- each
        # row's sample_idx (saved per layer below) points into this table,
        # so the original sequence a row was derived from is
        # base_seqs[sample_idx].
        h5f.create_dataset("base_seqs", data=base_seqs)

        datasets = {}
        for output_name in out_output_names[1:]:
            grp = h5f.create_group(output_name)
            grp.attrs["output_name"] = output_name
            datasets[output_name] = {
                "vec": grp.create_dataset(
                    "vec", shape=(n_rows, cfg.d_model), dtype="float32"
                ),
                "sample_idx": grp.create_dataset(
                    "sample_idx", shape=(n_rows,), dtype="int64"
                ),
                "pos": grp.create_dataset("pos", shape=(n_rows,), dtype="int64"),
                "vocab_id": grp.create_dataset(
                    "vocab_id", shape=(n_rows,), dtype="int64"
                ),
            }

        row_start = 0
        for c in tqdm.tqdm(range(n_chunks)):
            token_ids = perturbed_inputs[c]  # (cfg.batch, cfg.seq)
            # print(token_ids.shape)
            fx = run_iobinding_cpu_input_gpu_output(
                out_sess,
                out_input_names[0],
                token_ids,
                out_output_names,
            )

            chunk_rows = token_ids.shape[0]
            row_end = row_start + chunk_rows
            valid = min(row_end, n_rows) - row_start  # drop padding rows past n_rows

            chunk_pos = row_pos[row_start : row_start + valid]
            # gather the swept position's d_model vector for each row in the
            # chunk
            batch_arange = np.arange(valid)

            for i, output_name in enumerate(out_output_names[1:], start=1):
                layer_out = fx[i][:valid]  # (valid, cfg.seq, d_model)
                swept_vecs = (
                    layer_out[batch_arange, chunk_pos]
                    .to(torch.float32)
                    .detach()
                    .cpu()
                    .numpy()
                )

                ds = datasets[output_name]
                ds["vec"][row_start : row_start + valid] = swept_vecs
                ds["sample_idx"][row_start : row_start + valid] = row_sample_idx[
                    row_start : row_start + valid
                ]
                ds["pos"][row_start : row_start + valid] = chunk_pos
                ds["vocab_id"][row_start : row_start + valid] = row_vocab_id[
                    row_start : row_start + valid
                ]

            row_start = row_end

            if (c + 1) % flush_every == 0 or c == n_chunks - 1:
                h5f.flush()

    print(f"Saved perturbed sweep outputs to {out_path}")

    if not no_push:
        push(out_path)


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
        default="onnx_models/smoll/perturbed_sweep.h5",
        help="Output h5 path for the perturbed sweep outputs",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=128,
        help="Number of base sequences (from the start of --base-seqs-h5) to perturb",
    )
    parser.add_argument(
        "--vocab-sample-size",
        type=int,
        default=1000,
        help="Number of random vocab ids sampled per (sample, position) slot",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=50,
        help="Flush the output h5 file every this many chunks",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the vocab-sampling RNG",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Don't push --out to the remote store when finished",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    ensure_local(args.base_seqs_h5)
    with h5py.File(args.base_seqs_h5, "r") as h5f:
        base_seqs = h5f["input_ids"][:]
    print(f"base_seqs: {base_seqs.shape}")

    run_perturbed_sweep(
        cfg,
        base_seqs,
        args.out,
        n_samples=args.n_samples,
        vocab_sample_size=args.vocab_sample_size,
        flush_every=args.flush_every,
        seed=args.seed,
        no_push=args.no_push,
    )


if __name__ == "__main__":
    main()
