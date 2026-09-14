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


def build_chunk_rows(
    cfg,
    base_seqs: np.ndarray,
    row_start: int,
    row_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one chunk's (row_end - row_start, cfg.seq) token-id block plus
    its per-row (sample_idx, pos, vocab_id) annotation, without ever
    materializing all n_rows of the sweep at once (n_rows = n_samples *
    cfg.seq * cfg.vocab_size can be in the hundreds of millions).

    Row order is unravelled from (sample_idx, pos, vocab_id), matching the
    original dense-array construction this replaces.
    """
    n_samples = base_seqs.shape[0]
    row_sample_idx, row_pos, row_vocab_id = np.unravel_index(
        np.arange(row_start, row_end), (n_samples, cfg.seq, cfg.vocab_size),
    )
    token_ids = base_seqs[row_sample_idx].copy()  # (chunk, cfg.seq)
    token_ids[np.arange(row_end - row_start), row_pos] = row_vocab_id
    return token_ids, row_sample_idx, row_pos, row_vocab_id


def run_perturbed_sweep(
    cfg,
    base_seqs: np.ndarray,
    out_path: str,
    n_samples: int,
    flush_every: int,
    no_push: bool = False,
) -> None:
    base_seqs = base_seqs[:n_samples]

    n_rows = n_samples * cfg.seq * cfg.vocab_size
    n_chunks = -(-n_rows // cfg.batch)  # ceil division
    print(f"n_rows: {n_rows}, n_chunks: {n_chunks}")

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

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as h5f:
        h5f.attrs["n_samples"] = n_samples
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
                    "vec", shape=(n_rows, cfg.d_model), dtype="float32",
                ),
                "sample_idx": grp.create_dataset(
                    "sample_idx", shape=(n_rows,), dtype="int64",
                ),
                "pos": grp.create_dataset("pos", shape=(n_rows,), dtype="int64"),
                "vocab_id": grp.create_dataset(
                    "vocab_id", shape=(n_rows,), dtype="int64",
                ),
            }

        for c in tqdm.tqdm(range(n_chunks)):
            row_start = c * cfg.batch
            row_end = min(row_start + cfg.batch, n_rows)
            valid = row_end - row_start

            token_ids, chunk_sample_idx, chunk_pos, chunk_vocab_id = build_chunk_rows(
                cfg, base_seqs, row_start, row_end,
            )
            if valid < cfg.batch:
                # pad the last chunk up to cfg.batch by repeating the last row
                pad = cfg.batch - valid
                token_ids = np.concatenate(
                    [token_ids, np.repeat(token_ids[-1:], pad, axis=0)], axis=0,
                )

            fx = run_iobinding_cpu_input_gpu_output(
                out_sess,
                out_input_names[0],
                token_ids,
                out_output_names,
            )

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
                ds["vec"][row_start:row_end] = swept_vecs
                ds["sample_idx"][row_start:row_end] = chunk_sample_idx
                ds["pos"][row_start:row_end] = chunk_pos
                ds["vocab_id"][row_start:row_end] = chunk_vocab_id

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
        "--flush-every",
        type=int,
        default=50,
        help="Flush the output h5 file every this many chunks",
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
        flush_every=args.flush_every,
        no_push=args.no_push,
    )


if __name__ == "__main__":
    main()
