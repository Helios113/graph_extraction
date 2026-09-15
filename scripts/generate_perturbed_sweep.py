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
from execute.sharded_writer import ShardedLayerWriter, load_base_seqs

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
    out_dir: str,
    n_samples: int,
    flush_every: int,
    shard_rows: int,
    no_push: bool = False,
    project_root: str = ".",
) -> None:
    base_seqs = base_seqs[:n_samples]

    batch = cfg.batch_for_node()
    n_rows = n_samples * cfg.seq * cfg.vocab_size
    n_chunks = -(-n_rows // batch)  # ceil division
    print(f"n_rows: {n_rows}, n_chunks: {n_chunks} (batch={batch})")

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

    # original (unperturbed) sequences, indexed by sample_idx -- each row's
    # sample_idx (saved per layer below) points into this table, so the
    # original sequence a row was derived from is base_seqs[sample_idx]. This
    # table is tiny (n_samples rows) relative to the sweep itself, so it's
    # kept as one small NFS-side file per run rather than sharded.
    base_seqs_path = os.path.join(project_root, out_dir, "base_seqs.h5")
    os.makedirs(os.path.dirname(base_seqs_path), exist_ok=True)
    if not os.path.exists(base_seqs_path):
        with h5py.File(base_seqs_path, "w") as f:
            f.attrs["n_samples"] = n_samples
            f.attrs["seq"] = cfg.seq
            f.attrs["d_model"] = cfg.d_model
            f.create_dataset("base_seqs", data=base_seqs)

    writers = {
        output_name: ShardedLayerWriter(
            layer_key=os.path.join(out_dir, output_name),
            n_rows=n_rows,
            schema={
                "vec": ((cfg.d_model,), "float32"),
                "sample_idx": ((), "int64"),
                "pos": ((), "int64"),
                "vocab_id": ((), "int64"),
            },
            shard_rows=shard_rows,
            flush_every=flush_every,
            attrs={"n_samples": n_samples, "seq": cfg.seq, "d_model": cfg.d_model,
                   "output_name": output_name},
            no_push=no_push,
            project_root=project_root,
        )
        for output_name in out_output_names[1:]
    }
    # All layers advance through the same chunk loop in lockstep, but may
    # have reached different next_row values before a crash (e.g. one
    # layer's shard finished uploading, another's didn't) -- resume from the
    # earliest, and each writer skips rows it already has.
    resume_row = min(w.next_row for w in writers.values())

    for c in tqdm.tqdm(range(n_chunks)):
        row_start = c * batch
        row_end = min(row_start + batch, n_rows)
        if row_end <= resume_row:
            continue
        valid = row_end - row_start

        token_ids, chunk_sample_idx, chunk_pos, chunk_vocab_id = build_chunk_rows(
            cfg, base_seqs, row_start, row_end,
        )
        if valid < batch:
            # pad the last chunk up to batch by repeating the last row
            pad = batch - valid
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
            writer = writers[output_name]
            if row_end <= writer.next_row:
                continue  # this layer already has this chunk from a prior run

            layer_out = fx[i][:valid]  # (valid, cfg.seq, d_model)
            swept_vecs = (
                layer_out[batch_arange, chunk_pos]
                .to(torch.float32)
                .detach()
                .cpu()
                .numpy()
            )

            writer.write_rows(
                row_start, row_end,
                vec=swept_vecs,
                sample_idx=chunk_sample_idx,
                pos=chunk_pos,
                vocab_id=chunk_vocab_id,
            )

    for output_name, writer in writers.items():
        writer.close()

    print(f"Saved perturbed sweep outputs under {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str, help="Path to a toml ModelConfig")
    parser.add_argument(
        "--base-seqs-h5",
        type=str,
        default="data/smoll_sequences",
        help="Output layer key (from scripts.generate_base_sequences) for a sharded "
        "input_ids dataset of (n_samples, seq) token ids",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="onnx_models/smoll/perturbed_sweep",
        help="Output directory -- one sharded output per layer is written here",
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
        help="Flush the current shard's h5 file to local scratch disk every this many chunks",
    )
    parser.add_argument(
        "--shard-rows",
        type=int,
        default=100_000,
        help="Rows per shard -- each shard is written to local scratch disk, "
        "then pushed to the remote store and deleted locally as soon as it's full",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Keep finished shards on local scratch disk instead of pushing to the remote store",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    base_seqs = load_base_seqs(args.base_seqs_h5)
    print(f"base_seqs: {base_seqs.shape}")

    run_perturbed_sweep(
        cfg,
        base_seqs,
        args.out_dir,
        n_samples=args.n_samples,
        flush_every=args.flush_every,
        shard_rows=args.shard_rows,
        no_push=args.no_push,
    )


if __name__ == "__main__":
    main()
