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
from execute.sharded_writer import ShardedLayerWriter, load_base_seqs

DEVICE = torch.device("cuda")


def bf16_torch_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Bit-reinterpret a torch bfloat16 tensor as an ml_dtypes.bfloat16 numpy
    array (torch tensors can't go through .numpy() directly in this bf16
    dtype -- see the plain uint16 view/roundtrip this relies on)."""
    return t.contiguous().view(torch.uint16).cpu().numpy().view(ml_dtypes.bfloat16)


def _save_base_seqs_once(out_dir: str, base_seqs: np.ndarray) -> None:
    """base_seqs is tiny relative to the activations and shared across every
    layer -- kept as one small NFS-side file per run rather than duplicated
    into every layer's shards."""
    path = os.path.join(out_dir, "base_seqs.h5")
    if os.path.exists(path):
        return
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(path, "w") as h5f:
        h5f.attrs["n_samples"] = base_seqs.shape[0]
        h5f.create_dataset("base_seqs", data=base_seqs)


def collect_and_save_activations(
    cfg,
    base_seqs: np.ndarray,
    out_dir: str,
    shard_rows: int,
    flush_every: int,
    no_push: bool,
) -> None:
    """Runs base_seqs through the model in cfg.batch-sized chunks and writes
    each layer's activations incrementally via a ShardedLayerWriter (rows =
    samples), instead of accumulating every layer's full (n_samples, seq,
    d_model) tensor on GPU before writing anything once at the end -- so a
    crash partway through only loses the current chunk, and a re-run resumes
    rather than redoing the whole pass."""
    n_samples = base_seqs.shape[0]

    ensure_base_model(cfg, force=True, verbose=False)

    extra_outputs = sorted(
        {s.input for s in cfg.sublayers} | {s.output for s in cfg.sublayers}
    )

    extra_outputs_model_path = ensure_extra_outputs(cfg, extra_outputs, force=True)

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
    layer_names = [name for name in extra_output_names if name != "output"]

    _save_base_seqs_once(out_dir, base_seqs)

    writers = {
        name: ShardedLayerWriter(
            layer_key=os.path.join(out_dir, "activations", name),
            n_rows=n_samples,
            schema={"activation": ((cfg.seq, cfg.d_model), "uint16")},
            shard_rows=shard_rows,
            flush_every=flush_every,
            attrs={"n_samples": n_samples, "seq": cfg.seq, "d_model": cfg.d_model, "layer": name},
            no_push=no_push,
            project_root=".",
        )
        for name in layer_names
    }
    resume_row = min(w.next_row for w in writers.values())
    if resume_row >= n_samples:
        print("all layers' activations already complete, nothing to do")
        for w in writers.values():
            w.close()
        return

    # Run base sequences through the session, chunked by the node's fixed
    # ONNX batch dim (see cfg.batch_for_node). Pad the last chunk by
    # repeating the final row, then trim the padding rows back off before
    # writing.
    batch = cfg.batch_for_node()
    for start in tqdm.tqdm(range(resume_row, n_samples, batch)):
        end = min(start + batch, n_samples)
        chunk = base_seqs[start:end]
        pad = batch - chunk.shape[0]
        if pad:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], pad, axis=0)], axis=0)

        fx = run_iobinding_cpu_input_gpu_output(
            extra_outputs_sess, extra_input_name, chunk, extra_output_names
        )
        for name, val in zip(extra_output_names, fx):
            if name == "output":
                continue
            writer = writers[name]
            if end <= writer.next_row:
                continue  # this layer already has this chunk from a prior run
            val = val[: end - start]  # trim padding rows back off
            activation_np = bf16_torch_to_numpy(val).view(np.uint16)
            writer.write_rows(start, end, activation=activation_np)

    for writer in writers.values():
        writer.close()


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
        default="onnx_models/smoll/base_activations",
        help="Output directory -- base_seqs.h5 plus one sharded output per "
        "layer (under activations/<layer>/) is written here",
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

    base_seqs = load_base_seqs(args.base_seqs_h5)
    print(f"base_seqs: {base_seqs.shape}")

    collect_and_save_activations(
        cfg, base_seqs, args.out_dir,
        shard_rows=args.shard_rows, flush_every=args.flush_every, no_push=args.no_push,
    )
    print(f"saved base_seqs + activations to {args.out_dir}")


if __name__ == "__main__":
    main()
