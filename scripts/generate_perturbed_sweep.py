import argparse
import dataclasses
import os
import queue
import threading

import h5py
import numpy as np
import onnxruntime as ort
import torch
import torch.utils.dlpack
import tqdm

from config import current_node_name, load_config
from construct.construct import ensure_base_model, ensure_extra_outputs, exported_batch_size
from execute.io_onnx import run_iobinding_cpu_input_gpu_output
from execute.remote_storage import AsyncPusher
from execute.sharded_writer import ShardedLayerWriter, load_base_seqs

DEVICE = torch.device("cuda")

# On OOM, cfg.batch_for_node() is shrunk by this factor and the sweep is
# retried (see run_perturbed_sweep) -- find_max_batch_size.py's probe can't
# perfectly predict the real workload's peak memory (different code path,
# concurrent uploads, ...), so this is the last-resort safety net rather
# than the primary way to pick a batch size.
_OOM_BATCH_SHRINK_FACTOR = 0.7
_OOM_MAX_RETRIES = 5


def build_chunk_rows(
    cfg,
    base_seqs: np.ndarray,
    row_start: int,
    row_end: int,
    vocab_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one chunk's (row_end - row_start, cfg.seq) token-id block plus
    its per-row (sample_idx, pos, vocab_id) annotation, without ever
    materializing all n_rows of the sweep at once (n_rows = n_samples *
    cfg.seq * n_vocab_sample can be in the hundreds of millions).

    Row order is unravelled from (sample_idx, pos, vocab_idx), matching the
    original dense-array construction this replaces. `vocab_ids`, if given,
    is a fixed subsample of token ids (see --n-vocab-sample) indexed by
    vocab_idx; the resulting per-row vocab_id array holds the actual token
    ids substituted in, not the indices into `vocab_ids`.
    """
    n_samples = base_seqs.shape[0]
    n_vocab = cfg.vocab_size if vocab_ids is None else len(vocab_ids)
    row_sample_idx, row_pos, row_vocab_idx = np.unravel_index(
        np.arange(row_start, row_end), (n_samples, cfg.seq, n_vocab),
    )
    row_vocab_id = row_vocab_idx if vocab_ids is None else vocab_ids[row_vocab_idx]
    token_ids = base_seqs[row_sample_idx].copy()  # (chunk, cfg.seq)
    token_ids[np.arange(row_end - row_start), row_pos] = row_vocab_id
    return token_ids, row_sample_idx, row_pos, row_vocab_id


def _prefetch_chunk_rows(
    cfg, base_seqs: np.ndarray, chunk_bounds, vocab_ids: np.ndarray | None = None, prefetch: int = 2,
):
    """Yields build_chunk_rows(cfg, base_seqs, row_start, row_end, vocab_ids)
    for each (row_start, row_end) in `chunk_bounds`, computed on a
    background thread up to `prefetch` chunks ahead of what's been yielded
    so far.

    build_chunk_rows is pure numpy/CPU work with no GPU or shared mutable
    state involved, so it's safe to run on a worker thread while the main
    thread is blocked in the GPU forward pass or writing the previous
    chunk's output -- this keeps the CPU busy building chunk i+1's rows
    instead of sitting idle for the whole GPU call + write of chunk i.
    """
    q: queue.Queue = queue.Queue(maxsize=prefetch)
    _SENTINEL = object()

    def _worker():
        try:
            for row_start, row_end in chunk_bounds:
                result = build_chunk_rows(cfg, base_seqs, row_start, row_end, vocab_ids)
                q.put((row_start, row_end, result, None))
        except Exception as e:  # surface the exception on the consumer side
            q.put((None, None, None, e))
        finally:
            q.put(_SENTINEL)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    try:
        while True:
            item = q.get()
            if item is _SENTINEL:
                return
            row_start, row_end, result, err = item
            if err is not None:
                raise err
            yield row_start, row_end, result
    finally:
        thread.join()


def run_perturbed_sweep(
    cfg,
    base_seqs: np.ndarray,
    out_dir: str,
    n_samples: int,
    flush_every: int,
    shard_rows: int,
    no_push: bool = False,
    project_root: str = ".",
    upload_workers: int = 4,
    config_path: str | None = None,
    n_vocab_sample: int | None = None,
    vocab_sample_seed: int = 0,
) -> None:
    """Retries _run_perturbed_sweep_once on OOM, shrinking cfg.batch_for_node()
    each time -- find_max_batch_size.py's probe measures a different code
    path (no concurrent uploads, no dlpack tensors held across a chunk's
    full gather+write) than the real sweep, so it can pick a batch that
    still OOMs here. Each retry re-exports at the smaller batch (the ONNX
    batch dim is fixed at export time) and resumes from writer.next_row, so
    only the failed attempt's in-flight chunk is wasted work.

    `config_path`: if given, a successful shrink is written back into
    [batch_by_node] in this TOML file, so the next invocation starts from
    the working batch size instead of re-discovering it via another OOM."""
    from scripts.find_max_batch_size import _is_oom_error, _update_batch_by_node

    # Drawn once up front (not inside _run_perturbed_sweep_once) so it's
    # identical across OOM retries -- resuming from writer.next_row relies on
    # row index -> vocab_id being a fixed mapping across attempts.
    vocab_ids = None
    if n_vocab_sample is not None:
        rng = np.random.default_rng(vocab_sample_seed)
        vocab_ids = np.sort(rng.choice(cfg.vocab_size, size=n_vocab_sample, replace=False))

    node = current_node_name()
    shrunk = False
    for attempt in range(_OOM_MAX_RETRIES + 1):
        try:
            _run_perturbed_sweep_once(
                cfg, base_seqs, out_dir, n_samples, flush_every, shard_rows,
                no_push=no_push, project_root=project_root, upload_workers=upload_workers,
                vocab_ids=vocab_ids,
            )
            break
        except RuntimeError as e:
            if not _is_oom_error(e) or attempt == _OOM_MAX_RETRIES:
                raise
            old_batch = cfg.batch_for_node(node)
            new_batch = max(1, int(old_batch * _OOM_BATCH_SHRINK_FACTOR))
            print(
                f"OOM at batch={old_batch} (attempt {attempt + 1}/{_OOM_MAX_RETRIES}), "
                f"shrinking to batch={new_batch} and retrying",
                flush=True,
            )
            cfg = dataclasses.replace(cfg, batch_by_node={**cfg.batch_by_node, node: new_batch})
            shrunk = True

    if shrunk and config_path is not None:
        _update_batch_by_node(config_path, node, cfg.batch_for_node(node))
        print(f"Wrote shrunk batch_by_node.{node} = {cfg.batch_for_node(node)} -> {config_path}")


def _run_perturbed_sweep_once(
    cfg,
    base_seqs: np.ndarray,
    out_dir: str,
    n_samples: int,
    flush_every: int,
    shard_rows: int,
    no_push: bool = False,
    project_root: str = ".",
    upload_workers: int = 4,
    vocab_ids: np.ndarray | None = None,
) -> None:
    base_seqs = base_seqs[:n_samples]

    batch = cfg.batch_for_node()
    n_vocab = cfg.vocab_size if vocab_ids is None else len(vocab_ids)
    n_rows = n_samples * cfg.seq * n_vocab
    n_chunks = -(-n_rows // batch)  # ceil division
    print(f"n_rows: {n_rows}, n_chunks: {n_chunks} (batch={batch})")

    # Force a re-export if the cached base model was built for a different
    # batch size than this node currently wants (e.g. cfg.batch_by_node was
    # updated, or this is a different node's cache) -- otherwise reuse it,
    # since re-exporting a multi-GB model on every invocation is wasteful.
    cached_batch = exported_batch_size(cfg.base_model_path)
    stale_batch = cached_batch != batch
    if stale_batch:
        print(
            f"Cached base model batch size doesn't match cfg.batch_for_node() "
            f"({cached_batch} != {batch}), forcing re-export",
        )
    ensure_base_model(cfg, force=stale_batch, verbose=False)
    extra_outputs = [sublayer.output for sublayer in cfg.sublayers]
    model_path = ensure_extra_outputs(cfg, extra_outputs, force=stale_batch)
    # input_shape is (batch, seq, d_model) of the sublayer's OUTPUT tensor --
    # not always cfg.d_model, e.g. GPT-2's MLP intermediate (c_fc) outputs
    # are 4*d_model wide. Keyed by output name since that's what out_sess's
    # outputs (and hence `writers` below) are indexed by.
    d_model_by_output = {sublayer.output: sublayer.input_shape[2] for sublayer in cfg.sublayers}

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.intra_op_num_threads = 8

    out_sess = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=["CUDAExecutionProvider"],
    )
    # No leading "output" (final logits) slot -- ensure_extra_outputs
    # extracts a subgraph whose outputs are exactly extra_outputs, dropping
    # the logits/lm_head computation entirely since nothing here reads it.
    out_output_names = [o.name for o in out_sess.get_outputs()]
    out_input_names = [o.name for o in out_sess.get_inputs()]
    print(out_output_names, out_input_names)

    # Reused across every chunk instead of allocating a new IOBinding per
    # call (run_iobinding_cpu_input_gpu_output clears its prior bindings
    # before rebinding, so this is safe across chunks with differing shapes,
    # e.g. the final padded chunk).
    io_binding = out_sess.io_binding()

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

    # Shared across every layer's writer below so their shard uploads run
    # concurrently (bounded by upload_workers) instead of one at a time --
    # each writer still caps itself at one of its own shards in flight, so
    # the GPU loop is never more than one shard ahead of what's been pushed
    # for a given layer.
    pusher = None if no_push else AsyncPusher(max_workers=upload_workers)

    writers = {
        output_name: ShardedLayerWriter(
            layer_key=os.path.join(out_dir, output_name),
            n_rows=n_rows,
            schema={
                "vec": ((d_model_by_output[output_name],), "float32"),
                "sample_idx": ((), "int64"),
                "pos": ((), "int64"),
                "vocab_id": ((), "int64"),
            },
            shard_rows=shard_rows,
            flush_every=flush_every,
            attrs={"n_samples": n_samples, "seq": cfg.seq, "d_model": d_model_by_output[output_name],
                   "output_name": output_name},
            no_push=no_push,
            project_root=project_root,
            pusher=pusher,
        )
        for output_name in out_output_names
    }
    # All layers advance through the same chunk loop in lockstep, but may
    # have reached different next_row values before a crash (e.g. one
    # layer's shard finished uploading, another's didn't) -- resume from the
    # earliest, and each writer skips rows it already has.
    resume_row = min(w.next_row for w in writers.values())

    chunk_bounds = []
    for c in range(n_chunks):
        row_start = c * batch
        row_end = min(row_start + batch, n_rows)
        if row_end <= resume_row:
            continue
        chunk_bounds.append((row_start, row_end))

    # build_chunk_rows for the next chunk(s) runs on a background thread
    # while this chunk's GPU forward pass + output gather/write happen here,
    # so the CPU-bound row construction is never on the critical path.
    prefetched = _prefetch_chunk_rows(cfg, base_seqs, chunk_bounds, vocab_ids=vocab_ids)

    for row_start, row_end, (token_ids, chunk_sample_idx, chunk_pos, chunk_vocab_id) in tqdm.tqdm(
        prefetched, total=len(chunk_bounds),
    ):
        valid = row_end - row_start
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
            io_binding=io_binding,
        )

        # gather the swept position's d_model vector for each row in the
        # chunk
        batch_arange = np.arange(valid)

        for i, output_name in enumerate(out_output_names):
            writer = writers[output_name]
            if row_end <= writer.next_row:
                continue  # this layer already has this chunk from a prior run

            # A chunk's boundaries come from the current --batch value, but
            # writer.next_row was left off at some prior run's (possibly
            # different) batch size, so it can fall strictly inside this
            # chunk rather than exactly at row_start. Only the sub-slice from
            # next_row onward is new for this layer -- write_rows requires
            # row_start == next_row, so trim the skip offset off every
            # per-row array before calling it.
            skip = max(0, writer.next_row - row_start)

            layer_out = fx[i][:valid]  # (valid, cfg.seq, d_model)
            swept_vecs = (
                layer_out[batch_arange, chunk_pos]
                .to(torch.float32)
                .detach()
                .cpu()
                .numpy()
            )

            writer.write_rows(
                row_start + skip, row_end,
                vec=swept_vecs[skip:],
                sample_idx=chunk_sample_idx[skip:],
                pos=chunk_pos[skip:],
                vocab_id=chunk_vocab_id[skip:],
            )

    for output_name, writer in writers.items():
        writer.close()  # waits for that layer's own last in-flight upload

    if pusher is not None:
        pusher.shutdown()

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
        default=64,
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
        default=1_000_000,
        help="Rows per shard -- each shard is written to local scratch disk, "
        "then pushed to the remote store and deleted locally as soon as it's full",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Keep finished shards on local scratch disk instead of pushing to the remote store",
    )
    parser.add_argument(
        "--upload-workers",
        type=int,
        default=4,
        help="Max number of shard uploads (across all layers) to run concurrently in "
        "separate processes, while the GPU sweep keeps computing",
    )
    parser.add_argument(
        "--n-vocab-sample",
        type=int,
        default=None,
        help="If set, sweep a fixed random subsample of this many token ids out of "
        "cfg.vocab_size, instead of every vocab id (e.g. 20000 out of 32000)",
    )
    parser.add_argument(
        "--vocab-sample-seed",
        type=int,
        default=0,
        help="RNG seed for --n-vocab-sample's random subsample",
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
        upload_workers=args.upload_workers,
        config_path=args.config,
        n_vocab_sample=args.n_vocab_sample,
        vocab_sample_seed=args.vocab_sample_seed,
    )


if __name__ == "__main__":
    main()
