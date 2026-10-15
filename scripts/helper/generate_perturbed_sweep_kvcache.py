"""KV-cache variant of scripts/python/generate_perturbed_sweep.py.

The perturbed sweep substitutes one token at a time into each base sequence
and records a sublayer's activation at the swept position, for every
(sample, position, vocab_id) triple. For a fixed (sample, position), every
one of the n_vocab (or n_vocab_sample) rows shares the exact same prefix
tokens[:position] -- generate_perturbed_sweep.py re-runs the full (batch,
seq) forward pass from scratch for every row regardless, recomputing that
shared prefix's attention/MLP compute n_vocab times over.

This script exploits the shared prefix instead: for each (sample, position),
run the prefix once through a KV-cache-enabled ONNX graph (see
construct/kv_cache_export.py, construct/kv_cache_subgraph.py) to get that
prefix's key/value cache, then run every vocab candidate as a length-1
continuation batched together, with the prefix's KV broadcast across that
batch via torch.Tensor.expand (a view, not a physical copy, until ORT reads
it). Verified numerically against a full forward pass (see the prototyping
session this script came out of): matches to fp32 rounding noise (~1e-7).

Kept as a separate script/pipeline from generate_perturbed_sweep.py rather
than folding the two together -- the two-call (prefix + continuation)
control flow, KV broadcasting, and (sample, position)-grouped chunking are
different enough from the flat-row-range chunking of the original that
merging them would make both harder to follow, and this way the existing
sweep is untouched.

Output format (ShardedLayerWriter schema, directory layout, resume
semantics) is unchanged from generate_perturbed_sweep.py, so downstream
loaders (load_shard_rows et al.) work on either script's output the same
way.
"""

import argparse
import os

import h5py
import numpy as np
import onnxruntime as ort
import torch
import torch.utils.dlpack
import tqdm

from config import load_config
from construct.kv_cache_export import ensure_kv_cache_model
from construct.kv_cache_subgraph import (
    ensure_kv_cache_extra_outputs,
    past_kv_input_names,
    present_kv_names,
)
from execute.sharded_writer import ShardedLayerWriter, load_base_seqs

_OOM_BATCH_SHRINK_FACTOR = 0.7
_OOM_MAX_RETRIES = 5


def _run_prefix(
    sess: ort.InferenceSession,
    io_binding: ort.IOBinding,
    prefix_ids: np.ndarray,
    n_layers: int,
) -> list[torch.Tensor]:
    """Runs the shared prefix (1, pos) through `sess` with an empty incoming
    KV cache, returning the n_layers*2 present.{i}.{key,value} GPU tensors
    (each (1, n_kv_heads, pos, head_dim)). Called once per (sample, position)
    -- must have pos >= 1: the exported graph cannot run an empty input_ids
    (ORT's Reshape fails on 0 elements), so pos == 0 skips this call and
    _run_continuation attends to a zero-length past instead."""
    if prefix_ids.shape[1] == 0:
        raise ValueError("_run_prefix needs pos >= 1; pass prefix_kv=None to _run_continuation for pos == 0")
    pos = prefix_ids.shape[1]
    inputs = {
        "input_ids": prefix_ids,
        "attention_mask": np.ones((1, pos), dtype=np.int64),
        "position_ids": np.arange(pos, dtype=np.int64)[None, :],
    }

    # past K/V shape is (batch, n_kv_heads, past_len=0, head_dim) -- n_kv_heads/head_dim
    # are concrete dims in the graph's declared input shape (only batch and
    # past_sequence_length are symbolic), so read them off sess.get_inputs()
    # rather than hardcoding them.
    for inp in sess.get_inputs():
        if inp.name.startswith("past_key_values."):
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]
            shape[0] = 1  # batch
            shape[2] = 0  # past_len
            inputs[inp.name] = np.zeros(shape, dtype=np.float32)

    io_binding.clear_binding_inputs()
    io_binding.clear_binding_outputs()
    for name, array in inputs.items():
        io_binding.bind_cpu_input(name, array)
    output_names = [sess.get_outputs()[0].name] + present_kv_names(n_layers)
    for name in output_names:
        io_binding.bind_output(name, device_type="cuda", device_id=0)
    sess.run_with_iobinding(io_binding)

    return [torch.utils.dlpack.from_dlpack(o._ortvalue.to_dlpack()) for o in io_binding.get_outputs()]


def _run_continuation(
    sess: ort.InferenceSession,
    io_binding: ort.IOBinding,
    candidate_ids: np.ndarray,
    pos: int,
    prefix_kv: list[torch.Tensor] | None,
    n_layers: int,
    sublayer_output_name: str,
) -> torch.Tensor:
    """Runs `candidate_ids` (n_candidates,) as independent length-1
    continuations, each attending back to the same shared `prefix_kv`
    (broadcast across the batch, not duplicated) plus itself. Returns the
    sublayer activation at the new position, (n_candidates, d_model).

    `prefix_kv=None` (pos == 0) binds a zero-length past instead."""
    n_candidates = candidate_ids.shape[0]

    io_binding.clear_binding_inputs()
    io_binding.clear_binding_outputs()
    io_binding.bind_cpu_input("input_ids", candidate_ids[:, None].astype(np.int64))
    io_binding.bind_cpu_input(
        "attention_mask", np.ones((n_candidates, pos + 1), dtype=np.int64),
    )
    io_binding.bind_cpu_input(
        "position_ids", np.full((n_candidates, 1), pos, dtype=np.int64),
    )

    # Bind each layer's broadcast prefix KV as a GPU-resident input directly
    # (no host round-trip) -- prefix_kv tensors are already on DEVICE from
    # _run_prefix, torch.expand gives a zero-copy broadcast view, and ORT's
    # bind_input (as opposed to bind_cpu_input) takes a device pointer
    # straight from that view's storage.
    present_names = present_kv_names(n_layers)
    past_names = past_kv_input_names(n_layers)
    expanded = []  # keep references alive until session.run_with_iobinding returns
    if prefix_kv is None:
        assert pos == 0
        for inp in sess.get_inputs():
            if inp.name.startswith("past_key_values."):
                shape = [d if isinstance(d, int) else 0 for d in inp.shape]
                shape[0] = n_candidates
                io_binding.bind_cpu_input(inp.name, np.zeros(shape, dtype=np.float32))
    for past_name, present_tensor in zip(past_names, prefix_kv[1:] if prefix_kv is not None else []):  # [0] is the sublayer output
        expanded_view = present_tensor.expand(n_candidates, *present_tensor.shape[1:]).contiguous()
        expanded.append(expanded_view)
        io_binding.bind_input(
            name=past_name,
            device_type="cuda",
            device_id=0,
            element_type=np.float32 if expanded_view.dtype == torch.float32 else _np_dtype(expanded_view.dtype),
            shape=tuple(expanded_view.shape),
            buffer_ptr=expanded_view.data_ptr(),
        )

    io_binding.bind_output(sublayer_output_name, device_type="cuda", device_id=0)
    for name in present_names:
        io_binding.bind_output(name, device_type="cuda", device_id=0)  # discarded -- required output

    sess.run_with_iobinding(io_binding)
    out = io_binding.get_outputs()[0]
    # (n_candidates, 1, d_model): drop the length-1 sequence dim
    return torch.utils.dlpack.from_dlpack(out._ortvalue.to_dlpack())[:, 0]


def _np_dtype(torch_dtype: torch.dtype):
    import ml_dtypes
    mapping = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.bfloat16: ml_dtypes.bfloat16,
    }
    return mapping[torch_dtype]


def run_perturbed_sweep_kvcache(
    cfg,
    base_seqs: np.ndarray,
    out_dir: str,
    n_samples: int,
    layer_idx: int,
    boundary: str,
    flush_every: int,
    shard_rows: int,
    candidate_batch: int,
    no_push: bool = False,
    defer_push: bool = False,
    auto_delete: bool = False,
    project_root: str = ".",
    max_in_flight: int = 1,
    n_vocab_sample: int | None = None,
    vocab_sample_seed: int = 0,
) -> None:
    """Retries once on OOM, shrinking `candidate_batch` -- same rationale as
    generate_perturbed_sweep.run_perturbed_sweep's OOM retry, but the batch
    dimension being shrunk here is the continuation step's candidate count,
    not an ONNX-export-time static batch (this graph's batch dim is dynamic,
    so no re-export is needed on shrink)."""
    for attempt in range(_OOM_MAX_RETRIES + 1):
        try:
            _run_perturbed_sweep_kvcache_once(
                cfg, base_seqs, out_dir, n_samples, layer_idx, boundary, flush_every, shard_rows,
                candidate_batch, no_push=no_push, defer_push=defer_push, auto_delete=auto_delete,
                project_root=project_root,
                max_in_flight=max_in_flight,
                n_vocab_sample=n_vocab_sample, vocab_sample_seed=vocab_sample_seed,
            )
            return
        except RuntimeError as e:
            from scripts.helper.find_max_batch_size import _is_oom_error
            if not _is_oom_error(e) or attempt == _OOM_MAX_RETRIES:
                raise
            old_batch = candidate_batch
            candidate_batch = max(1, int(candidate_batch * _OOM_BATCH_SHRINK_FACTOR))
            print(
                f"OOM at candidate_batch={old_batch} (attempt {attempt + 1}/{_OOM_MAX_RETRIES}), "
                f"shrinking to candidate_batch={candidate_batch} and retrying",
                flush=True,
            )


def _run_perturbed_sweep_kvcache_once(
    cfg,
    base_seqs: np.ndarray,
    out_dir: str,
    n_samples: int,
    layer_idx: int,
    boundary: str,
    flush_every: int,
    shard_rows: int,
    candidate_batch: int,
    no_push: bool = False,
    defer_push: bool = False,
    auto_delete: bool = False,
    project_root: str = ".",
    max_in_flight: int = 1,
    n_vocab_sample: int | None = None,
    vocab_sample_seed: int = 0,
) -> None:
    base_seqs = base_seqs[:n_samples]

    vocab_ids = None
    if n_vocab_sample is not None:
        rng = np.random.default_rng(vocab_sample_seed)
        vocab_ids = np.sort(rng.choice(cfg.vocab_size, size=n_vocab_sample, replace=False))
    n_vocab = cfg.vocab_size if vocab_ids is None else len(vocab_ids)
    n_rows = n_samples * cfg.seq * n_vocab
    print(f"n_rows: {n_rows} (n_samples={n_samples}, seq={cfg.seq}, n_vocab={n_vocab})")

    ensure_kv_cache_model(cfg)
    subgraph_path, sublayer_output_name, n_layers = ensure_kv_cache_extra_outputs(
        cfg, layer_idx, boundary,
    )

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.intra_op_num_threads = 8
    sess = ort.InferenceSession(
        str(subgraph_path), sess_options=sess_options, providers=["CUDAExecutionProvider"],
    )

    prefix_io_binding = sess.io_binding()
    continuation_io_binding = sess.io_binding()

    base_seqs_path = os.path.join(project_root, out_dir, "base_seqs.h5")
    os.makedirs(os.path.dirname(base_seqs_path), exist_ok=True)
    if not os.path.exists(base_seqs_path):
        with h5py.File(base_seqs_path, "w") as f:
            f.attrs["n_samples"] = n_samples
            f.attrs["seq"] = cfg.seq
            f.attrs["d_model"] = cfg.d_model
            f.create_dataset("base_seqs", data=base_seqs)

    writer = ShardedLayerWriter(
        layer_key=os.path.join(out_dir, sublayer_output_name.strip("/").replace("/", ".")),
        n_rows=n_rows,
        schema={
            "vec": ((cfg.d_model,), "float32"),
            "sample_idx": ((), "int64"),
            "pos": ((), "int64"),
            "vocab_id": ((), "int64"),
        },
        shard_rows=shard_rows,
        flush_every=flush_every,
        attrs={
            "n_samples": n_samples, "seq": cfg.seq, "d_model": cfg.d_model,
            "output_name": sublayer_output_name, "layer_idx": layer_idx, "boundary": boundary,
        },
        no_push=no_push,
        defer_push=defer_push,
        auto_delete=auto_delete,
        project_root=project_root,
        max_in_flight=max_in_flight,
    )

    # Row order matches generate_perturbed_sweep.py's unravel_index((sample_idx,
    # pos, vocab_idx), (n_samples, seq, n_vocab)) -- (sample_idx, pos) groups are
    # contiguous row ranges of size n_vocab, so resuming mid-group is just
    # skipping the already-written vocab_idx prefix within that group.
    resume_row = writer.next_row

    group_size = n_vocab
    total_groups = n_samples * cfg.seq
    start_group = resume_row // group_size

    pbar = tqdm.tqdm(total=total_groups, initial=start_group, unit="group")
    for group_idx in range(start_group, total_groups):
        sample_idx, pos = divmod(group_idx, cfg.seq)
        group_row_start = group_idx * group_size
        group_row_end = group_row_start + group_size
        if group_row_end <= writer.next_row:
            pbar.update(1)
            continue

        prefix_ids = base_seqs[sample_idx : sample_idx + 1, :pos]
        prefix_kv = _run_prefix(sess, prefix_io_binding, prefix_ids, n_layers) if pos > 0 else None

        skip_in_group = max(0, writer.next_row - group_row_start)
        for chunk_start in range(skip_in_group, group_size, candidate_batch):
            chunk_end = min(chunk_start + candidate_batch, group_size)
            vocab_idx_chunk = np.arange(chunk_start, chunk_end)
            candidate_ids = vocab_idx_chunk if vocab_ids is None else vocab_ids[vocab_idx_chunk]

            swept_vecs = _run_continuation(
                sess, continuation_io_binding, candidate_ids, pos, prefix_kv, n_layers,
                sublayer_output_name,
            ).to(torch.float32).detach().cpu().numpy()

            n_chunk = chunk_end - chunk_start
            writer.write_rows(
                group_row_start + chunk_start, group_row_start + chunk_end,
                vec=swept_vecs,
                sample_idx=np.full(n_chunk, sample_idx, dtype=np.int64),
                pos=np.full(n_chunk, pos, dtype=np.int64),
                vocab_id=candidate_ids.astype(np.int64),
            )

        pbar.update(1)
    pbar.close()

    writer.close()

    print(f"Saved KV-cache perturbed sweep outputs under {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str, help="Path to a toml ModelConfig")
    parser.add_argument("--layer-idx", type=int, required=True,
                         help="Transformer decoder layer index (0-based) to sweep")
    parser.add_argument("--boundary", type=str, choices=["attn_out", "mlp_out", "embed"], required=True,
                         help="Which residual-stream tensor within --layer-idx to record: "
                         "'attn_out' (post-attention residual) or 'mlp_out' (post-MLP residual, "
                         "also the residual stream entering the next layer), or 'embed' (token+position "
                         "embeddings; --layer-idx is ignored)")
    parser.add_argument(
        "--base-seqs-h5", type=str, default="data/smoll_sequences",
        help="Output layer key (from scripts.python.generate_base_sequences) for a sharded "
        "input_ids dataset of (n_samples, seq) token ids",
    )
    parser.add_argument(
        "--out-dir", type=str, default="onnx_models/smoll/perturbed_sweep_kvcache",
        help="Output directory -- one sharded output for the swept sublayer is written here",
    )
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--flush-every", type=int, default=50)
    parser.add_argument("--shard-rows", type=int, default=2_000_000)
    parser.add_argument(
        "--candidate-batch", type=int, default=2048,
        help="Max number of vocab candidates run as one continuation-step batch "
        "(independent of any prefix-pass batching -- the prefix pass always runs at batch=1)",
    )
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--defer-push", action="store_true")
    parser.add_argument(
        "--auto-delete", action="store_true",
        help="Delete a shard's local scratch copy once its upload is confirmed pushed "
        "(default: keep it on local scratch disk after pushing). Ignored if --no-push "
        "is also set.",
    )
    parser.add_argument("--max-in-flight", type=int, default=1)
    parser.add_argument("--n-vocab-sample", type=int, default=None)
    parser.add_argument("--vocab-sample-seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_seqs = load_base_seqs(args.base_seqs_h5)
    print(f"base_seqs: {base_seqs.shape}")

    run_perturbed_sweep_kvcache(
        cfg, base_seqs, args.out_dir,
        n_samples=args.n_samples,
        layer_idx=args.layer_idx,
        boundary=args.boundary,
        flush_every=args.flush_every,
        shard_rows=args.shard_rows,
        candidate_batch=args.candidate_batch,
        no_push=args.no_push,
        defer_push=args.defer_push,
        auto_delete=args.auto_delete,
        max_in_flight=args.max_in_flight,
        n_vocab_sample=args.n_vocab_sample,
        vocab_sample_seed=args.vocab_sample_seed,
    )


if __name__ == "__main__":
    main()
