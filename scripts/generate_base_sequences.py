"""Load sequences from HuggingFace using a ModelConfig's input_source and save
them to a sharded h5 dataset in tokenized form.

Tokenization is CPU-bound dataset iteration (no GPU), but for large --n this
can still run long enough to be worth resuming rather than restarting: rows
are written via a ShardedLayerWriter (one "layer" -- the whole output --
sharded to local scratch disk, pushed to the remote store and deleted
locally as each shard fills, with progress checkpointed on NFS), so a
crash partway through resumes token-by-token from the last completed shard
instead of re-tokenizing from scratch.

Usage:
    uv run python -m scripts.generate_base_sequences configs/smoll.toml --n 1000 --seq 8 --out-key data/smoll_sequences
"""
import argparse

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from config import load_config
from execute.sharded_writer import ShardedLayerWriter


def tokenize_and_save_sequences(
    tokenizer_name: str,
    dataset: str,
    dataset_config: str | None,
    split: str,
    text_column: str,
    n: int,
    seq_len: int,
    out_key: str,
    shard_rows: int,
    flush_every: int,
    no_push: bool,
    overwrite_remote: bool = False,
) -> None:
    writer = ShardedLayerWriter(
        layer_key=out_key,
        n_rows=n,
        schema={"input_ids": ((seq_len,), "int64")},
        shard_rows=shard_rows,
        flush_every=flush_every,
        attrs={"n": n, "seq_len": seq_len},
        no_push=no_push,
        force_push=overwrite_remote,
        project_root=".",
    )
    if writer.next_row >= n:
        print(f"already complete ({writer.next_row}/{n} rows), nothing to do")
        writer.close()
        return

    ds = load_dataset(dataset, dataset_config, split=split)
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    # Rows already written (from a prior run) must be skipped from the
    # dataset iterator too, or a resume would re-tokenize the same examples
    # into a *different* set of rows than a from-scratch run would have
    # produced (the source dataset order is stable, so skipping the same
    # count of already-accepted examples reproduces the original row set).
    skip = writer.next_row
    seen = 0
    rows_batch: list[list[int]] = []
    row_start = writer.next_row

    def flush_batch(row_start: int, rows_batch: list[list[int]]) -> int:
        if not rows_batch:
            return row_start
        chunk = np.asarray(rows_batch, dtype=np.int64)
        writer.write_rows(row_start, row_start + len(rows_batch), input_ids=chunk)
        return row_start + len(rows_batch)

    for example in ds:
        if writer.next_row + len(rows_batch) >= n:
            break
        encoded = tok.encode(example[text_column])
        if len(encoded) < seq_len:
            continue
        if seen < skip:
            seen += 1
            continue
        rows_batch.append(encoded[:seq_len])
        if len(rows_batch) >= flush_every:
            row_start = flush_batch(row_start, rows_batch)
            rows_batch = []

    row_start = flush_batch(row_start, rows_batch)

    if writer.next_row < n:
        writer.close()
        raise ValueError(
            f"dataset only yielded {writer.next_row} examples with >= {seq_len} tokens, need {n}"
        )

    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str, help="Path to a toml ModelConfig")
    parser.add_argument("--n", type=int, default=1000, help="Number of sequences to load")
    parser.add_argument("--seq", type=int, default=None, help="Sequence length (defaults to cfg.seq)")
    parser.add_argument(
        "--out-key",
        type=str,
        default=None,
        help="Output layer key (project-relative path, no extension) -- "
        "shards and progress/manifest JSON are written under this",
    )
    parser.add_argument(
        "--shard-rows",
        type=int,
        default=200_000,
        help="Rows (sequences) per shard -- each shard is written to local scratch disk, "
        "then pushed to the remote store and deleted locally as soon as it's full",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=1000,
        help="Flush the current shard's h5 file to local scratch disk every this many rows",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Keep finished shards on local scratch disk instead of pushing to the remote store",
    )
    parser.add_argument(
        "--overwrite-remote",
        action="store_true",
        help="Overwrite the remote store's version of each shard unconditionally on a content "
        "conflict, instead of prompting (or, non-interactively, deferring to `upload.py "
        "apply-overrides`). Use after --overwrite when a prior run already pushed stale "
        "content for this --out-key.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seq_len = args.seq if args.seq is not None else cfg.seq
    out_key = args.out_key or f"data/{cfg.name}_sequences"
    print(f"getting {args.n} sequences of length {seq_len}")

    tokenize_and_save_sequences(
        tokenizer_name=cfg.input_source.tokenizer,
        dataset=cfg.input_source.dataset,
        dataset_config=cfg.input_source.dataset_config,
        split=cfg.input_source.split,
        text_column=cfg.input_source.text_column,
        n=args.n,
        seq_len=seq_len,
        out_key=out_key,
        shard_rows=args.shard_rows,
        flush_every=args.flush_every,
        no_push=args.no_push,
        overwrite_remote=args.overwrite_remote,
    )
    print(f"saved {args.n} tokenized sequences to {out_key}")


if __name__ == "__main__":
    main()
