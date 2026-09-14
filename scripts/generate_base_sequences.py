"""Load sequences from HuggingFace using a ModelConfig's input_source and save
them to an h5 file in tokenized form.

Usage:
    uv run generate_data/tokenize_to_h5.py configs/smoll.toml --n 1000 --seq 8 --out data/smoll_sequences.h5
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from config import load_config


def tokenize_sequences(
    tokenizer_name: str,
    dataset: str,
    dataset_config: str | None,
    split: str,
    text_column: str,
    n: int,
    seq_len: int,
) -> np.ndarray:
    ds = load_dataset(dataset, dataset_config, split=split)
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    rows: list[list[int]] = []
    for example in ds:
        if len(rows) >= n:
            break
        encoded = tok.encode(example[text_column])
        if len(encoded) < seq_len:
            continue
        rows.append(encoded[:seq_len])

    if len(rows) < n:
        raise ValueError(
            f"dataset only yielded {len(rows)} examples with >= {seq_len} tokens, need {n}"
        )

    return np.asarray(rows, dtype=np.int64)


def save_sequences(sequences: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("input_ids", data=sequences)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str, help="Path to a toml ModelConfig")
    parser.add_argument("--n", type=int, default=1000, help="Number of sequences to load")
    parser.add_argument("--seq", type=int, default=None, help="Sequence length (defaults to cfg.seq)")
    parser.add_argument("--out", type=str, default=None, help="Output h5 path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seq_len = args.seq if args.seq is not None else cfg.seq
    out_path = Path(args.out) if args.out else Path("data") / f"{cfg.name}_sequences.h5"
    print(f"getting {args.n} sequences of length {seq_len}")
    sequences: np.ndarray[tuple[Any, ...], np.dtype[Any]] = tokenize_sequences(
        tokenizer_name=cfg.input_source.tokenizer,
        dataset=cfg.input_source.dataset,
        dataset_config=cfg.input_source.dataset_config,
        split=cfg.input_source.split,
        text_column=cfg.input_source.text_column,
        n=args.n,
        seq_len=seq_len,
    )

    save_sequences(sequences, out_path)
    print(f"saved {sequences.shape} tokenized sequences to {out_path}")


if __name__ == "__main__":
    main()
