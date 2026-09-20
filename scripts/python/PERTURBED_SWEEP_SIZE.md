# Sizing `generate_perturbed_sweep.py` output

## Row count

Each sublayer output writes one row per (sample, sequence position, swept
vocab id):

```
n_rows = n_samples * cfg.seq * n_vocab
```

where `n_vocab = cfg.vocab_size` unless `--n-vocab-sample` is set, in which
case `n_vocab = n_vocab_sample`.

This is computed **per sublayer output** (`cfg.sublayers`), and each
sublayer's rows are written to its own file/shard set
(`{out_dir}/{sublayer_output}/shard_*.h5`), so total data scales with the
number of sublayers too.

## Bytes per row

Each row's schema (`execute/sharded_writer.py`, set in
`_run_perturbed_sweep_once`):

| field       | shape          | dtype   | bytes |
|-------------|----------------|---------|-------|
| `vec`       | `(d_model,)`   | float32 | `4 * d_model` |
| `sample_idx`| scalar         | int64   | 8 |
| `pos`       | scalar         | int64   | 8 |
| `vocab_id`  | scalar         | int64   | 8 |

```
bytes_per_row = 4 * d_model + 24
```

(HDF5 adds a small amount of overhead per dataset/chunk; ignore for
estimation purposes, or add ~1-2% for chunk headers if being precise.)

## Total size

```
bytes_per_sublayer = n_rows * bytes_per_row
total_bytes        = bytes_per_sublayer * n_sublayers
```

Plus a small, fixed `base_seqs.h5` (`(n_samples, cfg.seq)` int token ids +
`(n_samples, cfg.seq, d_model)`-independent — it only stores the token ids,
not activations), which is negligible relative to the sweep itself.

## Worked example: `configs/smoll_jac.toml`

```
d_model    = 960
seq        = 8
vocab_size = 49152
sublayers  = 2   (add_188->add_192, add_192->add_194)
```

Full vocab sweep, `n_samples = 64` (the CLI default):

```
n_rows            = 64 * 8 * 49152        = 25,165,824 rows
bytes_per_row     = 4*960 + 24            = 3,864 bytes
bytes_per_sublayer= 25,165,824 * 3,864    ≈ 97.2 GB
total (2 sublayers)                       ≈ 194.3 GB
```

With `--n-vocab-sample 20000` instead of the full 49152-token vocab:

```
n_rows            = 64 * 8 * 20000        = 10,240,000 rows
bytes_per_sublayer= 10,240,000 * 3,864    ≈ 39.6 GB
total (2 sublayers)                       ≈ 79.1 GB
```

## Quick formula to plug in your own numbers

```
total_GB = n_samples * seq * n_vocab * (4*d_model + 24) * n_sublayers / 1e9
```

Scale `n_samples` and `--n-vocab-sample` linearly against this to hit a
target size -- e.g. halving either one halves total output size.

## Shard count (for reference, not size)

`--shard-rows` (default 2,000,000) only controls how output is chunked into
per-shard h5 files, not the total bytes written. At the full-vocab example
above, `25,165,824 / 2,000,000 ≈ 13` shards per sublayer.
