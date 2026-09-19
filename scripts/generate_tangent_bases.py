"""For each layer in a toml ModelConfig, estimates a local tangent basis at
every point of that layer's real (seq, n_vocab, d_model) activation cloud
(the same cloud autoencoder.data.load_sample_cloud pulls from
scripts.generate_perturbed_sweep's output) via k-NN + PCA, and saves
(basis, center) pairs to local scratch -- input for
scripts.generate_tangent_sweep, the tangent-space analogue of
scripts.generate_ambient_sweep.

Mechanism (per layer, per position `pos`):
  1. Flatten the (n_vocab, d_model) cloud at that position onto GPU.
  2. k-NN (k = d_model + 1 by default, so the row-space basis is full-rank)
     for every point in the cloud.
  3. PCA via eigh on each point's (k, d_model) neighbor scatter matrix,
     batched -- see batch_full_tangent_basis's docstring in
     scripts.generate_tangent_manifold_distances (this reuses the same
     routine) for why eigh beats svd here.

Output: one h5 file per layer, kept on local scratch only (never pushed --
these are an intermediate artifact for scripts.generate_tangent_sweep to
consume on the same node, not sweep shard data -- see CLAUDE.md), per the
usual per-sublayer(here: per-layer)/flush-every resilience pattern from
execute.sharded_writer.ShardedLayerWriter. Each row holds the full (d_model,
d_model) orthonormal basis and the (d_model,) center point for one
(pos, vocab_idx) point of the cloud.
"""

import argparse
import os

import torch
import tqdm

from autoencoder.data import load_sample_cloud, sweep_dir_for
from config import load_config
from execute.sharded_writer import ShardedLayerWriter, discover_layer_names
from scripts.generate_tangent_manifold_distances import batch_full_tangent_basis
from generate_data.datagen import knn_gpu

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_layer(
    sweep_dir: str,
    layer: str,
    sample_idx: int,
    layer_key: str,
    knn_k: int,
    query_batch: int,
    rows_per_block: int,
    flush_every: int,
) -> None:
    print(f"[{layer}] loading real cloud", flush=True)
    cloud = load_sample_cloud(sweep_dir, layer, sample_idx).to(DEVICE)  # (seq, n_vocab, d_model)
    seq, n_vocab, d_model = cloud.shape
    flat = cloud.reshape(seq * n_vocab, d_model).to(torch.float32)
    n_rows = flat.shape[0]

    print(f"[{layer}] knn (k={knn_k}) over {n_rows} points", flush=True)
    _, neighbor_idx = knn_gpu(flat, k=knn_k, query_block=query_batch)
    print(f"[{layer}] knn done", flush=True)

    writer = ShardedLayerWriter(
        layer_key=layer_key,
        n_rows=n_rows,
        schema={
            "basis": ((d_model, d_model), "float32"),
            "center": ((d_model,), "float32"),
            "pos": ((), "int64"),
            "vocab_id": ((), "int64"),
        },
        shard_rows=max(n_rows, 1),
        flush_every=flush_every,
        no_push=True,
        attrs=dict(sample_idx=sample_idx, seq=seq, n_vocab=n_vocab, d_model=d_model, knn_k=knn_k),
    )

    pos_all = torch.arange(seq, device=DEVICE).repeat_interleave(n_vocab)
    vocab_all = torch.arange(n_vocab, device=DEVICE).repeat(seq)

    for block_start in tqdm.tqdm(
        range(writer.next_row, n_rows, rows_per_block), desc=f"{layer} tangent bases"
    ):
        block_end = min(block_start + rows_per_block, n_rows)

        idx = neighbor_idx[block_start:block_end].to(DEVICE, non_blocking=True)
        neighbors = flat[idx]  # (b, knn_k, d_model)
        basis = batch_full_tangent_basis(neighbors)  # (b, d_model, d_model)
        centers = flat[block_start:block_end]  # (b, d_model)

        writer.write_rows(
            block_start, block_end,
            basis=basis.detach().cpu().numpy(),
            center=centers.detach().cpu().numpy(),
            pos=pos_all[block_start:block_end].cpu().numpy(),
            vocab_id=vocab_all[block_start:block_end].cpu().numpy(),
        )

    writer.close()
    print(f"[{layer}] done -> {layer_key}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=str, help="Path to a toml ModelConfig")
    p.add_argument("--sample-idx", type=int, default=0)
    p.add_argument("--layers", nargs="*", default=None, help="default: every layer in sweep_dir")
    p.add_argument("--knn-k", type=int, default=None, help="default: cfg.d_model + 1 (full-rank basis)")
    p.add_argument("--query-batch", type=int, default=4096, help="knn_gpu query block size")
    p.add_argument("--rows-per-block", type=int, default=500,
                   help="points processed together per batched eigh call")
    p.add_argument("--flush-every", type=int, default=20)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    knn_k = args.knn_k if args.knn_k is not None else cfg.d_model + 1
    sweep_dir = sweep_dir_for(cfg)
    layers = args.layers or discover_layer_names(sweep_dir)
    print(f"layers ({len(layers)}): {layers}", flush=True)

    for layer in layers:
        layer_key = os.path.join(args.out_dir, layer)
        run_layer(
            sweep_dir, layer, args.sample_idx, layer_key,
            knn_k=knn_k, query_batch=args.query_batch,
            rows_per_block=args.rows_per_block, flush_every=args.flush_every,
        )


if __name__ == "__main__":
    main()
