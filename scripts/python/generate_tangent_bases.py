"""For each layer in a toml ModelConfig, estimates a local tangent basis at
every point of that layer's real (seq, n_vocab, d_model) activation cloud
(the same cloud autoencoder.data.load_sample_cloud pulls from
scripts.python.generate_perturbed_sweep's output) via k-NN + PCA, and saves
(basis, center) pairs to local scratch -- input for
scripts.python.generate_tangent_sweep, the tangent-space analogue of
scripts.python.generate_ambient_sweep.

Mechanism (per layer, per position `pos`):
  1. Flatten the (n_vocab, d_model) cloud at that position onto GPU.
  2. k-NN (k = d_model + 1 by default, so the row-space basis is full-rank)
     for every point in the cloud.
  3. PCA via eigh on each point's (k, d_model) neighbor scatter matrix,
     batched -- see batch_full_tangent_basis's docstring in
     scripts.python.generate_tangent_manifold_distances (this reuses the same
     routine) for why eigh beats svd here. --basis-method selects primal
     ("full", the default, eigh on the (d_model, d_model) covariance matrix,
     requires knn_k >= d_model) vs. dual ("dual", eigh on the (knn_k, knn_k)
     Gram matrix via batch_dual_tangent_basis -- mathematically equivalent,
     but the only option that works, and cheaper, when knn_k < d_model, e.g.
     --knn-k 160 with a d_model=768+ config). The saved basis then has rank
     knn_k instead of d_model.

--sample-idx takes one or more samples; --samples-per-chunk groups them
purely for progress-reporting/organization (default 1). Each sample's
k-NN/PCA pass is fully independent (a sample-0 point's neighbors never
include sample-1 points -- different token contexts), so grouping never
changes what gets computed.

Output: one h5 file per (sample, layer) -- {layer}.sample{idx}.h5, per the
usual per-sublayer(here: per-layer)/flush-every resilience pattern from
execute.sharded_writer.ShardedLayerWriter. Pushed as soon as that (sample,
layer) finishes, same as the rest of this sweep family -- see CLAUDE.md's
per-sublayer push pattern -- unless --no-push is given, e.g. when running on
the same node scripts.python.generate_tangent_sweep will read it from
(--tangent-bases-dir), where a push is unnecessary round-trip work. Each row
holds the full (d_model, d_model) orthonormal basis and the (d_model,)
center point for one (pos, vocab_idx) point of the cloud.
"""

import argparse
import os

import torch
import tqdm

from autoencoder.data import load_sample_cloud, sweep_dir_for
from config import load_config
from execute.remote_storage import AsyncPusher as HetznerAsyncPusher
from execute.sharded_writer import Backend, ShardedLayerWriter, discover_layer_names
from scripts.python.generate_tangent_manifold_distances import (
    batch_dual_tangent_basis,
    batch_full_tangent_basis,
)
from generate_data.datagen import knn_gpu

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mem(label: str) -> None:
    if DEVICE.type != "cuda":
        return
    alloc = torch.cuda.memory_allocated(DEVICE) / 2**30
    peak = torch.cuda.max_memory_allocated(DEVICE) / 2**30
    reserved = torch.cuda.memory_reserved(DEVICE) / 2**30
    free, total = (x / 2**30 for x in torch.cuda.mem_get_info(DEVICE))
    print(
        f"[mem] {label}: allocated={alloc:.2f}GiB peak={peak:.2f}GiB "
        f"reserved={reserved:.2f}GiB free={free:.2f}GiB total={total:.2f}GiB",
        flush=True,
    )


def run_layer(
    sweep_dir: str,
    layer: str,
    sample_idx: int,
    layer_key: str,
    knn_k: int,
    basis_method: str,
    query_batch: int,
    rows_per_block: int,
    flush_every: int,
    no_push: bool,
    backend: Backend,
    pusher: object | None,
) -> None:
    print(f"[{layer}] loading real cloud", flush=True)
    cloud = load_sample_cloud(sweep_dir, layer, sample_idx).to(DEVICE)  # (seq, n_vocab, d_model)
    seq, n_vocab, d_model = cloud.shape
    flat = cloud.reshape(seq * n_vocab, d_model).to(torch.float32)
    n_rows = flat.shape[0]
    mem(f"[{layer}] after loading cloud ({n_rows} points, d_model={d_model})")

    print(f"[{layer}] knn (k={knn_k}) over {n_rows} points", flush=True)
    _, neighbor_idx = knn_gpu(flat, k=knn_k, query_block=query_batch)
    print(f"[{layer}] knn done", flush=True)
    mem(f"[{layer}] after knn")

    # basis_method="dual" (kernel/dual PCA, eigh on the (K, K) Gram matrix)
    # returns all knn_k tangent directions; "full" (primal PCA, eigh on the
    # (D, D) covariance matrix) requires knn_k >= d_model and returns
    # exactly d_model. See batch_dual_tangent_basis/batch_full_tangent_basis
    # docstrings for why/when each applies.
    basis_rank = knn_k if basis_method == "dual" else d_model

    writer = ShardedLayerWriter(
        layer_key=layer_key,
        n_rows=n_rows,
        schema={
            "basis": ((basis_rank, d_model), "float32"),
            "center": ((d_model,), "float32"),
            "pos": ((), "int64"),
            "vocab_id": ((), "int64"),
        },
        shard_rows=max(n_rows, 1),
        flush_every=flush_every,
        no_push=no_push,
        backend=backend,
        attrs=dict(
            sample_idx=sample_idx, seq=seq, n_vocab=n_vocab, d_model=d_model,
            knn_k=knn_k, basis_method=basis_method,
        ),
        pusher=pusher,
    )

    pos_all = torch.arange(seq, device=DEVICE).repeat_interleave(n_vocab)
    vocab_all = torch.arange(n_vocab, device=DEVICE).repeat(seq)

    n_blocks = -(-((n_rows - writer.next_row)) // rows_per_block)  # ceil div, for the mem-print cadence below
    mem_every = max(1, n_blocks // 10)  # ~10 memory prints per layer, not one per block

    for i, block_start in enumerate(tqdm.tqdm(
        range(writer.next_row, n_rows, rows_per_block), desc=f"{layer} tangent bases"
    )):
        block_end = min(block_start + rows_per_block, n_rows)

        idx = neighbor_idx[block_start:block_end].to(DEVICE, non_blocking=True)
        neighbors = flat[idx]  # (b, knn_k, d_model)
        if basis_method == "dual":
            basis = batch_dual_tangent_basis(neighbors)  # (b, knn_k, d_model)
        else:
            basis = batch_full_tangent_basis(neighbors)  # (b, d_model, d_model)
        centers = flat[block_start:block_end]  # (b, d_model)

        writer.write_rows(
            block_start, block_end,
            basis=basis.detach().cpu().numpy(),
            center=centers.detach().cpu().numpy(),
            pos=pos_all[block_start:block_end].cpu().numpy(),
            vocab_id=vocab_all[block_start:block_end].cpu().numpy(),
        )

        if i % mem_every == 0:
            mem(f"[{layer}] block {block_start}:{block_end} (rows_per_block={rows_per_block})")

    writer.close()
    print(f"[{layer}] done -> {layer_key}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=str, help="Path to a toml ModelConfig")
    p.add_argument("--sample-idx", type=int, nargs="+", default=[0], help="one or more samples to process")
    p.add_argument("--samples-per-chunk", type=int, default=1,
                   help="how many samples to process per progress-reporting group in one process -- purely "
                        "an overhead/organization knob (saves re-invoking the script per sample), since each "
                        "sample's knn/eigh pass is always independent: a sample-0 point's neighbors can never "
                        "legitimately include sample-1 points (different token contexts), so this never "
                        "changes what gets computed, only how the run is grouped/logged")
    p.add_argument("--layers", nargs="*", default=None, help="default: every layer in sweep_dir")
    p.add_argument("--knn-k", type=int, default=None, help="default: cfg.d_model + 1 (full-rank basis)")
    p.add_argument("--basis-method", choices=["full", "dual"], default="full",
                   help="'full' (default): primal PCA via eigh on the (d_model, d_model) covariance "
                        "matrix (batch_full_tangent_basis) -- requires --knn-k >= cfg.d_model + 1 for a "
                        "full-rank basis. 'dual': kernel/dual PCA via eigh on the (knn_k, knn_k) Gram "
                        "matrix (batch_dual_tangent_basis) -- mathematically equivalent eigenvectors, but "
                        "works (and is cheaper) when --knn-k < cfg.d_model, e.g. --knn-k 160; the saved "
                        "basis then has rank knn_k instead of d_model.")
    p.add_argument("--query-batch", type=int, default=4096, help="knn_gpu query block size")
    p.add_argument("--rows-per-block", type=int, default=1500,
                   help="points processed together per batched eigh call -- batch_full_tangent_basis's "
                        "eigh throughput is flat per-row past batch>=100, so a larger block amortizes "
                        "per-block launch/transfer overhead (each block round-trips a (rows_per_block, "
                        "knn_k, d_model) neighbor gather and a (rows_per_block, d_model, d_model) basis "
                        "tensor GPU<->CPU). Raise this until [mem] prints (see mem()) show reserved "
                        "approaching the GPU's total; lower it back down on a smaller-memory node "
                        "(e.g. mauao's A40 vs ruapehu's H100) if it OOMs")
    p.add_argument("--flush-every", type=int, default=20)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--no-push", action="store_true",
                   help="keep finished (sample, layer) shards on local scratch only -- e.g. when "
                        "scripts.python.generate_tangent_sweep will read them from --tangent-bases-dir on "
                        "this same node right after, where a push is unnecessary round-trip work")
    p.add_argument("--upload-workers", type=int, default=4)
    p.add_argument("--backend", choices=["hetzner", "nfs"], default="nfs",
                   help="transport ShardedLayerWriter pushes each finished (sample, layer)'s shards through "
                        "-- 'nfs' (default) via execute.node_relay's NFS-staging relay, 'hetzner' via "
                        "execute.remote_storage (not used by this project's own launchers anymore). "
                        "Ignored if --no-push is also set.")
    args = p.parse_args()

    cfg = load_config(args.config)
    knn_k = args.knn_k if args.knn_k is not None else cfg.d_model + 1
    if args.basis_method == "full" and knn_k < cfg.d_model:
        raise ValueError(
            f"--basis-method full requires --knn-k >= cfg.d_model ({cfg.d_model}) for a full-rank basis, "
            f"got --knn-k {knn_k}; use --basis-method dual for knn_k < d_model"
        )
    sweep_dir = sweep_dir_for(cfg)
    layers = args.layers or discover_layer_names(sweep_dir)
    print(f"layers ({len(layers)}): {layers}", flush=True)

    pusher_cls = HetznerAsyncPusher
    if args.backend == "nfs":
        from execute.node_relay import AsyncPusher as NfsAsyncPusher
        pusher_cls = NfsAsyncPusher
    pusher = None if args.no_push else pusher_cls(max_workers=args.upload_workers)

    groups = [
        args.sample_idx[i:i + args.samples_per_chunk]
        for i in range(0, len(args.sample_idx), args.samples_per_chunk)
    ]
    for group in groups:
        print(f"\n=== samples {group} ===", flush=True)
        for sample_idx in group:
            for layer in layers:
                layer_key = os.path.join(args.out_dir, f"{layer}.sample{sample_idx}")
                run_layer(
                    sweep_dir, layer, sample_idx, layer_key,
                    knn_k=knn_k, basis_method=args.basis_method, query_batch=args.query_batch,
                    rows_per_block=args.rows_per_block, flush_every=args.flush_every,
                    no_push=args.no_push, backend=args.backend, pusher=pusher,
                )


if __name__ == "__main__":
    main()
