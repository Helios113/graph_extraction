"""Fused counterpart of scripts.python.generate_tangent_bases +
scripts.python.generate_tangent_sweep: for each sublayer (src -> dst),
samples points from the local tangent ball at randomly-anchored real points
of src's activation cloud and runs them through the real ONNX sublayer,
WITHOUT ever materializing or saving a dense per-vocab-point tangent basis.

Why this exists: generate_tangent_bases.py computes and saves one
(basis, center) row per REAL point in the cloud (all ~n_vocab points per
position), so generate_tangent_sweep.py can later draw random anchors from
that saved dense basis for any --n/--radius/sublayer combo. At gpt2 scale
(d_model=768, n_vocab=50257) with --basis-method dual --knn-k 160, that
dense basis is ~92GB PER (layer, sample) (n_vocab*seq x knn_k x d_model
float32) -- a 6-layer/64-sample batch (scripts.python.run_layer_batched_tangent_bases's
own default demo) would be ~35TB of basis output, which is what caused a
real ruapehu OOM (page-cache/write-back pressure from writing tens of GB
per layer inside a 64GB memory cgroup -- confirmed via /usr/bin/time -v
showing per-process RSS is actually only ~2.5GB; this was never a Python
memory leak, purely an output-size problem).

This script avoids that entirely by flipping the order of operations:
  1. Per (sample, sublayer, position), draw --n anchor indices FIRST
     (uniformly, with replacement, same semantics as
     generate_tangent_sweep.tangent_sample's anchor draw) -- before any
     k-NN or eigh runs.
  2. k-NN only the drawn anchors (as QUERIES) against the FULL cloud (as
     REFERENCE) -- see knn_query_gpu below -- instead of knn-ing every
     point in the cloud against itself. Cost scales with --n (e.g. 1000),
     not n_vocab (e.g. 50257).
  3. Basis (dual or full, same batch_dual_tangent_basis/batch_full_tangent_basis
     as generate_tangent_bases.py) only for those --n anchors' neighbor sets.
  4. Immediately sample one tangent-ball point per anchor
     (sample_ball_in_tangent_basis_batch_gpu, same as
     generate_tangent_sweep.py) and discard the basis -- it's never written
     to disk.
  5. Splice each sampled point into src's real per-position context
     activations and run the real {src}.{dst} ONNX subgraph, exactly as
     generate_tangent_sweep.run_sublayer does.

Output schema/layer_key convention is IDENTICAL to
scripts.python.generate_tangent_sweep's ({src}.{dst}.radius{r}.sample{idx},
src_vec/dst_vec/pos/anchor/multiplicity) -- downstream consumers of tangent
-sweep output need no changes. There is no separate tangent-bases output at
all: this script needs no --tangent-bases-dir and produces none of
generate_tangent_bases.py's {layer}.sample{idx}.h5 files.

Trade-off vs. the two-script pipeline: generate_tangent_bases.py's saved
dense basis can be reused across many downstream generate_tangent_sweep.py
runs (different --n/--radius/sublayer combos) without recomputing k-NN/eigh.
This fused script recomputes k-NN/eigh from scratch every time it runs
(once per requested --radius, since each radius's anchor draw is
independent -- see run_sublayer_direct), which is fine because the k-NN is
only ever over --n points, not the full cloud, so the recompute is cheap
relative to what generate_tangent_bases.py was paying before. Use this
script when you don't need the dense basis for anything else; use the
original two-script pipeline (or --basis-method full for a config where
n_vocab is small enough for the dense basis to be affordable) if you do.
"""

import argparse
import os
import time

import numpy as np
import torch

from autoencoder.data import load_sample_cloud, sweep_dir_for
from config import ModelConfig, load_config
from execute.sharded_writer import Backend, ShardedLayerWriter, discover_layer_names
from scripts.python.generate_tangent_manifold_distances import (
    batch_dual_tangent_basis,
    batch_full_tangent_basis,
    sample_ball_in_tangent_basis_batch_gpu,
)
from scripts.python.generate_tangent_sweep import (
    context_activations,
    load_base_seq,
    make_subgraph_map,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def knn_query_gpu(query: torch.Tensor, ref: torch.Tensor, k: int, ref_block: int = 16384) -> torch.Tensor:
    """k nearest neighbors (by Euclidean distance) of each row in `query`
    (Q, D) against the full `ref` (N, D) cloud -- returns (Q, k) int64
    indices into `ref`, on `query`'s device.

    Unlike generate_data.datagen.knn_gpu (which knns a cloud against
    itself, cost scaling with N^2/N*k), this only computes distances from
    the (typically small, e.g. --n=1000) query set to the full reference
    cloud, cost O(Q*N) -- the whole point of sampling anchors BEFORE
    running k-NN, not after (see this module's docstring). ref_block chunks
    the reference dimension to bound peak memory for the (Q, ref_block)
    distance matrix at large N (e.g. n_vocab=50257), mirroring knn_gpu's
    own ref_block chunking.
    """
    Q = query.shape[0]
    N = ref.shape[0]
    q_sq = (query ** 2).sum(dim=1, keepdim=True)  # (Q, 1)

    best_dist, best_idx = None, None
    for ref_start in range(0, N, ref_block):
        ref_end = min(ref_start + ref_block, N)
        r = ref[ref_start:ref_end]
        r_sq = (r ** 2).sum(dim=1).unsqueeze(0)  # (1, ref_block)
        d2 = q_sq + r_sq - 2.0 * (query @ r.T)
        d2.clamp_min_(0.0)

        chunk_k = min(k, ref_end - ref_start)
        chunk_dist, chunk_local_idx = torch.topk(d2, chunk_k, dim=1, largest=False, sorted=True)
        chunk_global_idx = chunk_local_idx + ref_start

        if best_dist is None:
            best_dist, best_idx = chunk_dist, chunk_global_idx
        else:
            merged_dist = torch.cat([best_dist, chunk_dist], dim=1)
            merged_idx = torch.cat([best_idx, chunk_global_idx], dim=1)
            take_k = min(k, merged_dist.shape[1])
            best_dist, sel = torch.topk(merged_dist, take_k, dim=1, largest=False, sorted=True)
            best_idx = torch.gather(merged_idx, 1, sel)

    return best_idx.to(torch.int64)


def sample_tangent_points(
    flat_cloud: torch.Tensor,
    n: int,
    knn_k: int,
    basis_method: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draws `n` anchor indices (uniform, with replacement) into
    `flat_cloud` (n_vocab, D), computes a tangent basis ONLY for those
    anchors (knn against the full cloud + dual/full PCA -- never for the
    other n_vocab - n points), samples one point per anchor's tangent
    ball, and returns (points (n, D), anchor (n,)) -- discarding the basis
    immediately, mirroring generate_tangent_sweep.tangent_sample's
    signature/semantics but without ever reading a saved dense basis.
    """
    n_points = flat_cloud.shape[0]
    anchor = torch.randint(0, n_points, (n,), generator=generator, device=DEVICE)
    anchor_points = flat_cloud[anchor]  # (n, D)

    neighbor_idx = knn_query_gpu(anchor_points, flat_cloud, k=knn_k)  # (n, knn_k)
    neighbors = flat_cloud[neighbor_idx]  # (n, knn_k, D)

    if basis_method == "dual":
        basis = batch_dual_tangent_basis(neighbors)  # (n, knn_k, D)
    else:
        basis = batch_full_tangent_basis(neighbors)  # (n, D, D)

    return anchor_points, basis, anchor


def run_sublayer_direct(
    cfg: ModelConfig,
    sweep_dir: str,
    src: str,
    dst: str,
    sample_idx: int,
    base_ids: np.ndarray,
    knn_k: int,
    basis_method: str,
    n: int,
    radii: list[float],
    seed: int,
    out_dir: str,
    flush_every: int,
    no_push: bool,
    backend: Backend,
    pusher: object | None,
) -> None:
    seq = cfg.seq
    d_model = cfg.d_model

    print(f"[{src}->{dst}] loading real cloud for {src}", flush=True)
    cloud = load_sample_cloud(sweep_dir, src, sample_idx).to(DEVICE)  # (seq, n_vocab, d_model)
    cloud = cloud.to(torch.float32)

    print(f"[{src}->{dst}] getting real context activations for sample_idx={sample_idx}", flush=True)
    ctx = context_activations(cfg, src, base_ids)  # (seq, d_model), real values at every position

    map_fn = make_subgraph_map(cfg, src, dst)

    for radius in radii:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        layer_key = os.path.join(out_dir, f"{src}.{dst}.radius{radius:g}.sample{sample_idx}")
        writer = ShardedLayerWriter(
            layer_key=layer_key,
            n_rows=seq * n,
            schema={
                "src_vec": ((d_model,), "float32"),
                "dst_vec": ((d_model,), "float32"),
                "pos": ((), "int64"),
                "anchor": ((), "int64"),
                "multiplicity": ((), "int64"),
            },
            shard_rows=max(seq * n, 1),
            flush_every=flush_every,
            no_push=no_push,
            backend=backend,
            attrs=dict(
                sample_idx=sample_idx, n_samples=n, seq=seq, d_model=d_model, radius=radius,
                knn_k=knn_k, basis_method=basis_method,
            ),
            pusher=pusher,
        )

        row = writer.next_row
        for pos in range(seq):
            anchor_points, basis, anchor = sample_tangent_points(
                cloud[pos], n, knn_k, basis_method, generator,
            )  # anchor_points/basis on GPU, anchor: (n,)
            sampled = sample_ball_in_tangent_basis_batch_gpu(
                basis, anchor_points, radius, n_samples=1, generator=generator,
            ).squeeze(1)  # (n, d_model)
            pts = sampled.detach().cpu().numpy().astype(np.float32)
            anchor_np = anchor.cpu().numpy().astype(np.int64)

            spliced = np.broadcast_to(ctx, (n, seq, d_model)).copy()  # (n, seq, d_model)
            spliced[:, pos] = pts
            out = map_fn(spliced.astype(np.float32))  # (n, seq, d_model)
            dst_vecs = np.ascontiguousarray(out[:, pos])  # (n, d_model)

            writer.write_rows(
                row, row + n,
                src_vec=pts, dst_vec=dst_vecs,
                pos=np.full(n, pos, dtype=np.int64),
                anchor=anchor_np, multiplicity=np.ones(n, dtype=np.int64),
            )
            row += n
            print(f"[{src}->{dst}] radius={radius:g} pos={pos}: sampled {n} tangent-space points (basis computed for {n} anchors only, not the full cloud)", flush=True)

        writer.close()
        print(f"[{src}->{dst}] radius={radius:g} done -> {layer_key}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=str, help="Path to a toml ModelConfig")
    p.add_argument("--sample-idx", type=int, nargs="+", default=[0],
                   help="one or more base sequences' real activations to splice into")
    p.add_argument("--samples-per-chunk", type=int, default=1,
                   help="how many samples to process per progress-reporting group in one process -- purely "
                        "an overhead/organization knob, same as generate_tangent_sweep.py's flag")
    p.add_argument("--layers", nargs="*", default=None, help="default: every layer in sweep_dir")
    p.add_argument("--sublayers", nargs="*", default=None,
                   help="only run these src.dst sublayers (default: every cfg.sublayers entry whose input is in --layers)")
    p.add_argument("--knn-k", type=int, default=None, help="default: cfg.d_model + 1 (full-rank basis)")
    p.add_argument("--basis-method", choices=["full", "dual"], default="dual",
                   help="'dual' (default here, unlike generate_tangent_bases.py) since this script's whole "
                        "point is avoiding the dense full-rank basis's cost -- 'full' still requires "
                        "--knn-k >= cfg.d_model. See batch_dual_tangent_basis/batch_full_tangent_basis "
                        "docstrings in generate_tangent_manifold_distances.py.")
    p.add_argument("--n", type=int, default=1000, help="samples per (sublayer, position) -- also the number "
                        "of anchors a tangent basis is ever computed for (never the full cloud)")
    p.add_argument("--radius", type=float, nargs="+", default=[0.1],
                   help="one or more radii of the tangent-space ball to sample from -- each gets its own "
                        "output layer_key and its own independent anchor draw/k-NN/basis computation (unlike "
                        "generate_tangent_sweep.py, which can reuse one saved dense basis across radii -- "
                        "this script has no saved basis to reuse, so recomputes per radius; cheap since the "
                        "k-NN is only ever over --n points)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--flush-every", type=int, default=10)
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--upload-workers", type=int, default=4)
    p.add_argument("--backend", choices=["nfs"], default="nfs",
                   help="transport ShardedLayerWriter pushes each finished sublayer's shards through -- "
                        "NFS staging via execute.node_relay only (Hetzner is banned for new scripts, see "
                        "CLAUDE.md).")
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

    sublayers = [(s.input, s.output) for s in cfg.sublayers if s.input in layers]
    if args.sublayers is not None:
        wanted = set(args.sublayers)
        sublayers = [(src, dst) for src, dst in sublayers if f"{src}.{dst}" in wanted]
    print(f"{len(sublayers)} sublayer(s) to run: {sublayers}", flush=True)

    from execute.node_relay import AsyncPusher as NfsAsyncPusher
    pusher = None if args.no_push else NfsAsyncPusher(max_workers=args.upload_workers)

    groups = [
        args.sample_idx[i:i + args.samples_per_chunk]
        for i in range(0, len(args.sample_idx), args.samples_per_chunk)
    ]
    for group in groups:
        print(f"\n=== samples {group} ===", flush=True)
        for sample_idx in group:
            base_ids = load_base_seq(sweep_dir, sample_idx)
            for src, dst in sublayers:
                t0 = time.time()
                run_sublayer_direct(
                    cfg, sweep_dir, src, dst, sample_idx, base_ids,
                    knn_k=knn_k, basis_method=args.basis_method,
                    n=args.n, radii=args.radius, seed=args.seed,
                    out_dir=args.out_dir, flush_every=args.flush_every,
                    no_push=args.no_push, backend=args.backend, pusher=pusher,
                )
                print(f"[sample {sample_idx}] [{src}->{dst}] took {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
