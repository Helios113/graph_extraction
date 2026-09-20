import argparse
import os

import numpy as np
import onnxruntime as ort
import torch
import tqdm

from config import load_config
from construct.construct import ensure_base_model, ensure_subgraphs
from execute.io_onnx import run_iobinding_gpu
from execute.sharded_writer import ShardedLayerWriter, load_activations, load_shard_rows
from generate_data.datagen import knn_gpu

DEVICE = torch.device("cuda")


def batch_full_tangent_basis(neighbors: torch.Tensor) -> torch.Tensor:
    """Batched equivalent of estimate_local_tangent_space_p_var(..., p=1) --
    i.e. the full-rank (p=1) case only, since that's all this script uses.

    neighbors: (B, K, D), with K >= D (as guaranteed by knn_k = d_model + 1),
    so the full-rank row-space basis spans all D dimensions. Returns (B, D, D)
    orthonormal row-bases equivalent to Vh from
    estimate_local_tangent_space_p_var(..., p=1)'s torch.linalg.svd.

    Uses eigh on the (B, D, D) covariance/scatter matrix instead of svd on
    the (B, K, D) neighbor matrix directly: torch.linalg.svd's batched GPU
    path does not parallelize across the batch (empirically ~58ms/row
    regardless of batch size for D=960, K=961), while batched eigh on the
    much-smaller D x D matrix does (~1.5ms/row at batch>=100) -- a ~35x
    speedup for the tangent-basis step, which otherwise dominates this
    script's runtime. Eigenvectors come back in ascending eigenvalue order;
    flipped here to match svd's descending-singular-value row order (not
    load-bearing for p=1's full-rank basis, kept for consistency).
    """
    centered = neighbors - neighbors.mean(dim=1, keepdim=True)
    cov = torch.matmul(centered.transpose(-1, -2), centered)  # (B, D, D)
    _, evecs = torch.linalg.eigh(cov)
    return evecs.transpose(-1, -2).flip(dims=[1])  # (B, D, D)


def sample_ball_in_tangent_basis_batch_gpu(
    basis: torch.Tensor,
    center: torch.Tensor,
    radius: float,
    n_samples: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Batched version of sample_ball_in_tangent_basis_gpu.

    basis: (B, k, D). center: (B, D). Returns (B, n_samples, D).
    """
    B, k, D = basis.shape
    device = basis.device

    g = torch.randn((B, n_samples, k), device=device, dtype=torch.float32, generator=generator)
    directions = g / g.norm(dim=-1, keepdim=True)
    u = torch.rand((B, n_samples, 1), device=device, dtype=torch.float32, generator=generator)
    radii = radius * u.pow(1.0 / k)
    local_points = directions * radii  # (B, n_samples, k)

    return torch.bmm(local_points, basis) + center.unsqueeze(1)


def run_sublayer_batch(
    sess: ort.InferenceSession,
    input_name: str,
    output_name: str,
    seqs: torch.Tensor,
    pos: np.ndarray,
    batch: int,
) -> torch.Tensor:
    """Run (n, seq, d_model) bf16 sequences through the sublayer's forward
    subgraph in chunks of `batch` rows (its fixed input batch dim), and
    return just the d_model vector at each row's swept position, stacked as
    (n, d_model) float32.

    Padding rows (to fill out the last chunk to `batch`) repeat the last row
    and are dropped from the result.

    `n` is expected to span many sweep rows' samples at once (a whole block,
    not a single row) so each ORT call does substantial work -- ORT session
    call overhead is the dominant cost otherwise.
    """
    n = seqs.shape[0]
    pad = (-n) % batch
    if pad:
        seqs = torch.cat([seqs, seqs[-1:].repeat(pad, 1, 1)], dim=0)
        pos = np.concatenate([pos, np.repeat(pos[-1:], pad)])

    # Keep the row-selection index on-device: indexing a CUDA tensor with a
    # numpy array forces an implicit host->device copy (and a sync) on every
    # chunk, which dominates wall-clock time when there are many chunks.
    pos_t = torch.from_numpy(pos).to(DEVICE, non_blocking=True)
    arange_batch = torch.arange(batch, device=DEVICE)

    out = torch.empty((seqs.shape[0], seqs.shape[2]), dtype=torch.float32, device=DEVICE)
    for start in range(0, seqs.shape[0], batch):
        chunk = seqs[start : start + batch]
        chunk_pos = pos_t[start : start + batch]
        fx = run_iobinding_gpu(sess, input_name, chunk, output_name)
        out[start : start + batch] = fx[arange_batch[: chunk.shape[0]], chunk_pos].to(torch.float32)

    return out[:n]


def run_layer(
    cfg,
    sweep_out_dir: str,
    base_seqs: np.ndarray,
    base_activations: dict[str, np.ndarray],
    sublayer_input: str,
    sublayer_output: str,
    subgraph_path: str,
    layer_key: str,
    n_samples: int,
    radius: float,
    knn_k: int,
    query_batch: int,
    seed: int,
    rows_per_block: int,
    flush_every: int,
    shard_rows: int,
    no_push: bool,
    project_root: str,
) -> None:
    print("in run layer", flush=True)
    print("reading sweep shards", flush=True)
    sweep_data = load_shard_rows(
        project_root, os.path.join(sweep_out_dir, sublayer_input),
        dataset_names=["vec", "sample_idx", "pos", "vocab_id"],
    )
    pinned = torch.from_numpy(sweep_data["vec"]).pin_memory()
    print("moving to device", flush=True)
    flat = pinned.to(DEVICE, non_blocking=True)
    row_sample_idx = sweep_data["sample_idx"]
    row_pos = sweep_data["pos"]
    row_vocab_id = sweep_data["vocab_id"]
    d_model = flat.shape[1]
    n_rows = flat.shape[0]

    print("doing knn", flush=True)
    kns = knn_gpu(flat, k=knn_k, query_block=query_batch)
    print("knn done")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.intra_op_num_threads = 8
    sub_sess = ort.InferenceSession(
        str(subgraph_path),
        sess_options=sess_options,
        providers=["CUDAExecutionProvider"],
    )

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    writer = ShardedLayerWriter(
        layer_key=layer_key,
        n_rows=n_rows,
        schema={
            "input_distance": ((n_samples,), "float32"),
            "output_distance": ((n_samples,), "float32"),
            "k": ((), "int64"),
            "sample_idx": ((), "int64"),
            "pos": ((), "int64"),
            "vocab_id": ((), "int64"),
        },
        shard_rows=shard_rows,
        flush_every=flush_every,
        attrs={
            "n_samples": n_samples, "radius": radius, "knn_k": knn_k, "d_model": d_model,
            "input": sublayer_input, "output": sublayer_output,
        },
        no_push=no_push,
        project_root=project_root,
    )

    base_input_activation = base_activations[sublayer_input]  # (n_base, seq, d_model) bf16
    seq_len, d_model_check = base_input_activation.shape[1], base_input_activation.shape[2]
    assert d_model_check == d_model

    # Process rows in blocks so the SVD and the ONNX forward pass are each
    # issued as a handful of large batched calls instead of n_rows tiny ones
    # -- per-call (kernel launch / ORT session) overhead otherwise dominates
    # wall-clock time at n_rows ~ 800k. rows_per_block is independent of the
    # node's fixed ORT input batch dim (see cfg.batch_for_node):
    # run_sublayer_batch internally chunks the block's
    # (rows_per_block * (n_samples+1)) sequences into batch-sized ORT calls.
    rows_per_block = max(1, rows_per_block)
    batch = cfg.batch_for_node()

    for block_i, block_start in enumerate(tqdm.tqdm(
        range(writer.next_row, n_rows, rows_per_block), desc=f"{sublayer_input}.{sublayer_output}"
    )):
        block_end = min(block_start + rows_per_block, n_rows)
        idx = np.arange(block_start, block_end)  # host-side: used only for h5/numpy slicing
        b = idx.shape[0]

        # kns[1] (neighbor indices) lives in pinned CPU memory -- knn_gpu
        # keeps the full (n_rows, knn_k) index matrix off-GPU since at
        # n_rows in the millions it wouldn't fit alongside `flat` and the
        # rest of this loop's working set. Move just this block's slice over.
        neighbor_idx = kns[1][block_start:block_end].to(DEVICE, non_blocking=True)
        neighbors = flat[neighbor_idx]  # (b, knn_k, d_model)
        basis = batch_full_tangent_basis(neighbors)  # (b, k, d_model)
        k = basis.shape[1]
        centers = flat[block_start:block_end]  # (b, d_model)

        sampled = sample_ball_in_tangent_basis_batch_gpu(
            basis, centers, radius, n_samples, generator=generator
        )  # (b, n_samples, d_model)

        block_sample_idx = row_sample_idx[idx]
        block_pos = row_pos[idx]

        # Full-sequence base activations for this block, one per row, with
        # the swept position overwritten by each row's samples / center in
        # turn. Laid out as (b, n_samples + 1, seq, d_model) then flattened
        # to (b * (n_samples + 1), seq, d_model) for the ORT call.
        base_seq = torch.from_numpy(
            base_input_activation[block_sample_idx].astype(np.float32)
        ).to(DEVICE)  # (b, seq, d_model)
        seqs = base_seq.unsqueeze(1).repeat(1, n_samples + 1, 1, 1).to(torch.bfloat16)
        row_ar = np.arange(b)
        points = torch.cat([sampled, centers.unsqueeze(1)], dim=1)  # (b, n_samples + 1, d_model)
        seqs[row_ar[:, None], np.arange(n_samples + 1)[None, :], block_pos[:, None], :] = (
            points.to(torch.bfloat16)
        )
        pos_arr = np.repeat(block_pos, n_samples + 1)

        out_vecs = run_sublayer_batch(
            sub_sess,
            sublayer_input,
            sublayer_output,
            seqs.reshape(b * (n_samples + 1), seq_len, d_model),
            pos_arr,
            batch,
        ).reshape(b, n_samples + 1, d_model)
        sampled_out, center_out = out_vecs[:, :-1], out_vecs[:, -1]

        input_dist = (sampled - centers.unsqueeze(1)).norm(dim=-1)  # (b, n_samples)
        output_dist = (sampled_out - center_out.unsqueeze(1)).norm(dim=-1)  # (b, n_samples)

        writer.write_rows(
            block_start,
            block_end,
            input_distance=input_dist.detach().cpu().numpy(),
            output_distance=output_dist.detach().cpu().numpy(),
            k=np.full(b, k, dtype=np.int64),
            sample_idx=block_sample_idx,
            pos=block_pos,
            vocab_id=row_vocab_id[idx],
        )

    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str, help="Path to a toml ModelConfig")
    parser.add_argument(
        "--perturbed-sweep-dir",
        type=str,
        default="onnx_models/smoll/perturbed_sweep",
        help="Output directory produced by scripts/python/generate_perturbed_sweep.py "
        "(one sharded output per layer)",
    )
    parser.add_argument(
        "--base-activations-dir",
        type=str,
        default="onnx_models/smoll/base_activations",
        help="Output directory produced by scripts/python/generate_base_activations.py",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="onnx_models/smoll/tangent_manifold_distances",
        help="Output directory -- one sharded output per sublayer (under "
        "{input}.{output}/) is written here",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of points sampled from the tangent ball per sweep row",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.1,
        help="Radius of the tangent-space ball to sample from",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=None,
        help="Number of neighbors used to estimate each row's tangent basis "
        "(defaults to cfg.d_model + 1)",
    )
    parser.add_argument(
        "--query-batch",
        type=int,
        default=4096,
        help="knn_gpu query block size",
    )
    parser.add_argument(
        "--rows-per-block",
        type=int,
        default=500,
        help="Number of sweep rows processed together per batched SVD / ONNX "
        "forward call. Higher amortizes per-call overhead better but uses "
        "more GPU memory.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=20,
        help="Flush the current shard's h5 file to local scratch disk every this many blocks",
    )
    parser.add_argument(
        "--shard-rows",
        type=int,
        default=200_000,
        help="Rows per shard -- each shard is written to local scratch disk, "
        "then pushed to the remote store and deleted locally as soon as it's full",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for tangent-ball sampling")
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Keep finished shards on local scratch disk instead of pushing to the remote store",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    knn_k = args.knn_k if args.knn_k is not None else cfg.d_model + 1

    ensure_base_model(cfg, force=False, verbose=False)
    sub_graphs = ensure_subgraphs(cfg, force=False)

    base_seqs, base_activations = load_activations(args.base_activations_dir)
    print(f"base_seqs: {base_seqs.shape}, activations: {sorted(base_activations)}")

    for subgraph_path, sublayer_conf in tqdm.tqdm(sub_graphs, desc="sublayers"):
        layer_name = f"{sublayer_conf.input}.{sublayer_conf.output}"
        layer_key = os.path.join(args.out_dir, layer_name)

        run_layer(
            cfg,
            args.perturbed_sweep_dir,
            base_seqs,
            base_activations,
            sublayer_conf.input,
            sublayer_conf.output,
            subgraph_path,
            layer_key,
            n_samples=args.n_samples,
            radius=args.radius,
            knn_k=knn_k,
            query_batch=args.query_batch,
            seed=args.seed,
            rows_per_block=args.rows_per_block,
            flush_every=args.flush_every,
            shard_rows=args.shard_rows,
            no_push=args.no_push,
            project_root=".",
        )

        print(f"{layer_name}: saved tangent manifold distances under {layer_key}")


if __name__ == "__main__":
    main()
