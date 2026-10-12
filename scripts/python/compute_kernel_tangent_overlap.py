"""How much of each sublayer Jacobian's (near-)kernel lies in the local
tangent space of the activation manifold, at every real perturbed point.

For a sublayer `{in}.{out}` and a perturbed-sweep row (sample_idx, pos,
vocab_id), J = d out[pos] / d in[pos] is the d_model x d_model diagonal
Jacobian block (same block, same ONNX pullback as
compute_perturbed_jacobian_det.py). Its right singular vectors v_i live in
`in`-space and its left singular vectors u_i in `out`-space; the smallest
ones (v_min, u_min) are the kernel / cokernel directions (J is never exactly
singular at a real point, so "kernel" means the sigma_min direction).

T_pM at the point is estimated exactly as generate_tangent_bases.py does it:
k-NN (k = d_model + 1) over that sample's flattened (seq * n_vocab, d_model)
real cloud of the layer, then PCA of the neighbours. The PCA is recomputed
here (cheap at one sample's ~32k points) instead of read from tangent_bases/,
because those files don't keep the PCA eigenvalues, which are needed to
choose a tangent dimension. For `in`-space this uses the `in` layer's cloud,
for `out`-space the `out` layer's cloud (the row's out[pos] is exactly the
`out` cloud's point for the same row -- checked against the sweep's stored
vec, see --verify-atol/--verify-rtol).

Saved per row (one ShardedLayerWriter per sublayer at --out-dir/{in}.{out}):
  row_idx, sample_idx, pos, vocab_id   the perturbed-sweep row
  sigma      (D,)          singular values of J, ascending
  in_evals   (D,)          local PCA eigenvalues of the `in` cloud, descending
  out_evals  (D,)          same for the `out` cloud
  v_coef2    (n_vec, D)    squared coefficients of right singular vectors in
                           the `in` PCA basis (column j = j-th principal dir)
  u_coef2    (n_vec, D)    same for left singular vectors, `out` PCA basis
The n_vec singular vectors are the --n-bottom smallest (ascending: index 0 =
v_min) followed by the largest one (v_max, as a contrast). Each row of
*_coef2 sums to 1, and cumsum(v_coef2[0])[d-1] = ||P_{T_d} v_min||^2 = cos^2 of
the angle between the kernel line and the top-d tangent space, for every d
at once -- choosing d (e.g. by explained variance of in_evals) is left to the
plot script, scripts/analysis/plot_kernel_tangent_overlap.py.

The input_ids -> embedding sublayer is skipped (token-id input, no Jacobian).

Usage (mauao, where simple_stories' perturbed shards live):
    uv run python -m scripts.python.compute_kernel_tangent_overlap configs/simple_stories.toml \\
        --samples 0 --out-dir onnx_models/simple_stories/kernel_tangent_overlap_samples_0
"""
import argparse
import os

import h5py
import numpy as np
import torch
import tqdm

from autoencoder.data import _load_row_range, load_sample_cloud
from config import load_config
from construct.construct import ensure_base_model, ensure_extra_outputs, ensure_subgraphs, exported_batch_size
from execute.io_onnx import run_iobinding_cpu_input_gpu_output
from execute.sharded_writer import ShardedLayerWriter
from generate_data.datagen import knn_gpu
from scripts.python.compute_perturbed_jacobian_det import DEVICE, _session, jacobian_chunk, open_pullback


def local_pca(flat: torch.Tensor, neighbor_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """PCA of each point's k-NN neighbourhood, as batch_full_tangent_basis
    (generate_tangent_manifold_distances.py) but keeping the eigenvalues.

    flat: (N, D) GPU cloud; neighbor_idx: (b, k) GPU long rows of it.
    Returns (evals (b, D) descending, evecs (b, D, D) with column j = j-th
    principal direction), float64."""
    nb = flat[neighbor_idx].to(torch.float64)  # (b, k, D)
    centered = nb - nb.mean(dim=1, keepdim=True)
    cov = centered.transpose(-1, -2) @ centered / (nb.shape[1] - 1)
    evals, evecs = torch.linalg.eigh(cov)
    return evals.flip(-1).clamp_min(0), evecs.flip(-1)


def singular_dirs(jac: torch.Tensor, n_bottom: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """sigma (ascending), and the n_bottom smallest + the largest right (v)
    and left (u) singular vectors of each (D, D) block, via float64 eigh of
    J^T J and J J^T (batched GPU svd doesn't parallelise over the batch).
    Returns (sigma (n, D), v (n, D, n_vec), u (n, D, n_vec))."""
    j = jac.to(torch.float64)
    jt = j.transpose(-1, -2)
    ev_v, vec_v = torch.linalg.eigh(jt @ j)  # ascending
    _, vec_u = torch.linalg.eigh(j @ jt)
    pick = list(range(n_bottom)) + [j.shape[-1] - 1]
    return ev_v.clamp_min(0).sqrt(), vec_v[..., pick], vec_u[..., pick]


def sample_cloud_knn(sweep_dir: str, layer: str, sample_idx: int, knn_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(flat (seq * n_vocab, D) GPU float32 cloud, its (N, knn_k) CPU int32
    k-NN indices) -- the same neighbourhoods generate_tangent_bases.py uses."""
    cloud = load_sample_cloud(sweep_dir, layer, sample_idx)
    flat = cloud.reshape(-1, cloud.shape[-1]).to(DEVICE, torch.float32)
    _, idx = knn_gpu(flat, k=knn_k)
    return flat, idx


def run_sublayer(
    cfg, fwd_path, sublayer, base_seqs, extra, sweep_dir, out_dir, samples, knn_k, n_bottom,
    rows_per_block, points_per_call, flush_every, no_push, verify_atol, verify_rtol,
) -> None:
    extra_sess, extra_io, extra_input_name, seq_batch = extra
    layer_name = f"{sublayer.input}.{sublayer.output}"
    _, seq, d_model = sublayer.input_shape
    fixed_batch = exported_batch_size(fwd_path)
    ppc = points_per_call or fixed_batch // d_model
    if ppc * d_model > fixed_batch:
        raise ValueError(f"--points-per-call {ppc} * d_model {d_model} exceeds the fixed batch {fixed_batch}")
    rows_per_block = min(rows_per_block, seq_batch)
    knn_k = knn_k or d_model + 1
    n_vec = n_bottom + 1

    in_key = os.path.join(sweep_dir, sublayer.input)
    n_per_sample = len(load_sample_cloud(sweep_dir, sublayer.input, samples[0]).reshape(-1, d_model))

    writer = ShardedLayerWriter(
        layer_key=os.path.join(out_dir, layer_name),
        n_rows=n_per_sample * len(samples),
        schema={
            "row_idx": ((), "int64"),
            "sample_idx": ((), "int64"),
            "pos": ((), "int64"),
            "vocab_id": ((), "int64"),
            "sigma": ((d_model,), "float32"),
            "in_evals": ((d_model,), "float32"),
            "out_evals": ((d_model,), "float32"),
            "v_coef2": ((n_vec, d_model), "float32"),
            "u_coef2": ((n_vec, d_model), "float32"),
        },
        shard_rows=n_per_sample * len(samples),
        flush_every=flush_every,
        no_push=no_push,
        project_root=".",
        attrs=dict(
            seq=seq, d_model=d_model, sublayer_input=sublayer.input, sublayer_output=sublayer.output,
            perturbed_sweep_dir=sweep_dir, samples=list(samples), knn_k=knn_k, n_bottom=n_bottom,
        ),
    )
    if writer.next_row >= writer.n_rows:
        print(f"{layer_name}: already complete", flush=True)
        return

    pb_sess, pb_io, inp_buf, mask_buf = open_pullback(fwd_path, sublayer)
    max_err = 0.0
    max_tol_frac = 0.0  # max |recomputed - stored| / (atol + rtol max|stored row|); > 1 fails

    def check(recomputed, stored):
        nonlocal max_err, max_tol_frac
        diff = (recomputed - stored).abs()
        max_err = max(max_err, diff.max().item())
        # rounding error in a coordinate scales with the whole vector, not that coordinate
        scale = stored.abs().amax(dim=-1, keepdim=True)
        max_tol_frac = max(max_tol_frac, (diff / (verify_atol + verify_rtol * scale)).max().item())

    for si, s in enumerate(samples):
        base = si * n_per_sample
        if writer.next_row >= base + n_per_sample:
            continue
        print(f"{layer_name}: sample {s}: knn (k={knn_k}) over both clouds", flush=True)
        in_flat, in_nn = sample_cloud_knn(sweep_dir, sublayer.input, s, knn_k)
        out_flat, out_nn = sample_cloud_knn(sweep_dir, sublayer.output, s, knn_k)
        g0 = s * n_per_sample  # global sweep row of this sample's first point
        meta = {
            k: _load_row_range(in_key, g0, g0 + n_per_sample, k) for k in ("sample_idx", "pos", "vocab_id")
        }
        assert (meta["sample_idx"] == s).all(), f"{layer_name}: sweep rows for sample {s} aren't contiguous"

        for start in tqdm.trange(max(0, writer.next_row - base), n_per_sample, rows_per_block,
                                 desc=f"{layer_name} s{s}"):
            end = min(start + rows_per_block, n_per_sample)
            n = end - start
            pos = meta["pos"][start:end]
            token_ids = np.zeros((seq_batch, seq), dtype=np.int64)
            token_ids[:n] = base_seqs[s]
            token_ids[np.arange(n), pos] = meta["vocab_id"][start:end]
            (in_act,) = run_iobinding_cpu_input_gpu_output(
                extra_sess, extra_input_name, token_ids, [sublayer.input], io_binding=extra_io,
            )
            in_act = in_act[:n].to(torch.float32).clone()
            pos_t = torch.from_numpy(pos).to(DEVICE)

            res = {k: [] for k in ("sigma", "in_evals", "out_evals", "v_coef2", "u_coef2")}
            for a in range(0, n, ppc):
                b = min(a + ppc, n)
                jac, out = jacobian_chunk(
                    pb_sess, pb_io, in_act[a:b], pos_t[a:b], sublayer.input, sublayer.output, inp_buf, mask_buf,
                )
                rows = torch.arange(start + a, start + b, device=DEVICE)
                check(out, out_flat[rows])
                # in_act[pos] must be the `in` cloud point too, or the PCA is of the wrong neighbourhood
                in_at_pos = in_act[a:b][torch.arange(b - a), pos_t[a:b]]
                check(in_at_pos, in_flat[rows])

                sigma, v, u = singular_dirs(jac, n_bottom)
                in_ev, in_vec = local_pca(in_flat, in_nn[start + a:start + b].to(DEVICE).long())
                out_ev, out_vec = local_pca(out_flat, out_nn[start + a:start + b].to(DEVICE).long())
                # (b, n_vec, D): coefficient of singular vector i on principal direction j
                v_c = (in_vec.transpose(-1, -2) @ v).transpose(-1, -2) ** 2
                u_c = (out_vec.transpose(-1, -2) @ u).transpose(-1, -2) ** 2
                for k, t in zip(res, (sigma, in_ev, out_ev, v_c, u_c)):
                    res[k].append(t.float().cpu().numpy())
                del jac
            if max_tol_frac > 1:
                raise RuntimeError(
                    f"{layer_name} sample {s} rows [{start}, {end}): recomputed activations differ from the "
                    f"perturbed sweep's stored vecs by {max_err:.3g},"
                    f" beyond --verify-atol {verify_atol} + --verify-rtol {verify_rtol} * max|stored row|"
                )
            writer.write_rows(
                base + start, base + end,
                row_idx=np.arange(g0 + start, g0 + end, dtype=np.int64),
                sample_idx=meta["sample_idx"][start:end], pos=pos, vocab_id=meta["vocab_id"][start:end],
                **{k: np.concatenate(v) for k, v in res.items()},
            )
        del in_flat, out_flat, in_nn, out_nn
        torch.cuda.empty_cache()

    writer.close()
    print(f"{layer_name}: done, max |recomputed - stored vec| = {max_err:.3g} "
          f"({max_tol_frac:.2f} of tolerance) -> {writer.layer_key}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=str)
    p.add_argument("--perturbed-sweep-dir", default="onnx_models/simple_stories/perturbed_sweep")
    p.add_argument("--out-dir", required=True, help="project-relative ShardedLayerWriter prefix")
    p.add_argument("--sublayers", nargs="+", default=None, help="{in}.{out}; default all but input_ids.embedding")
    p.add_argument("--samples", type=int, nargs="+", default=[0])
    p.add_argument("--knn-k", type=int, default=None, help="default d_model + 1, as generate_tangent_bases.py")
    p.add_argument("--n-bottom", type=int, default=3, help="smallest singular vectors to keep (plus the largest)")
    p.add_argument("--rows-per-block", type=int, default=4096,
                   help="perturbed rows per extra-outputs forward / write (capped at its fixed batch)")
    p.add_argument("--points-per-call", type=int, default=None, help="default fixed_batch // d_model")
    p.add_argument("--flush-every", type=int, default=2)
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--verify-atol", type=float, default=1e-3)
    p.add_argument("--verify-rtol", type=float, default=1e-2,
                   help="relative part of the recomputed-vs-stored check, times each row's max |stored|. The "
                   "perturbed sweep was generated with ORT's default TF32 matmuls (this recompute has TF32 off), "
                   "so the stored vecs carry ~1e-3 relative noise that grows with depth (~4e-3 by add_30); a "
                   "wrong point differs at O(1)")
    args = p.parse_args()

    cfg = load_config(args.config)
    with h5py.File(os.path.join(args.perturbed_sweep_dir, "base_seqs.h5"), "r") as f:
        base_seqs = f["base_seqs"][:]
    ensure_base_model(cfg, force=False, verbose=False)
    extra_path = ensure_extra_outputs(cfg, [s.output for s in cfg.sublayers], force=False)
    extra_sess = _session(extra_path)
    extra = (extra_sess, extra_sess.io_binding(), extra_sess.get_inputs()[0].name, exported_batch_size(extra_path))

    for fwd_path, sublayer in ensure_subgraphs(cfg, force=False):
        name = f"{sublayer.input}.{sublayer.output}"
        if sublayer.input == cfg.input_name or (args.sublayers and name not in args.sublayers):
            continue
        run_sublayer(
            cfg, fwd_path, sublayer, base_seqs, extra, args.perturbed_sweep_dir, args.out_dir, args.samples,
            args.knn_k, args.n_bottom, args.rows_per_block, args.points_per_call, args.flush_every,
            args.no_push, args.verify_atol, args.verify_rtol,
        )


if __name__ == "__main__":
    main()
