"""GPU-batched full-sweep version of
scripts/analysis/lipschitz_ratio_bootstrap_eqd_sweep.py -- same Algorithm 1
(Bootstrap Expected Quantile Discrepancy threshold selection), same sweep
structure and h5 output schema, but running the pairwise-ratio sampling and
GPD/EQD fitting on GPU instead of CPU/scipy.

Self-contained: the GPU pipeline (compute_lipschitz_ratios_gpu,
bootstrap_eqd_statistic_batched, select_bootstrap_eqd_threshold_gpu_multi,
fit_gpd_and_c_hat_gpu) and the perturbed-sweep shard-reading helpers
(load_layer_manifest, load_row_range) are defined directly in this file
rather than imported from scripts/analysis/bootstrap_eqd_gpu.py /
lipschitz_ratio_eqd_fit.py -- only the underlying batched GPD MLE fitter
(scripts/analysis/batched_gpd_mle.fit_gpd_batched) and core project
infrastructure (config.load_config, execute.sharded_writer._scratch_root)
remain external imports.

Positions are batched across (sample, pos) pairs into groups of
--group-size (default 16, the benchmarked sweet spot at the default
--eqd-n-boot -- see scripts/analysis/bench_batched_gpd_scale.py: time-per-
position on an A40 drops from ~1.4s at group size 1 to ~0.78s at group size
16, with only marginal further improvement out to 64). A group can span
multiple samples (e.g. group_size=16 covers two full samples at
simple_stories' seq_len=8), since every position shares the same n_tail
candidate grid regardless of which sample/position it comes from -- see
select_bootstrap_eqd_threshold_gpu_multi.

Note --group-size and --eqd-n-boot scale GPU cost very differently: grouping
MORE POSITIONS is nearly free (positions share the same candidate grid, so
batching them adds parallel rows without adding real problem difficulty --
group_size 1->16 only doubled memory, ~2.6GB->~7.9GB in testing), but raising
eqd_n_boot (bootstrap resamples per candidate -- see --eqd-n-boot's help
text for why bootstrapping matters at all, not just direct-fit scoring)
scales both time AND memory roughly LINEARLY, since it's a genuinely new
axis of independent random draws. If you raise --eqd-n-boot, lower
--group-size proportionally or you'll hit an out-of-memory error.

Output layout matches lipschitz_ratio_eqd_fit.py /
lipschitz_ratio_bootstrap_eqd_sweep.py exactly (one h5 per (sublayer pair,
sample), one group per position, same dataset/attr names), so
scripts/python/plot_lipschitz_ratio_eqd_fit.py works unmodified against this
script's --out-dir. Because positions are batched ACROSS samples rather than
processed one full sample at a time, a sample's h5 file is only written once
every position in that sample has been fit -- samples are buffered in memory
until complete, then flushed to disk in sample-index order.

Usage (on mauao, via srun -- reads perturbed-sweep shards straight off local
scratch, same as lipschitz_ratio_eqd_fit.py; needs a GPU, --gres=gpu:a40:1
or similar):
    uv run python -m scripts.analysis.lipschitz_ratio_bootstrap_eqd_sweep_gpu \\
        configs/simple_stories.toml --perturbed-sweep-dir onnx_models/simple_stories/perturbed_sweep

sbatch version: scripts/batch_scripts/lipschitz_ratio_bootstrap_eqd_sweep_gpu_mauao_sbatch.sh
"""

import argparse
import functools
import json
import os

import h5py
import numpy as np
import torch
import tqdm

from config import load_config
from execute.sharded_writer import _scratch_root
from scripts.lipschitz.batched_gpd_mle import fit_gpd_batched

# sbatch buffers stdout by default -- flush every print so progress is
# visible in the .out log while the job is still running, not just at exit
# (see scripts/analysis/bench_batched_gpd_scale.py for the same fix and why
# it matters for a multi-hour job).
print = functools.partial(print, flush=True)


# --- inlined from scripts/analysis/lipschitz_ratio_eqd_fit.py, so this
# script only depends on execute.sharded_writer for shard I/O, not on
# another scripts/analysis module -----------------------------------------


def load_layer_manifest(project_root: str, layer_key: str) -> dict:
    manifest_path = os.path.join(project_root, layer_key + ".manifest.json")
    with open(manifest_path) as f:
        return json.load(f)


def local_shard_path(layer_key: str, shard_idx: int) -> str:
    """Path a shard was (or would be) written to on this node's local
    scratch by ShardedLayerWriter, regardless of whether it's since been
    pushed and/or deleted from scratch -- see ShardedLayerWriter._scratch_dir."""
    return os.path.join(_scratch_root(), layer_key, f"shard_{shard_idx:05d}.h5")


def load_row_range(
    layer_key: str, manifest: dict, row_start: int, row_end: int, dataset_names: list[str],
) -> dict[str, np.ndarray]:
    """Reads only rows [row_start, row_end) of `layer_key`'s sharded output
    directly from this node's local scratch, i.e. only the shards that
    overlap that range -- not every shard for the whole layer.

    No NFS/Hetzner pull: this is meant to run via srun on the same node
    (mauao) the sweep itself ran on, so every shard should already be
    sitting on local scratch. Raises with a clear message if a needed
    shard isn't there (e.g. scratch was cleaned up, or this is running on
    the wrong node) instead of silently reaching for a remote transport.

    Relies on the same invariant `load_shard_rows` does (shards cover
    disjoint, contiguous, sorted row ranges) plus generate_perturbed_sweep's
    row layout (rows unravelled from (sample_idx, pos, vocab_id) in that
    order), which is what lets a caller ask for one sample's contiguous row
    range in the first place.
    """
    out: dict[str, list[np.ndarray]] = {name: [] for name in dataset_names}
    for shard in sorted(manifest["shards"], key=lambda s: s["shard_idx"]):
        s_start, s_end = shard["row_start"], shard["row_end"]
        if s_end <= row_start or s_start >= row_end:
            continue
        local_path = local_shard_path(layer_key, shard["shard_idx"])
        if not os.path.exists(local_path):
            raise FileNotFoundError(
                f"{local_path!r} not found on local scratch (node scratch root "
                f"{_scratch_root()!r}) -- this script must run on the same node the "
                f"perturbed sweep ran on (mauao), with a matching $NEW_JAC_SCRATCH, and "
                "does not pull from NFS staging or Hetzner. If the sweep's scratch copy "
                "was cleaned up, relay it back with scripts/helper/sync_node_scratch.sh "
                "or re-run generate_perturbed_sweep.py."
            )
        lo, hi = max(row_start, s_start) - s_start, min(row_end, s_end) - s_start
        with h5py.File(local_path, "r") as h5f:
            for name in dataset_names:
                out[name].append(h5f[name][lo:hi])
    return {name: np.concatenate(chunks, axis=0) for name, chunks in out.items()}


# --- inlined from scripts/analysis/bootstrap_eqd_gpu.py, so this script's
# GPU pipeline is self-contained rather than importing that module -------


def compute_lipschitz_ratios_gpu(
    vec_in: torch.Tensor, vec_out: torch.Tensor, n_pairs: int, generator: torch.Generator,
    return_pairs: bool = False,
) -> tuple[torch.Tensor, ...]:
    """GPU torch equivalent of lipschitz_ratio_eqd_fit.py's
    compute_lipschitz_ratios. vec_in, vec_out: (n_points, d) tensors on the
    target device. Returns (input_dist, output_dist, ratio), each (n_pairs,)
    float64 tensors on the same device, plus the (i, j) point indices of
    each pair if return_pairs."""
    device = vec_in.device
    n_points = vec_in.shape[0]
    max_pairs = n_points * (n_points - 1) // 2
    n_pairs = min(n_pairs, max_pairs)

    i = torch.randint(0, n_points, (n_pairs,), device=device, generator=generator)
    j = torch.randint(0, n_points, (n_pairs,), device=device, generator=generator)
    same = i == j
    while same.any():
        n_same = int(same.sum())
        j[same] = torch.randint(0, n_points, (n_same,), device=device, generator=generator)
        same = i == j

    input_dist = torch.linalg.norm((vec_in[i] - vec_in[j]).double(), dim=-1)
    output_dist = torch.linalg.norm((vec_out[i] - vec_out[j]).double(), dim=-1)
    ratio = output_dist / input_dist
    if return_pairs:
        return input_dist, output_dist, ratio, i, j
    return input_dist, output_dist, ratio


def gpd_model_quantile_batched(p: torch.Tensor, xi: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Batched GPD(xi, sigma, loc=0) quantile function, xi/sigma (...,), p
    (m,) broadcasting to (..., m): (sigma/xi)[(1-p)^(-xi) - 1], xi -> 0 limit
    -sigma*log(1-p)."""
    xi_ = xi.unsqueeze(-1)
    sigma_ = sigma.unsqueeze(-1)
    near_zero = xi_.abs() < 1e-10
    xi_safe = torch.where(near_zero, torch.full_like(xi_, 1e-10), xi_)
    q_general = (sigma_ / xi_safe) * ((1 - p) ** (-xi_safe) - 1)
    q_limit = -sigma_ * torch.log(1 - p)
    return torch.where(near_zero, q_limit, q_general)


def bootstrap_eqd_statistic_batched(
    exceedances: torch.Tensor,
    exceedances_mask: torch.Tensor,
    p_grid: torch.Tensor,
    n_boot: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """d_hat_E(u) for every candidate threshold at once, plus the individual
    per-replicate d_b(u) values it's averaged from. exceedances: (n_candidates,
    max_n_tail) float64, each row's first `n_tail_c` entries (per
    exceedances_mask) are that candidate's exceedances, zero-padded past
    that. exceedances_mask: (n_candidates, max_n_tail) bool.

    Draws n_boot non-parametric bootstrap resamples of EVERY candidate's
    exceedances simultaneously (shape (n_candidates, n_boot, max_n_tail)),
    fits a GPD to every (candidate, replicate) pair in one batched call to
    fit_gpd_batched, computes the mean absolute deviation between each
    fit's model quantiles and its own resample's empirical quantiles, and
    averages over replicates.

    Returns (d_hat_e, d_b):
      d_hat_e: (n_candidates,) -- the averaged statistic used to pick the
        best threshold (inf for a candidate where every replicate's fit
        failed to converge).
      d_b: (n_candidates, n_boot) -- every individual replicate's own d_b(u)
        value before averaging (inf where that specific replicate's fit
        didn't converge), so a caller can inspect the spread behind each
        d_hat_e point rather than only the mean -- e.g. to see how much an
        unstable candidate's selection depended on which bootstrap draws it
        happened to get."""
    n_candidates, max_n_tail = exceedances.shape
    device = exceedances.device
    n_tail_per_candidate = exceedances_mask.sum(dim=-1)  # (n_candidates,)

    # Bootstrap resample indices: for candidate c, replicate b, draw
    # n_tail_per_candidate[c] indices in [0, n_tail_per_candidate[c]) with
    # replacement -- padded entries (index >= n_tail_c) are masked out below
    # rather than avoided at draw time, since n_tail varies per row.
    resample_idx = (
        torch.rand(n_candidates, n_boot, max_n_tail, device=device, generator=generator)
        * n_tail_per_candidate.view(n_candidates, 1, 1)
    ).long().clamp(max=max_n_tail - 1)
    # Expanded (not just broadcast) to (n_candidates, n_boot, max_n_tail) --
    # every replicate of a given candidate shares the same n_tail, but the
    # flat reshape below needs an actual (n_candidates*n_boot, max_n_tail)
    # tensor, which a broadcast (n_candidates, 1, max_n_tail) shape can't
    # produce (observed: reshape size mismatch, 1/n_boot the expected size).
    resample_mask = (
        torch.arange(max_n_tail, device=device).view(1, 1, -1) < n_tail_per_candidate.view(n_candidates, 1, 1)
    ).expand(n_candidates, n_boot, max_n_tail)

    exc_expanded = exceedances.unsqueeze(1).expand(n_candidates, n_boot, max_n_tail)
    resamples = torch.gather(exc_expanded, -1, resample_idx)
    resamples = torch.where(resample_mask, resamples, torch.zeros_like(resamples))

    flat_shape = (n_candidates * n_boot, max_n_tail)
    xi, sigma, converged = fit_gpd_batched(resamples.reshape(flat_shape), resample_mask.reshape(flat_shape))
    xi = xi.reshape(n_candidates, n_boot)
    sigma = sigma.reshape(n_candidates, n_boot)
    converged = converged.reshape(n_candidates, n_boot)

    # Different candidates have different n_tail (ragged row lengths), so
    # the quantile function can't use a plain shared-length quantile helper
    # -- sort with padding pushed to +inf, then index each row's quantile
    # position using ONLY that row's own n_tail_per_candidate, not the
    # shared max_n_tail.
    resamples_sorted, _ = torch.sort(
        torch.where(resample_mask, resamples, torch.full_like(resamples, float("inf"))), dim=-1,
    )
    model_q = gpd_model_quantile_batched(p_grid, xi, sigma)  # (n_candidates, n_boot, m)

    # Per-row quantile using only the first n_tail_c (non-inf) entries.
    n_tail_exp = n_tail_per_candidate.view(n_candidates, 1, 1).expand(-1, n_boot, -1).double()
    pos = p_grid.view(1, 1, -1) * (n_tail_exp - 1)
    lo = pos.floor().long().clamp(min=0)
    hi = pos.ceil().long().clamp(min=0)
    lo = torch.minimum(lo, (n_tail_per_candidate.view(n_candidates, 1, 1) - 1).expand_as(lo))
    hi = torch.minimum(hi, (n_tail_per_candidate.view(n_candidates, 1, 1) - 1).expand_as(hi))
    frac = (pos - lo.double())
    lo_vals = torch.gather(resamples_sorted, -1, lo)
    hi_vals = torch.gather(resamples_sorted, -1, hi)
    sample_q = lo_vals + frac * (hi_vals - lo_vals)

    d_b = torch.abs(model_q - sample_q).mean(dim=-1)  # (n_candidates, n_boot)
    d_b = torch.where(converged, d_b, torch.full_like(d_b, float("inf")))

    n_converged = converged.sum(dim=-1)
    d_hat_e = torch.where(
        n_converged > 0,
        torch.where(converged, d_b, torch.zeros_like(d_b)).sum(dim=-1) / n_converged.clamp(min=1),
        torch.full((n_candidates,), float("inf"), device=device, dtype=d_b.dtype),
    )
    return d_hat_e, d_b


def fit_gpd_and_c_hat_gpu(g_sorted: torch.Tensor, n_tail: int, threshold: float) -> dict:
    """Final exact-fit GPD + c_hat at the selected threshold, single-item
    (not batched) but still on GPU via fit_gpd_batched's batch-of-1 path --
    keeps this on-device rather than round-tripping to scipy, for
    consistency with the rest of the batched pipeline.

    Reports whatever (xi, sigma) fit_gpd_batched's default n_iters=10 budget
    reaches even when its strict `converged` flag (tol=1e-4 gradient norm)
    doesn't trigger, rather than discarding the position as NaN -- per
    batched_gpd_mle.py's own tuning notes, `converged=False` at n_iters=10
    routinely still means "within 0.5-5% of scipy's exact MLE," just short
    of the strict gradient-norm bar. Observed on real data: without this,
    positions selecting a large n_tail (~1900, where convergence is a bit
    slower) came back all-NaN even though the underlying fit was fine --
    losing a position's result entirely is worse than reporting its
    best-effort estimate."""
    exceedances = threshold - g_sorted[:n_tail]
    xi, sigma, _converged = fit_gpd_batched(exceedances.unsqueeze(0))
    xi, sigma = xi.item(), sigma.item()
    c_hat = float(threshold + sigma / xi) if xi < 0 else float("nan")
    return {"shape_xi": float(xi), "scale_sigma": float(sigma), "c_hat": c_hat}


def select_bootstrap_eqd_threshold_gpu_multi(
    g_sorted_list: list[torch.Tensor],
    n_tail_min: int,
    n_tail_max: int,
    n_tail_step: int,
    n_boot: int,
    m_plotting_points: int,
    generator: torch.Generator,
    group_size: int = 16,
) -> list[dict]:
    """Runs Algorithm 1 (bootstrap-EQD threshold selection) over MULTIPLE
    (sample, pos) ratio clouds at once, batching `group_size` positions
    together into each fit_gpd_batched call (this is what the sweep loop
    below calls instead of scoring one position at a time).

    Every position shares the same n_tail candidate grid (n_tail_min/max/step
    are fixed CLI-level parameters, not data-dependent), so their
    exceedances/masks stack cleanly along a new leading axis and one call to
    bootstrap_eqd_statistic_batched can score every position's candidates in
    parallel -- this is what group_size=16 (the benchmarked sweet spot, see
    module docstring) exploits.

    g_sorted_list: list of (n_points,) float64 tensors, each ascending-sorted,
    on the target device -- n_points may differ between positions (a
    position's own g_sorted.shape[0] only affects that position's own
    n_tail_max clamp below).

    Returns a list of per-position result dicts (n_tail_values, d_hat_e,
    d_b, best_n_tail, threshold, best_d_hat_e), one per input g_sorted (in
    the same order)."""
    if not g_sorted_list:
        return []
    device = g_sorted_list[0].device
    n_tail_values = np.arange(n_tail_min, n_tail_max, n_tail_step)
    if n_tail_values.size == 0:
        raise ValueError(f"empty n_tail sweep range [{n_tail_min}, {n_tail_max})")
    max_n_tail = int(n_tail_values.max())
    n_candidates = n_tail_values.size
    n_tail_t = torch.tensor(n_tail_values, device=device, dtype=torch.long)
    p_grid = torch.arange(1, m_plotting_points + 1, device=device, dtype=torch.float64) / (m_plotting_points + 1)

    results: list[dict] = [None] * len(g_sorted_list)  # type: ignore[list-item]

    for group_start in range(0, len(g_sorted_list), group_size):
        group = g_sorted_list[group_start : group_start + group_size]
        n_group = len(group)

        # Every position in the group must actually have >= max_n_tail
        # points for the shared candidate grid to be valid for it.
        for g in group:
            if g.shape[0] <= max_n_tail:
                raise ValueError(
                    f"position with {g.shape[0]} points can't support n_tail_max={max_n_tail} "
                    "in a batched group -- reduce --n-tail-max or drop this position"
                )

        # Stack this group's exceedances/masks along a new leading axis, then
        # flatten (n_group, n_candidates) -> one batch dim for
        # bootstrap_eqd_statistic_batched, same as a single position's
        # (n_candidates, max_n_tail) but with n_group more positions along
        # for the ride.
        g_head_stack = torch.stack([g[:max_n_tail] for g in group])  # (n_group, max_n_tail)
        u_stack = torch.stack([g[n_tail_t] for g in group])  # (n_group, n_candidates)
        k_idx = torch.arange(max_n_tail, device=device)
        exceedances_mask_1 = k_idx.view(1, -1) < n_tail_t.view(-1, 1)  # (n_candidates, max_n_tail)
        exceedances_mask = exceedances_mask_1.unsqueeze(0).expand(n_group, n_candidates, max_n_tail)
        exceedances = torch.where(
            exceedances_mask,
            u_stack.unsqueeze(-1) - g_head_stack.unsqueeze(1),
            torch.zeros(1, device=device, dtype=torch.float64),
        )

        flat_shape = (n_group * n_candidates, max_n_tail)
        d_hat_e_flat, d_b_flat = bootstrap_eqd_statistic_batched(
            exceedances.reshape(flat_shape), exceedances_mask.reshape(flat_shape), p_grid, n_boot, generator,
        )
        d_hat_e_group = d_hat_e_flat.reshape(n_group, n_candidates)
        d_b_group = d_b_flat.reshape(n_group, n_candidates, n_boot)

        best_idx = torch.argmin(d_hat_e_group, dim=-1)  # (n_group,)
        for gi in range(n_group):
            bi = int(best_idx[gi].item())
            results[group_start + gi] = {
                "n_tail_values": n_tail_values,
                "d_hat_e": d_hat_e_group[gi].cpu().numpy(),
                "d_b": d_b_group[gi].cpu().numpy(),  # (n_candidates, n_boot)
                "best_n_tail": int(n_tail_values[bi]),
                "threshold": float(u_stack[gi, bi].item()),
                "best_d_hat_e": float(d_hat_e_group[gi, bi].item()),
            }

    return results


def gather_position(
    perturbed_sweep_dir: str,
    in_layer_key: str,
    out_layer_key: str,
    in_manifest: dict,
    out_manifest: dict,
    sample_idx: int,
    pos: int,
    seq_len: int,
    vocab_size: int,
    n_pairs: int,
    n_pairs_save: int,
    device: str,
    generator: torch.Generator,
) -> dict:
    """Loads one (sample, pos)'s vocab-sweep vec rows, computes the full
    co-Lipschitz ratio cloud on GPU, and returns everything downstream needs:
    the sorted ratio cloud (for threshold selection) plus a small
    n_pairs_save-sized scatter subsample (for the h5 output, same as
    lipschitz_ratio_eqd_fit.py's run_sample)."""
    rows_per_sample = seq_len * vocab_size
    row_start = sample_idx * rows_per_sample
    row_end = row_start + rows_per_sample

    in_vec = load_row_range(in_layer_key, in_manifest, row_start, row_end, ["vec"])["vec"]
    out_vec = load_row_range(out_layer_key, out_manifest, row_start, row_end, ["vec"])["vec"]
    in_vec = in_vec.reshape(seq_len, vocab_size, -1)[pos]
    out_vec = out_vec.reshape(seq_len, vocab_size, -1)[pos]

    vec_in_t = torch.tensor(in_vec, device=device)
    vec_out_t = torch.tensor(out_vec, device=device)
    input_dist, output_dist, ratio = compute_lipschitz_ratios_gpu(vec_in_t, vec_out_t, n_pairs, generator)
    g_sorted = torch.sort(ratio).values

    n_save = min(n_pairs_save, ratio.shape[0])
    save_idx = torch.randperm(ratio.shape[0], device=device, generator=generator)[:n_save]

    return {
        "sample_idx": sample_idx,
        "pos": pos,
        "g_sorted": g_sorted,
        "input_dist_save": input_dist[save_idx].float().cpu().numpy(),
        "output_dist_save": output_dist[save_idx].float().cpu().numpy(),
        "ratio_save": ratio[save_idx].float().cpu().numpy(),
    }


def write_sample_h5(
    sublayer_input: str,
    sublayer_output: str,
    sample_idx: int,
    positions: list[dict],
    seq_len: int,
    n_pairs: int,
    n_pairs_save: int,
    eqd_n_boot: int,
    m_plotting_points: int,
    out_path: str,
) -> None:
    """Writes one sample's h5 file -- same schema as
    lipschitz_ratio_eqd_fit.py's run_sample / lipschitz_ratio_bootstrap_eqd_sweep.py's
    run_sample. `positions` is a list of per-position dicts already carrying
    both the raw scatter subsample (from gather_position) and the fitted
    threshold/GPD result (attached by the caller after batched fitting)."""
    assert len(positions) == seq_len, f"expected {seq_len} positions, got {len(positions)}"

    tmp_path = out_path + ".tmp"
    with h5py.File(tmp_path, "w") as h5f:
        h5f.attrs["sublayer_input"] = sublayer_input
        h5f.attrs["sublayer_output"] = sublayer_output
        h5f.attrs["sample_idx"] = sample_idx
        h5f.attrs["n_pairs"] = n_pairs
        h5f.attrs["n_pairs_save"] = n_pairs_save
        h5f.attrs["eqd_n_boot"] = eqd_n_boot
        h5f.attrs["m_plotting_points"] = m_plotting_points

        for p in sorted(positions, key=lambda p: p["pos"]):
            grp = h5f.create_group(f"pos_{p['pos']:04d}")
            grp.attrs["pos"] = p["pos"]
            grp.create_dataset("input_dist", data=p["input_dist_save"])
            grp.create_dataset("output_dist", data=p["output_dist_save"])
            grp.create_dataset("ratio", data=p["ratio_save"])

            if p.get("fit_error") is not None:
                grp.attrs["fit_error"] = p["fit_error"]
                continue

            grp.create_dataset("eqd_n_tail_values", data=p["eqd_n_tail_values"])
            grp.create_dataset("eqd_sweep", data=p["eqd_sweep"].astype(np.float32))
            # (n_candidates, n_boot): every individual bootstrap replicate's
            # own d_b(u) value at every candidate threshold, before
            # averaging into eqd_sweep/d_hat_e -- inf where that replicate's
            # GPD fit didn't converge, same convention as eqd_sweep itself.
            grp.create_dataset("eqd_d_b", data=p["eqd_d_b"].astype(np.float32))
            grp.attrs["eqd_best_n_tail"] = p["eqd_best_n_tail"]
            grp.attrs["eqd_best_eqd"] = p["eqd_best_eqd"]
            grp.attrs["threshold"] = p["threshold"]
            grp.attrs["shape_xi"] = p["shape_xi"]
            grp.attrs["scale_sigma"] = p["scale_sigma"]
            grp.attrs["g_min_observed"] = p["g_min_observed"]
            grp.attrs["c_hat"] = p["c_hat"]

        h5f.attrs["complete"] = True

    os.replace(tmp_path, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=str, help="Path to a toml ModelConfig")
    parser.add_argument(
        "--perturbed-sweep-dir", type=str, default="onnx_models/smoll/perturbed_sweep",
        help="Output directory produced by scripts/python/generate_perturbed_sweep.py",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Directory to write per-(sample, layer-pair) h5 result files to "
        "(default: <perturbed-sweep-dir>/../lipschitz_ratio_bootstrap_eqd_sweep_gpu, on NFS).",
    )
    parser.add_argument(
        "--sublayers", nargs="*", default=None,
        help="Only run these src.dst sublayer pairs (e.g. add_4.add_6) -- default: every "
        "cfg.sublayers entry. Lets a driver (e.g. one relaying input data node-to-node in "
        "batches, see scripts/python/run_lipschitz_ratio_bootstrap_eqd_sweep_gpu.py) restrict "
        "one invocation to a subset of pairs whose input data is already present.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="Number of base samples to process (default: all samples in the sweep)",
    )
    parser.add_argument("--n-pairs", type=int, default=200_000)
    parser.add_argument("--n-pairs-save", type=int, default=10_000)
    parser.add_argument("--n-tail-min", type=int, default=200)
    parser.add_argument("--n-tail-max", type=int, default=1100)
    parser.add_argument("--n-tail-step", type=int, default=200)
    parser.add_argument(
        "--eqd-n-boot", type=int, default=200,
    )
    parser.add_argument("--m-plotting-points", type=int, default=50)
    parser.add_argument(
        "--group-size", type=int, default=8,
        help="Number of (sample, pos) positions batched into one GPU fit call -- see module "
        "docstring for the benchmarked speed/memory tradeoff (default 16 stays safely under an "
        "A40's ~47.7GB at the default --eqd-n-boot=200, ~24.6GB measured; raise if profiling "
        "shows GPU memory headroom and time-per-position still dropping, lower if you hit an "
        "out-of-memory error or raise --eqd-n-boot further above the default).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cpu":
        raise SystemExit(
            "no GPU available -- this script requires CUDA (srun --gres=gpu:...). "
            "Use lipschitz_ratio_bootstrap_eqd_sweep.py for a CPU-only run."
        )

    cfg = load_config(args.config)
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.perturbed_sweep_dir.rstrip("/")), "lipschitz_ratio_bootstrap_eqd_sweep_gpu",
    )
    os.makedirs(out_dir, exist_ok=True)

    sublayer_pairs = [(s.input, s.output) for s in cfg.sublayers]
    if args.sublayers is not None:
        wanted = set(args.sublayers)
        sublayer_pairs = [(src, dst) for src, dst in sublayer_pairs if f"{src}.{dst}" in wanted]
        missing = wanted - {f"{src}.{dst}" for src, dst in sublayer_pairs}
        if missing:
            raise SystemExit(f"--sublayers requested pairs not in {args.config}: {sorted(missing)}")
    print(f"sublayer pairs: {sublayer_pairs}")

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    for sublayer_input, sublayer_output in sublayer_pairs:
        print(f"\n=== {sublayer_input} -> {sublayer_output} ===")

        in_layer_key = os.path.join(args.perturbed_sweep_dir, sublayer_input)
        out_layer_key = os.path.join(args.perturbed_sweep_dir, sublayer_output)
        if not os.path.exists(in_layer_key + ".manifest.json"):
            print(f"skipping {sublayer_input} -> {sublayer_output}: "
                  f"no perturbed-sweep output for {sublayer_input!r}")
            continue
        in_manifest = load_layer_manifest(".", in_layer_key)
        out_manifest = load_layer_manifest(".", out_layer_key)

        n_samples_total = in_manifest["attrs"]["n_samples"]
        seq_len = in_manifest["attrs"]["seq"]
        n_rows_total = in_manifest["n_rows"]
        vocab_size = n_rows_total // (n_samples_total * seq_len)

        n_samples = args.n_samples or n_samples_total
        n_samples = min(n_samples, n_samples_total)

        pair_dir = os.path.join(out_dir, f"{sublayer_input}.{sublayer_output}")
        os.makedirs(pair_dir, exist_ok=True)

        # Skip samples whose output already exists (same resumability as
        # lipschitz_ratio_bootstrap_eqd_sweep.py) -- built up front so the
        # (sample, pos) work list below only contains what actually needs
        # computing, keeping batched groups from wasting slots on
        # already-done samples.
        samples_to_run = [
            s for s in range(n_samples)
            if args.overwrite or not os.path.exists(os.path.join(pair_dir, f"sample_{s:05d}.h5"))
        ]
        if not samples_to_run:
            print(f"{sublayer_input}.{sublayer_output}: all {n_samples} samples already done, skipping")
            continue
        print(f"{len(samples_to_run)}/{n_samples} samples need computing "
              f"({seq_len} positions each = {len(samples_to_run) * seq_len} total positions)")

        work_items = [(s, pos) for s in samples_to_run for pos in range(seq_len)]

        # Gather every position's ratio cloud + scatter subsample first
        # (cheap relative to fitting -- pair sampling is a single index_select
        # + norm per position, not the batched-fit bottleneck), buffered by
        # sample so a sample's h5 can be written as soon as all its
        # positions' fits land.
        pending_by_sample: dict[int, list[dict]] = {s: [] for s in samples_to_run}

        for group_start in tqdm.tqdm(
            range(0, len(work_items), args.group_size),
            desc=f"{sublayer_input}.{sublayer_output}",
        ):
            group_items = work_items[group_start : group_start + args.group_size]

            gathered = [
                gather_position(
                    args.perturbed_sweep_dir, in_layer_key, out_layer_key, in_manifest, out_manifest,
                    s, pos, seq_len, vocab_size, args.n_pairs, args.n_pairs_save, device, gen,
                )
                for s, pos in group_items
            ]

            try:
                fit_results = select_bootstrap_eqd_threshold_gpu_multi(
                    [g["g_sorted"] for g in gathered],
                    n_tail_min=args.n_tail_min,
                    n_tail_max=args.n_tail_max,
                    n_tail_step=args.n_tail_step,
                    n_boot=args.eqd_n_boot,
                    m_plotting_points=args.m_plotting_points,
                    generator=gen,
                    group_size=args.group_size,
                )
            except ValueError as e:
                # Whole group's n_tail range invalid (e.g. a position with
                # too few points) -- record the same fit_error on every
                # position in this group, matching the CPU sweep's
                # per-position ValueError handling.
                for g in gathered:
                    pending_by_sample[g["sample_idx"]].append({**g, "fit_error": str(e)})
                continue

            for g, fit_result in zip(gathered, fit_results):
                gpd_result = fit_gpd_and_c_hat_gpu(g["g_sorted"], fit_result["best_n_tail"], fit_result["threshold"])
                pending_by_sample[g["sample_idx"]].append({
                    **g,
                    "fit_error": None,
                    "eqd_n_tail_values": fit_result["n_tail_values"],
                    "eqd_sweep": fit_result["d_hat_e"],
                    "eqd_d_b": fit_result["d_b"],
                    "eqd_best_n_tail": fit_result["best_n_tail"],
                    "eqd_best_eqd": fit_result["best_d_hat_e"],
                    "threshold": fit_result["threshold"],
                    "shape_xi": gpd_result["shape_xi"],
                    "scale_sigma": gpd_result["scale_sigma"],
                    "g_min_observed": float(g["g_sorted"][0].item()),
                    "c_hat": gpd_result["c_hat"],
                })

            # Flush any sample whose every position has now landed.
            for s in list(pending_by_sample.keys()):
                if len(pending_by_sample[s]) == seq_len:
                    out_path = os.path.join(pair_dir, f"sample_{s:05d}.h5")
                    write_sample_h5(
                        sublayer_input, sublayer_output, s, pending_by_sample.pop(s), seq_len,
                        args.n_pairs, args.n_pairs_save, args.eqd_n_boot, args.m_plotting_points, out_path,
                    )

        if pending_by_sample:
            # Should only happen if group boundaries left a sample partially
            # gathered at the very end of this sublayer pair's work list --
            # flush whatever's left rather than silently dropping it.
            for s, positions in pending_by_sample.items():
                if len(positions) == seq_len:
                    out_path = os.path.join(pair_dir, f"sample_{s:05d}.h5")
                    write_sample_h5(
                        sublayer_input, sublayer_output, s, positions, seq_len,
                        args.n_pairs, args.n_pairs_save, args.eqd_n_boot, args.m_plotting_points, out_path,
                    )
                else:
                    print(f"WARNING: sample {s} only has {len(positions)}/{seq_len} positions, not writing")

    print(f"\nSaved GPU-batched bootstrap-EQD lipschitz ratio fits under {out_dir}")


if __name__ == "__main__":
    main()
