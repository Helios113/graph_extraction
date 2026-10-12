"""GPU-batched implementation of scripts/analysis/lipschitz_ratio_bootstrap_eqd_fit.py's
Algorithm 1 (bootstrap Expected Quantile Discrepancy threshold selection),
covering the whole per-position pipeline on GPU, not just the GPD fit:

  1. Pairwise ratio sampling (compute_lipschitz_ratios_gpu): gather random
     point-pair distances via torch index_select + norm, same rejection-free
     i<j sampling as lipschitz_ratio_eqd_fit.py's compute_lipschitz_ratios.
  2. Bootstrap resampling of each candidate threshold's exceedances
     (torch.randint-indexed gather, fully vectorized across candidates x
     bootstrap replicates at once -- no Python-level loop over B).
  3. Batched GPD MLE fit of every (candidate, replicate) resample at once
     (scripts/analysis/batched_gpd_mle.py's fit_gpd_batched).
  4. Batched linear-interpolated empirical quantiles + GPD model quantiles,
     mean absolute deviation, averaged over replicates per candidate.

All of (1)-(4) run as torch tensor ops on one device end to end -- the only
CPU/numpy round-trip is the final small (n_candidates,)-sized d_hat_E(u)
array used to pick the best threshold. See batched_gpd_mle.py's docstring
for why float64 is required (float32 Newton convergence was ~10% on an A40
in testing here, vs 100% at float64).

Batches across MULTIPLE (sample, pos) positions at once too, via
select_bootstrap_eqd_threshold_gpu_multi -- benchmarked on real
co-Lipschitz ratio data on an A40 (scripts/analysis/bench_batched_gpd_scale.py):
time-per-position drops from ~1.4s at group size 1 to ~0.78s at group size
16, with only marginal further improvement out to 64 (and GPU memory scaling
~linearly with group size, ~2.6GB at 1 position vs ~24.8GB at 64) -- so
group_size=16 is the default, capturing nearly all the available speedup
without pushing memory usage unnecessarily. Same `chunk`-style grouping
pattern used elsewhere in this codebase, e.g. autoencoder/batched_autoencoder.py's
knn.
"""

import numpy as np
import torch

from scripts.lipschitz.batched_gpd_mle import fit_gpd_batched


def compute_lipschitz_ratios_gpu(
    vec_in: torch.Tensor, vec_out: torch.Tensor, n_pairs: int, generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPU torch equivalent of lipschitz_ratio_eqd_fit.py's
    compute_lipschitz_ratios. vec_in, vec_out: (n_points, d) tensors on the
    target device. Returns (input_dist, output_dist, ratio), each (n_pairs,)
    float64 tensors on the same device."""
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
    return input_dist, output_dist, ratio


def batched_quantile(x_sorted: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Batched linear-interpolated sample quantile function, matching
    np.quantile(..., method="linear") / torch.quantile's default -- computed
    manually (rather than torch.quantile, which caps input size at 2**24
    elements and doesn't broadcast a per-row p the way this needs) over the
    last dim of x_sorted (..., n), evaluated at p (m,), giving (..., m).

    x_sorted must already be ascending-sorted along its last dim."""
    n = x_sorted.shape[-1]
    pos = p * (n - 1)
    lo = pos.floor().long().clamp(0, n - 1)
    hi = pos.ceil().long().clamp(0, n - 1)
    frac = (pos - lo.double()).to(x_sorted.dtype)
    lo_exp = lo.expand(*x_sorted.shape[:-1], -1)
    hi_exp = hi.expand(*x_sorted.shape[:-1], -1)
    lo_vals = torch.gather(x_sorted, -1, lo_exp)
    hi_vals = torch.gather(x_sorted, -1, hi_exp)
    return lo_vals + frac * (hi_vals - lo_vals)


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
    # the quantile function can't use plain batched_quantile (which assumes
    # every row's real entries span the same length) -- sort with padding
    # pushed to +inf, then index each row's quantile position using ONLY
    # that row's own n_tail_per_candidate, not the shared max_n_tail.
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


def select_bootstrap_eqd_threshold_gpu(
    g_sorted: torch.Tensor,
    n_tail_min: int,
    n_tail_max: int,
    n_tail_step: int,
    n_boot: int,
    m_plotting_points: int,
    generator: torch.Generator,
) -> dict:
    """GPU-batched equivalent of
    lipschitz_ratio_bootstrap_eqd_fit.py's select_bootstrap_eqd_threshold --
    same candidate grid and selection rule, but every candidate's bootstrap
    EQD statistic is computed in one batched GPU call instead of a Python
    loop over candidates (each doing its own Python loop over B replicates).

    All candidates are padded to the single largest candidate's n_tail and
    fit in one batched call. This was initially suspected to waste too much
    compute on small-n_tail candidates (padding a 20-point candidate out to
    a 2000-point row), and a bucketed-by-n_tail variant was tried to bound
    that waste -- but benchmarking on an A40 showed bucketing was a net
    LOSS (smaller buckets were slower: 44.5s/position at bucket_size=4 vs.
    15.7s/position unbucketed), because splitting into more batches just
    multiplies per-call fixed overhead (kernel launches, sync points) by the
    number of buckets. The real bottleneck turned out to be
    fit_gpd_batched's multi-start grid running all 5 starts' full Newton
    loops over the WHOLE batch regardless of whether the first start
    already converged (fixed in batched_gpd_mle.py: later starts now only
    run on rows the earlier starts left unconverged) -- that fix alone took
    a same-shape isolated fit benchmark from ~17x to ~330x over sequential
    scipy, making padding waste comparatively minor. See git history/PR
    discussion for the bucketing dead end if reviving it is ever tempting.

    g_sorted: (n_points,) float64 tensor, ascending-sorted, on the target
    device."""
    device = g_sorted.device
    n_tail_max = min(n_tail_max, g_sorted.shape[0] - 1)
    n_tail_values = np.arange(n_tail_min, n_tail_max, n_tail_step)
    if n_tail_values.size == 0:
        raise ValueError(
            f"empty n_tail sweep range [{n_tail_min}, {n_tail_max}) for {g_sorted.shape[0]} points"
        )

    max_n_tail = int(n_tail_values.max())
    n_candidates = n_tail_values.size
    n_tail_t = torch.tensor(n_tail_values, device=device, dtype=torch.long)

    # exceedances[c, k] = u_c - g_sorted[k] for k < n_tail_values[c], else 0
    # (masked). u_c = g_sorted[n_tail_values[c]].
    u = g_sorted[n_tail_t]  # (n_candidates,)
    k_idx = torch.arange(max_n_tail, device=device)
    exceedances_mask = k_idx.view(1, -1) < n_tail_t.view(-1, 1)
    g_head = g_sorted[:max_n_tail].view(1, -1).expand(n_candidates, -1)
    exceedances = torch.where(
        exceedances_mask, u.view(-1, 1) - g_head, torch.zeros(1, device=device, dtype=g_sorted.dtype),
    )

    p_grid = torch.arange(1, m_plotting_points + 1, device=device, dtype=g_sorted.dtype) / (m_plotting_points + 1)

    d_hat_e, d_b = bootstrap_eqd_statistic_batched(exceedances, exceedances_mask, p_grid, n_boot, generator)

    best_idx = int(torch.argmin(d_hat_e).item())
    best_n_tail = int(n_tail_values[best_idx])
    threshold = float(u[best_idx].item())

    return {
        "n_tail_values": n_tail_values,
        "d_hat_e": d_hat_e.cpu().numpy(),
        "d_b": d_b.cpu().numpy(),  # (n_candidates, n_boot) -- every individual bootstrap's own fitness
        "best_n_tail": best_n_tail,
        "threshold": threshold,
        "best_d_hat_e": float(d_hat_e[best_idx].item()),
    }


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
    """Runs select_bootstrap_eqd_threshold_gpu's Algorithm 1 over MULTIPLE
    (sample, pos) ratio clouds at once, batching `group_size` positions
    together into each fit_gpd_batched call (the driver a sweep script
    should call instead of looping select_bootstrap_eqd_threshold_gpu one
    position at a time).

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

    Returns a list of per-position result dicts, same shape/keys as
    select_bootstrap_eqd_threshold_gpu's single-position return, one per
    input g_sorted (in the same order)."""
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
        # points for the shared candidate grid to be valid for it -- same
        # per-position requirement select_bootstrap_eqd_threshold_gpu
        # enforces individually (n_tail_max clamp / empty-range check), just
        # verified up front here since the whole group is built as one
        # stacked tensor rather than per-position.
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
