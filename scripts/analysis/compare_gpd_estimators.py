"""Compares scripts/lipschitz/batched_gpd_mle.py's Newton MLE against
scripts/analysis/gpd_pwm_fit.py's PWM/L-moments estimator on the SAME
exceedances, for positions in a lipschitz_ratio_bootstrap_eqd_sweep_gpu.py
(or CPU lipschitz_ratio_eqd_fit.py) output directory.

Motivation: GPD MLE is known to behave poorly near xi=0 (flat/multimodal
likelihood) and is non-regular for xi < -0.5 -- a near-zero or sign-flipped
MLE xi (see scripts/analysis/list_flagged_positions.py) could be a real
feature of the ratio cloud's tail, or a Newton-solver artifact of that
region. PWM/L-moments doesn't touch the likelihood at all (closed-form
sample-moment estimator), so a large disagreement between the two on the
same data is a signal the MLE fit in that region shouldn't be trusted at
face value.

The h5 sweep output only stores a random n_pairs_save-sized ratio
subsample, not the full n_pairs cloud the original MLE fit's exceedances
were drawn from (see gather_position/fit_gpd_and_c_hat_gpu in
lipschitz_ratio_bootstrap_eqd_sweep_gpu.py) -- so this script regenerates a
fresh full-resolution ratio cloud from the perturbed-sweep shards via
lipschitz_ratio_bootstrap_eqd_fit.py's load_ratio_cloud (same n_pairs, a
fresh seed), then reuses the h5's saved threshold/eqd_best_n_tail to build
exceedances the same way fit_gpd_and_c_hat_gpu does. This is not a
bit-identical replay of the original run's random pairs (that would need
replaying the whole run's shared torch.Generator state) -- it's a fresh
i.i.d. draw from the same (sample, pos) distribution, which is what the
estimator-comparison question actually needs.

Usage:
    # Every flagged position in the pair dir:
    uv run python -m scripts.analysis.compare_gpd_estimators \\
        onnx_models/simple_stories/lipschitz_ratio_bootstrap_eqd_sweep_gpu/embedding.add_4 \\
        --perturbed-sweep-dir onnx_models/simple_stories/perturbed_sweep \\
        --flagged-only

    # One specific (sample, pos), plus a 5-panel plot (ratio histogram,
    # tail histogram, EQD n_tail sweep, exceedance QQ plot, c_hat-vs-n_tail
    # curve) -- --plot requires --sample-idx/--pos since it only makes
    # sense for one position at a time. --subsample-n trades cloud
    # resolution for speed uniformly across every panel (the EQD sweep and
    # c_hat panel are the expensive parts, O(n_tail_max) GPD fits each, on
    # CPU only in a plain srun with no --gres=gpu):
    uv run python -m scripts.analysis.compare_gpd_estimators \\
        onnx_models/simple_stories/lipschitz_ratio_bootstrap_eqd_sweep_gpu/embedding.add_4 \\
        --perturbed-sweep-dir onnx_models/simple_stories/perturbed_sweep \\
        --sample-idx 5 --pos 7 --plot detail.png --subsample-n 10000
"""

import argparse
import glob
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from scripts.analysis.gpd_pwm_fit import gpd_pwm_fit
from scripts.analysis.lipschitz_ratio_bootstrap_eqd_fit import load_ratio_cloud
from scripts.lipschitz.batched_gpd_mle import fit_gpd_batched, fit_gpd_c_hat_nonneg_batched
from scripts.lipschitz.lipschitz_ratio_bootstrap_eqd_sweep_gpu import (
    gpd_model_quantile_batched,
    select_bootstrap_eqd_threshold_gpu_multi,
)


def load_one_position(pair_dir: str, sample_idx: int, pos: int) -> tuple[float, int, float]:
    """Reads (threshold, n_tail, xi) for a single (sample_idx, pos) straight
    out of its sample_*.h5 file, without scanning the rest of pair_dir."""
    path = os.path.join(pair_dir, f"sample_{sample_idx:05d}.h5")
    with h5py.File(path, "r") as h5f:
        grp_name = f"pos_{pos:04d}"
        if grp_name not in h5f:
            raise KeyError(f"{grp_name!r} not found in {path}")
        grp = h5f[grp_name]
        if "fit_error" in grp.attrs:
            raise ValueError(f"{path}:{grp_name} has fit_error={grp.attrs['fit_error']!r}, no fit to compare")
        return float(grp.attrs["threshold"]), int(grp.attrs["eqd_best_n_tail"]), float(grp.attrs["shape_xi"])


def iter_positions(pair_dir: str, flagged_only: bool):
    """Yields (sample_idx, pos, threshold, n_tail, xi) for every position in
    pair_dir with a valid fit -- filtered to xi>=0 or c_hat<0 (same
    criterion as list_flagged_positions.py) if flagged_only."""
    paths = sorted(glob.glob(os.path.join(pair_dir, "sample_*.h5")))
    if not paths:
        raise SystemExit(f"no sample_*.h5 files found under {pair_dir!r}")

    for path in paths:
        sample_idx = int(os.path.basename(path).removeprefix("sample_").removesuffix(".h5"))
        with h5py.File(path, "r") as h5f:
            for pos_key in sorted(k for k in h5f.keys() if k.startswith("pos_")):
                grp = h5f[pos_key]
                if "fit_error" in grp.attrs:
                    continue
                xi = float(grp.attrs["shape_xi"])
                c_hat = float(grp.attrs["c_hat"])
                if flagged_only and not (xi >= 0 or c_hat < 0):
                    continue
                yield sample_idx, int(grp.attrs["pos"]), float(grp.attrs["threshold"]), int(grp.attrs["eqd_best_n_tail"]), xi


def plot_detail(
    out_path: str,
    sublayer_input: str,
    sublayer_output: str,
    sample_idx: int,
    pos: int,
    g_sorted: np.ndarray,
    n_tail: int,
    threshold: float,
    xi_mle: float,
    sigma_mle: float,
    xi_pwm: float,
    sigma_pwm: float,
    n_tail_min: int,
    n_tail_max: int,
    n_tail_step: int,
    eqd_n_boot: int,
    m_plotting_points: int,
    device: str,
    input_dist: np.ndarray | None = None,
    output_dist: np.ndarray | None = None,
) -> None:
    """8-panel grid (2x4): ratio distribution, tail histogram (all n_tail
    exceedances), a recomputed EQD n_tail sweep, a fit-quality-vs-n_tail
    curve (mean |model_q - sample_q| over the plotting-position grid,
    computed directly -- no bootstrap, unlike the EQD panel's d_hat_e), an
    exceedance QQ plot (empirical vs. MLE/PWM fitted GPD quantiles) for the
    one picked n_tail, an xi-vs-n_tail curve, a c_hat-vs-n_tail curve, and
    (if input_dist/output_dist are given) a scatter of every pair's input
    distance ||x-y|| vs. output distance ||f(x)-f(y)|| -- the two raw
    quantities the ratio g=output/input is built from, letting you see
    directly whether the ratio cloud's shape comes from a tight linear
    relationship (points near a y=g*x line) or genuine scatter/heteroscedasticity
    a single ratio histogram can't show. The xi and c_hat panels each also
    show a THIRD curve: the xi<0, c_hat>=0 CONSTRAINED fit
    (scripts/lipschitz/batched_gpd_mle.py's fit_gpd_c_hat_nonneg_batched) --
    this codebase's ratios g>=0 always, so an unconstrained MLE landing on a
    negative implied infimum (c_hat<0, which does happen -- see
    scripts/analysis/list_flagged_positions.py) isn't physically valid; the
    constrained curve instead reports the best-likelihood fit still
    satisfying c_hat>=0, coinciding with the unconstrained MLE wherever
    that's already feasible. All of the candidate-grid panels (EQD,
    fit-quality, xi, c_hat) share one candidate grid, with the picked
    n_tail/threshold marked throughout. g_sorted is whatever cloud the
    caller passes in -- the full regenerated cloud, or a subsample of it
    via main()'s --subsample-n (all panels use the same g_sorted, so
    subsampling applies uniformly). The EQD sweep and the candidate-grid
    panels are O(n_tail_max * eqd_n_boot) batched GPD fits --
    device="cuda" runs them on GPU (same batched ops
    lipschitz_ratio_bootstrap_eqd_sweep_gpu.py uses), which is an order of
    magnitude faster than CPU for this workload; requires the srun/sbatch
    job to have requested a GPU (--gres=gpu:...)."""
    exceedances = threshold - g_sorted[:n_tail]

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    ax_hist, ax_tail, ax_eqd, ax_fit = axes[0]
    ax_qq, ax_xi, ax_chat, ax_scatter = axes[1]

    ax_hist.hist(g_sorted, bins=100, color="steelblue", alpha=0.8)
    ax_hist.axvline(threshold, color="orange", linestyle="--", label=f"threshold u={threshold:.4g}")
    if xi_mle < 0:
        c_hat_mle = threshold + sigma_mle / xi_mle
        ax_hist.axvline(c_hat_mle, color="red", linestyle="--", label=f"c_hat (MLE)={c_hat_mle:.4g}")
    ax_hist.legend()
    ax_hist.set_xlabel("ratio g = ||f(x)-f(y)|| / ||x-y||")
    ax_hist.set_ylabel("count")
    ax_hist.set_title(f"ratio distribution (N={g_sorted.size})")

    ax_tail.hist(exceedances, bins=50, color="steelblue", alpha=0.8)
    ax_tail.set_xlabel("exceedance = threshold - g")
    ax_tail.set_ylabel("count")
    ax_tail.set_title(f"full tail (n_tail={n_tail})\nxi_mle={xi_mle:.4f}  xi_pwm={xi_pwm:.4f}")

    g_sorted_t = torch.tensor(g_sorted, dtype=torch.float64, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    [fit_result] = select_bootstrap_eqd_threshold_gpu_multi(
        [g_sorted_t], n_tail_min=n_tail_min, n_tail_max=n_tail_max, n_tail_step=n_tail_step,
        n_boot=eqd_n_boot, m_plotting_points=m_plotting_points, generator=gen, group_size=1,
    )
    ax_eqd.plot(fit_result["n_tail_values"], fit_result["d_hat_e"], color="steelblue", lw=1.5)
    ax_eqd.axvline(
        n_tail, color="orange", linestyle="--", label=f"this fit's n_tail={n_tail}",
    )
    if fit_result["best_n_tail"] != n_tail:
        ax_eqd.axvline(
            fit_result["best_n_tail"], color="green", linestyle=":",
            label=f"EQD-best n_tail={fit_result['best_n_tail']}",
        )
    ax_eqd.legend()
    ax_eqd.set_xlabel("n_tail")
    ax_eqd.set_ylabel("EQD statistic")
    ax_eqd.set_title("EQD threshold sweep")

    # QQ plot: empirical exceedance quantiles (plotting-position convention
    # p_j = j/(n_tail+1), same as bootstrap_eqd_statistic_batched's p_grid)
    # against the MLE- and PWM-fitted GPD's own model quantiles at those
    # same p -- shows where the two estimators' fitted tails diverge across
    # the full exceedance range, not just in their single xi/sigma values.
    exceedances_sorted = np.sort(exceedances)
    p_qq = np.arange(1, n_tail + 1) / (n_tail + 1)
    p_qq_t = torch.tensor(p_qq, dtype=torch.float64, device=device)
    q_mle = gpd_model_quantile_batched(
        p_qq_t, torch.tensor([xi_mle], dtype=torch.float64, device=device),
        torch.tensor([sigma_mle], dtype=torch.float64, device=device),
    )[0].cpu().numpy()
    q_pwm = gpd_model_quantile_batched(
        p_qq_t, torch.tensor([xi_pwm], dtype=torch.float64, device=device),
        torch.tensor([sigma_pwm], dtype=torch.float64, device=device),
    )[0].cpu().numpy()
    ax_qq.plot(exceedances_sorted, exceedances_sorted, color="gray", lw=1, linestyle=":", label="y=x")
    ax_qq.plot(exceedances_sorted, q_mle, color="tab:red", lw=1.5, label=f"MLE fit (xi={xi_mle:.4f})")
    ax_qq.plot(exceedances_sorted, q_pwm, color="tab:green", lw=1.5, label=f"PWM fit (xi={xi_pwm:.4f})")
    ax_qq.legend()
    ax_qq.set_xlabel("empirical exceedance quantile")
    ax_qq.set_ylabel("model exceedance quantile")
    ax_qq.set_title(f"exceedance QQ plot (n_tail={n_tail})")

    # c_hat vs n_tail: fit a final (non-bootstrapped) GPD at every candidate
    # n_tail in the same grid the EQD sweep used, on the same (possibly
    # subsampled) cloud, batching every candidate's MLE fit into one
    # fit_gpd_batched call (ragged tail lengths handled by a mask, same
    # pattern fit_gpd_and_c_hat_gpu/bootstrap_eqd_statistic_batched use) --
    # shows how sensitive c_hat is to the threshold choice, not just its
    # value at the one picked n_tail.
    n_tail_grid = fit_result["n_tail_values"]
    max_n_tail_grid = int(n_tail_grid.max())
    n_candidates_grid = n_tail_grid.size
    n_tail_grid_t = torch.tensor(n_tail_grid, dtype=torch.long, device=device)
    u_grid = g_sorted[n_tail_grid]  # threshold at each candidate n_tail
    k_idx = torch.arange(max_n_tail_grid, device=device)
    mask_grid = k_idx.view(1, -1) < n_tail_grid_t.view(-1, 1)
    exc_grid = np.where(
        mask_grid.cpu().numpy(),
        u_grid[:, None] - g_sorted[:max_n_tail_grid][None, :],
        0.0,
    )
    exc_grid_t = torch.tensor(exc_grid, dtype=torch.float64, device=device)
    xi_grid, sigma_grid, conv_grid = fit_gpd_batched(exc_grid_t, mask_grid)
    xi_grid, sigma_grid, conv_grid = xi_grid.cpu().numpy(), sigma_grid.cpu().numpy(), conv_grid.cpu().numpy()
    # xi_mle_grid/c_hat_mle_grid are NOT filtered on conv_grid (fit_gpd_batched's
    # strict tol=1e-4 gradient-norm convergence check) -- same pattern
    # lipschitz_ratio_bootstrap_eqd_sweep_gpu.py's fit_gpd_and_c_hat_gpu uses,
    # per batched_gpd_mle.py's own tuning notes: converged=False at the
    # default n_iters=10 budget routinely still means "within 0.5-5% of
    # scipy's exact MLE," not "bad fit" -- dropping those points to NaN left
    # real gaps in this curve over ranges of n_tail that had a perfectly
    # usable best-effort fit. conv_grid is kept only to mark those points
    # visually (open/faded, not omitted) so the strict-vs-best-effort
    # distinction is still visible.
    xi_mle_grid = xi_grid
    sigma_mle_grid = sigma_grid
    c_hat_mle_grid = np.where(
        xi_grid < 0, u_grid + sigma_grid / np.where(xi_grid < 0, xi_grid, -1), np.nan,
    )
    xi_pwm_grid = np.full(n_candidates_grid, np.nan)
    sigma_pwm_grid = np.full(n_candidates_grid, np.nan)
    c_hat_pwm_grid = np.full(n_candidates_grid, np.nan)
    for i in range(n_candidates_grid):
        xi_c, sigma_c = gpd_pwm_fit(exc_grid[i, : n_tail_grid[i]])
        xi_pwm_grid[i] = xi_c
        sigma_pwm_grid[i] = sigma_c
        if xi_c < 0:
            c_hat_pwm_grid[i] = u_grid[i] + sigma_c / xi_c

    # xi<0, c_hat>=0 constrained fit: this codebase's ratios g>=0 always, so
    # a fitted tail with a negative implied infimum (c_hat<0, which the
    # unconstrained MLE can and does land on -- see
    # scripts/analysis/list_flagged_positions.py) isn't physically valid.
    # fit_gpd_c_hat_nonneg_batched keeps the unconstrained MLE wherever it's
    # already feasible and only re-solves (on the c_hat=0 boundary) rows
    # where the constraint is actually binding.
    u_grid_t = torch.tensor(u_grid, dtype=torch.float64, device=device)
    xi_nonneg_grid, sigma_nonneg_grid, was_constrained_grid = fit_gpd_c_hat_nonneg_batched(
        exc_grid_t, u_grid_t, mask_grid,
    )
    xi_nonneg_grid = xi_nonneg_grid.cpu().numpy()
    sigma_nonneg_grid = sigma_nonneg_grid.cpu().numpy()
    c_hat_nonneg_grid = np.where(
        np.isfinite(xi_nonneg_grid) & (xi_nonneg_grid < 0),
        u_grid + sigma_nonneg_grid / np.where(xi_nonneg_grid < 0, xi_nonneg_grid, -1),
        np.nan,
    )

    # marker="x" for MLE (distinct from PWM/constrained's "." dots) so it
    # stays identifiable even when its line is visually buried under
    # another curve -- the constrained fit often nearly coincides with
    # unconstrained MLE wherever the constraint isn't binding (see
    # fit_gpd_c_hat_nonneg_batched's docstring), which can hide a
    # same-shaped marker entirely.
    ax_xi.plot(n_tail_grid, xi_mle_grid, color="tab:red", marker="x", ms=6, lw=1.2, label="xi (MLE)")
    if (~conv_grid).any():
        # Newton didn't hit the strict tol=1e-4 gradient-norm bar at these
        # candidates -- still plotted above (best-effort fit, not dropped),
        # but marked here so it's visible which points are strict-converged
        # vs. best-effort.
        ax_xi.scatter(
            n_tail_grid[~conv_grid], xi_mle_grid[~conv_grid], facecolors="none", edgecolors="tab:red",
            marker="o", s=80, label="MLE not strictly converged",
        )
    ax_xi.plot(n_tail_grid, xi_pwm_grid, color="tab:green", marker=".", lw=1.2, label="xi (PWM)")
    ax_xi.plot(
        n_tail_grid, xi_nonneg_grid, color="tab:purple", marker=".", lw=1.2, linestyle="--",
        label="xi (xi<0, c_hat>=0 constrained)",
    )
    ax_xi.axhline(0, color="gray", lw=1, linestyle=":")
    ax_xi.axvline(n_tail, color="orange", linestyle="--", label=f"picked n_tail={n_tail}")
    ax_xi.legend(fontsize=8)
    ax_xi.set_xlabel("n_tail")
    ax_xi.set_ylabel("xi")
    ax_xi.set_title("xi vs. n_tail")

    ax_chat.plot(n_tail_grid, c_hat_mle_grid, color="tab:red", marker="x", ms=6, lw=1.2, label="c_hat (MLE)")
    not_converged_finite = ~conv_grid & np.isfinite(c_hat_mle_grid)
    if not_converged_finite.any():
        ax_chat.scatter(
            n_tail_grid[not_converged_finite], c_hat_mle_grid[not_converged_finite], facecolors="none",
            edgecolors="tab:red", marker="o", s=80, label="MLE not strictly converged",
        )
    ax_chat.plot(n_tail_grid, c_hat_pwm_grid, color="tab:green", marker=".", lw=1.2, label="c_hat (PWM)")
    ax_chat.plot(
        n_tail_grid, c_hat_nonneg_grid, color="tab:purple", marker=".", lw=1.2, linestyle="--",
        label="c_hat (constrained)",
    )
    ax_chat.axvline(n_tail, color="orange", linestyle="--", label=f"picked n_tail={n_tail}")
    ax_chat.legend(fontsize=8)
    ax_chat.set_xlabel("n_tail")
    ax_chat.set_ylabel("c_hat")
    ax_chat.set_title("c_hat vs. n_tail")

    # Fit quality vs n_tail: same quantity the QQ panel shows visually for
    # ONE n_tail (mean |model_q(p) - sample_q(p)| over the plotting-position
    # grid p_j=j/(n_tail_c+1)), computed directly here -- no bootstrap
    # resampling, unlike the EQD panel's d_hat_e -- at every candidate
    # n_tail, for both estimators. Lower is a better-matching fit at that
    # threshold; this is literally the QQ plot's y-x gap, averaged into one
    # number per n_tail.
    fit_mle_grid = np.full(n_candidates_grid, np.nan)
    fit_pwm_grid = np.full(n_candidates_grid, np.nan)
    for i in range(n_candidates_grid):
        n_tail_c = int(n_tail_grid[i])
        exc_c_sorted = np.sort(exc_grid[i, :n_tail_c])
        p_c = np.arange(1, n_tail_c + 1) / (n_tail_c + 1)
        if np.isfinite(xi_mle_grid[i]) and np.isfinite(sigma_mle_grid[i]):
            p_c_t = torch.tensor(p_c, dtype=torch.float64, device=device)
            q_mle_c = gpd_model_quantile_batched(
                p_c_t, torch.tensor([xi_mle_grid[i]], dtype=torch.float64, device=device),
                torch.tensor([sigma_mle_grid[i]], dtype=torch.float64, device=device),
            )[0].cpu().numpy()
            fit_mle_grid[i] = np.mean(np.abs(q_mle_c - exc_c_sorted))
        if np.isfinite(xi_pwm_grid[i]) and np.isfinite(sigma_pwm_grid[i]):
            q_pwm_c = gpd_model_quantile_batched(
                torch.tensor(p_c, dtype=torch.float64), torch.tensor([xi_pwm_grid[i]], dtype=torch.float64),
                torch.tensor([sigma_pwm_grid[i]], dtype=torch.float64),
            )[0].numpy()
            fit_pwm_grid[i] = np.mean(np.abs(q_pwm_c - exc_c_sorted))

    ax_fit.plot(n_tail_grid, fit_mle_grid, color="tab:red", marker="x", ms=6, lw=1.2, label="fit error (MLE)")
    ax_fit.plot(n_tail_grid, fit_pwm_grid, color="tab:green", marker=".", lw=1.2, label="fit error (PWM)")
    ax_fit.axvline(n_tail, color="orange", linestyle="--", label=f"picked n_tail={n_tail}")
    ax_fit.legend()
    ax_fit.set_xlabel("n_tail")
    ax_fit.set_ylabel("mean |model_q - sample_q|")
    ax_fit.set_title("fit quality vs. n_tail")

    if input_dist is not None and output_dist is not None:
        # Raw (input_dist, output_dist) pairs the ratio g=output/input was
        # computed from -- shows whether the ratio cloud's shape comes from
        # a tight roughly-linear relationship (points hugging a y=g*x line
        # through the origin) or real scatter/heteroscedasticity a ratio
        # histogram alone can't distinguish. Subsampled for plotting (these
        # arrays can be n_pairs=200k+ long) -- a scatter that dense would
        # be an unreadable solid blob and slow to render regardless.
        max_scatter_points = 20_000
        if input_dist.size > max_scatter_points:
            idx = np.random.default_rng(0).choice(input_dist.size, size=max_scatter_points, replace=False)
            in_plot, out_plot = input_dist[idx], output_dist[idx]
        else:
            in_plot, out_plot = input_dist, output_dist
        ax_scatter.scatter(in_plot, out_plot, s=2, alpha=0.15, color="steelblue")
        ax_scatter.set_xlabel("input distance ||x-y||")
        ax_scatter.set_ylabel("output distance ||f(x)-f(y)||")
        ax_scatter.set_title(f"input vs. output distance (n={in_plot.size})")
    else:
        ax_scatter.axis("off")

    fig.suptitle(
        f"{sublayer_input} -> {sublayer_output}: sample {sample_idx}, pos {pos}  "
        f"(N={g_sorted.size} pairs, n_tail={n_tail} exceedances)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved figure to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pair_dir", type=str, help="e.g. onnx_models/.../embedding.add_4")
    parser.add_argument(
        "--perturbed-sweep-dir", type=str, required=True,
        help="Same --perturbed-sweep-dir the sweep that produced pair_dir was run with.",
    )
    parser.add_argument("--flagged-only", action="store_true", help="Only compare xi>=0 or c_hat<0 positions.")
    parser.add_argument(
        "--sample-idx", type=int, default=None,
        help="Compare only this one sample (requires --pos too) instead of scanning the whole pair_dir.",
    )
    parser.add_argument("--pos", type=int, default=None, help="Compare only this one position (requires --sample-idx too).")
    parser.add_argument(
        "--n-tail", type=int, default=None,
        help="Override the tail size used for both fits, instead of each position's saved eqd_best_n_tail. "
        "Threshold is always recomputed as this run's own regenerated g_sorted[n_tail] (never the h5's saved "
        "threshold value, which was computed against a different random draw of the ratio cloud and would "
        "produce spurious negative exceedances if reused here).",
    )
    parser.add_argument(
        "--n-pairs", type=int, default=200_000,
        help="Pairs to draw for the regenerated ratio cloud. Matches the original sweep's --n-pairs by default; "
        "lower this (e.g. via --subsample-n, which overrides it) to speed everything up -- load_ratio_cloud's "
        "own pairwise-distance computation, the tail fit, and (if --plot) the EQD sweep and c_hat panel all "
        "scale with this, on CPU only (no GPU in a plain srun).",
    )
    parser.add_argument("--seed", type=int, default=1, help="Fresh seed for the regenerated ratio cloud.")
    parser.add_argument(
        "--subsample-n", type=int, default=None,
        help="Shortcut for --n-pairs: draws this many pairs for EVERYTHING (histograms, tail fit, EQD sweep, "
        "c_hat panel alike) instead of the full --n-pairs cloud. Must stay > --n-tail-max. Use this instead of "
        "--n-pairs directly when you just want 'make it faster' without also having to reduce --n-tail-max.",
    )
    parser.add_argument(
        "--plot", type=str, default=None,
        help="Save a 5-panel figure (ratio histogram, tail histogram, recomputed EQD n_tail sweep, exceedance "
        "QQ plot, c_hat-vs-n_tail curve) to this path. Requires --sample-idx/--pos (one position at a time).",
    )
    parser.add_argument(
        "--eqd-n-boot", type=int, default=50, help="Bootstrap replicates for the --plot EQD sweep panel "
        "(matches lipschitz_ratio_bootstrap_eqd_sweep_gpu.py's --eqd-n-boot default).",
    )
    parser.add_argument("--n-tail-min", type=int, default=20, help="--plot EQD sweep candidate grid lower bound.")
    parser.add_argument("--n-tail-max", type=int, default=3_000, help="--plot EQD sweep candidate grid upper bound.")
    parser.add_argument("--n-tail-step", type=int, default=50, help="--plot EQD sweep candidate grid step.")
    parser.add_argument("--m-plotting-points", type=int, default=50, help="--plot EQD sweep quantile grid size.")
    parser.add_argument(
        "--device", type=str, default=None, choices=["cuda", "cpu"],
        help="Device for the GPD fits (default: cuda if available, else cpu). The --plot EQD sweep and c_hat "
        "panel are O(n_tail_max * eqd_n_boot) batched GPD fits and are an order of magnitude faster on GPU -- "
        "request one in the srun/sbatch command (--gres=gpu:...) to use it.",
    )
    args = parser.parse_args()
    if (args.sample_idx is None) != (args.pos is None):
        raise SystemExit("--sample-idx and --pos must be given together")
    if args.plot is not None and args.sample_idx is None:
        raise SystemExit("--plot requires --sample-idx/--pos (plots one position at a time)")
    n_pairs = args.subsample_n if args.subsample_n is not None else args.n_pairs
    if args.plot is not None and n_pairs <= args.n_tail_max:
        raise SystemExit(f"--n-pairs/--subsample-n ({n_pairs}) must be > --n-tail-max ({args.n_tail_max})")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    sublayer_input, sublayer_output = os.path.basename(args.pair_dir.rstrip("/")).split(".", 1)

    if args.sample_idx is not None:
        threshold, n_tail, xi_h5 = load_one_position(args.pair_dir, args.sample_idx, args.pos)
        positions = [(args.sample_idx, args.pos, threshold, n_tail, xi_h5)]
    else:
        positions = list(iter_positions(args.pair_dir, args.flagged_only))
    if not positions:
        raise SystemExit("no positions to compare (try without --flagged-only)")

    rows = []
    for sample_idx, pos, threshold, n_tail, xi_h5 in tqdm.tqdm(positions, desc="comparing"):
        input_dist, output_dist, ratio = load_ratio_cloud(
            args.perturbed_sweep_dir, sublayer_input, sublayer_output, sample_idx, pos,
            n_pairs=n_pairs, seed=args.seed, return_dists=True,
        )
        g_sorted = np.sort(ratio)
        if args.n_tail is not None:
            n_tail = args.n_tail
        # threshold is ALWAYS recomputed from this position's freshly
        # regenerated g_sorted, never reused from the h5 (whose saved
        # threshold was computed on the ORIGINAL sweep's random draw of
        # ratio pairs). g_sorted here is a fresh i.i.d. draw with its own
        # seed (see load_ratio_cloud call above) -- mixing the old
        # threshold with this new cloud's sorted values let some of
        # g_sorted[:n_tail] land ABOVE the old threshold (pure sampling
        # noise between the two draws), producing spurious negative
        # exceedances (threshold - g_sorted[i] < 0) that don't belong in a
        # left-tail GPD fit. n_tail (a count, not a data-dependent value) is
        # still safe to reuse across draws -- only the threshold VALUE
        # needs to match the draw it's applied to.
        threshold = float(g_sorted[n_tail])
        exceedances = threshold - g_sorted[:n_tail]

        xi_pwm, sigma_pwm = gpd_pwm_fit(exceedances)

        z_t = torch.tensor(exceedances, dtype=torch.float64, device=device).unsqueeze(0)
        xi_mle_t, sigma_mle_t, converged_t = fit_gpd_batched(z_t)
        xi_mle, sigma_mle, converged = xi_mle_t.item(), sigma_mle_t.item(), bool(converged_t.item())

        rows.append({
            "sample": sample_idx, "pos": pos, "n_pairs": g_sorted.size, "n_tail": n_tail,
            "xi_h5": xi_h5, "xi_mle_refit": xi_mle, "xi_pwm": xi_pwm,
            "sign_flip": (xi_mle >= 0) != (xi_pwm >= 0),
            "mle_converged": converged,
        })

        if args.plot is not None:
            plot_detail(
                args.plot, sublayer_input, sublayer_output, sample_idx, pos, g_sorted, n_tail, threshold,
                xi_mle, sigma_mle, xi_pwm, sigma_pwm,
                n_tail_min=args.n_tail_min, n_tail_max=args.n_tail_max, n_tail_step=args.n_tail_step,
                eqd_n_boot=args.eqd_n_boot, m_plotting_points=args.m_plotting_points, device=device,
                input_dist=input_dist, output_dist=output_dist,
            )

    print(f"\n{'sample':>8} {'pos':>5} {'n_pairs':>9} {'n_tail':>7} {'xi_h5':>10} {'xi_mle':>10} {'xi_pwm':>10} "
          f"{'sign_flip':>10} {'converged':>10}")
    for r in rows:
        print(f"{r['sample']:>8} {r['pos']:>5} {r['n_pairs']:>9} {r['n_tail']:>7} {r['xi_h5']:>10.4f} "
              f"{r['xi_mle_refit']:>10.4f} {r['xi_pwm']:>10.4f} {str(r['sign_flip']):>10} "
              f"{str(r['mle_converged']):>10}")

    n_flip = sum(r["sign_flip"] for r in rows)
    print(f"\n{n_flip}/{len(rows)} positions disagree on sign(xi) between MLE and PWM")


if __name__ == "__main__":
    main()
