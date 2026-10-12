"""Bootstrap Expected Quantile Discrepancy threshold selection, with
double-bootstrap uncertainty propagation, for one (sample, pos) co-Lipschitz
ratio cloud -- an exact implementation of the two algorithms below, as
opposed to scripts/analysis/lipschitz_ratio_eqd_fit.py's deterministic
closed-form EQD approximation (single statistic per candidate threshold, no
resampling; see that script's module docstring and CLAUDE.md discussion for
why it trades bootstrap accuracy for sweep-wide speed).

1. Automated threshold selection (Algorithm 1, Bootstrap EQD):
   For each candidate threshold u (a grid of exceedance counts n_tail), draw
   B bootstrap resamples of the n_tail exceedances x_u, fit a GPD to each
   resample by MLE, and compute the mean absolute deviation between that
   fit's model quantiles and x_u^b's own empirical (linear-interpolated)
   quantiles over a grid of m plotting-position probabilities:

       d_b(u) = (1/m) sum_j | (sigma^b/xi^b) [(1-p_j)^(-xi^b) - 1]
                              - Q(p_j; x_u^b) |

   averaged over the B resamples to give d_hat_E(u) = mean_b d_b(u). The
   selected threshold u_hat minimizes d_hat_E(u) over the candidate grid.

2. Double-bootstrap uncertainty propagation (Algorithm 2) for c_hat, this
   codebase's co-Lipschitz lower-bound estimate (c_hat = u + sigma/xi when
   the fitted shape xi < 0, same quantity as
   scripts/analysis/lipschitz_ratio_eqd_fit.py's fit_gpd_at_threshold and
   scripts/python/estimate_colipschitz_gpd.py's fit_gpd_tail):
     - Outer loop (B2 iterations): resample n values from the *original*
       ratio cloud with replacement, capturing threshold-selection
       uncertainty.
     - Re-run Algorithm 1 on each outer resample to get a re-estimated
       threshold u_hat_b.
     - Inner loop (B1 iterations): parametric-bootstrap the GPD fit at
       u_hat_b's exceedances (draw B1 samples from the fitted GPD itself,
       refit each), capturing parameter estimation uncertainty.
     - Compute c_hat for all B1*B2 realizations; their empirical quantiles
       give a threshold-uncertainty-aware confidence interval for c_hat,
       instead of estimate_colipschitz_gpd.py's fixed-threshold delta-method
       CI.

This is a single-(sublayer pair, sample, pos) tool, not a full-sweep
replacement: the nested bootstrap (candidate grid x B, then B2 x (Algorithm 1
+ B1)) is orders of magnitude more expensive than
lipschitz_ratio_eqd_fit.py's closed-form sweep, so it's meant to be pointed
at specific cases of interest (e.g. the negative-c_hat / positive-xi
positions already flagged from that script's output) rather than run as a
full config sweep.

Usage (on mauao, via srun -- reads perturbed-sweep shards straight off local
scratch, same as lipschitz_ratio_eqd_fit.py -- see that script's module
docstring and CLAUDE.md's Hetzner-ban section for why no NFS/Hetzner pull
happens here):
    uv run python -m scripts.analysis.lipschitz_ratio_bootstrap_eqd_fit \\
        configs/gpt2.toml --perturbed-sweep-dir onnx_models/gpt2/perturbed_sweep \\
        --sublayer-input add_5 --sublayer-output add_6 \\
        --sample-idx 0 --pos 0 \\
        --out-json onnx_models/gpt2/lipschitz_ratio_bootstrap_eqd_fit/add_5.add_6.sample_00000.pos_0000.json
"""

import argparse
import json
import os

import h5py
import numpy as np
import tqdm
from scipy.stats import genpareto

from config import load_config
from execute.sharded_writer import _scratch_root
from scripts.analysis.lipschitz_ratio_eqd_fit import (
    compute_lipschitz_ratios,
    load_layer_manifest,
    load_row_range,
)


def sample_quantile_function(x_sorted: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Linear-interpolated sample quantile function Q(p; x) of the already-
    sorted 1-D sample x, evaluated at probabilities p, matching np.quantile's
    default (linear) interpolation -- used as Q(p_j; x_u^b, q) in d_b(u)."""
    return np.quantile(x_sorted, p, method="linear")


def gpd_model_quantile(p: np.ndarray, xi: float, sigma: float) -> np.ndarray:
    """GPD(xi, sigma, loc=0) quantile function (sigma/xi)[(1-p)^(-xi) - 1],
    xi -> 0 limit -sigma*log(1-p)."""
    if abs(xi) < 1e-10:
        return -sigma * np.log(1 - p)
    return (sigma / xi) * ((1 - p) ** (-xi) - 1)


def bootstrap_eqd_statistic(
    exceedances: np.ndarray, p_grid: np.ndarray, n_boot: int, rng: np.random.Generator,
) -> float:
    """d_hat_E(u): mean over n_boot non-parametric bootstrap resamples of
    exceedances of the mean absolute deviation between the resample's own
    fitted-GPD model quantiles and its own empirical quantile function
    (Algorithm 1's inner bootstrap goodness-of-fit metric, at one candidate
    threshold u whose exceedances are `exceedances`)."""
    n_exc = exceedances.size
    d_b = np.empty(n_boot)
    for b in range(n_boot):
        resample = rng.choice(exceedances, size=n_exc, replace=True)
        resample_sorted = np.sort(resample)
        try:
            xi_b, _loc_b, sigma_b = genpareto.fit(resample_sorted, floc=0)
        except Exception:
            d_b[b] = np.inf
            continue
        if not (sigma_b > 0) or not np.isfinite(xi_b):
            d_b[b] = np.inf
            continue
        model_q = gpd_model_quantile(p_grid, xi_b, sigma_b)
        sample_q = sample_quantile_function(resample_sorted, p_grid)
        d_b[b] = float(np.mean(np.abs(model_q - sample_q)))
    return float(np.mean(d_b))


def select_bootstrap_eqd_threshold(
    g_sorted: np.ndarray,
    n_tail_min: int,
    n_tail_max: int,
    n_tail_step: int,
    n_boot: int,
    m_plotting_points: int,
    rng: np.random.Generator,
    show_progress: bool = False,
) -> dict:
    """Algorithm 1: sweeps candidate thresholds u = g_sorted[n_tail] over
    n_tail in [n_tail_min, n_tail_max), computes the bootstrap EQD statistic
    d_hat_E(u) at each (bootstrap_eqd_statistic, B=n_boot resamples of that
    candidate's exceedances), and selects the n_tail minimizing it.

    g_sorted: ascending-sorted co-Lipschitz ratios (exceedances are the
    left-tail deficit threshold - g, same convention as
    lipschitz_ratio_eqd_fit.py/estimate_colipschitz_gpd.py)."""
    n_tail_max = min(n_tail_max, g_sorted.size - 1)
    n_tail_values = np.arange(n_tail_min, n_tail_max, n_tail_step)
    if n_tail_values.size == 0:
        raise ValueError(
            f"empty n_tail sweep range [{n_tail_min}, {n_tail_max}) for {g_sorted.size} points"
        )

    p_grid = np.arange(1, m_plotting_points + 1) / (m_plotting_points + 1)

    d_hat_e = np.empty(n_tail_values.size)
    iterator = tqdm.tqdm(n_tail_values, desc="bootstrap EQD sweep") if show_progress else n_tail_values
    for i, n_tail in enumerate(iterator):
        u = g_sorted[n_tail]
        exceedances = u - g_sorted[:n_tail]
        d_hat_e[i] = bootstrap_eqd_statistic(exceedances, p_grid, n_boot, rng)

    best_idx = int(np.argmin(d_hat_e))
    best_n_tail = int(n_tail_values[best_idx])
    threshold = float(g_sorted[best_n_tail])

    return {
        "n_tail_values": n_tail_values,
        "d_hat_e": d_hat_e,
        "best_n_tail": best_n_tail,
        "threshold": threshold,
        "best_d_hat_e": float(d_hat_e[best_idx]),
    }


def fit_gpd_and_c_hat(g_sorted: np.ndarray, n_tail: int, threshold: float) -> dict:
    """Exact MLE GPD fit to the n_tail smallest values of g (as exceedances
    below `threshold`), plus c_hat = threshold + sigma/xi when xi < 0 (see
    lipschitz_ratio_eqd_fit.py's fit_gpd_at_threshold for the same
    derivation)."""
    exceedances = threshold - g_sorted[:n_tail]
    shape, _loc, scale = genpareto.fit(exceedances, floc=0)
    c_hat = float(threshold + scale / shape) if shape < 0 else np.nan
    return {"shape_xi": float(shape), "scale_sigma": float(scale), "c_hat": c_hat}


def double_bootstrap_c_hat(
    g: np.ndarray,
    n_tail_min: int,
    n_tail_max: int,
    n_tail_step: int,
    eqd_n_boot: int,
    m_plotting_points: int,
    b1_inner: int,
    b2_outer: int,
    rng: np.random.Generator,
) -> dict:
    """Algorithm 2: propagates both threshold-selection uncertainty (outer
    loop, B2 resamples of the full ratio cloud g, each re-running Algorithm 1
    to get its own u_hat_b) and GPD parameter-estimation uncertainty (inner
    loop, B1 parametric bootstraps of the GPD fitted at that outer sample's
    selected threshold) into an empirical distribution of c_hat over
    B1*B2 realizations.

    g: the full (non-sorted) co-Lipschitz ratio sample for this (sample,
    pos) -- the "original dataset" Algorithm 2 resamples n values from with
    replacement in its outer loop."""
    n = g.size
    c_hats = []
    outer_thresholds = []
    outer_n_tails = []
    skipped_outer = 0

    for b2 in tqdm.tqdm(range(b2_outer), desc="double-bootstrap outer (B2)"):
        outer_resample = np.sort(rng.choice(g, size=n, replace=True))

        try:
            eqd_result_b = select_bootstrap_eqd_threshold(
                outer_resample,
                n_tail_min=n_tail_min,
                n_tail_max=n_tail_max,
                n_tail_step=n_tail_step,
                n_boot=eqd_n_boot,
                m_plotting_points=m_plotting_points,
                rng=rng,
            )
        except ValueError:
            skipped_outer += 1
            continue

        u_hat_b = eqd_result_b["threshold"]
        n_tail_b = eqd_result_b["best_n_tail"]
        outer_thresholds.append(u_hat_b)
        outer_n_tails.append(n_tail_b)

        exceedances_b = u_hat_b - outer_resample[:n_tail_b]
        try:
            xi_b, _loc_b, sigma_b = genpareto.fit(exceedances_b, floc=0)
        except Exception:
            continue
        if not (sigma_b > 0) or not np.isfinite(xi_b):
            continue

        # Inner loop: parametric bootstrap -- draw from the FITTED GPD
        # itself (not a resample of exceedances_b), refit, recompute c_hat.
        n_exc_b = exceedances_b.size
        for _b1 in range(b1_inner):
            parametric_sample = genpareto.rvs(
                xi_b, loc=0, scale=sigma_b, size=n_exc_b, random_state=rng,
            )
            try:
                xi_1, _loc_1, sigma_1 = genpareto.fit(parametric_sample, floc=0)
            except Exception:
                continue
            if xi_1 < 0:
                c_hats.append(float(u_hat_b + sigma_1 / xi_1))
            else:
                c_hats.append(np.nan)

    c_hats = np.array(c_hats, dtype=np.float64)
    finite_c_hats = c_hats[np.isfinite(c_hats)]

    result = {
        "n_realizations": int(c_hats.size),
        "n_finite_c_hat": int(finite_c_hats.size),
        "skipped_outer_iterations": skipped_outer,
        "outer_threshold_mean": float(np.mean(outer_thresholds)) if outer_thresholds else None,
        "outer_threshold_std": float(np.std(outer_thresholds)) if outer_thresholds else None,
        "outer_n_tail_mean": float(np.mean(outer_n_tails)) if outer_n_tails else None,
        "outer_n_tail_std": float(np.std(outer_n_tails)) if outer_n_tails else None,
        "c_hat_samples": c_hats,
    }

    if finite_c_hats.size:
        result["c_hat_mean"] = float(np.mean(finite_c_hats))
        result["c_hat_median"] = float(np.median(finite_c_hats))
        result["c_hat_std"] = float(np.std(finite_c_hats))
        for level in (0.90, 0.95, 0.99):
            alpha = 1 - level
            lo, hi = np.quantile(finite_c_hats, [alpha / 2, 1 - alpha / 2])
            result[f"ci_{level:.2f}"] = (float(lo), float(hi))
    else:
        result["c_hat_mean"] = None
        result["note"] = "no finite c_hat realizations (all inner/outer fits had shape xi >= 0)"

    return result


def load_ratio_cloud(
    perturbed_sweep_dir: str,
    sublayer_input: str,
    sublayer_output: str,
    sample_idx: int,
    pos: int,
    n_pairs: int,
    seed: int,
) -> np.ndarray:
    """Loads this (sample, pos)'s full vocab-sweep input/output vec rows
    directly from local scratch (same path lipschitz_ratio_eqd_fit.py's
    run_sample uses) and recomputes the full-resolution ratio array -- not
    the n_pairs_save-sized subsample already written to that script's h5
    output, since bootstrapping needs the full --n-pairs density."""
    in_layer_key = os.path.join(perturbed_sweep_dir, sublayer_input)
    out_layer_key = os.path.join(perturbed_sweep_dir, sublayer_output)
    in_manifest = load_layer_manifest(".", in_layer_key)
    out_manifest = load_layer_manifest(".", out_layer_key)

    n_samples_total = in_manifest["attrs"]["n_samples"]
    seq_len = in_manifest["attrs"]["seq"]
    n_rows_total = in_manifest["n_rows"]
    vocab_size = n_rows_total // (n_samples_total * seq_len)

    rows_per_sample = seq_len * vocab_size
    row_start = sample_idx * rows_per_sample
    row_end = (sample_idx + 1) * rows_per_sample

    in_vec = load_row_range(in_layer_key, in_manifest, row_start, row_end, ["vec"])["vec"]
    out_vec = load_row_range(out_layer_key, out_manifest, row_start, row_end, ["vec"])["vec"]

    in_vec = in_vec.reshape(seq_len, vocab_size, -1)[pos]
    out_vec = out_vec.reshape(seq_len, vocab_size, -1)[pos]

    rng = np.random.default_rng(seed + sample_idx)
    _input_dist, _output_dist, ratio = compute_lipschitz_ratios(in_vec, out_vec, n_pairs=n_pairs, rng=rng)
    return ratio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=str, help="Path to a toml ModelConfig")
    parser.add_argument(
        "--perturbed-sweep-dir", type=str, default="onnx_models/smoll/perturbed_sweep",
        help="Output directory produced by scripts/python/generate_perturbed_sweep.py",
    )
    parser.add_argument("--sublayer-input", type=str, required=True)
    parser.add_argument("--sublayer-output", type=str, required=True)
    parser.add_argument("--sample-idx", type=int, required=True)
    parser.add_argument("--pos", type=int, required=True)
    parser.add_argument(
        "--n-pairs", type=int, default=200_000,
        help="Number of random point pairs to sample for the ratio cloud (same meaning as "
        "lipschitz_ratio_eqd_fit.py's --n-pairs)",
    )
    parser.add_argument("--n-tail-min", type=int, default=20, help="Smallest candidate n_tail")
    parser.add_argument("--n-tail-max", type=int, default=2_000, help="Largest candidate n_tail (exclusive)")
    parser.add_argument("--n-tail-step", type=int, default=50, help="Step size in the candidate n_tail grid")
    parser.add_argument(
        "--eqd-n-boot", type=int, default=100,
        help="B: number of bootstrap resamples per candidate threshold in Algorithm 1's "
        "bootstrap EQD statistic",
    )
    parser.add_argument(
        "--m-plotting-points", type=int, default=50,
        help="m: number of equally-spaced plotting-position probabilities used in the "
        "mean absolute deviation d_b(u)",
    )
    parser.add_argument(
        "--b1-inner", type=int, default=200,
        help="B1: parametric GPD bootstrap iterations per outer resample in Algorithm 2's "
        "inner loop (parameter estimation uncertainty)",
    )
    parser.add_argument(
        "--b2-outer", type=int, default=200,
        help="B2: resamples of the full ratio cloud in Algorithm 2's outer loop "
        "(threshold-selection uncertainty)",
    )
    parser.add_argument(
        "--skip-double-bootstrap", action="store_true",
        help="Only run Algorithm 1 (single threshold selection on the observed data) and "
        "skip Algorithm 2 -- useful for a quick look before committing to the B1*B2 cost",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-json", type=str, default=None,
        help="Path to write results as JSON (arrays included; c_hat_samples lets you "
        "re-derive any other quantile/CI without re-running)",
    )
    args = parser.parse_args()

    load_config(args.config)  # validates the config path; sublayer names come from the CLI

    print(
        f"loading ratio cloud: {args.sublayer_input} -> {args.sublayer_output}, "
        f"sample {args.sample_idx}, pos {args.pos}, n_pairs={args.n_pairs}"
    )
    rng = np.random.default_rng(args.seed)
    ratio = load_ratio_cloud(
        args.perturbed_sweep_dir,
        args.sublayer_input,
        args.sublayer_output,
        args.sample_idx,
        args.pos,
        n_pairs=args.n_pairs,
        seed=args.seed,
    )
    g_sorted = np.sort(ratio)
    print(f"loaded {g_sorted.size} ratio pairs, min={g_sorted[0]:.6g}, max={g_sorted[-1]:.6g}")

    print("\n=== Algorithm 1: bootstrap EQD threshold selection ===")
    eqd_result = select_bootstrap_eqd_threshold(
        g_sorted,
        n_tail_min=args.n_tail_min,
        n_tail_max=args.n_tail_max,
        n_tail_step=args.n_tail_step,
        n_boot=args.eqd_n_boot,
        m_plotting_points=args.m_plotting_points,
        rng=rng,
        show_progress=True,
    )
    print(
        f"selected threshold u_hat={eqd_result['threshold']:.6g} "
        f"(n_tail={eqd_result['best_n_tail']}, d_hat_E={eqd_result['best_d_hat_e']:.6g})"
    )

    gpd_result = fit_gpd_and_c_hat(g_sorted, eqd_result["best_n_tail"], eqd_result["threshold"])
    print(f"GPD fit at u_hat: xi={gpd_result['shape_xi']:.4g}, sigma={gpd_result['scale_sigma']:.4g}")
    if np.isfinite(gpd_result["c_hat"]):
        print(f"c_hat (fixed-threshold point estimate) = {gpd_result['c_hat']:.6g}")
    else:
        print("c_hat undefined at u_hat (shape xi >= 0)")

    out = {
        "sublayer_input": args.sublayer_input,
        "sublayer_output": args.sublayer_output,
        "sample_idx": args.sample_idx,
        "pos": args.pos,
        "n_pairs": args.n_pairs,
        "eqd_n_tail_values": eqd_result["n_tail_values"].tolist(),
        "eqd_d_hat_e": eqd_result["d_hat_e"].tolist(),
        "eqd_best_n_tail": eqd_result["best_n_tail"],
        "threshold": eqd_result["threshold"],
        "eqd_best_d_hat_e": eqd_result["best_d_hat_e"],
        "shape_xi": gpd_result["shape_xi"],
        "scale_sigma": gpd_result["scale_sigma"],
        "c_hat_point_estimate": gpd_result["c_hat"],
        "g_min_observed": float(g_sorted[0]),
    }

    if not args.skip_double_bootstrap:
        print(f"\n=== Algorithm 2: double-bootstrap (B1={args.b1_inner}, B2={args.b2_outer}) ===")
        db_result = double_bootstrap_c_hat(
            ratio,
            n_tail_min=args.n_tail_min,
            n_tail_max=args.n_tail_max,
            n_tail_step=args.n_tail_step,
            eqd_n_boot=args.eqd_n_boot,
            m_plotting_points=args.m_plotting_points,
            b1_inner=args.b1_inner,
            b2_outer=args.b2_outer,
            rng=rng,
        )
        print(
            f"{db_result['n_finite_c_hat']}/{db_result['n_realizations']} realizations had "
            f"finite c_hat ({db_result['skipped_outer_iterations']} outer iterations skipped: "
            "empty n_tail sweep range for that resample)"
        )
        if db_result["c_hat_mean"] is not None:
            print(f"c_hat: mean={db_result['c_hat_mean']:.6g}, median={db_result['c_hat_median']:.6g}, "
                  f"std={db_result['c_hat_std']:.6g}")
            for level in (0.90, 0.95, 0.99):
                lo, hi = db_result[f"ci_{level:.2f}"]
                print(f"  {level:.0%} CI: [{lo:.6g}, {hi:.6g}]")
            print(f"threshold-selection spread across outer resamples: "
                  f"u_hat mean={db_result['outer_threshold_mean']:.6g}, "
                  f"std={db_result['outer_threshold_std']:.6g}")
        else:
            print(f"  {db_result['note']}")

        out["double_bootstrap"] = {k: v for k, v in db_result.items() if k != "c_hat_samples"}
        out["double_bootstrap"]["c_hat_samples"] = db_result["c_hat_samples"].tolist()

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nsaved results to {args.out_json}")


if __name__ == "__main__":
    main()
