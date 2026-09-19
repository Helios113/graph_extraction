"""Estimate a co-Lipschitz constant via extreme value theory (peaks-over-threshold).

For a sublayer f, let g(x) = sigma_min(Df(x)) be the smallest singular value of
its Jacobian at input x, so the co-Lipschitz constant is c = inf_x g(x). Given
samples g_1, ..., g_N (here: every (sample, seq position) entry of
`smallest_singular_value` from `scripts/compute_base_jacobian_stats.py`'s
output), this fits a Generalized Pareto Distribution to the exceedances
z_i = u - g_i for g_i < u, for a lower threshold u (Peaks-Over-Threshold / POT
method, applied to the left tail of g by working with -g).

GPD CDF: F(z) = 1 - (1 + xi*z/sigma)^(-1/xi), fit with loc fixed at 0 (z is
already an exceedance above 0 by construction).

If the fitted shape xi < 0, the tail has a finite right endpoint at
sigma/(-xi), giving an estimate of the true infimum. Since exceedances are
z = u - g, that caps g from below at u - (that endpoint) = u - sigma/(-xi):

    c_hat = u + sigma / xi      (xi < 0, so sigma/xi < 0, i.e. c_hat < u)

A confidence interval for c_hat is obtained via the delta method: the GPD's
(xi, sigma) MLEs have an asymptotic covariance given by the inverse observed
Fisher information (the Hessian of the negative log-likelihood at the MLE);
propagating that through the map (xi, sigma) -> c_hat = u + sigma/xi gives
Var(c_hat) via its gradient, hence a normal-approximation CI. This is the
probabilistic statement: e.g. "P(true inf >= c_0) >= 0.95" for
c_0 = c_hat - z_{0.95} * se(c_hat).

Threshold selection: by default u is set to the --n-tail-min'th smallest
value of g, i.e. a fixed exceedance count. --threshold-quantile instead pins
u to a given quantile of g, and --threshold pins it to an exact value. There
is no automatic search (e.g. mean-residual-life) over candidate thresholds --
sensitivity to u should be checked by re-running with a few different
values.

Usage:
    uv run python -m scripts.estimate_colipschitz_gpd \\
        onnx_models/smoll/jacobian_stats
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import genpareto, norm

from execute.sharded_writer import load_shard_rows


def discover_layer_keys(jacobian_stats_dir: str) -> list[str]:
    """A layer's manifest lives at <out_dir>/<layer_name>.manifest.json
    (scripts.compute_base_jacobian_stats' ShardedLayerWriter output) --
    discover every layer under `jacobian_stats_dir` by globbing those."""
    manifests = sorted(glob.glob(os.path.join(jacobian_stats_dir, "*.manifest.json")))
    return [os.path.basename(p)[: -len(".manifest.json")] for p in manifests]


def load_smallest_singular_values(project_root: str, layer_key: str) -> np.ndarray:
    sv = load_shard_rows(project_root, layer_key, dataset_names=["smallest_singular_value"])[
        "smallest_singular_value"
    ]
    return sv.reshape(-1).astype(np.float64)


def gpd_negloglik_hessian(shape: float, scale: float, exceedances: np.ndarray) -> np.ndarray:
    """Observed Fisher information (Hessian of the negative log-likelihood)
    of the GPD(shape, scale, loc=0) at the given parameters, evaluated on
    `exceedances`. Returns the (2, 2) Hessian in (xi, sigma) order.

    Analytic second derivatives of the GPD log-density
        l = -log(sigma) - (1 + 1/xi) * log(1 + xi*z/sigma)
    (obtained symbolically), summed over exceedances and negated. Using the
    closed form rather than finite differences avoids evaluating l outside
    its support {1 + xi*z/sigma > 0}, which happens easily near xi ~ -1
    where that support is a tight interval.
    """
    xi, sigma = shape, scale
    z = exceedances
    s = sigma + xi * z  # = sigma * (1 + xi*z/sigma), assumed > 0 (within GPD support)

    d_xixi = (xi**2 * z**2 * (xi + 1) + 2 * xi * z * s - 2 * s**2 * np.log(s / sigma)) / (
        xi**3 * s**2
    )
    d_sigmasigma = (sigma**2 - 2 * sigma * z - xi * z**2) / (sigma**2 * (s**2))
    d_xisigma = z * (s - z * (xi + 1)) / (sigma * s**2)

    hessian = -np.array(
        [
            [d_xixi.sum(), d_xisigma.sum()],
            [d_xisigma.sum(), d_sigmasigma.sum()],
        ]
    )
    return hessian


def fit_gpd_tail(
    g: np.ndarray,
    threshold: float | None,
    threshold_quantile: float | None,
    n_tail_min: int,
) -> dict:
    """Fits a GPD to the exceedances z = u - g[g < u] and, if the fitted
    shape is negative, estimates the finite lower endpoint c_hat = u -
    sigma/xi along with its confidence interval via the delta method."""
    g = np.sort(g)
    n = g.size

    if threshold is not None:
        u = threshold
    elif threshold_quantile is not None:
        u = np.quantile(g, threshold_quantile)
    else:
        # default: threshold s.t. at least n_tail_min points fall below it
        u = g[min(n_tail_min, n - 1)]

    exceedances = u - g[g < u]
    n_exc = exceedances.size
    if n_exc < 10:
        raise ValueError(
            f"only {n_exc} exceedances below threshold u={u:.6g} (need >= 10); "
            "lower --threshold-quantile or raise --threshold"
        )

    shape, _loc, scale = genpareto.fit(exceedances, floc=0)

    result = {
        "n_samples": int(n),
        "threshold": float(u),
        "n_exceedances": int(n_exc),
        "shape_xi": float(shape),
        "scale_sigma": float(scale),
        "g_min_observed": float(g[0]),
    }

    if shape < 0:
        c_hat = u + scale / shape

        hessian = gpd_negloglik_hessian(shape, scale, exceedances)
        try:
            cov = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            cov = np.full((2, 2), np.nan)

        # gradient of c_hat = u + sigma/xi w.r.t. (xi, sigma)
        grad = np.array([-scale / shape**2, 1.0 / shape])
        var_c_hat = float(grad @ cov @ grad)

        result["c_hat"] = float(c_hat)
        result["cov_xi_sigma"] = cov.tolist()
        result["var_c_hat"] = var_c_hat

        if var_c_hat > 0 and np.isfinite(var_c_hat):
            se_c_hat = float(np.sqrt(var_c_hat))
            result["se_c_hat"] = se_c_hat
            for level in (0.90, 0.95, 0.99):
                z = norm.ppf(level)
                result[f"lower_bound_{level:.2f}"] = c_hat - z * se_c_hat
        else:
            result["se_c_hat"] = None
    else:
        result["c_hat"] = None
        result["note"] = (
            f"shape xi={shape:.4g} >= 0: tail is unbounded below (or heavy-tailed) "
            "under the fitted GPD -- no finite infimum estimate"
        )

    return result


def plot_sigma_min(g: np.ndarray, layer_name: str, result: dict, out_path: str) -> None:
    """Histogram of sigma_min (g) for one layer, with the fitted threshold u
    and, if available, the estimated infimum c_hat marked."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(g, bins=200, color="steelblue", alpha=0.8)
    ax.axvline(result["threshold"], color="orange", linestyle="--", label=f"threshold u={result['threshold']:.4g}")
    if result["c_hat"] is not None:
        ax.axvline(result["c_hat"], color="red", linestyle="--", label=f"c_hat={result['c_hat']:.4g}")
    ax.set_xlabel("sigma_min")
    ax.set_ylabel("count")
    ax.set_title(f"{layer_name}: sigma_min distribution (N={g.size})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "jacobian_stats_dir",
        type=str,
        help="Output directory from scripts.compute_base_jacobian_stats "
        "(one sharded output + manifest per layer), e.g. onnx_models/smoll/jacobian_stats",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed threshold u on g (smallest singular value). Overrides "
        "--threshold-quantile / --n-tail-min if given.",
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=None,
        help="Quantile of g to use as threshold u (e.g. 0.1 for the 10th percentile).",
    )
    parser.add_argument(
        "--n-tail-min",
        type=int,
        default=200,
        help="Default threshold selection: u is set so this many of the smallest "
        "g values are used as exceedances (default: 200).",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Optional path to write results as JSON (one entry per input file).",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Optional directory to save a per-layer sigma_min histogram PNG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    layer_keys = discover_layer_keys(args.jacobian_stats_dir)
    if not layer_keys:
        raise SystemExit(f"no *.manifest.json found under {args.jacobian_stats_dir!r}")

    all_results = {}
    for layer_name in layer_keys:
        layer_key = os.path.join(args.jacobian_stats_dir, layer_name)
        g = load_smallest_singular_values(".", layer_key)

        result = fit_gpd_tail(
            g,
            threshold=args.threshold,
            threshold_quantile=args.threshold_quantile,
            n_tail_min=args.n_tail_min,
        )
        all_results[layer_name] = result

        if args.plot_dir:
            os.makedirs(args.plot_dir, exist_ok=True)
            plot_path = os.path.join(args.plot_dir, f"{layer_name}.sigma_min.png")
            plot_sigma_min(g, layer_name, result, plot_path)
            print(f"saved plot to {plot_path}")

        print(f"\n=== {layer_name} ({layer_key}) ===")
        print(f"  N = {result['n_samples']} samples, observed min g = {result['g_min_observed']:.6g}")
        print(
            f"  threshold u = {result['threshold']:.6g} "
            f"({result['n_exceedances']} exceedances)"
        )
        print(f"  fitted GPD: xi = {result['shape_xi']:.4g}, sigma = {result['scale_sigma']:.4g}")
        if result["c_hat"] is not None:
            print(f"  c_hat (estimated inf g) = {result['c_hat']:.6g}")
            if result.get("se_c_hat") is not None:
                print(f"  se(c_hat) = {result['se_c_hat']:.4g}")
                for level in (0.90, 0.95, 0.99):
                    lb = result[f"lower_bound_{level:.2f}"]
                    print(f"  P(true inf >= {lb:.6g}) >= {level:.2f}")
            else:
                print("  se(c_hat) unavailable (singular Fisher information)")
        else:
            print(f"  {result['note']}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nsaved results to {args.out_json}")


if __name__ == "__main__":
    main()
