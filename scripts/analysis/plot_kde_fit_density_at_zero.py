"""Loads one (sample, pos) ratio distribution from a
lipschitz_ratio_bootstrap_eqd_sweep_gpu.py (or CPU lipschitz_ratio_eqd_fit.py)
output h5, fits a Gaussian KDE to it, plots the fit over the histogram, and
evaluates the fitted density at g=0.

Motivation: the GPD tail fit already done by that sweep (see
scripts/analysis/lipschitz_ratio_eqd_fit.py) only models the lower tail
below a fitted threshold u. A KDE fit to the *whole* ratio distribution and
evaluated at zero gives a quick, assumption-light sanity check of how much
density a smooth estimator places at/near g=0 -- useful as a rough
cross-check against the GPD-based c_hat (estimated infimum of the ratio),
not a replacement for it.

Usage:
    uv run python -m scripts.analysis.plot_kde_fit_density_at_zero \\
        onnx_models/simple_stories/lipschitz_ratio_bootstrap_eqd_sweep_gpu/embedding.add_4 \\
        --sample-idx 10 --pos 0 --out kde_fit.png
"""

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def load_ratio_histogram(pair_dir: str, sample_idx: int, pos: int) -> tuple[np.ndarray, dict]:
    path = os.path.join(pair_dir, f"sample_{sample_idx:05d}.h5")
    grp_name = f"pos_{pos:04d}"
    with h5py.File(path, "r") as h5f:
        if grp_name not in h5f:
            raise KeyError(f"{grp_name!r} not found in {path}")
        grp = h5f[grp_name]
        ratio = grp["ratio"][:].astype(np.float64)
        meta = dict(h5f.attrs)
        meta.update(grp.attrs)
    return ratio, meta


def fit_kde(ratio: np.ndarray, bw_method: str | float | None) -> gaussian_kde:
    return gaussian_kde(ratio, bw_method=bw_method)


def plot_kde_fit(
    ratio: np.ndarray,
    kde: gaussian_kde,
    meta: dict,
    sample_idx: int,
    pos: int,
    out_path: str | None,
) -> float:
    density_at_zero = float(kde(0.0)[0])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratio, bins=100, density=True, color="steelblue", alpha=0.6, label="histogram")

    lo = min(0.0, ratio.min())
    xs = np.linspace(lo, ratio.max(), 1000)
    ax.plot(xs, kde(xs), color="red", lw=1.5, label="Gaussian KDE fit")

    ax.axvline(0.0, color="black", linestyle=":", lw=1)
    ax.plot([0.0], [density_at_zero], "o", color="black", zorder=5, label=f"density(0)={density_at_zero:.4g}")

    ax.set_xlabel("ratio g = ||f(x)-f(y)|| / ||x-y||")
    ax.set_ylabel("density")
    ax.legend()

    sublayer_input = meta.get("sublayer_input", "?")
    sublayer_output = meta.get("sublayer_output", "?")
    ax.set_title(f"{sublayer_input} -> {sublayer_output}: sample {sample_idx}, pos {pos} (N={ratio.size})")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"saved figure to {out_path}")
    else:
        plt.show()

    return density_at_zero


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "pair_dir", type=str,
        help="One sublayer-pair directory from the sweep's --out-dir, e.g. "
        "onnx_models/simple_stories/lipschitz_ratio_bootstrap_eqd_sweep_gpu/embedding.add_4",
    )
    parser.add_argument("--sample-idx", type=int, default=0, help="which sample_*.h5 to load")
    parser.add_argument("--pos", type=int, default=0, help="which sequence position to load")
    parser.add_argument(
        "--bw-method", type=str, default=None,
        help="scipy.stats.gaussian_kde bw_method override (e.g. a float scale factor, "
        "or 'scott'/'silverman'); default lets scipy pick (Scott's rule)",
    )
    parser.add_argument("--out", default=None, help="if set, saves the figure here instead of plt.show()")
    args = parser.parse_args()

    bw_method: str | float | None = args.bw_method
    if bw_method is not None:
        try:
            bw_method = float(bw_method)
        except ValueError:
            pass

    ratio, meta = load_ratio_histogram(args.pair_dir, args.sample_idx, args.pos)
    kde = fit_kde(ratio, bw_method)
    density_at_zero = plot_kde_fit(ratio, kde, meta, args.sample_idx, args.pos, args.out)

    print(f"N={ratio.size}  min(ratio)={ratio.min():.6g}  max(ratio)={ratio.max():.6g}")
    print(f"KDE density at g=0: {density_at_zero:.6g}")


if __name__ == "__main__":
    main()
