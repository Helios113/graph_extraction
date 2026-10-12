"""Plots for scripts/analysis/lipschitz_ratio_eqd_fit.py's output: one h5 per
(sublayer pair, sample), each holding one group per sequence position with a
ratio-scatter subsample, the EQD threshold-sweep curve, and the fitted
GPD/threshold/c_hat scalars (see that script's module docstring for the field
definitions).

Two modes:

  --mode detail (default): for one --sample-idx/--pos, plots the ratio
    histogram (with fitted threshold u and c_hat marked, mirroring
    estimate_colipschitz_gpd.py's plot_sigma_min) side by side with the EQD
    n_tail sweep curve that picked that threshold.

  --mode summary: aggregates c_hat/shape_xi/threshold across every
    (sample, pos) found in the directory into histograms, so you can see the
    spread of the fitted co-Lipschitz lower bound over the whole sweep at a
    glance.

Usage:
  uv run python -m scripts.python.plot_lipschitz_ratio_eqd_fit \\
      onnx_models/simple_stories/lipschitz_ratio_eqd_fit/add_4.add_6 \\
      --mode detail --sample-idx 0 --pos 0 --out detail.png

  uv run python -m scripts.python.plot_lipschitz_ratio_eqd_fit \\
      onnx_models/simple_stories/lipschitz_ratio_eqd_fit/add_4.add_6 \\
      --mode summary --out summary.png
"""

import argparse
import glob
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_position(pair_dir: str, sample_idx: int, pos: int) -> tuple[h5py.File, str, str]:
    path = os.path.join(pair_dir, f"sample_{sample_idx:05d}.h5")
    h5f = h5py.File(path, "r")
    grp_name = f"pos_{pos:04d}"
    if grp_name not in h5f:
        h5f.close()
        raise KeyError(f"{grp_name!r} not found in {path}")
    return h5f, path, grp_name


def plot_detail(pair_dir: str, sample_idx: int, pos: int, out_path: str | None) -> None:
    h5f, path, grp_name = load_position(pair_dir, sample_idx, pos)
    with h5f:
        grp = h5f[grp_name]
        ratio = grp["ratio"][:]

        fig, (ax_hist, ax_eqd) = plt.subplots(1, 2, figsize=(12, 5))

        ax_hist.hist(ratio, bins=100, color="steelblue", alpha=0.8)
        if "threshold" in grp.attrs:
            # threshold/c_hat both describe the LEFT tail (GPD fit to the
            # n_tail smallest ratios below threshold u) -- they mark small-g
            # territory, not a central cutoff, even though u often falls
            # near the bulk for a tight ratio distribution like this one.
            ax_hist.axvline(
                grp.attrs["threshold"], color="orange", linestyle="--",
                label=f"lower-tail threshold u={grp.attrs['threshold']:.4g}",
            )
            c_hat = grp.attrs["c_hat"]
            if np.isfinite(c_hat):
                ax_hist.axvline(c_hat, color="red", linestyle="--", label=f"c_hat (est. infimum)={c_hat:.4g}")
            ax_hist.legend()
        ax_hist.set_xlabel("ratio g = ||f(x)-f(y)|| / ||x-y||")
        ax_hist.set_ylabel("count")
        ax_hist.set_title(f"ratio distribution (N={ratio.size})")

        if "eqd_sweep" in grp:
            n_tail_values = grp["eqd_n_tail_values"][:]
            eqd_sweep = grp["eqd_sweep"][:]
            ax_eqd.plot(n_tail_values, eqd_sweep, color="steelblue", lw=1.5)
            ax_eqd.axvline(
                grp.attrs["eqd_best_n_tail"], color="orange", linestyle="--",
                label=f"best n_tail={int(grp.attrs['eqd_best_n_tail'])}",
            )
            ax_eqd.legend()
        else:
            ax_eqd.text(0.5, 0.5, grp.attrs.get("fit_error", "no fit"), ha="center", va="center")
        ax_eqd.set_xlabel("n_tail")
        ax_eqd.set_ylabel("EQD statistic")
        ax_eqd.set_title("EQD threshold sweep")

        sublayer_input = h5f.attrs["sublayer_input"]
        sublayer_output = h5f.attrs["sublayer_output"]
        fig.suptitle(f"{sublayer_input} -> {sublayer_output}: sample {sample_idx}, pos {pos}")
        fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"saved figure to {out_path}")
    else:
        plt.show()


def load_summary(pair_dir: str) -> dict[str, np.ndarray]:
    """Collects scalar fit results (c_hat, shape_xi, threshold, g_min_observed)
    across every position in every sample_*.h5 file in pair_dir."""
    fields = ["c_hat", "shape_xi", "threshold", "g_min_observed"]
    out: dict[str, list[float]] = {f: [] for f in fields}

    paths = sorted(glob.glob(os.path.join(pair_dir, "sample_*.h5")))
    if not paths:
        raise FileNotFoundError(f"no sample_*.h5 files found under {pair_dir}")

    for path in paths:
        with h5py.File(path, "r") as h5f:
            for grp_name in h5f:
                grp = h5f[grp_name]
                if "eqd_sweep" not in grp:
                    continue  # fit_error case, skipped
                for f in fields:
                    out[f].append(grp.attrs[f])

    return {f: np.array(v, dtype=np.float64) for f, v in out.items()}


def plot_summary(pair_dir: str, out_path: str | None) -> None:
    stats = load_summary(pair_dir)
    n = stats["threshold"].size

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    finite_c_hat = stats["c_hat"][np.isfinite(stats["c_hat"])]
    ax = axes[0, 0]
    if finite_c_hat.size:
        # A handful of near-zero-shape fits blow up to large-magnitude c_hat
        # outliers (c_hat = threshold + scale/shape diverges as shape -> 0),
        # which would otherwise crush the informative range into one bin.
        lo, hi = np.quantile(finite_c_hat, [0.01, 0.99])
        n_clipped = int(np.sum((finite_c_hat < lo) | (finite_c_hat > hi)))
        ax.hist(finite_c_hat, bins=60, range=(lo, hi), color="steelblue", alpha=0.8)
        if n_clipped:
            ax.text(
                0.98, 0.95, f"{n_clipped} outside [1st, 99th] pct range",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color="gray",
            )
    ax.set_xlabel("c_hat (estimated co-Lipschitz lower bound)")
    ax.set_ylabel("count")
    ax.set_title(f"c_hat across positions (finite: {finite_c_hat.size}/{n})")

    ax = axes[0, 1]
    ax.hist(stats["shape_xi"], bins=60, color="steelblue", alpha=0.8)
    ax.axvline(0, color="orange", linestyle="--", label="xi=0")
    ax.set_xlabel("shape xi")
    ax.set_ylabel("count")
    ax.set_title("GPD shape parameter across positions")
    ax.legend()

    ax = axes[1, 0]
    ax.hist(stats["threshold"], bins=60, color="steelblue", alpha=0.8)
    ax.set_xlabel("EQD-selected threshold u")
    ax.set_ylabel("count")
    ax.set_title("Selected threshold across positions")

    ax = axes[1, 1]
    ax.hist(stats["g_min_observed"], bins=60, color="steelblue", alpha=0.8)
    ax.set_xlabel("min observed ratio")
    ax.set_ylabel("count")
    ax.set_title("Smallest observed ratio across positions")

    pair_name = os.path.basename(pair_dir.rstrip("/"))
    fig.suptitle(f"{pair_name}: fit summary over {n} (sample, pos) positions")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"saved figure to {out_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "pair_dir", type=str,
        help="One sublayer-pair directory from scripts/analysis/lipschitz_ratio_eqd_fit.py's "
        "--out-dir, e.g. onnx_models/simple_stories/lipschitz_ratio_eqd_fit/add_4.add_6",
    )
    parser.add_argument("--mode", choices=["detail", "summary"], default="detail")
    parser.add_argument("--sample-idx", type=int, default=0, help="[detail mode] which sample_*.h5 to plot")
    parser.add_argument("--pos", type=int, default=0, help="[detail mode] which sequence position to plot")
    parser.add_argument("--out", default=None, help="if set, saves the figure here instead of plt.show()")
    args = parser.parse_args()

    if args.mode == "detail":
        plot_detail(args.pair_dir, args.sample_idx, args.pos, args.out)
    else:
        plot_summary(args.pair_dir, args.out)


if __name__ == "__main__":
    main()
