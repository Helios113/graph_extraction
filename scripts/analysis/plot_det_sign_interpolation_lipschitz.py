"""Lipschitz scatter/histogram on the interpolated points written by
`scripts/python/generate_det_sign_interpolation.py`.

For every pair (i, j) of the file's n interpolated points (every pair, not
a random subsample -- n is small -- except pairs closer than --min-dt in t,
whose ratio is dominated by output noise), computes the
same ratio the Lipschitz scripts use
(scripts/lipschitz/lipschitz_ratio_bootstrap_eqd_sweep_gpu.py's
compute_lipschitz_ratios_gpu):

    g = ||vec_out_i - vec_out_j|| / ||vec_in_i - vec_in_j||

and plots:
  - left: scatter of ||f(x_i) - f(x_j)|| against ||x_i - x_j|| for those
    pairs, with lines y = g_min x and y = g_max x (the observed lower/upper
    Lipschitz bounds) through the origin;
  - middle: histogram of g over all pairs at least --min-dt apart in t, with
    its min and max marked;
  - right: the local ratio over a --min-dt step along the path
    (||y(t + dt) - y(t)|| / ||x(t + dt) - x(t)||, i.e. the directional
    stretch along the path) against t, to show where along A -> B the ratio is
    smallest/largest -- endpoint A has det > 0, B has det < 0, so the path
    crosses a singular Jacobian somewhere.

Writes one PNG next to each input h5 ({stem}_lipschitz_hist.png) unless
--out is given (single input only), and prints ratio quantiles.

Usage:
    uv run python -m scripts.analysis.plot_det_sign_interpolation_lipschitz \\
        onnx_models/simple_stories/det_sign_interpolation/add_4.add_6/row26685_row26140.h5
"""
import argparse
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES = "#2a78d6"  # dataviz reference palette, categorical slot 1
ACCENT = "#eb6834"  # slot 2 -- min/max markers
INK = "#52514e"


def all_pair_dists(
    vec_in: np.ndarray, vec_out: np.ndarray, t: np.ndarray, min_dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(||x_i - x_j||, ||f(x_i) - f(x_j)||) for every i < j with
    |t_j - t_i| >= min_dt, float64. Row-by-row so memory stays O(n * d)."""
    x = vec_in.astype(np.float64)
    y = vec_out.astype(np.float64)
    in_dists, out_dists = [], []
    for i in range(len(x) - 1):
        j = i + 1 + np.flatnonzero(t[i + 1:] - t[i] >= min_dt - 1e-12)
        if j.size == 0:
            break
        in_dists.append(np.linalg.norm(x[j] - x[i], axis=1))
        out_dists.append(np.linalg.norm(y[j] - y[i], axis=1))
    return np.concatenate(in_dists), np.concatenate(out_dists)


def plot_file(path: str, out_path: str | None, bins: int, min_dt: float) -> str:
    with h5py.File(path, "r") as f:
        t = f["t"][:]
        vec_in = f["vec_in"][:]
        vec_out = f["vec_out"][:]
        attrs = dict(f.attrs)

    in_dist, out_dist = all_pair_dists(vec_in, vec_out, t, min_dt)
    g = out_dist / in_dist
    # local ratio over a stride of min_dt in t, the same noise floor as g
    stride = max(1, int(round(min_dt / (t[1] - t[0]))))
    x64, y64 = vec_in.astype(np.float64), vec_out.astype(np.float64)
    local = np.linalg.norm(y64[stride:] - y64[:-stride], axis=1) / np.linalg.norm(
        x64[stride:] - x64[:-stride], axis=1
    )
    t_mid = 0.5 * (t[stride:] + t[:-stride])

    q = np.quantile(g, [0.0, 0.01, 0.5, 0.99, 1.0])
    print(f"{path}: {g.size} pairs with |dt| >= {min_dt:g}, g min {q[0]:.4g}, 1% {q[1]:.4g}, median {q[2]:.4g}, "
          f"99% {q[3]:.4g}, max {q[4]:.4g}; local ratio min {local.min():.4g} at t={t_mid[local.argmin()]:.3f}, "
          f"max {local.max():.4g} at t={t_mid[local.argmax()]:.3f}")

    fig, (ax_scat, ax_hist, ax_path) = plt.subplots(1, 3, figsize=(19, 5.2))
    for ax in (ax_scat, ax_hist, ax_path):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK)

    ax_scat.scatter(in_dist, out_dist, s=1, color=SERIES, alpha=0.15, linewidths=0, rasterized=True)
    x_line = np.array([0.0, in_dist.max()])
    for val, name in ((q[0], "min"), (q[4], "max")):
        ax_scat.plot(x_line, val * x_line, color=ACCENT, linestyle="--", linewidth=1.5)
        # max label above the upper line's end; min label below the lower
        # line's midpoint, clear of the point cloud that sits above it
        x_lab = x_line[1] if name == "max" else 0.6 * x_line[1]
        ax_scat.annotate(f"slope {name} g = {val:.3g}", (x_lab, val * x_lab),
                         xytext=(-4, 6) if name == "max" else (6, -16), textcoords="offset points",
                         ha="right" if name == "max" else "left", color=INK, fontsize=9)
    ax_scat.set_xlim(left=0)
    ax_scat.set_ylim(bottom=0)
    ax_scat.set_xlabel("||x_i - x_j||  (input, at pos)")
    ax_scat.set_ylabel("||f(x_i) - f(x_j)||  (output, at pos)")
    ax_scat.set_title(f"Output vs input distance, all pairs |dt| >= {min_dt:g}", loc="left", fontsize=11)

    ax_hist.hist(g, bins=bins, color=SERIES, edgecolor="white", linewidth=0.5)
    for val, name in ((q[0], "min"), (q[4], "max")):
        ax_hist.axvline(val, color=ACCENT, linestyle="--", linewidth=1.5)
        ax_hist.annotate(f"{name} {val:.3g}", xy=(val, 1), xycoords=("data", "axes fraction"),
                         xytext=(4 if name == "min" else -4, -12), textcoords="offset points",
                         ha="left" if name == "min" else "right", color=INK, fontsize=9)
    ax_hist.set_xlabel("g = ||f(x_i) - f(x_j)|| / ||x_i - x_j||")
    ax_hist.set_ylabel("pairs")
    ax_hist.set_title(f"Ratio g = slope (N = {g.size:,}, median {q[2]:.3g})",
                      loc="left", fontsize=11)

    ax_path.plot(t_mid, local, color=SERIES, linewidth=2)
    k_min, k_max = local.argmin(), local.argmax()
    for k in (k_min, k_max):
        ax_path.plot(t_mid[k], local[k], "o", color=ACCENT, markersize=8,
                     markeredgecolor="white", markeredgewidth=2)
        ax_path.annotate(f"{local[k]:.3g} at t={t_mid[k]:.2f}", (t_mid[k], local[k]),
                         xytext=(10, -3) if k == k_min else (0, 9), textcoords="offset points",
                         ha="left" if k == k_min else "center", color=INK, fontsize=9)
    ax_path.set_xlabel(f"t  (A: det {'+' if attrs['a_sign'] > 0 else '-'} at t=0  ->  "
                       f"B: det {'+' if attrs['b_sign'] > 0 else '-'} at t=1)")
    ax_path.set_ylabel(f"||y(t+dt) - y(t)|| / ||x(t+dt) - x(t)||, dt = {stride * (t[1] - t[0]):.3g}")
    ax_path.set_title("Local ratio along the path", loc="left", fontsize=11)
    ax_path.margins(y=0.1)

    fig.suptitle(
        f"{attrs['sublayer_input']} -> {attrs['sublayer_output']}, sample {attrs['a_sample_idx']}, "
        f"pos {attrs['pos']}: row {attrs['a_row']} (vocab {attrs['a_vocab_id']}, "
        f"log|det| {attrs['a_log_abs_det']:.3g}) -> row {attrs['b_row']} "
        f"(vocab {attrs['b_vocab_id']}, log|det| {attrs['b_log_abs_det']:.3g})",
        x=0.01, ha="left", fontsize=11,
    )
    fig.tight_layout()

    out_path = out_path or os.path.splitext(path)[0] + "_lipschitz_hist.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("h5", nargs="+", help="generate_det_sign_interpolation.py output file(s)")
    parser.add_argument("--out", default=None, help="PNG path (single input only); default next to the h5")
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument(
        "--min-dt",
        type=float,
        default=0.01,
        help="Drop pairs closer than this in t (and use it as the local-ratio stride). "
        "Neighbouring points are only |x_B - x_A| / (n - 1) apart, so output noise makes their "
        "ratios noisy -- a few %% at n = 1000 for files generated with TF32 on (before "
        "generate_det_sign_interpolation.py disabled it), far less without; 0 keeps every pair",
    )
    args = parser.parse_args()
    if args.out and len(args.h5) > 1:
        parser.error("--out only works with a single input file")
    for path in args.h5:
        plot_file(path, args.out, args.bins, args.min_dt)


if __name__ == "__main__":
    main()
