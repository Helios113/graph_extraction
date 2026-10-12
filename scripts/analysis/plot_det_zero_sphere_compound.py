"""Compound plot: points along the critical direction v_min vs points along a
control direction, on shared axes, for a `scripts/python/generate_det_zero_sphere.py`
file.

v_min is the right singular vector of the smallest singular value of the
sublayer's Jacobian J at the det = 0 centre c (the direction the layer
collapses); the control is random points in the ball around c, each with
its own random ambient direction (see generate_det_zero_sphere.py): points
uniform in the ball of radius r, plus points at the v_min line's log-spaced
radii (--line-min-s to r) with a fresh random direction each.

Plots (2 x 3, last slot empty):
  - scatter of ||f(x) - f(c)|| against ||x - c|| for every point on both
    lines (uniform + log-spaced), log-log, with a k s^2 guide through the
    v_min line's far end;
  - histogram of the Lipschitz ratio g = ||f(x_i) - f(x_j)|| / ||x_i - x_j||
    over every pair of points within each set (the v_min uniform line; the
    points uniform in the ball), both on
    shared linear axes. The
    log-spaced points are left out: their near-coincident pairs near c are
    output-noise dominated.
  - the log-spaced v_min points again, with noise-tilted versions c + s v_min
    + eps |s| u (u random orthogonal to v_min, one set per --noise-levels eps,
    light to dark orange), and dotted linear-leak lines g_perp eps s, g_perp
    = the random ball points' median centre-to-point ratio;
  - heatmap of log10 ||f(x) - f(c)|| over the (v_min, v_2) plane grid, with
    the det J = 0 contour where it crosses the plane;
  - histogram of the centre-to-point ratio ||f(x) - f(c)|| / ||x - c|| over
    every point on both lines (incl. log-spaced).
Points and pairs whose output change is below float32 precision are removed
from every panel (plane pixels blanked): the threshold is NOISE_BAND x the
float32 output floor, estimated by plot_det_zero_sphere_lipschitz.fp32_threshold
from the v_min points; the removed band is shaded grey on the log-log scatters.
Green = v_min, violet = random points in the ball. The last two panels are hidden for files
made before the noisy / plane samples were added.

Writes {stem}_compound.png next to each input h5 (or --out).

Usage:
    uv run python -m scripts.analysis.plot_det_zero_sphere_compound \\
        onnx_models/simple_stories/det_zero_sphere/add_4.add_6/row26685_row26140_sphere.h5
"""
import argparse
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.analysis.plot_det_zero_sphere_lipschitz import (
    GRID,
    REF,
    SING_COLOR,
    _style,
    all_pair_dists,
    fp32_threshold,
)

CTRL_COLOR = "#4a3aa7"  # dataviz slot 7 -- control direction (slots 1/2 mean det sign in the main plot)
SERIES = (
    ("singular", SING_COLOR, "along v_min (critical direction)"),
    ("control", CTRL_COLOR, "random points in the ball"),
)


def plot_file(path: str, out_path: str | None, bins: int) -> str:
    with h5py.File(path, "r") as f:
        if "line_control" not in f:
            raise ValueError(f"{path} has no control-direction line -- regenerate it with "
                             f"scripts/python/generate_det_zero_sphere.py")
        center_in, center_out = f["center_in"][:], f["center_out"][:]
        svals = f["singular_values"][:]
        attrs = dict(f.attrs)
        lines = {
            name: (f[f"line_{name}/vec_in"][:], f[f"line_{name}/vec_out"][:])
            for name in ("singular", "control", "singular_log", "control_log")
            if f"line_{name}" in f
        }
        noisy = {k: f[f"noisy/{k}"][:] for k in ("eps", "vec_in", "vec_out")} if "noisy" in f else None
        plane = {k: f[f"plane/{k}"][:] for k in ("ab", "vec_out", "sign")} if "plane" in f else None
    r = attrs["radius"]

    fig, axes = plt.subplots(2, 3, figsize=(22, 11))
    (ax_dist, ax_hist, ax_ratio), (ax_noisy, ax_plane, ax_unused) = axes
    ax_unused.set_visible(False)
    for ax in axes.flat:
        _style(ax)

    # distance from the centre for every point on each line (uniform + log-spaced)
    dist = {}
    for name, _, _ in SERIES:
        parts = [lines[name]] + ([lines[f"{name}_log"]] if f"{name}_log" in lines else [])
        d_in = np.linalg.norm(np.concatenate([p[0] for p in parts]).astype(np.float64) - center_in, axis=1)
        d_out = np.linalg.norm(np.concatenate([p[1] for p in parts]).astype(np.float64) - center_out, axis=1)
        keep = (d_out > 0) & (d_in > 0)  # log axes; drops the s = 0 sample
        # the uniform line's end point (|s| = r) -- the log-spaced line may reach past r
        far = np.argmin(np.abs(d_in - r))
        dist[name] = (d_in[keep], d_out[keep], d_out[far] / d_in[far])
    # sigma_min * s is far below the float32 output floor, so the guide is the
    # second-order k s^2 through the v_min line's point at |s| = r. Points
    # (and pairs) whose output change is below the float32 threshold are
    # dropped everywhere below; the grey band marks where they were
    k2, floor, band_top, n_floor = fp32_threshold(dist["singular"][0], dist["singular"][1], r)
    print(f"{path}: float32 output floor ~{floor:.2g} ({n_floor} rounding-dominated v_min points); "
          f"dropping output changes < {band_top:.2g}")
    removed = {}
    for name in dist:
        d_in, d_out, at_r = dist[name]
        above = d_out >= band_top
        removed[name] = int((~above).sum())
        dist[name] = (d_in[above], d_out[above], at_r)
    x_ref = np.geomspace(dist["singular"][0].min(), dist["singular"][0].max(), 50)

    def shade_floor(ax) -> None:
        if floor > 0:
            ax.axhspan(ax.get_ylim()[0] if ax.get_yscale() == "log" else 0, band_top,
                       color=GRID, alpha=0.7, zorder=0, linewidth=0,
                       label=f"below float32 precision, removed (< {band_top:.2g})")

    for name, color, label in SERIES:
        d_in, d_out, at_r = dist[name]
        ax_dist.scatter(d_in, d_out, s=8, color=color, linewidths=0,
                        label=f"{label}; {at_r:.3g} x at r ({removed[name]} pts removed)")
    ax_dist.plot(x_ref, k2 * x_ref**2, ":", color=REF, linewidth=1.2,
                 label=f"quadratic guide {k2:.2g} s^2, fit at r (sigma_min = {svals[-1]:.1e})")
    ax_dist.set_xscale("log")
    ax_dist.set_yscale("log")
    shade_floor(ax_dist)
    ax_dist.grid(axis="x", color=GRID, linewidth=0.8)
    ax_dist.legend(frameon=False, fontsize=9, loc="upper left")
    ax_dist.set_xlabel("||x - c||  (input distance from the det = 0 centre)")
    ax_dist.set_ylabel("||f(x) - f(c)||  (output distance)")
    ax_dist.set_title("Input vs output distance from the centre", loc="left", fontsize=11)

    # pairwise ratios, dropping pairs whose output difference is below float32 precision
    g_lines, pairs_removed = {}, {}
    for name, _, _ in SERIES:
        pin, pout = all_pair_dists(*lines[name])
        ok = pout >= band_top
        g_lines[name], pairs_removed[name] = pout[ok] / pin[ok], int((~ok).sum())
    g_all = np.concatenate(list(g_lines.values()))
    edges = np.linspace(0.0, g_all.max(), 3 * bins + 1)
    for name, color, label in SERIES:
        gl = g_lines[name]
        ql = np.quantile(gl, [0.0, 0.5, 1.0])
        print(f"{path}: {name} line, {gl.size} pairs, g min {ql[0]:.3g}, median {ql[1]:.3g}, max {ql[2]:.3g}")
        ax_hist.hist(gl, bins=edges, histtype="stepfilled", color=color, alpha=0.8,
                     label=f"{label} (median {ql[1]:.3g}, range {ql[0]:.2g}-{ql[2]:.3g}; "
                           f"{pairs_removed[name]} pairs below float32 removed)")
    ax_hist.legend(frameon=False, fontsize=9, loc="upper left")
    ax_hist.set_xlabel("g = ||f(x_i) - f(x_j)|| / ||x_i - x_j||,  x_i, x_j in the same set")
    ax_hist.set_ylabel("pairs")
    ax_hist.set_title(f"Pairwise Lipschitz ratio (v_min line; uniform in the ball; {len(lines['singular'][0])} points each)",
                      loc="left", fontsize=11)

    # centre-to-point ratio ||f(x) - f(c)|| / ||x - c|| over every point on
    # both lines (uniform + log-spaced) above float32 precision
    ratios = {name: dist[name][1] / dist[name][0] for name, _, _ in SERIES}
    r_all = np.concatenate(list(ratios.values()))
    r_edges = np.geomspace(r_all.min(), r_all.max(), bins + 1)
    for name, color, label in SERIES:
        short = "v_min" if name == "singular" else "ball"
        ax_ratio.hist(ratios[name], bins=r_edges, histtype="step", color=color, linewidth=2,
                      label=f"{short} ({ratios[name].size} pts, median {np.median(ratios[name]):.3g}; "
                            f"{removed[name]} below float32 removed)")
    ax_ratio.set_xscale("log")
    ax_ratio.set_yscale("log")  # the large-s spike near 1 dwarfs the rest
    lo, hi = ax_ratio.get_ylim()
    ax_ratio.set_ylim(lo, hi * 30)  # headroom so the legend sits above the bars
    ax_ratio.legend(frameon=False, fontsize=9, loc="upper left")
    ax_ratio.set_xlabel("||f(x) - f(c)|| / ||x - c||  (centre-to-point ratio, all points incl. log-spaced)")
    ax_ratio.set_ylabel("points")
    ax_ratio.set_title("Centre-to-point ratio (below float32 precision removed)", loc="left", fontsize=11)

    # noisy kernel direction: pure v_min vs v_min tilted by ~eps rad
    # (c + s v_min + eps |s| u, u random orthogonal to v_min per point)
    if noisy is not None:
        def dists(v_in, v_out):
            d_in = np.linalg.norm(v_in.astype(np.float64) - center_in, axis=1)
            d_out = np.linalg.norm(v_out.astype(np.float64) - center_out, axis=1)
            keep = (d_in > 0) & (d_out >= band_top)  # drop points below float32 precision
            return d_in[keep], d_out[keep]

        ax_noisy.scatter(*dists(*lines["singular_log"]), s=8, color=SING_COLOR, linewidths=0,
                         label="along v_min, no noise")
        noise_colors = plt.get_cmap("Oranges")(np.linspace(0.45, 0.95, len(noisy["eps"])))
        # a tilt of eps leaks a linear term ~ g_perp eps s, g_perp = the median
        # centre-to-point ratio of the random ball points (a generic direction's stretch)
        c_in, c_out = lines["control"]
        g_perp = float(np.median(np.linalg.norm(c_out.astype(np.float64) - center_out, axis=1)
                                 / np.linalg.norm(c_in.astype(np.float64) - center_in, axis=1)))
        for eps, v_in, v_out, color in zip(noisy["eps"], noisy["vec_in"], noisy["vec_out"], noise_colors):
            d_in, d_out = dists(v_in, v_out)
            ax_noisy.scatter(d_in, d_out, s=8, color=color, linewidths=0,
                             label=f"v_min + noise, eps = {eps:g} (tilt ~{eps:g} rad)")
            x_lin = np.geomspace(d_in.min(), d_in.max(), 50)
            ax_noisy.plot(x_lin, g_perp * eps * x_lin, ":", color=color, linewidth=1.2)
        ax_noisy.plot([], [], ":", color=REF, label=f"linear leak {g_perp:.2g} x eps x s (dotted, per eps)")
        ax_noisy.set_xscale("log")
        ax_noisy.set_yscale("log")
        shade_floor(ax_noisy)
        ax_noisy.grid(axis="x", color=GRID, linewidth=0.8)
        ax_noisy.legend(frameon=False, fontsize=9, loc="upper left")
        ax_noisy.set_xlabel("||x - c||  (input distance from the det = 0 centre)")
        ax_noisy.set_ylabel("||f(x) - f(c)||  (output distance)")
        ax_noisy.set_title("Kernel direction with noise", loc="left", fontsize=11)
    else:
        ax_noisy.set_visible(False)

    # the (v_min, v_2) plane: log10 output distance from f(c) over the grid,
    # with the det = 0 set where it crosses the plane
    if plane is not None:
        ab = plane["ab"]
        d_out = np.linalg.norm(plane["vec_out"].astype(np.float64) - center_out, axis=-1)
        # pixels below float32 precision are blanked
        img = ax_plane.imshow(
            np.ma.masked_less(np.log10(np.maximum(d_out, 1e-300)), np.log10(band_top)).T,
            origin="lower", cmap="Greens",
            extent=(ab[0], ab[-1], ab[0], ab[-1]), aspect="equal",
        )
        fig.colorbar(img, ax=ax_plane, label="log10 ||f(x) - f(c)||  (blank: below float32 precision)")
        if (plane["sign"] > 0).any() and (plane["sign"] < 0).any():
            ax_plane.contour(ab, ab, plane["sign"].T.astype(float), levels=[0.0], colors=REF, linewidths=1.5)
            ax_plane.plot([], [], color=REF, label="det J = 0")
            ax_plane.legend(frameon=False, fontsize=9, loc="upper right")
        ax_plane.plot(0, 0, "o", color="white", markeredgecolor=REF, markersize=6)
        ax_plane.grid(False)
        ax_plane.set_xlabel(f"a, along v_min (sigma = {svals[-1]:.1e})")
        ax_plane.set_ylabel(f"b, along v_2 (sigma = {svals[-2]:.2g})")
        ax_plane.set_title("Output distance over the (v_min, v_2) plane", loc="left", fontsize=11)
    else:
        ax_plane.set_visible(False)

    fig.suptitle(
        f"{attrs['sublayer_input']} -> {attrs['sublayer_output']}, sample {attrs['a_sample_idx']}, "
        f"pos {attrs['pos']}: lines through the det = 0 point x(t*) [crossing {attrs['crossing']}/{len(attrs['crossings_t']) - 1}], t* = {attrs['t_star']:.4f}, "
        f"|s| <= r = {r:.3g} (rows {attrs['a_row']} -> {attrs['b_row']})",
        x=0.01, ha="left", fontsize=11,
    )
    fig.tight_layout()

    out_path = out_path or os.path.splitext(path)[0] + "_compound.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("h5", nargs="+", help="generate_det_zero_sphere.py output file(s)")
    parser.add_argument("--out", default=None, help="PNG path (single input only); default next to the h5")
    parser.add_argument("--bins", type=int, default=100, help="Histogram bins (x3 on the shared axis)")
    args = parser.parse_args()
    if args.out and len(args.h5) > 1:
        parser.error("--out only works with a single input file")
    for path in args.h5:
        plot_file(path, args.out, args.bins)


if __name__ == "__main__":
    main()
