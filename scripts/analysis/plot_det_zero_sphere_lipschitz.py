"""Lipschitz analysis of the sphere points written by
`scripts/python/generate_det_zero_sphere.py` (points sampled uniformly on a
sphere around the det = 0 crossing between two opposite-sign perturbed
points).

Plots (2 x 4, last slot empty):
  - scatter of ||f(x_i) - f(x_j)|| against ||x_i - x_j|| over every pair of
    sphere points, with lines y = g_min x and y = g_max x through the origin
    (same ratio g as the Lipschitz scripts' compute_lipschitz_ratios_gpu);
  - histogram of g over those pairs;
  - centre-to-point stretch ||f(x_k) - f(c)|| / r for every sphere point,
    split by the sign of det J at x_k -- r = ||x_k - c|| is the radius, so
    this is the ratio from the singular centre outwards;
  - log|det J| along the original A -> B path, split by sign, with the
    crossing t* the sphere is centred on;
  - input vs output distance from the centre, ||f(x) - f(c)|| against
    ||x - c||, for points on the line through c in J's smallest singular
    direction v_min (uniform s in [-r, r] plus log-spaced |s| down to
    --line-min-s), and a k s^2 guide through the v_min line's far end (along v_min the
    linear term sigma_min s is usually below the float32 output floor, so
    the change is second order);
  - histogram of g over every pair of points on the v_min line;
  (v_min vs a control direction on shared axes: plot_det_zero_sphere_compound.py)
  - the singular values of J at the centre.
Blue = det > 0, orange = det < 0 in the stretch and path panels; green =
the v_min direction.

Writes {stem}_lipschitz.png next to each input h5 (or --out) and prints
ratio quantiles.

Usage:
    uv run python -m scripts.analysis.plot_det_zero_sphere_lipschitz \\
        onnx_models/simple_stories/det_zero_sphere/add_4.add_6/row26685_row26140_sphere.h5
"""
import argparse
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

POS_COLOR = "#2a78d6"  # dataviz reference palette, categorical slot 1 -- det > 0 / single series
NEG_COLOR = "#eb6834"  # slot 2 -- det < 0
SING_COLOR = "#1baf7a"  # slot 3 -- singular direction v_min
REF = "#52514e"        # reference lines + annotation ink
GRID = "#e4e3df"


def all_pair_dists(vec_in: np.ndarray, vec_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(||x_i - x_j||, ||f(x_i) - f(x_j)||) for every i < j, float64.
    Row-by-row so memory stays O(n * d)."""
    x = vec_in.astype(np.float64)
    y = vec_out.astype(np.float64)
    in_d, out_d = [], []
    for i in range(len(x) - 1):
        in_d.append(np.linalg.norm(x[i + 1:] - x[i], axis=1))
        out_d.append(np.linalg.norm(y[i + 1:] - y[i], axis=1))
    return np.concatenate(in_d), np.concatenate(out_d)


NOISE_BAND = 2.0  # output changes below this x the estimated float32 floor are dropped as noise


def fp32_threshold(d_in: np.ndarray, d_out: np.ndarray, r: float) -> tuple[float, float, float, int]:
    """(k2, floor, threshold, n_floor_pts) from the v_min points' distances
    from the centre (d_in, d_out > 0). k2 is the s^2 coefficient fitted at the
    point nearest |s| = r; the float32 output floor is the median output
    change over the points where k2 s^2 predicts < 10% of what's measured
    (rounding-dominated); anything with output change below
    threshold = NOISE_BAND x floor is below float32 precision and is dropped
    from the plots."""
    far = np.argmin(np.abs(d_in - r))
    k2 = d_out[far] / d_in[far] ** 2
    floor_pts = k2 * d_in**2 < 0.1 * d_out
    floor = float(np.median(d_out[floor_pts])) if floor_pts.any() else 0.0
    return k2, floor, NOISE_BAND * floor, int(floor_pts.sum())


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=REF)


def plot_file(path: str, out_path: str | None, bins: int) -> str:
    with h5py.File(path, "r") as f:
        vec_in, vec_out = f["vec_in"][:], f["vec_out"][:]
        sign, center_in, center_out = f["sign"][:], f["center_in"][:], f["center_out"][:]
        path_t, path_sign, path_ld = f["path_t"][:], f["path_sign"][:], f["path_log_abs_det"][:]
        attrs = dict(f.attrs)
        svals = f["singular_values"][:]
        # (vec_in, vec_out) of each line through the centre present in the file:
        # singular / control on the uniform s grid, *_log on log-spaced |s|
        # (older files lack control and/or the *_log lines)
        lines = {
            name: (f[f"line_{name}/vec_in"][:], f[f"line_{name}/vec_out"][:])
            for name in ("singular", "control", "singular_log", "control_log")
            if f"line_{name}" in f
        }

    in_d, out_d = all_pair_dists(vec_in, vec_out)
    g = out_d / in_d
    r_k = np.linalg.norm(vec_in.astype(np.float64) - center_in, axis=1)
    stretch = np.linalg.norm(vec_out.astype(np.float64) - center_out, axis=1) / r_k

    q = np.quantile(g, [0.0, 0.01, 0.5, 0.99, 1.0])
    print(f"{path}: {g.size} pairs, g min {q[0]:.4g}, 1% {q[1]:.4g}, median {q[2]:.4g}, "
          f"99% {q[3]:.4g}, max {q[4]:.4g}")
    for s, name in ((1, "det > 0"), (-1, "det < 0")):
        m = sign == s
        if m.any():
            print(f"  centre-to-point stretch, {name} ({m.sum()} pts): min {stretch[m].min():.4g}, "
                  f"median {np.median(stretch[m]):.4g}, max {stretch[m].max():.4g}")

    fig, axes = plt.subplots(2, 4, figsize=(27, 10))
    (ax_scat, ax_hist, ax_lines, ax_line_hist), (ax_center, ax_path, ax_spec, ax_unused) = axes
    ax_unused.set_visible(False)
    for ax in axes.flat:
        _style(ax)

    ax_scat.scatter(in_d, out_d, s=1, color=POS_COLOR, alpha=0.15, linewidths=0, rasterized=True)
    x_line = np.array([0.0, in_d.max()])
    for val, name in ((q[0], "min"), (q[4], "max")):
        ax_scat.plot(x_line, val * x_line, color=REF, linestyle="--", linewidth=1.2)
        # max label above the upper line's end; min label below the lower
        # line's midpoint, clear of the point cloud near the lines' far end
        x_lab = x_line[1] if name == "max" else 0.5 * x_line[1]
        ax_scat.annotate(f"slope {name} g = {val:.3g}", (x_lab, val * x_lab),
                         xytext=(-4, 6) if name == "max" else (6, -16), textcoords="offset points",
                         ha="right" if name == "max" else "left", color=REF, fontsize=9)
    ax_scat.set_xlim(left=0)
    ax_scat.set_ylim(bottom=0)
    ax_scat.set_xlabel("||x_i - x_j||  (input, at pos)")
    ax_scat.set_ylabel("||f(x_i) - f(x_j)||  (output, at pos)")
    ax_scat.set_title(f"Output vs input distance, all sphere pairs (N = {g.size:,})", loc="left", fontsize=11)

    ax_hist.hist(g, bins=bins, color=POS_COLOR, edgecolor="white", linewidth=0.5)
    for val, name in ((q[0], "min"), (q[4], "max")):
        ax_hist.axvline(val, color=REF, linestyle="--", linewidth=1.2)
        ax_hist.annotate(f"{name} {val:.3g}", xy=(val, 1), xycoords=("data", "axes fraction"),
                         xytext=(4 if name == "min" else -4, -12), textcoords="offset points",
                         ha="left" if name == "min" else "right", color=REF, fontsize=9)
    ax_hist.set_xlabel("g = ||f(x_i) - f(x_j)|| / ||x_i - x_j||")
    ax_hist.set_ylabel("pairs")
    ax_hist.set_title(f"Pairwise ratio g (median {q[2]:.3g})", loc="left", fontsize=11)

    edges = np.histogram_bin_edges(stretch, bins=min(bins, 60))
    for s, color, name in ((1, POS_COLOR, "det > 0"), (-1, NEG_COLOR, "det < 0")):
        m = sign == s
        if m.any():
            ax_center.hist(stretch[m], bins=edges, histtype="step", color=color, linewidth=2,
                           label=f"{name} ({m.sum()} pts)")
    ax_center.legend(frameon=False, fontsize=9)
    ax_center.set_xlabel(f"||f(x_k) - f(c)|| / r,  r = {attrs['radius']:.3g}")
    ax_center.set_ylabel("sphere points")
    ax_center.set_title("Stretch from the det = 0 centre, by det sign at x_k", loc="left", fontsize=11)

    for s, color, name in ((1, POS_COLOR, "det > 0"), (-1, NEG_COLOR, "det < 0")):
        m = path_sign == s
        ax_path.scatter(path_t[m], path_ld[m], s=10, color=color, linewidths=0, label=name)
    ax_path.axvline(attrs["t_star"], color=REF, linestyle="--", linewidth=1.2)
    ax_path.annotate(f"t* = {attrs['t_star']:.4f}", xy=(attrs["t_star"], 1), xycoords=("data", "axes fraction"),
                     xytext=(4, -12), textcoords="offset points", color=REF, fontsize=9)
    ax_path.legend(frameon=False, fontsize=9, loc="upper right", markerscale=2)
    ax_path.set_xlabel("t along A -> B")
    ax_path.set_ylabel("log|det J|")
    ax_path.set_title("det along the path (sphere centred at t*)", loc="left", fontsize=11)

    # input vs output distance from the centre, points along v_min only
    # (uniform line + log-spaced |s| down to --line-min-s); the control-direction
    # comparison is in plot_det_zero_sphere_compound.py
    r = attrs["radius"]
    parts = [lines["singular"]] + ([lines["singular_log"]] if "singular_log" in lines else [])
    d_in = np.linalg.norm(np.concatenate([p[0] for p in parts]).astype(np.float64) - center_in, axis=1)
    d_out = np.linalg.norm(np.concatenate([p[1] for p in parts]).astype(np.float64) - center_out, axis=1)
    keep = (d_out > 0) & (d_in > 0)  # log axes; drops the s = 0 sample
    d_in, d_out = d_in[keep], d_out[keep]
    # sigma_min * x (the linear term along v_min) is typically far below the
    # float32 output floor (~1e-6), so draw a quadratic guide through the
    # v_min line's point at |s| = r instead -- the collapsed direction's output
    # change is second order, k s^2 -- and drop points below the float32 floor
    k2, floor, thresh, _ = fp32_threshold(d_in, d_out, r)
    far = np.argmin(np.abs(d_in - r))
    at_r = d_out[far] / d_in[far]
    above = d_out >= thresh
    ax_lines.scatter(d_in[above], d_out[above], s=8, color=SING_COLOR, linewidths=0,
                     label=f"along v_min (smallest singular direction), both signs of s; {at_r:.3g} x at r "
                           f"({int((~above).sum())} pts below the float32 floor removed)")
    x_ref = np.geomspace(d_in[above].min(), d_in.max(), 50)
    ax_lines.plot(x_ref, k2 * x_ref**2, ":", color=REF, linewidth=1.2,
                  label=f"quadratic guide {k2:.2g} s^2, fit at r (linear term sigma_min = {svals[-1]:.1e})")
    ax_lines.set_xscale("log")
    ax_lines.set_yscale("log")
    ax_lines.grid(axis="x", color=GRID, linewidth=0.8)
    ax_lines.legend(frameon=False, fontsize=9, loc="upper left")
    ax_lines.set_xlabel("||x - c||  (input distance from the det = 0 centre)")
    ax_lines.set_ylabel("||f(x) - f(c)||  (output distance)")
    ax_lines.set_title("Input vs output distance from the centre, log-log", loc="left", fontsize=11)

    # Lipschitz ratio g over every pair of v_min-line points -- same ratio as
    # the sphere histogram, restricted to the collapsed direction
    # pairs whose output difference is below the float32 threshold are dropped
    pin, pout = all_pair_dists(*lines["singular"])
    g_line = pout[pout >= thresh] / pin[pout >= thresh]
    ql = np.quantile(g_line, [0.0, 0.5, 1.0])
    print(f"  v_min line: {g_line.size} pairs, g min {ql[0]:.3g}, median {ql[1]:.3g}, max {ql[2]:.3g}")
    ax_line_hist.hist(g_line, bins=bins, color=SING_COLOR, edgecolor="white", linewidth=0.5)
    for val, name in ((ql[0], "min"), (ql[2], "max")):
        ax_line_hist.axvline(val, color=REF, linestyle="--", linewidth=1.2)
        ax_line_hist.annotate(f"{name} {val:.3g}", xy=(val, 1), xycoords=("data", "axes fraction"),
                              xytext=(4 if name == "min" else -4, -12), textcoords="offset points",
                              ha="left" if name == "min" else "right", color=REF, fontsize=9)
    ax_line_hist.set_xlabel(f"g = ||f(x_i) - f(x_j)|| / ||x_i - x_j||,  x_i, x_j on the v_min line "
                            f"({pin.size - g_line.size} pairs below the float32 floor removed)")
    ax_line_hist.set_ylabel("pairs")
    ax_line_hist.set_title(f"Lipschitz ratio along v_min (N = {g_line.size:,}, median {ql[1]:.3g})",
                           loc="left", fontsize=11)

    ax_spec.semilogy(np.arange(1, len(svals) + 1), svals, color=SING_COLOR, linewidth=2)
    ax_spec.plot(len(svals), svals[-1], "o", color=SING_COLOR, markersize=8,
                 markeredgecolor="white", markeredgewidth=2)
    ax_spec.annotate(f"sigma_min = {svals[-1]:.2g}\n(next {svals[-2]:.2g})", (len(svals), svals[-1]),
                     xytext=(-8, 4), textcoords="offset points", ha="right", color=REF, fontsize=9)
    ax_spec.set_xlabel("singular value index")
    ax_spec.set_ylabel("sigma_i of J at the centre")
    ax_spec.set_title(f"Singular values of J at x(t*) (sigma_max {svals[0]:.3g})", loc="left", fontsize=11)

    kind = "ball" if attrs.get("ball") else "sphere"
    fig.suptitle(
        f"{attrs['sublayer_input']} -> {attrs['sublayer_output']}, sample {attrs['a_sample_idx']}, pos {attrs['pos']}: "
        f"crossing {attrs['crossing']}/{len(attrs['crossings_t']) - 1}, {len(vec_in)} points uniform on a {kind} around x(t*), "
        + (f"fixed radius = {attrs['radius']:g}" if attrs["radius_to"] == "fixed"
           else f"radius = distance to {str(attrs['radius_to']).upper()}")
        + f" (rows {attrs['a_row']} -> {attrs['b_row']})",
        x=0.01, ha="left", fontsize=11,
    )
    fig.tight_layout()

    out_path = out_path or os.path.splitext(path)[0] + "_lipschitz.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("h5", nargs="+", help="generate_det_zero_sphere.py output file(s)")
    parser.add_argument("--out", default=None, help="PNG path (single input only); default next to the h5")
    parser.add_argument("--bins", type=int, default=100)
    args = parser.parse_args()
    if args.out and len(args.h5) > 1:
        parser.error("--out only works with a single input file")
    for path in args.h5:
        plot_file(path, args.out, args.bins)


if __name__ == "__main__":
    main()
