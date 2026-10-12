"""Publication (ICLR, 5.5in text width) figures for one
scripts/python/generate_det_zero_manifold.py file -- the critical (det J = 0)
point c between two opposite-sign perturbed points, its nearest manifold point
p, and the sample sets around them:

  A   p: on the manifold (convex neighbour samples)      blue,       solid, o
  B   p: ambient ball                                    dark grey,  solid, o
  C   p: along v_min(p), J's smallest singular direction dark red,   solid, o
  cB  c: ambient ball                                    light grey, dashed, ^
  cC  c: along v_min(c), the kernel direction            red,        dashed, ^

Colours follow the other *_paper.py figures (kernel direction v_min red
#f0614a and random direction grey as in plot_kernel_tangent_overlap_paper.py;
the EVT figure is drawn by plot_lipschitz_evt_paper.draw_evt_row itself);
the centre is encoded twice (line style / marker as well as shade), so the
figure reads without colour.

{--out}_main.pdf/.png -- seven panels and the legend on a 4 x 2 grid:
  (a) kernel score u of the samples;
  (b) chart residual vs u per sample;
  (c) ||f(x) - f(centre)|| vs ||x - centre||          (x in [1e-8, 1e2]);
  (d) each sample's K_L value ||f(x) - f(centre)|| / ||x - centre|| vs ||x - centre||
                                                       (x in [1e-8, 1e2]);
  (e) histogram of that K_L value;
  (f) log|det J| vs ||x - centre||;
  (g) pairwise ratio g = ||f(x_i) - f(x_j)|| / ||x_i - x_j|| within each set, min dotted.
{--out}_evt.pdf/.png -- the EVT lower-Lipschitz figure (same panels as
  lipschitz_evt.pdf), one row per set, fitted with the sweep's own
  bootstrap-EQD + GPD pipeline.

Every panel except (c) and (d), and the EVT figure, uses only the samples whose
distance from their centre lies in --dist-range (default 1e-6 to 1e1: below
~1e-6 the float32 forward pass's rounding dominates ||f(x) - f(centre)||, and
beyond ~10 the sublayer is the identity far from the data). Uniform and
log-spaced samples are pooled, so the panels see every scale in the range.
Histograms are scaled to a peak of 1 per set and show the --hist-q (default 1st-99th)
percentile range of all sets pooled, on a log height axis down to 1e-3 of the peak; the
per-set minimum pairwise ratio is in the legend. No suptitle -- the caption goes
in the paper. CPU; ~1 min.

    uv run python -m scripts.analysis.plot_det_zero_manifold_paper \\
        onnx_models/simple_stories/det_zero_manifold_minS1e-12/add_4.add_6/row26685_row26140_manifold_r100_c05.h5
"""
import argparse
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullLocator

from scripts.analysis.plot_kernel_tangent_overlap_paper import C_MIN, C_RAND, GRID, INK, PAPER_RC, legend_fs, save_figure
from scripts.analysis.plot_lipschitz_evt_paper import all_pair_dists, draw_evt_row, finish_evt_figure
from scripts.lipschitz.lipschitz_ratio_bootstrap_eqd_sweep_gpu import (
    fit_gpd_and_c_hat_gpu,
    select_bootstrap_eqd_threshold_gpu_multi,
)

plt.rcParams.update(PAPER_RC)

HIST_YMIN = 1e-3  # lowest relative height shown on the (log) histogram axes
HIST_Q = (1.0, 99.0)  # --hist-q: percentile range of the histogram panels (a), (e), (g)
C_MANIFOLD = "#2a78d6"  # the EVT figure's tail blue
C_DARK_GREY = "#7a7a7a"
C_DARK_RED = "#a5321c"
STYLE = {
    "A": dict(label=r"$p$: on manifold", color=C_MANIFOLD, ls="-", marker="o", centre="p"),
    "B": dict(label=r"$p$: ambient", color=C_DARK_GREY, ls="-", marker="o", centre="p"),
    "C": dict(label=r"$p$: along $v_{\min}(p)$", color=C_DARK_RED, ls="-", marker="o", centre="p"),
    "cB": dict(label=r"$c$: ambient", color=C_RAND, ls=(0, (4, 1.5)), marker="^", centre="c"),
    "cC": dict(label=r"$c$: along $v_{\min}(c)$ (kernel)", color=C_MIN, ls=(0, (4, 1.5)), marker="^", centre="c"),
}
# Scatter figures (output / pairwise distance): one marker for every set, so the sets must be told apart
# by colour alone -- five distinct hues (validated: CVD dE >= 9.6, normal-vision dE >= 15.9), the kernel
# line keeping the paper's v_min red
SCATTER_COLOR = {"A": "#2a78d6", "B": "#1baf7a", "C": "#4a3aa7", "cB": "#eda100", "cC": C_MIN}
EVT_LABEL = {
    "A": "$p$: on\nmanifold", "B": "$p$: ambient", "C": "$p$: along\n$v_{\\min}(p)$",
    "cB": "$c$: ambient", "cC": "$c$: kernel\n$v_{\\min}(c)$",
}


def load(path: str) -> dict:
    with h5py.File(path, "r") as f:
        d = {"attrs": dict(f.attrs), "p_out": f["p_out"][:], "c_out": f["c_out"][:],
             "sv_min_c": float(f["singular_values_c"][-1]), "sv_min_p": float(f["singular_values_p"][-1])}
        for g in STYLE:
            parts = [{k: f[name][k][:] for k in ("vec_in", "vec_out", "dist", "u", "residual", "log_abs_det")}
                     for name in (g, g + "_log")]
            d[g] = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    for g in STYLE:
        s = d[g]
        s["df"] = np.linalg.norm(s["vec_out"] - d[STYLE[g]["centre"] + "_out"], axis=1)
    return d


def _style_ax(ax, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="both", which="major", pad=1.5)


def _hist_peak1(ax, values: dict[str, np.ndarray], n_bins: int, q: tuple[float, float] | None = None,
                filled: bool = True) -> tuple[float, float]:
    """Filled histograms on shared bins over the q-th percentiles of all sets pooled (so a few
    extreme samples don't squash the rest; values outside are left out, not piled into the
    end bins), each scaled to peak 1. Returns the (lo, hi) range shown."""
    allv = np.concatenate([v[np.isfinite(v)] for v in values.values()])
    lo, hi = (float(x) for x in np.percentile(allv, q or HIST_Q))
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    bins = np.linspace(lo, hi, n_bins + 1)
    for g, v in values.items():
        counts, edges = np.histogram(v[np.isfinite(v)], bins=bins)
        if counts.max() > 0:
            st = STYLE[g]
            h = counts / counts.max()
            # filled bars (translucent, so overlapping sets stay readable) with a thin outline that keeps
            # the set's line style; on the log axis the bars start at the axis floor, not at 0
            if filled:
                ax.stairs(h, edges, baseline=HIST_YMIN, fill=True, color=st["color"], alpha=0.28, linewidth=0)
            ax.stairs(h, edges, baseline=HIST_YMIN, color=st["color"], linestyle=st["ls"],
                      linewidth=0.8 if filled else 1.1)
    # log heights: the far-field samples all land in one bin (same u, ratio ~1), and on a linear
    # axis that spike flattens everything else
    ax.set_yscale("log")
    ax.set_ylim(HIST_YMIN, 1.6)
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(lo, hi)
    return lo, hi


def _scatter(ax, x: np.ndarray, y: np.ndarray, g: str) -> None:
    st = STYLE[g]
    ax.scatter(x, y, s=2.2 if st["marker"] == "o" else 2.6, marker=st["marker"], color=st["color"],
               linewidths=0, alpha=0.75, rasterized=True, zorder=3 if g in ("C", "cC") else 2)


def main_figure(d: dict, rng_lo: float, rng_hi: float, wide_lo: float, wide_hi: float, out: str) -> None:
    a = d["attrs"]
    in_rng = {g: (d[g]["dist"] >= rng_lo) & (d[g]["dist"] <= rng_hi) for g in STYLE}
    wide = {g: (d[g]["dist"] >= wide_lo) & (d[g]["dist"] <= wide_hi) for g in STYLE}

    fig, axs = plt.subplots(4, 2, figsize=(5.5, 7.6))
    axs = axs.ravel()

    # (a) kernel score
    ax = axs[0]
    _hist_peak1(ax, {g: d[g]["u"][in_rng[g]] for g in STYLE}, 30, filled=False)
    ax.set_xlabel(r"Kernel score $u$  ($1$ = on manifold)", color=INK)
    ax.set_ylabel("Count / peak", color=INK)
    ax.set_title("(a) On-manifold check", color=INK)

    # (b) residual vs u
    ax = axs[1]
    for g in STYLE:
        m = in_rng[g] & (d[g]["residual"] > 0)
        _scatter(ax, d[g]["residual"][m], d[g]["u"][m], g)
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Chart residual (local spacings)", color=INK)
    ax.set_ylabel(r"Kernel score $u$", color=INK)
    ax.set_title("(b) Two validators per sample", color=INK)

    # (c) output vs input distance
    ax = axs[2]
    for g in STYLE:
        m = wide[g] & (d[g]["df"] > 0)
        _scatter(ax, d[g]["dist"][m], d[g]["df"][m], g)
    xs = np.array([wide_lo, wide_hi])
    ax.plot(xs, xs, color=INK, linewidth=0.8, linestyle=(0, (1, 1.5)), zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(wide_lo, wide_hi)
    ax.set_xlabel(r"$\|x-\mathrm{centre}\|$", color=INK)
    ax.set_ylabel(r"$\|f(x)-f(\mathrm{centre})\|$", color=INK)
    ax.set_title("(c) Output vs input distance", color=INK)

    # (d) per-sample K_L value vs distance
    ax = axs[3]
    for g in STYLE:
        m = wide[g] & (d[g]["df"] > 0) & (d[g]["dist"] > 0)
        _scatter(ax, d[g]["dist"][m], d[g]["df"][m] / d[g]["dist"][m], g)
    ax.axhline(1.0, color=INK, linewidth=0.8, linestyle=(0, (1, 1.5)), zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(wide_lo, wide_hi)
    ax.set_xlabel(r"$\|x-\mathrm{centre}\|$", color=INK)
    ax.set_ylabel(r"$\|f(x)-f(\mathrm{centre})\|\,/\,\|x-\mathrm{centre}\|$", color=INK)
    ax.set_title(r"(d) $K_L$ value of each sample", color=INK)

    # (e) histogram of the per-sample K_L value
    ax = axs[4]
    vals = {}
    for g in STYLE:
        m = in_rng[g] & (d[g]["df"] > 0) & (d[g]["dist"] > 0)
        vals[g] = np.log10(d[g]["df"][m] / d[g]["dist"][m])
    _hist_peak1(ax, vals, 50)
    ax.set_xlabel(r"$\log_{10}\,\|f(x)-f(\mathrm{centre})\|\,/\,\|x-\mathrm{centre}\|$", color=INK)
    ax.set_ylabel("Count / peak", color=INK)
    ax.set_title(r"(e) Distribution of $K_L$ values", color=INK)

    # (f) log|det J| vs distance
    ax = axs[5]
    for g in STYLE:
        _scatter(ax, d[g]["dist"][in_rng[g]], d[g]["log_abs_det"][in_rng[g]], g)
    ax.axhline(a["p_log_abs_det"], color=INK, linewidth=0.8, linestyle=(0, (1, 1.5)), zorder=1)
    ax.axhline(a["c_log_abs_det"], color=INK, linewidth=0.8, linestyle=(0, (4, 1.5)), zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(rng_lo, rng_hi)
    ax.set_xlabel(r"$\|x-\mathrm{centre}\|$", color=INK)
    ax.set_ylabel(r"$\log|\det J|$", color=INK)
    ax.set_title(r"(f) $\log|\det J|$ vs distance", color=INK)

    # (g) pairwise ratio within each set
    ax = axs[6]
    pair_vals, mins = {}, {}
    for g in STYLE:
        m = in_rng[g]
        din, dout = all_pair_dists(d[g]["vec_in"][m], d[g]["vec_out"][m])
        ok = (din > 0) & (dout > 0)
        pair_vals[g] = np.log10(dout[ok] / din[ok])
        mins[g] = pair_vals[g].min()
    lo, hi = _hist_peak1(ax, pair_vals, 60)
    for g in STYLE:  # the minima are the extreme values, so they're marked only when inside the range shown
        if lo <= mins[g] <= hi:
            ax.axvline(mins[g], color=STYLE[g]["color"], linewidth=0.8, linestyle=(0, (1, 1.5)))
    ax.set_xlabel(r"$\log_{10} g,\ g=\|f(x_i)-f(x_j)\|\,/\,\|x_i-x_j\|$", color=INK)
    ax.set_ylabel("Count / peak", color=INK)
    ax.set_title("(g) Pairwise ratio", color=INK)

    for ax in axs[:7]:
        _style_ax(ax, "both" if ax.get_xscale() == "log" else "y")
    for ax in (axs[2], axs[3]):
        ax.xaxis.set_major_locator(LogLocator(numticks=6))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
    axs[5].xaxis.set_minor_locator(NullLocator())

    # (h) legend slot
    leg = axs[7]
    leg.axis("off")
    handles = [Line2D([], [], color=STYLE[g]["color"], linestyle=STYLE[g]["ls"], linewidth=1.1,
                      marker=STYLE[g]["marker"], markersize=3.5, label=STYLE[g]["label"]
                      + f"  (min $g$ {10 ** mins[g]:.2g})") for g in STYLE]
    handles += [
        Line2D([], [], color=INK, linewidth=0.8, linestyle=(0, (1, 1.5)), label=r"(c) $y=x$; (d) $y=1$; (f) at $p$"),
        Line2D([], [], color=INK, linewidth=0.8, linestyle=(0, (4, 1.5)), label=r"(f) $\log|\det J|$ at $c$"),
    ]
    leg.legend(handles=handles, loc="center", frameon=False, handlelength=2.4, fontsize=legend_fs(7), labelspacing=0.7)
    print(f"caption numbers: |c - p| = {a['dist_c_p']:.3g} ({a['dist_c_p_units']:.2f} local spacings); "
          f"sigma_min(J) {d['sv_min_c']:.2e} at c, {d['sv_min_p']:.2e} at p")

    fig.tight_layout(h_pad=0.9, w_pad=1.2)
    fig.align_ylabels(axs[[0, 2, 4, 6]])
    save_figure(fig, out + "_main")
    plt.close(fig)


def evt_figure(d: dict, rng_lo: float, rng_hi: float, args, out: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    fig, axs = plt.subplots(len(STYLE), 3, figsize=(5.5, 1.22 * len(STYLE) + 0.6))
    n_violations = 0
    for r, g in enumerate(STYLE):
        m = (d[g]["dist"] >= rng_lo) & (d[g]["dist"] <= rng_hi)
        din, dout = all_pair_dists(d[g]["vec_in"][m], d[g]["vec_out"][m])
        ok = (din > 0) & (dout > 0)
        din, dout = din[ok], dout[ok]
        gv = dout / din
        fit_idx = rng.choice(gv.size, args.n_pairs, replace=False) if gv.size > args.n_pairs else slice(None)
        g_sorted = torch.sort(torch.from_numpy(gv[fit_idx]).to(device, torch.float64)).values
        n_tail_max = min(args.n_tail_max, int(0.25 * g_sorted.numel()) + args.n_tail_step)
        [sel] = select_bootstrap_eqd_threshold_gpu_multi(
            [g_sorted], args.n_tail_min, n_tail_max, args.n_tail_step, args.eqd_n_boot,
            args.m_plotting_points, gen, group_size=1,
        )
        fit = fit_gpd_and_c_hat_gpu(g_sorted, sel["best_n_tail"], sel["threshold"])
        u, xi, sigma, k_hat = sel["threshold"], fit["shape_xi"], fit["scale_sigma"], fit["c_hat"]
        print(f"  EVT {g:3s} samples {int(m.sum())} pairs {gv.size}  min g {gv.min():.4g}  n_tail "
              f"{sel['best_n_tail']}  u {u:.4g}  xi {xi:.3f}  K_L-hat {k_hat:.4g}  below {(gv < k_hat).sum()}")
        n_violations += draw_evt_row(axs[r], din, dout, u, xi, sigma, k_hat, EVT_LABEL[g], rng, args.n_scatter,
                                     args.n_scatter_tail, args.outlier_q)
    finish_evt_figure(fig, axs, n_violations, out + "_evt")
    plt.close(fig)


def main() -> None:
    global HIST_Q
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", help="generate_det_zero_manifold.py output h5 (one crossing)")
    p.add_argument("--dist-range", type=float, nargs=2, default=[1e-6, 1e1],
                   help="samples used by every panel except (c), (d), and by the EVT figure")
    p.add_argument("--hist-q", type=float, nargs=2, default=list(HIST_Q),
                   help="percentile range of the histogram panels (a), (e), (g), pooled over all sets")
    p.add_argument("--wide-range", type=float, nargs=2, default=[1e-8, 1e2], help="x range of panels (c), (d)")
    p.add_argument("--n-pairs", type=int, default=200_000)
    p.add_argument("--n-tail-min", type=int, default=200)
    p.add_argument("--n-tail-max", type=int, default=1100)
    p.add_argument("--n-tail-step", type=int, default=200)
    p.add_argument("--eqd-n-boot", type=int, default=200)
    p.add_argument("--m-plotting-points", type=int, default=50)
    p.add_argument("--n-scatter", type=int, default=20_000)
    p.add_argument("--n-scatter-tail", type=int, default=3_000)
    p.add_argument("--outlier-q", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="path prefix (default figures/paper/det_zero_manifold_c{NN})")
    args = p.parse_args()

    HIST_Q = tuple(args.hist_q)
    d = load(args.file)
    out = args.out or os.path.join("figures", "paper", f"det_zero_manifold_c{int(d['attrs']['crossing']):02d}")
    main_figure(d, *args.dist_range, *args.wide_range, out)
    evt_figure(d, *args.dist_range, args, out)


if __name__ == "__main__":
    main()
