"""Publication (ICLR, full page) figure of the extreme-value estimate of the
lower Lipschitz constant K_L, for the same sublayers as
plot_kernel_tangent_overlap_paper.py: rows = layers --layers (default 1 3 6)
x {Attention, FFN}, columns =
  (a) scatter of ||x - y|| vs ||f(x) - f(y)|| over pairs of perturbed points,
      lower-tail pairs (g < u) highlighted, with the line of slope K_L-hat;
  (b) histogram of the ratio g = ||f(x) - f(y)|| / ||x - y||, threshold u and
      K_L-hat marked;
  (c) the lower tail g < u: empirical conditional density against the fitted
      GPD density, whose finite endpoint is K_L-hat = u + sigma / xi (xi < 0).

One (sample, pos) per panel row -- the same one for every sublayer
(--sample-idx/--pos, default sample 47 pos 2: every one of the six default
sublayers has a valid fit there, 0 < K_L-hat <= min observed g, and each
K_L-hat sits within the 25th-75th percentile of that sublayer's valid fits).
The fit (u, xi, sigma, K_L-hat) is read as-is from
lipschitz_ratio_bootstrap_eqd_sweep_gpu.py's output. That fit used n_pairs
(200k) random pairs of the position's 4019 vocab-substitution points, of which
it only saved 10k; this script recomputes ALL ~8M pairs from the perturbed
sweep (the population those 200k were drawn from), so the tail panel has a
dense empirical tail, and it reports whether any pair falls below K_L-hat.
Needs the perturbed sweep shards locally -- run on mauao.

Panels (b) and (c) drop ratios below the --outlier-q quantile (default 1e-5, ~80 of ~8M pairs) for
display only; panel (a), the printed stats and the fit itself use every pair.

Writes {--out}.pdf (TrueType fonts) / .png and prints per-row fit stats.

    uv run python -m scripts.analysis.plot_lipschitz_evt_paper configs/simple_stories.toml
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
from matplotlib.patches import Patch
from matplotlib.ticker import NullLocator

from autoencoder.data import load_sample_cloud
from config import load_config
from scripts.analysis.plot_kernel_tangent_overlap_paper import PAPER_RC, legend_fs, legend_ncol, save_figure, \
    transformer_blocks

plt.rcParams.update(PAPER_RC)
C_BULK, C_TAIL, C_BOUND, C_FIT, C_U, INK, GRID = "#b0b0b0", "#2a78d6", "#f0614a", "#222222", "#6b6b6b", "#222222", \
    "#e6e6e6"
C_TRUE = "#222222"  # known true K_L (analytical models only)


def all_pair_dists(vec_in: np.ndarray, vec_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(input_dist, output_dist) over every unordered pair i < j, float64."""
    a, b = torch.from_numpy(vec_in).double(), torch.from_numpy(vec_out).double()
    iu = torch.triu_indices(len(a), len(a), offset=1)
    din = torch.cdist(a, a)[iu[0], iu[1]].numpy()
    dout = torch.cdist(b, b)[iu[0], iu[1]].numpy()
    return din, dout


def gpd_tail_density(g: np.ndarray, u: float, xi: float, sigma: float) -> np.ndarray:
    """Density of g given g < u under the lower-tail GPD on y = u - g."""
    y = u - g
    z = 1 + xi * y / sigma
    out = np.zeros_like(g)
    ok = (y >= 0) & (z > 0)
    out[ok] = z[ok] ** (-1 / xi - 1) / sigma
    return out


def draw_evt_row(axs_row, din: np.ndarray, dout: np.ndarray, u: float, xi: float, sigma: float, k_hat: float,
                 label: str, rng: np.random.Generator, n_scatter: int, n_scatter_tail: int, outlier_q: float,
                 true_kl: float | None = None) -> int:
    """Draws one figure row -- (a) distance scatter, (b) ratio histogram, (c) lower tail vs GPD fit -- for
    pairs with input/output distances din/dout and a lower-tail GPD fit (threshold u, xi, sigma) with
    endpoint k_hat (NaN when xi >= 0: no finite bound). true_kl, if given, is a known true K_L marked in (b),
    whose x-range is then extended to include it. Returns the number of pairs with g < k_hat."""
    g = dout / din
    tail = g < u
    finite = bool(np.isfinite(k_hat))
    below = np.flatnonzero(g < k_hat) if finite else np.array([], dtype=int)

    # (a) scatter
    ax = axs_row[0]
    bulk = rng.choice(np.flatnonzero(~tail), min(n_scatter, (~tail).sum()), replace=False)
    tl = np.flatnonzero(tail)
    tl = rng.choice(tl, min(n_scatter_tail, tl.size), replace=False)
    ax.scatter(din[bulk], dout[bulk], s=0.6, color=C_BULK, linewidths=0, rasterized=True, zorder=2)
    ax.scatter(din[tl], dout[tl], s=0.9, color=C_TAIL, linewidths=0, rasterized=True, zorder=2)
    # always drawn: pairs that violate the estimated bound
    ax.scatter(din[below], dout[below], s=14, facecolors="none", edgecolors=C_BOUND, linewidths=0.8, zorder=5)
    xmax = np.percentile(din, 99.9)
    if finite:
        # under the points: in high dimension the whole cloud can sit on the line
        ax.plot([0, xmax], [0, k_hat * xmax], color=C_BOUND, linewidth=1.1, zorder=1.5)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, np.percentile(dout, 99.9))
    ax.set_ylabel(label + "\n" + r"$\|f(x)-f(y)\|$", color=INK)

    # (b) ratio histogram
    ax = axs_row[1]
    g_lo = np.quantile(g, outlier_q)
    shown = g >= g_lo
    print(f"    display: dropped {(~shown).sum()} outlier pairs below g = {g_lo:.4f} from (b), (c)")
    lo, hi = min(g_lo, k_hat) if finite else g_lo, np.percentile(g, 99.9)
    if true_kl is not None:
        lo = min(lo, true_kl)
    bins = np.linspace(lo - 0.02 * (hi - lo), hi, 80)
    ax.hist(g[~tail & shown], bins=bins, color=C_BULK, linewidth=0)
    ax.hist(g[tail & shown], bins=bins, color=C_TAIL, linewidth=0)
    ax.axvline(u, color=C_U, linewidth=0.9, linestyle=(0, (3, 1.5)))
    if finite:
        ax.axvline(k_hat, color=C_BOUND, linewidth=1.1)
    if true_kl is not None:
        ax.axvline(true_kl, color=C_TRUE, linewidth=1.1, linestyle=(0, (1, 1.5)))
    ax.set_yscale("log")
    ax.set_ylabel("Count", color=INK)

    # (c) lower tail: empirical conditional density vs fitted GPD
    ax = axs_row[2]
    gt = g[tail & shown]
    t_lo = min(k_hat, gt.min()) if finite else gt.min()
    pad = 0.04 * (u - t_lo)
    x_left = t_lo - pad
    tbins = np.linspace(x_left, u, 45)
    dens, _, _ = ax.hist(gt, bins=tbins, density=True, color=C_TAIL, alpha=0.55, linewidth=0)
    grid = np.linspace(x_left, u, 2000)
    model = gpd_tail_density(grid, u, xi, sigma)
    ax.plot(grid, np.where(model > 0, model, np.nan), color=C_FIT, linewidth=1.1)
    # log density: shows the tail thinning out towards the GPD's finite endpoint K_L-hat
    ax.set_yscale("log")
    ax.set_ylim(0.3 * dens[dens > 0].min(), 3 * max(dens.max(), np.nanmax(model)))
    if finite:
        ax.axvline(k_hat, color=C_BOUND, linewidth=1.1)
    ax.axvline(u, color=C_U, linewidth=0.9, linestyle=(0, (3, 1.5)))
    ax.set_xlim(x_left, u + pad)
    ax.set_ylabel("Density", color=INK)
    # just right of the K_L-hat line: the tail only rises towards u, so the top-left is empty
    x_text = max(0.06, (k_hat - x_left) / (u + pad - x_left) + 0.04) if finite else 0.06
    k_txt = rf"$\hat K_L={k_hat:.3f}$" if finite else r"no finite $\hat K_L$"
    ax.text(x_text, 0.95, k_txt + "\n" + rf"$\xi={xi:.3f}$", transform=ax.transAxes,
            ha="left", va="top", fontsize=7, color=INK, linespacing=1.3)

    for ax in axs_row[1:]:
        ax.yaxis.set_minor_locator(NullLocator())  # log axes: major decades only
    for ax in axs_row:
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="both", which="major", pad=1.5)
    return below.size


def finish_evt_figure(fig, axs, n_violations: int, out: str, true_kl: float | None = None) -> None:
    """Column titles, axis labels, the shared legend, layout, and the .pdf/.png write."""
    axs[0, 0].set_title("(a) Pairwise distances", fontsize=8.5, color=INK)
    axs[0, 1].set_title("(b) Ratio distribution", fontsize=8.5, color=INK)
    axs[0, 2].set_title("(c) Lower tail and GPD fit", fontsize=8.5, color=INK)
    axs[-1, 0].set_xlabel(r"$\|x-y\|$", color=INK)
    axs[-1, 1].set_xlabel(r"$g=\|f(x)-f(y)\|\,/\,\|x-y\|$", color=INK)
    axs[-1, 2].set_xlabel(r"$g$", color=INK)

    handles = [
        Patch(color=C_BULK, label="Pairs"),
        Patch(color=C_TAIL, label=r"Lower tail ($g<u$)"),
        Line2D([], [], color=C_U, linewidth=0.9, linestyle=(0, (3, 1.5)), label=r"Threshold $u$"),
        Line2D([], [], color=C_FIT, linewidth=1.1, label="GPD fit"),
        Line2D([], [], color=C_BOUND, linewidth=1.1, label=r"Lower bound $\hat K_L$"),
    ]
    if true_kl is not None:
        handles.append(Line2D([], [], color=C_TRUE, linewidth=1.1, linestyle=(0, (1, 1.5)),
                              label=rf"True $K_L={true_kl:g}$"))
    if n_violations:
        handles.append(Line2D([], [], marker="o", linestyle="none", markersize=4, markerfacecolor="none",
                              markeredgecolor=C_BOUND, markeredgewidth=0.8, label=r"Pairs with $g<\hat K_L$"))
    fig.legend(handles=handles, loc="upper center", ncol=legend_ncol(3), frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.8, columnspacing=1.4, fontsize=legend_fs(7.5))
    n_legend_rows = -(-len(handles) // legend_ncol(3))
    row_h = 0.2 * (1 + 0.4 * (legend_fs(7.5) / 7.5 - 1))  # inches per legend row, grows with the legend font
    # a bigger (presentation) legend gets extra figure height instead of squeezing the panels
    fig.set_figheight(fig.get_figheight() + row_h * n_legend_rows - 0.2 * -(-len(handles) // 3))
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.08 + row_h * n_legend_rows) / fig.get_figheight()), h_pad=0.6,
                     w_pad=0.8)
    fig.align_ylabels(axs[:, 0])
    save_figure(fig, out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--fit-dir", default="onnx_models/simple_stories/lipschitz_ratio_bootstrap_eqd_sweep_gpu")
    p.add_argument("--perturbed-sweep-dir", default="onnx_models/simple_stories/perturbed_sweep")
    p.add_argument("--layers", type=int, nargs="+", default=[1, 3, 6], help="1-based transformer layers")
    p.add_argument("--sample-idx", type=int, default=47)
    p.add_argument("--pos", type=int, default=2)
    p.add_argument("--n-scatter", type=int, default=20_000, help="bulk pairs drawn in the scatter")
    p.add_argument("--n-scatter-tail", type=int, default=3_000, help="max tail pairs drawn in the scatter")
    p.add_argument("--outlier-q", type=float, default=1e-5,
                   help="display only: panels (b) and (c) drop ratios below this quantile (isolated stragglers "
                   "that stretch the x-axis); the fit and panel (a) are unaffected")
    p.add_argument("--max-points", type=int, default=None,
                   help="random subset of each position's cloud to take all pairs over (default: every point; "
                   "GPT-2's ~50k-point clouds need e.g. 4019, simple_stories' size, to fit in memory)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="figures/paper/lipschitz_evt")
    args = p.parse_args()

    cfg = load_config(args.config)
    blocks = transformer_blocks(cfg)
    rng = np.random.default_rng(args.seed)
    rows = [(layer, kind) for layer in args.layers for kind in ("attn", "ffn")]
    n_violations = 0

    fig, axs = plt.subplots(len(rows), 3, figsize=(5.5, 1.22 * len(rows) + 0.6),
                            gridspec_kw=dict(width_ratios=[1, 1, 1]))
    for r, (layer, kind) in enumerate(rows):
        name = blocks[layer - 1][kind]
        layer_in, layer_out = name.split(".")
        with h5py.File(os.path.join(args.fit_dir, name, f"sample_{args.sample_idx:05d}.h5")) as f:
            a = dict(f[f"pos_{args.pos:04d}"].attrs)
            n_pairs_fit = int(f.attrs["n_pairs"])
        u, xi, sigma, k_hat = a["threshold"], a["shape_xi"], a["scale_sigma"], a["c_hat"]
        if not (np.isfinite(k_hat) and k_hat > 0):
            raise SystemExit(f"{name} sample {args.sample_idx} pos {args.pos}: no valid finite bound "
                             f"(xi={xi:.3g}, K_L-hat={k_hat:.3g}) -- pick another --sample-idx/--pos")

        vin = load_sample_cloud(args.perturbed_sweep_dir, layer_in, args.sample_idx)[args.pos].numpy()
        vout = load_sample_cloud(args.perturbed_sweep_dir, layer_out, args.sample_idx)[args.pos].numpy()
        if args.max_points is not None and len(vin) > args.max_points:
            keep = np.sort(rng.choice(len(vin), args.max_points, replace=False))
            vin, vout = vin[keep], vout[keep]
        din, dout = all_pair_dists(vin, vout)
        g = dout / din
        tail = g < u
        print(f"layer {layer} {kind:4s} {name:<15} pairs {g.size}  u {u:.4f}  xi {xi:.3f}  sigma {sigma:.4f}  "
              f"K_L-hat {k_hat:.4f}  | frac below u: all-pairs {tail.mean():.2e} vs fit "
              f"{a['eqd_best_n_tail'] / n_pairs_fit:.2e}  | min g: all-pairs {g.min():.4f}, fit sample "
              f"{a['g_min_observed']:.4f}  | pairs below K_L-hat: {(g < k_hat).sum()}")

        label = f"Layer {layer}\n{'Attention' if kind == 'attn' else 'FFN'}"
        n_violations += draw_evt_row(axs[r], din, dout, u, xi, sigma, k_hat, label, rng, args.n_scatter,
                                     args.n_scatter_tail, args.outlier_q)

    finish_evt_figure(fig, axs, n_violations, args.out)


if __name__ == "__main__":
    main()
