"""Pairwise version of det_zero_output_distance_c05: ||f(x) - f(y)|| vs ||x - y|| over every pair within
each sample set around p and c (crossing 5), log-log -- the same pairs as det_zero_evt_c05's column (a)
(radius-1-spacing run, uniform samples). A random 20k pairs per set are drawn. Same colours as the
other det_zero_manifold paper figures.

    uv run python -m scripts.analysis.plot_det_zero_pairwise_distance
"""
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullLocator

from scripts.analysis.plot_det_zero_manifold_paper import GRID, INK, PAPER_RC, SCATTER_COLOR, STYLE
from scripts.analysis.plot_kernel_tangent_overlap_paper import legend_fs, legend_ncol, save_figure
from scripts.analysis.plot_lipschitz_evt_paper import all_pair_dists

plt.rcParams.update(PAPER_RC)

# one figure per model: (input file, output prefix)
RUNS = [
    ("onnx_models/simple_stories/det_zero_manifold/add_4.add_6/row26685_row26140_manifold_c05.h5", "figures/paper/det_zero_pairwise_distance_c05"),  # simple_stories add_4.add_6 (layer-1 FFN), crossing 5
    ("onnx_models/gpt2/det_zero_manifold/add_2.add_5/row182959_row176262_manifold_c04.h5", "figures/paper/det_zero_pairwise_distance_gpt2_c04"),  # GPT-2 add_2.add_5 (layer-1 FFN), crossing 4
]

for FILE, OUT in RUNS:
    N_DRAW = 20_000

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    lo, hi = np.inf, 0.0
    with h5py.File(FILE, "r") as f:
        for g, st in STYLE.items():  # uniform sets only, as in the EVT figure
            din, dout = all_pair_dists(f[g]["vec_in"][:], f[g]["vec_out"][:])
            ok = (din > 0) & (dout > 0)
            din, dout = din[ok], dout[ok]
            lo, hi = min(lo, din.min()), max(hi, din.max())  # x range: the input distances
            pick = rng.choice(din.size, min(N_DRAW, din.size), replace=False)
            ax.scatter(din[pick], dout[pick], s=1.5, marker="o", color=SCATTER_COLOR[g], linewidths=0,
                       alpha=0.6, rasterized=True, zorder=3 if g in ("C", "cC") else 2)
    ax.plot([lo / 1.5, hi * 1.5], [lo / 1.5, hi * 1.5], color=INK, linewidth=0.8, linestyle=(0, (1, 1.5)), zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(lo / 1.5, hi * 1.5)
    ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(numticks=6))
        axis.set_minor_locator(NullLocator())
    ax.set_xlabel(r"$\|x-y\|$", color=INK)
    ax.set_ylabel(r"$\|f(x)-f(y)\|$", color=INK)
    ax.grid(color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [Line2D([], [], color=SCATTER_COLOR[g], linestyle="none", marker="o", markersize=4, label=st["label"])
               for g, st in STYLE.items()]
    handles.append(Line2D([], [], color=INK, linewidth=0.8, linestyle=(0, (1, 1.5)), label=r"$\|f(x)-f(y)\|=\|x-y\|$"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=legend_ncol(3), frameon=False, fontsize=legend_fs(7),
              columnspacing=1.4)
    fig.tight_layout()
    save_figure(fig, OUT)
