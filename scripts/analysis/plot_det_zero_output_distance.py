"""Output vs input distance from the centre (p or c), log-log, for the sample sets around p and c
(crossing 5, uniform + log-spaced samples) -- the radius-1-spacing run, the same data as
det_zero_evt_c05 and det_zero_kernel_score_c05. Same colours as the other
det_zero_manifold paper figures.

    uv run python -m scripts.analysis.plot_det_zero_output_distance
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullLocator

from scripts.analysis.plot_det_zero_manifold_paper import GRID, INK, PAPER_RC, SCATTER_COLOR, STYLE, load
from scripts.analysis.plot_kernel_tangent_overlap_paper import legend_fs, legend_ncol, save_figure

plt.rcParams.update(PAPER_RC)

# one figure per model: (input file, output prefix)
RUNS = [
    ("onnx_models/simple_stories/det_zero_manifold/add_4.add_6/row26685_row26140_manifold_c05.h5", "figures/paper/det_zero_output_distance_c05"),  # simple_stories add_4.add_6 (layer-1 FFN), crossing 5
    ("onnx_models/gpt2/det_zero_manifold/add_2.add_5/row182959_row176262_manifold_c04.h5", "figures/paper/det_zero_output_distance_gpt2_c04"),  # GPT-2 add_2.add_5 (layer-1 FFN), crossing 4
]

for FILE, OUT in RUNS:
    d = load(FILE)
    all_dist = np.concatenate([d[g]["dist"][d[g]["dist"] > 0] for g in STYLE])
    X_LO, X_HI = all_dist.min() / 1.5, all_dist.max() * 1.5  # the full range of the data
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    for g, st in STYLE.items():
        m = (d[g]["dist"] >= X_LO) & (d[g]["dist"] <= X_HI) & (d[g]["df"] > 0)
        ax.scatter(d[g]["dist"][m], d[g]["df"][m], s=2.5, marker="o", color=SCATTER_COLOR[g], linewidths=0,
                   alpha=0.75, rasterized=True, zorder=3 if g in ("C", "cC") else 2)
    ax.plot([X_LO, X_HI], [X_LO, X_HI], color=INK, linewidth=0.8, linestyle=(0, (1, 1.5)), zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(X_LO, X_HI)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(numticks=6))
        axis.set_minor_locator(NullLocator())
    ax.set_xlabel(r"$\|x-\mathrm{centre}\|$", color=INK)
    ax.set_ylabel(r"$\|f(x)-f(\mathrm{centre})\|$", color=INK)
    ax.grid(color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [Line2D([], [], color=SCATTER_COLOR[g], linestyle="none", marker="o", markersize=4, label=st["label"])
               for g, st in STYLE.items()]
    handles.append(Line2D([], [], color=INK, linewidth=0.8, linestyle=(0, (1, 1.5)), label=r"$\|f(x)-f(\mathrm{centre})\|=\|x-\mathrm{centre}\|$"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=legend_ncol(3), frameon=False, fontsize=legend_fs(7),
              columnspacing=1.4)
    fig.tight_layout()
    save_figure(fig, OUT)
