"""Kernel score u of each sample vs its distance from the centre (p or c), simple_stories add_4.add_6 crossing 5
and GPT-2 add_2.add_5 crossing 4 (logr runs: uniform + log-spaced samples -- the same data as det_zero_kl_value_*).
The grey band is where held-out real points score (5th-95th percentile), i.e. what "on the manifold" looks like.
Same colours (SCATTER_COLOR) as the other det_zero scatter figures.

    uv run python -m scripts.analysis.plot_det_zero_kernel_distance
"""
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullLocator

from scripts.analysis.plot_det_zero_manifold_paper import GRID, INK, PAPER_RC, SCATTER_COLOR, STYLE, load
from scripts.analysis.plot_kernel_tangent_overlap_paper import legend_fs, legend_ncol, save_figure

plt.rcParams.update(PAPER_RC)

# one figure per model: (input file, output prefix)
RUNS = [
    ("onnx_models/simple_stories/det_zero_manifold_paper/add_4.add_6/row26685_row26140_manifold_r100_logr_c05.h5", "figures/paper/det_zero_kernel_distance_c05"),  # simple_stories add_4.add_6 (layer-1 FFN), crossing 5
    ("onnx_models/gpt2/det_zero_manifold_paper/add_2.add_5/row182959_row176262_manifold_r1000_logr_c04.h5", "figures/paper/det_zero_kernel_distance_gpt2_c04"),  # GPT-2 add_2.add_5 (layer-1 FFN), crossing 4
]

for FILE, OUT in RUNS:
    d = load(FILE)
    with h5py.File(FILE, "r") as f:
        held = f["calib/u_heldout"][:]
    band = np.percentile(held, [5, 95])
    all_dist = np.concatenate([d[g]["dist"][d[g]["dist"] > 0] for g in STYLE])
    X_LO, X_HI = all_dist.min() / 1.5, all_dist.max() * 1.5  # the full range of the data

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.axhspan(*band, color=GRID, alpha=0.8, linewidth=0, zorder=0)
    ax.axhline(1.0, color=INK, linewidth=0.8, linestyle=(0, (1, 1.5)), zorder=1)
    for g in STYLE:
        m = d[g]["dist"] > 0
        ax.scatter(d[g]["dist"][m], d[g]["u"][m], s=2.5, marker="o", color=SCATTER_COLOR[g], linewidths=0, alpha=0.75,
                   rasterized=True, zorder=3 if g in ("C", "cC") else 2)
    ax.set_xscale("log")
    ax.set_xlim(X_LO, X_HI)
    ax.xaxis.set_major_locator(LogLocator(numticks=7))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r"$\|x-\mathrm{centre}\|$", color=INK)
    ax.set_ylabel(r"Kernel score $u$  ($1$ = on manifold)", color=INK)
    ax.grid(color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [Line2D([], [], color=SCATTER_COLOR[g], linestyle="none", marker="o", markersize=4, label=st["label"])
               for g, st in STYLE.items()]
    handles += [Patch(color=GRID, alpha=0.8, label="held-out real points (5th-95th pct.)"),
                Line2D([], [], color=INK, linewidth=0.8, linestyle=(0, (1, 1.5)), label="$u=1$")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=legend_ncol(3), frameon=False,
              fontsize=legend_fs(7), columnspacing=1.4)
    fig.tight_layout()
    save_figure(fig, OUT)
    plt.close(fig)
    print(f"  held-out band {band[0]:.3f}..{band[1]:.3f}")
