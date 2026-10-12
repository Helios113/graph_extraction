"""Kernel score u of the samples around p and c (crossing 5), from the radius-1-spacing run:
each set's uniform samples sit ~one local spacing from their centre, so every set is one
compact bump. Linear heights, each set scaled to peak 1. Same colours as the other
det_zero_manifold paper figures.

    uv run python -m scripts.analysis.plot_det_zero_kernel_score
"""
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from scripts.analysis.plot_det_zero_manifold_paper import GRID, INK, PAPER_RC, STYLE
from scripts.analysis.plot_kernel_tangent_overlap_paper import legend_fs, legend_ncol, save_figure

plt.rcParams.update(PAPER_RC)

# one figure per model: (input file, output prefix)
RUNS = [
    ("onnx_models/simple_stories/det_zero_manifold/add_4.add_6/row26685_row26140_manifold_c05.h5", "figures/paper/det_zero_kernel_score_c05"),  # simple_stories add_4.add_6 (layer-1 FFN), crossing 5
    ("onnx_models/gpt2/det_zero_manifold/add_2.add_5/row182959_row176262_manifold_c04.h5", "figures/paper/det_zero_kernel_score_gpt2_c04"),  # GPT-2 add_2.add_5 (layer-1 FFN), crossing 4
]

for FILE, OUT in RUNS:
    with h5py.File(FILE, "r") as f:
        u = {g: f[g]["u"][:] for g in STYLE}  # uniform sets only
    lo = min(v.min() for v in u.values())
    hi = max(v.max() for v in u.values())
    bins = np.linspace(lo - 0.01, hi + 0.01, 51)

    fig, ax = plt.subplots(figsize=(5.5, 2.7))
    for g, st in STYLE.items():
        counts, edges = np.histogram(u[g], bins=bins)
        ax.stairs(counts / counts.max(), edges, color=st["color"], linestyle=st["ls"], linewidth=1.3)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"Kernel score $u$  ($1$ = on manifold)", color=INK)
    ax.set_ylabel("Count / peak", color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Line2D([], [], color=st["color"], linestyle=st["ls"], linewidth=1.3, label=st["label"])
                       for st in STYLE.values()], loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=legend_ncol(3),
              frameon=False, fontsize=legend_fs(7), handlelength=2.2, columnspacing=1.4)
    fig.tight_layout()
    save_figure(fig, OUT)
