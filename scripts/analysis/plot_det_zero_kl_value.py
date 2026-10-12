"""K_L value of each sample, ||f(x) - f(centre)|| / ||x - centre||, vs ||x - centre||, log-log, for the sample
sets around p and c (crossing 5, uniform + log-spaced samples) -- the logr run (log-spaced samples from 1e-8 to
1e2, uniform ones log-uniform in 1e-6..10), x range = the full range of the data. Same colours (SCATTER_COLOR) as the other det_zero scatter figures.

The float32 precision limit: the forward pass resolves ||f(x) - f(centre)|| only down to a floor F (rounding
accumulated through the sublayer), so no K_L value below F / ||x - centre|| can be told apart from noise.
F is measured here: the median output change of the two v_min lines over their smallest steps, where the
output change no longer scales with the step.

    uv run python -m scripts.analysis.plot_det_zero_kl_value
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

# one figure per model: (input file, output prefix, float32-floor window, far-field line). The floor
# window is where the v_min lines' output change is flat (the forward pass's rounding floor -- it scales
# with the output magnitude: ~1e-6 for simple_stories, ~5e-5 for GPT-2, whose outputs are ~60x larger);
# the far-field line is where every set's ratio has settled to ~1 (the residual stream dominates).
RUNS = [
    ("onnx_models/simple_stories/det_zero_manifold_paper/add_4.add_6/row26685_row26140_manifold_r100_logr_c05.h5", "figures/paper/det_zero_kl_value_c05", (1e-6, 5e-5), 1.0),
    ("onnx_models/gpt2/det_zero_manifold_paper/add_2.add_5/row182959_row176262_manifold_r1000_logr_c04.h5", "figures/paper/det_zero_kl_value_gpt2_c04", (1e-4, 1e-2), 1e2),
]

for FILE, OUT, (F_LO, F_HI), X_FAR in RUNS:
    d = load(FILE)
    all_dist = np.concatenate([d[g]["dist"][d[g]["dist"] > 0] for g in STYLE])
    X_LO, X_HI = all_dist.min() / 1.5, all_dist.max() * 1.5  # the full range of the data
    floor_pts = np.concatenate([d[g]["df"][(d[g]["dist"] >= F_LO) & (d[g]["dist"] <= F_HI)] for g in ("C", "cC")])
    fp32_floor = float(np.median(floor_pts))
    print(f"float32 output floor: {fp32_floor:.3g} (median ||f(x) - f(centre)|| of {floor_pts.size} v_min samples "
          f"between {F_LO:g} and {F_HI:g} from their centre)")

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    for g in STYLE:
        m = (d[g]["dist"] >= X_LO) & (d[g]["dist"] <= X_HI) & (d[g]["df"] > 0)
        ax.scatter(d[g]["dist"][m], d[g]["df"][m] / d[g]["dist"][m], s=2.5, marker="o", color=SCATTER_COLOR[g],
                   linewidths=0, alpha=0.75, rasterized=True, zorder=3 if g in ("C", "cC") else 2)
    xs = np.array([X_LO, X_HI])
    ax.plot(xs, fp32_floor / xs, color=INK, linewidth=0.9, linestyle=(0, (4, 1.5)), zorder=4)
    ax.fill_between(xs, 1e-12, fp32_floor / xs, color=GRID, alpha=0.6, linewidth=0, zorder=0)
    ax.axhline(1.0, color=INK, linewidth=0.8, linestyle=(0, (1, 1.5)), zorder=1)
    ax.axvspan(X_FAR, X_HI, color="#e3f1e6", linewidth=0, zorder=0)  # light tint of the paper green #00663a
    ax.axvline(X_FAR, color=INK, linewidth=0.8, zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(X_LO, X_HI)
    ys = np.concatenate([d[g]["df"][(d[g]["dist"] >= X_LO) & (d[g]["df"] > 0)] / d[g]["dist"][(d[g]["dist"] >= X_LO) & (d[g]["df"] > 0)]
                         for g in STYLE])
    ax.set_ylim(0.5 * ys.min(), 1.5 * ys.max())  # the full range of the data
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(numticks=7))
        axis.set_minor_locator(NullLocator())
    ax.set_xlabel(r"$\|x-\mathrm{centre}\|$", color=INK)
    ax.set_ylabel(r"$\|f(x)-f(\mathrm{centre})\|\,/\,\|x-\mathrm{centre}\|$", color=INK)
    ax.grid(color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [Line2D([], [], color=SCATTER_COLOR[g], linestyle="none", marker="o", markersize=4, label=st["label"])
               for g, st in STYLE.items()]
    handles += [
        Line2D([], [], color=INK, linewidth=0.9, linestyle=(0, (4, 1.5)),
               label="float32 precision"),
        Line2D([], [], color=INK, linewidth=0.8, linestyle=(0, (1, 1.5)), label="ratio $=1$"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=legend_ncol(3), frameon=False, fontsize=legend_fs(7),
              columnspacing=1.4)
    fig.tight_layout()
    save_figure(fig, OUT)
