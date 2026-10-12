"""Publication (ICLR, 5.5in text width) version of
plot_kernel_tangent_overlap.py's histogram figure: a 3 x 2 grid, rows = three
transformer layers (--layers, 1-based, default 1 3 6), columns = that layer's
attention and FFN sublayer. Each panel: overlap ||P_T v||^2 of the Jacobian's
smallest (v_min, red) and largest (v_max, green, dashed) singular vectors with
the local tangent space, against the random-direction Beta(d/2, (D-d)/2)
baseline (grey), d = dim explaining --var-frac of local PCA variance.

Attention vs FFN is read from each sublayer's ONNX forward subgraph (Softmax
present = attention), not assumed from the name. No suptitle -- the caption
goes in the paper. Writes {--out}.pdf (TrueType fonts embedded) and {--out}.png.

    uv run python -m scripts.analysis.plot_kernel_tangent_overlap_paper configs/simple_stories.toml \\
        onnx_models/simple_stories/kernel_tangent_overlap_samples_0 --side in
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnx
from matplotlib.lines import Line2D

from config import load_config
from execute.sharded_writer import load_shard_rows
from scripts.analysis.plot_kernel_tangent_overlap import intrinsic_dim

C_MIN, C_MAX, C_RAND, INK, GRID = "#f0614a", "#00663a", "#bdbdbd", "#222222", "#e6e6e6"

# shared with the other *_paper.py figures so they match in the paper
PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# FIG_MODE=presentation (env) re-renders every *_paper.py figure for slides instead: PNG only at
# PRESENTATION_DPI, legend text scaled by PRESENTATION_LEGEND_SCALE, written under figures/presentation/
# instead of figures/paper/ (so the paper's PDFs are never touched)
PRESENTATION = os.environ.get("FIG_MODE", "paper") == "presentation"
PRESENTATION_DPI = 600
PRESENTATION_LEGEND_SCALE = 1.6
if PRESENTATION:
    PAPER_RC["legend.fontsize"] *= PRESENTATION_LEGEND_SCALE
plt.rcParams.update(PAPER_RC)


def legend_fs(size: float) -> float:
    """Legend font size: `size` for the paper, scaled up in presentation mode."""
    return size * PRESENTATION_LEGEND_SCALE if PRESENTATION else size


def legend_ncol(ncol: int) -> int:
    """Legend columns: `ncol` for the paper; at most 2 in presentation mode, where the bigger text of 3
    columns runs past the figure edge."""
    return min(ncol, 2) if PRESENTATION else ncol


def save_figure(fig, out: str) -> str:
    """Paper: {out}.pdf (dpi 400 for rasterized parts) + {out}.png at 300 dpi. Presentation: only a
    PRESENTATION_DPI png, with figures/paper/ in `out` swapped for figures/presentation/. Returns the
    prefix written."""
    if PRESENTATION:
        out = out.replace(os.path.join("figures", "paper"), os.path.join("figures", "presentation"), 1)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if PRESENTATION:
        fig.savefig(out + ".png", dpi=PRESENTATION_DPI, bbox_inches="tight", pad_inches=0.05)
        print(f"wrote {out}.png")
    else:
        fig.savefig(out + ".pdf", dpi=400)
        fig.savefig(out + ".png", dpi=300)
        print(f"wrote {out}.pdf / .png")
    return out


def sublayer_kind(cfg, sublayer) -> str:
    path = os.path.join("onnx_models", cfg.name, "sub_blocks",
                        f"{sublayer.input}.{sublayer.output}.subgraph.forward.onnx")
    if not os.path.exists(path):
        # no exported subgraph (e.g. GPT-2): fall back to the pre-LN layout every config here has --
        # attention and FFN alternate, starting with attention right after the embedding sublayer
        body = [s for s in cfg.sublayers if s.input != cfg.input_name]
        return "attn" if body.index(sublayer) % 2 == 0 else "ffn"
    ops = {n.op_type for n in onnx.load(path, load_external_data=False).graph.node}
    return "attn" if "Softmax" in ops else "ffn"


def transformer_blocks(cfg) -> list[dict[str, str]]:
    """[{"attn": "{in}.{out}", "ffn": "{in}.{out}"}, ...], one per transformer
    layer: the L-th attention sublayer and the FFN right after it."""
    blocks, cur = [], {}
    for s in cfg.sublayers:
        if s.input == cfg.input_name:
            continue
        cur[sublayer_kind(cfg, s)] = f"{s.input}.{s.output}"
        if "attn" in cur and "ffn" in cur:
            blocks.append(cur)
            cur = {}
    return blocks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("data_dir", help="compute_kernel_tangent_overlap.py --out-dir (project-relative)")
    p.add_argument("--side", choices=["in", "out"], default="in")
    p.add_argument("--layers", type=int, nargs=3, default=[1, 3, 6], help="1-based transformer layers, one per row")
    p.add_argument("--var-frac", type=float, default=0.95)
    p.add_argument("--bins", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="path prefix (default figures/paper/kernel_tangent_overlap_{side})")
    args = p.parse_args()

    blocks = transformer_blocks(load_config(args.config))

    coef_name, ev_name = ("v_coef2", "in_evals") if args.side == "in" else ("u_coef2", "out_evals")
    vec = "v" if args.side == "in" else "u"
    space = r"T_pM" if args.side == "in" else r"T_{f(p)}M"
    rng = np.random.default_rng(args.seed)

    panels = {}
    for layer in args.layers:
        for kind in ("attn", "ffn"):
            name = blocks[layer - 1][kind]
            rows = load_shard_rows(".", os.path.join(args.data_dir, name), dataset_names=[coef_name, ev_name])
            coef2 = rows[coef_name].astype(np.float64)
            n, _, dim = coef2.shape
            d = intrinsic_dim(rows[ev_name].astype(np.float64), args.var_frac)
            at_d = np.arange(n), d - 1
            panels[layer, kind] = dict(
                vmin=np.cumsum(coef2[:, 0], axis=1)[at_d],
                vmax=np.cumsum(coef2[:, -1], axis=1)[at_d],
                rand=rng.beta(d / 2, np.maximum(dim - d, 1) / 2),
            )
            med = {k: np.median(x) for k, x in panels[layer, kind].items()}
            print(f"layer {layer} {kind:4s} {name:<16} median vmin {med['vmin']:.3f} "
                  f"rand {med['rand']:.3f} vmax {med['vmax']:.3f}")

    # shared x range: up to the 99.9th percentile of everything shown, rounded up to 0.1
    xmax = min(1.0, np.ceil(10 * max(np.percentile(np.concatenate(list(v.values())), 99.9)
                                     for v in panels.values())) / 10)
    bins = np.linspace(0, xmax, args.bins + 1)

    fig, axs = plt.subplots(3, 2, figsize=(5.5, 4.8), sharex=True)
    for r, layer in enumerate(args.layers):
        for c, kind in enumerate(("attn", "ffn")):
            ax, v = axs[r, c], panels[layer, kind]
            ax.hist(v["rand"], bins=bins, density=True, color=C_RAND, linewidth=0, label="Random direction")
            ax.hist(v["vmax"], bins=bins, density=True, histtype="step", color=C_MAX, linewidth=1.3,
                    linestyle=(0, (4, 1.5)), label=r"$v_{\max}$ (largest $\sigma$)")
            ax.hist(v["vmin"], bins=bins, density=True, histtype="step", color=C_MIN, linewidth=1.3,
                    label=r"$v_{\min}$ (kernel)" if args.side == "in" else r"$u_{\min}$ (cokernel)")
            # reference: a kernel direction lying entirely in the tangent space has overlap exactly 1
            ax.axvline(1.0, color=INK, linewidth=1.1, linestyle=(0, (1, 1.5)),
                       label=rf"${vec}_{{\min}}\subset {space}$ (overlap $=1$)")
            ax.set_xlim(0, 1.03)
            ax.set_xticks(np.arange(0, 1.01, 0.2))
            ax.grid(axis="y", color=GRID, linewidth=0.5)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if r == 0:
                ax.set_title("Attention" if kind == "attn" else "FFN", fontsize=9, color=INK, pad=4)
            if c == 0:
                ax.set_ylabel(f"Layer {layer}\nDensity", color=INK)
            if r == 2:
                ax.set_xlabel(rf"$\|P_{{{space}}}\,{vec}\|^2$", color=INK)

    # added in order: random, v_max, v_min, reference line. Step histograms get box legend handles by
    # default; lines read better
    (h_rand, h_max, h_min, h_ref), (l_rand, l_max, l_min, l_ref) = axs[0, 0].get_legend_handles_labels()
    as_line = [Line2D([], [], color=h.get_edgecolor(), linewidth=1.3, linestyle=h.get_linestyle())
               for h in (h_min, h_max)]
    fig.legend([*as_line, h_rand, h_ref], [l_min, l_max, l_rand, l_ref], loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=2.0, fontsize=legend_fs(10))
    fig.tight_layout(rect=(0, 0, 1, 0.84 if PRESENTATION else 0.895), h_pad=0.8, w_pad=1.0)

    save_figure(fig, args.out or os.path.join("figures", "paper", f"kernel_tangent_overlap_{args.side}"))


if __name__ == "__main__":
    main()
