"""Distribution of the overlap between each sublayer Jacobian's kernel
direction and the local tangent space T_pM, at every real perturbed point --
plots compute_kernel_tangent_overlap.py's output.

overlap = ||P_{T_d} v||^2 = cos^2(angle between the line through v and T_d),
where T_d is the span of the top-d local PCA directions at the point. d is
chosen per point as the smallest d whose PCA eigenvalues explain
--var-frac of the neighbourhood variance. A uniformly random unit direction
has overlap ~ Beta(d/2, (D-d)/2) (mean d/D) -- drawn per point with that
point's d as the baseline: kernel overlap below it means the kernel avoids
the manifold, above it means the kernel lies along it.

--side in (default): right singular vectors v vs T_pM of the sublayer's input
cloud (the kernel proper -- input directions J collapses).
--side out: left singular vectors u vs T_{f(p)}M of the output cloud (the
cokernel -- output directions J can't reach).

Writes two PNGs next to --out (default {data_dir}/kernel_tangent_overlap_{side}):
  *_hist.png     one panel per sublayer: overlap histograms of the smallest
                 singular vector, the largest one (contrast), and the
                 random-direction baseline.
  *_profile.png  one panel per sublayer: ||P_{T_d} v_min||^2 as a function of
                 d (median and IQR over points) against the random d/D line --
                 independent of the --var-frac choice.
and prints a per-sublayer summary table. CPU-only.

    uv run python -m scripts.analysis.plot_kernel_tangent_overlap \\
        onnx_models/simple_stories/kernel_tangent_overlap_samples_0
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from execute.sharded_writer import load_shard_rows

INK, INK2, GRID, SURFACE = "#1a1a19", "#5f5e58", "#e4e3dc", "#fcfcfb"
C_MIN, C_MAX, C_RAND = "#2a78d6", "#eb6834", "#9a998f"


def intrinsic_dim(evals: np.ndarray, var_frac: float) -> np.ndarray:
    """Per-row smallest d with cumsum(evals)[:d] >= var_frac * total."""
    cum = np.cumsum(evals, axis=1)
    return (cum < var_frac * cum[:, -1:]).sum(axis=1) + 1


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_dir", help="compute_kernel_tangent_overlap.py --out-dir (project-relative)")
    p.add_argument("--side", choices=["in", "out"], default="in")
    p.add_argument("--var-frac", type=float, default=0.95, help="explained variance defining T_pM's dimension")
    p.add_argument("--bins", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="output path prefix")
    args = p.parse_args()

    def depth(key):  # config order: position of the sublayer's input in the residual chain
        a = json.load(open(key + ".manifest.json"))["attrs"]["sublayer_input"]
        return -1 if a == "embedding" else int(a.split("_")[-1])

    keys = sorted(
        (m[: -len(".manifest.json")] for m in glob.glob(os.path.join(args.data_dir, "*.manifest.json"))),
        key=depth,
    )
    if not keys:
        raise SystemExit(f"no *.manifest.json under {args.data_dir}")

    coef_name, ev_name = ("v_coef2", "in_evals") if args.side == "in" else ("u_coef2", "out_evals")
    vec = "v" if args.side == "in" else "u"
    space = "T_pM (input cloud)" if args.side == "in" else "T_f(p)M (output cloud)"
    rng = np.random.default_rng(args.seed)

    n = len(keys)
    ncol = min(4, n)
    nrow = -(-n // ncol)
    figsize = (max(3.6 * ncol, 10), 2.8 * nrow + 0.8)
    fig_h, axs_h = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False, facecolor=SURFACE)
    fig_p, axs_p = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False, facecolor=SURFACE)

    print(f"side={args.side}  var_frac={args.var_frac}")
    print(f"{'sublayer':<16}{'n':>7}{'d med':>7}{'min med':>9}{'rand med':>9}{'max med':>9}"
          f"{'P(min<rand)':>12}{'s_min med':>11}")
    bins = np.linspace(0, 1, args.bins + 1)
    for i, key in enumerate(keys):
        name = os.path.basename(key)
        rows = load_shard_rows(".", key, dataset_names=[coef_name, ev_name, "sigma"])
        coef2 = rows[coef_name].astype(np.float64)  # (N, n_vec, D)
        N, _, D = coef2.shape
        d = intrinsic_dim(rows[ev_name].astype(np.float64), args.var_frac)
        cum_min = np.cumsum(coef2[:, 0], axis=1)
        cum_max = np.cumsum(coef2[:, -1], axis=1)
        ov_min = cum_min[np.arange(N), d - 1]
        ov_max = cum_max[np.arange(N), d - 1]
        ov_rand = rng.beta(d / 2, np.maximum(D - d, 1) / 2)

        ax = axs_h.flat[i]
        style(ax)
        ax.hist(ov_rand, bins=bins, density=True, color=C_RAND, alpha=0.35, label="random direction")
        ax.hist(ov_max, bins=bins, density=True, histtype="step", color=C_MAX, linewidth=2,
                label=f"{vec}_max (largest σ)")
        ax.hist(ov_min, bins=bins, density=True, histtype="step", color=C_MIN, linewidth=2,
                label=f"{vec}_min (kernel)")
        ax.set_title(f"{name}   median d = {int(np.median(d))}/{D}", fontsize=9, color=INK)

        ax = axs_p.flat[i]
        style(ax)
        dd = np.arange(1, D + 1)
        q = np.percentile(cum_min, [25, 50, 75], axis=0)
        ax.fill_between(dd, q[0], q[2], color=C_MIN, alpha=0.2, linewidth=0)
        ax.plot(dd, q[1], color=C_MIN, linewidth=2, label=f"{vec}_min median (IQR band)")
        ax.plot(dd, np.median(cum_max, axis=0), color=C_MAX, linewidth=2, label=f"{vec}_max median")
        ax.plot(dd, dd / D, color=C_RAND, linewidth=1.5, linestyle="--", label="random: d/D")
        ax.axvline(np.median(d), color=INK2, linewidth=1, linestyle=":", label=f"median d @ {args.var_frac:.0%} var")
        ax.set_xlim(1, D)
        ax.set_ylim(0, 1)
        ax.set_title(name, fontsize=9, color=INK)

        print(f"{name:<16}{N:>7}{np.median(d):>7.0f}{np.median(ov_min):>9.3f}{np.median(ov_rand):>9.3f}"
              f"{np.median(ov_max):>9.3f}{np.mean(ov_min < ov_rand):>12.3f}{np.median(rows['sigma'][:, 0]):>11.3g}")

    for axs in (axs_h, axs_p):
        for ax in axs.flat[n:]:
            ax.set_visible(False)
    for ax in axs_h[-1]:
        ax.set_xlabel(f"‖P_T {vec}‖²  (cos² angle to T)", fontsize=8, color=INK2)
    for ax in axs_h[:, 0]:
        ax.set_ylabel("density", fontsize=8, color=INK2)
    for ax in axs_p[-1]:
        ax.set_xlabel("tangent dim d (top-d PCA directions)", fontsize=8, color=INK2)
    for ax in axs_p[:, 0]:
        ax.set_ylabel(f"‖P_(T_d) {vec}‖²", fontsize=8, color=INK2)

    out = args.out or os.path.join(args.data_dir, f"kernel_tangent_overlap_{args.side}")
    for fig, axs, suffix, title in (
        (fig_h, axs_h, "hist", f"Overlap of Jacobian singular directions with {space}, "
                               f"d = dim explaining {args.var_frac:.0%} of local variance"),
        (fig_p, axs_p, "profile", f"‖P_(T_d) {vec}_min‖² vs tangent dimension d, {space}"),
    ):
        handles, labels = axs.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9,
                   bbox_to_anchor=(0.5, 0.965))
        fig.suptitle(title, fontsize=11, color=INK, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(f"{out}_{suffix}.png", dpi=130, facecolor=SURFACE)
        print(f"wrote {out}_{suffix}.png")


if __name__ == "__main__":
    main()
