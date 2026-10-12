"""Toy check of the ratio-histogram / tail / distance-scatter view on a
high-dimensional map with a known singularity:

    f: R^d -> R^d,   f(x)_0 = x_0^2,   f(x)_k = x_k  for k >= 1   (d = --dim, 386)

J_f = diag(2 x_0, 1, ..., 1), so det J = 2 x_0 and f is singular exactly on
the hyperplane x_0 = 0 (and folds R^d across it: f(x) = f(x') for x' the
reflection of x in that hyperplane).

For a pair (x, y) with dx_0 = x_0 - y_0 and dr = the other coordinates' difference,

    g^2 = ||f(x)-f(y)||^2 / ||x-y||^2 = ((x_0 + y_0)^2 dx_0^2 + |dr|^2) / (dx_0^2 + |dr|^2),

a weighted average of (x_0 + y_0)^2 and 1, so the true lower Lipschitz
constant over the sampled region is K_L = min(1, inf |x_0 + y_0|): reached by
pairs differing only along x_0 (dr = 0). K_L = 0 whenever the region contains
both x_0 and -x_0, i.e. whenever it straddles the singularity (near or far).

Three sampling regimes for x_0 (the other d-1 coordinates are always
Uniform(-rest_scale, rest_scale)):

  one_side  x_0 ~ U(--one-side-range), below the singularity only
            (default [-1, -0.1], true K_L = 0.2)
  near      x_0 ~ U(-w, w), both sides close to it (w = --near-width, 0.1)
  far       x_0 ~ U(a, b) or U(-b, -a) with prob 1/2 each, both sides far
            from it (--far-range a b, default 0.5 1.0), with no points near 0

Each regime runs the same pipeline as
scripts/lipschitz/lipschitz_ratio_bootstrap_eqd_sweep_gpu.py (random pairs,
bootstrap-EQD threshold, GPD fit, c_hat) and gets one figure row: the domain vs.
image of the sampled points along x_0, the ratio histogram, the lower tail
(g <= u) against the true K_L and c_hat, and a scatter of ||x-y|| against
||f(x)-f(y)||.

Note: with isotropic sampling in d = 386 dims, |dr|^2 ~ (d-1) * (2/3) rest_scale^2
dwarfs dx_0^2, so random pairs almost never see the singular direction and g
concentrates near 1 in all three regimes. Lower --rest-scale (e.g. 0.01) to
make the x_0 fold dominate the pair geometry.

Usage:
    uv run python -m scripts.analysis.toy_singular_identity --out toy_singular_identity.png
    uv run python -m scripts.analysis.toy_singular_identity --rest-scale 0.01
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.lipschitz.lipschitz_ratio_bootstrap_eqd_sweep_gpu import (
    compute_lipschitz_ratios_gpu,
    fit_gpd_and_c_hat_gpu,
    select_bootstrap_eqd_threshold_gpu_multi,
)


def f(x: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    out[:, 0] = x[:, 0] ** 2
    return out


def sample_x0(regime: str, n: int, args: argparse.Namespace, gen: torch.Generator, device: str) -> torch.Tensor:
    u = torch.rand(n, device=device, generator=gen, dtype=torch.float64)
    if regime == "one_side":
        lo, hi = args.one_side_range
        return lo + (hi - lo) * u
    if regime == "near":
        return args.near_width * (2 * u - 1)
    if regime == "far":
        a, b = args.far_range
        sign = torch.where(torch.rand(n, device=device, generator=gen) < 0.5, -1.0, 1.0).double()
        return sign * (a + (b - a) * u)
    raise ValueError(regime)


def true_k_l(regime: str, args: argparse.Namespace) -> float:
    """min(1, inf |x_0 + y_0|) over the regime's x_0 support."""
    if regime == "one_side":
        lo, hi = args.one_side_range
        inf_sum = 2 * min(abs(lo), abs(hi)) if lo * hi > 0 else 0.0
    else:
        inf_sum = 0.0  # support is symmetric about 0, so x_0 = -y_0 is reachable
    return min(1.0, inf_sum)


def run_regime(regime: str, args: argparse.Namespace, gen: torch.Generator, device: str) -> dict:
    x = args.rest_scale * (2 * torch.rand(args.n, args.dim, device=device, generator=gen, dtype=torch.float64) - 1)
    x[:, 0] = sample_x0(regime, args.n, args, gen, device)
    y = f(x)

    input_dist, output_dist, ratio, _, _ = compute_lipschitz_ratios_gpu(x, y, args.n_pairs, gen, return_pairs=True)
    g_sorted = torch.sort(ratio).values
    [fit] = select_bootstrap_eqd_threshold_gpu_multi(
        [g_sorted], n_tail_min=args.n_tail_min, n_tail_max=args.n_tail_max, n_tail_step=args.n_tail_step,
        n_boot=args.eqd_n_boot, m_plotting_points=args.m_plotting_points, generator=gen, group_size=1,
    )
    gpd = fit_gpd_and_c_hat_gpu(g_sorted, fit["best_n_tail"], fit["threshold"])
    return {
        "x0": x[:, 0].cpu().numpy(),
        "input_dist": input_dist.cpu().numpy(),
        "output_dist": output_dist.cpu().numpy(),
        "ratio": ratio.cpu().numpy(),
        "g_sorted": g_sorted.cpu().numpy(),
        "n_tail": fit["best_n_tail"],
        "threshold": fit["threshold"],
        "xi": gpd["shape_xi"],
        "c_hat": gpd["c_hat"],
        "k_l": true_k_l(regime, args),
    }


def mark_k_l(ax, k_l: float, data: np.ndarray, c_hat: float) -> None:
    """Draws the true K_L line only if it's near the plotted data; otherwise
    (e.g. rest_scale ~ 1, where g sits in a narrow band just below 1) a far-off
    K_L would squash the histogram to a single bar, so it goes in the legend only."""
    lo, hi = data.min(), data.max()
    if np.isfinite(c_hat):
        lo, hi = min(lo, c_hat), max(hi, c_hat)
    pad = 0.05 * (hi - lo) or 1e-6
    if lo - 0.5 * (hi - lo) <= k_l <= hi:
        ax.axvline(k_l, color="black", lw=1.5, label=f"true K_L={k_l:.4g}")
    else:
        ax.plot([], [], " ", label=f"true K_L={k_l:.4g} (off-axis)")
        ax.set_xlim(lo - pad, hi + pad)


def plot_row(axes, regime: str, r: dict, n_scatter: int, seed: int) -> None:
    ax_dom, ax_hist, ax_tail, ax_scatter = axes
    threshold, c_hat, k_l = r["threshold"], r["c_hat"], r["k_l"]

    grid = np.linspace(min(r["x0"].min(), -abs(r["x0"]).max()), max(r["x0"].max(), abs(r["x0"]).max()), 200)
    ax_dom.plot(grid, grid**2, color="lightgray", lw=1.5, label="f(x)_0 = x_0^2")
    ax_dom.scatter(r["x0"], r["x0"] ** 2, s=4, color="steelblue", alpha=0.5, label=f"samples (n={r['x0'].size})")
    ax_dom.axvline(0, color="black", ls=":", lw=1, label="singularity x_0 = 0")
    ax_dom.set_xlabel("x_0 (domain)")
    ax_dom.set_ylabel("f(x)_0 (image)")
    ax_dom.set_title(f"[{regime}] sampled domain / image along x_0")
    ax_dom.legend(fontsize=7)

    ax_hist.hist(r["g_sorted"], bins=100, color="steelblue", alpha=0.8)
    ax_hist.axvline(threshold, color="orange", ls="--", label=f"threshold u={threshold:.4g}")
    if np.isfinite(c_hat):
        ax_hist.axvline(c_hat, color="red", ls="--", label=f"c_hat={c_hat:.4g}")
    mark_k_l(ax_hist, k_l, r["g_sorted"], c_hat)
    ax_hist.set_xlabel("ratio g = ||f(x)-f(y)|| / ||x-y||")
    ax_hist.set_ylabel("count")
    ax_hist.set_title(f"[{regime}] ratio distribution (N={r['g_sorted'].size:,})")
    ax_hist.legend(fontsize=7)

    tail = r["g_sorted"][: r["n_tail"]]
    ax_tail.hist(tail, bins=50, color="steelblue", alpha=0.8)
    ax_tail.axvline(threshold, color="orange", ls="--", label=f"u={threshold:.4g}")
    if np.isfinite(c_hat):
        ax_tail.axvline(c_hat, color="red", ls="--", label=f"c_hat={c_hat:.4g}")
    mark_k_l(ax_tail, k_l, tail, c_hat)
    ax_tail.set_xlabel("ratio g (g <= u)")
    ax_tail.set_ylabel("count")
    ax_tail.set_title(f"[{regime}] lower tail (n_tail={r['n_tail']}, xi={r['xi']:.3f})")
    ax_tail.legend(fontsize=7)

    in_tail = r["ratio"] <= threshold
    rng = np.random.default_rng(seed)
    bulk = rng.choice(np.flatnonzero(~in_tail), size=min(n_scatter, int((~in_tail).sum())), replace=False)
    ax_scatter.scatter(r["input_dist"][bulk], r["output_dist"][bulk], s=2, color="lightgray", alpha=0.5,
                       label=f"pairs ({bulk.size:,} of {r['ratio'].size:,} shown)")
    ax_scatter.scatter(r["input_dist"][in_tail], r["output_dist"][in_tail], s=5, color="red",
                       label=f"tail pairs, g <= u ({int(in_tail.sum())})")
    d_max = r["input_dist"].max()
    ax_scatter.plot([0, d_max], [0, d_max], color="black", ls=":", lw=1, label="slope 1")
    ax_scatter.plot([0, d_max], [0, k_l * d_max], color="black", lw=1, label=f"slope true K_L={k_l:.3g}")
    ax_scatter.set_xlabel("||x - y||")
    ax_scatter.set_ylabel("||f(x) - f(y)||")
    ax_scatter.set_title(f"[{regime}] input vs output distance")
    ax_scatter.legend(fontsize=7)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dim", type=int, default=386, help="ambient dimension d")
    parser.add_argument("--n", type=int, default=1000, help="points per regime")
    parser.add_argument("--rest-scale", type=float, default=1.0,
                        help="coordinates 1..d-1 ~ Uniform(-rest_scale, rest_scale)")
    parser.add_argument("--one-side-range", type=float, nargs=2, default=[-1.0, -0.1], metavar=("LO", "HI"),
                        help="x_0 range for the one_side regime (must not contain 0)")
    parser.add_argument("--near-width", type=float, default=0.1, help="near regime: x_0 ~ U(-w, w)")
    parser.add_argument("--far-range", type=float, nargs=2, default=[0.5, 1.0], metavar=("A", "B"),
                        help="far regime: |x_0| ~ U(A, B), random sign")
    parser.add_argument("--regimes", nargs="+", default=["one_side", "near", "far"],
                        choices=["one_side", "near", "far"])
    # The rest match lipschitz_ratio_bootstrap_eqd_sweep_gpu.py's defaults.
    parser.add_argument("--n-pairs", type=int, default=200_000)
    parser.add_argument("--n-tail-min", type=int, default=200)
    parser.add_argument("--n-tail-max", type=int, default=300)
    parser.add_argument("--n-tail-step", type=int, default=200)
    parser.add_argument("--eqd-n-boot", type=int, default=200)
    parser.add_argument("--m-plotting-points", type=int, default=50)
    parser.add_argument("--n-scatter", type=int, default=20_000, help="bulk pairs drawn in the scatter panel")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="toy_singular_identity.png")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    args = parser.parse_args()
    
    

    lo, hi = args.one_side_range
    if lo * hi <= 0:
        parser.error("--one-side-range must lie strictly on one side of 0")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(args.seed)

    fig, axes = plt.subplots(len(args.regimes), 4, figsize=(24, 5 * len(args.regimes)), squeeze=False)
    print(f"f(x) = (x_0^2, x_1, ..., x_{args.dim - 1}), n={args.n}, rest_scale={args.rest_scale:g}, device={device}")
    for row, regime in zip(axes, args.regimes):
        r = run_regime(regime, args, gen, device)
        print(f"  [{regime:8s}] x_0 in [{r['x0'].min():+.3f}, {r['x0'].max():+.3f}]  true K_L={r['k_l']:.4f}  "
              f"min g={r['g_sorted'][0]:.5f}  u={r['threshold']:.5f} (n_tail={r['n_tail']})  "
              f"xi={r['xi']:.4f}  c_hat={r['c_hat']:.5f}")
        plot_row(row, regime, r, args.n_scatter, args.seed)

    fig.suptitle(f"f(x)_0 = x_0^2, identity elsewhere, on R^{args.dim} (rest_scale={args.rest_scale:g})")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    plt.close(fig)
    print(f"  plot -> {args.out}")


if __name__ == "__main__":
    main()
