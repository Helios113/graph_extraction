"""Toy sanity check of the K_L (lower Lipschitz / co-Lipschitz constant)
pipeline on f(x) = x^2, where the right answer is known in closed form.

Runs exactly the steps scripts/lipschitz/lipschitz_ratio_bootstrap_eqd_sweep_gpu.py
runs on real model outputs, just on a 1-D cloud:

  1. sample n points x ~ Uniform(--lo, --hi), map them through f(x) = x^2,
  2. pairwise ratios g = |f(x) - f(y)| / |x - y| over --n-pairs random pairs
     (compute_lipschitz_ratios_gpu),
  3. bootstrap-EQD threshold selection over the n_tail grid
     (select_bootstrap_eqd_threshold_gpu_multi),
  4. GPD fit to the lower-tail exceedances u - g and
     c_hat = u + sigma/xi (fit_gpd_and_c_hat_gpu), plus the PWM fit for
     comparison, then compare_gpd_estimators.plot_detail's 8-panel figure.

For f(x) = x^2 the ratio is |x^2 - y^2| / |x - y| = |x + y|, so:
  - true K_L = inf |x + y| over the domain = 2*lo when lo >= 0 (0 if the
    domain straddles 0, since x = -y is then reachable),
  - with x, y ~ U(lo, hi), x + y has a triangular density near 2*lo -- it
    rises linearly from zero -- so the lower-tail CDF is ~ z^2 and the
    true GPD shape is xi = -1/2. A working pipeline should give
    xi ~ -0.5 and c_hat ~ 2*lo.
  - if the domain straddles 0, |x + y| has a flat (nonzero) density at 0,
    so the lower-tail CDF is ~ z and the true shape is xi = -1, c_hat ~ 0.

Usage:
    uv run python -m scripts.analysis.toy_kl_x_squared --plot toy_kl_x_squared.png
    uv run python -m scripts.analysis.toy_kl_x_squared --lo -1 --hi 1   # true K_L = 0
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.analysis.compare_gpd_estimators import plot_detail
from scripts.analysis.gpd_pwm_fit import gpd_pwm_fit
from scripts.lipschitz.lipschitz_ratio_bootstrap_eqd_sweep_gpu import (
    compute_lipschitz_ratios_gpu,
    fit_gpd_and_c_hat_gpu,
    select_bootstrap_eqd_threshold_gpu_multi,
)


def f(x: torch.Tensor) -> torch.Tensor:
    return x**2


def true_k_l(lo: float, hi: float) -> float:
    """inf over x != y in [lo, hi] of |x + y|."""
    if lo >= 0:
        return 2 * lo
    if hi <= 0:
        return 2 * abs(hi)
    return 0.0


def plot_toy(out_path: str, x: np.ndarray, x_i: np.ndarray, x_j: np.ndarray, ratio: np.ndarray,
             g_sorted: np.ndarray, threshold: float, c_hat: float, k_l: float,
             n_scatter: int = 20_000, seed: int = 0) -> None:
    """The toy-specific view plot_detail doesn't have: the function and its
    samples, which (x, y) pairs the ratios came from (tail pairs, g <= u,
    highlighted), and the lower tail of g against the known true K_L."""
    fig, (ax_f, ax_pairs, ax_tail) = plt.subplots(1, 3, figsize=(16, 4.6))

    grid = np.linspace(x.min(), x.max(), 200)
    ax_f.plot(grid, grid**2, color="steelblue", lw=1.5, label="f(x) = x^2")
    ax_f.scatter(x, x**2, s=4, color="black", alpha=0.3, label=f"samples (n={x.size})")
    ax_f.set_xlabel("x")
    ax_f.set_ylabel("f(x)")
    ax_f.set_title("function and sampled points")
    ax_f.legend()

    in_tail = ratio <= threshold
    rng = np.random.default_rng(seed)
    bulk = rng.choice(np.flatnonzero(~in_tail), size=min(n_scatter, int((~in_tail).sum())), replace=False)
    ax_pairs.scatter(x_i[bulk], x_j[bulk], s=2, color="lightgray", alpha=0.5,
                     label=f"all pairs ({bulk.size:,} of {ratio.size:,} shown)")
    ax_pairs.scatter(x_i[in_tail], x_j[in_tail], s=6, color="red",
                     label=f"tail pairs, g <= u ({int(in_tail.sum())}, all shown)")
    lim = (x.min(), x.max())
    # g = |x + y| for x^2, so the tail is the band between the lines |x + y| = K_L and |x + y| = u
    for level, style, name in [(k_l, "-", "true K_L"), (threshold, "--", "u")]:
        for sign in ([1, -1] if level > 0 else [1]):
            ax_pairs.plot(lim, [sign * level - lim[0], sign * level - lim[1]], color="black", ls=style, lw=1,
                          label=f"|x + y| = {name}" if sign == 1 else None)
    ax_pairs.set_xlim(lim)
    ax_pairs.set_ylim(lim)
    ax_pairs.set_aspect("equal")
    ax_pairs.set_xlabel("x (first point of pair)")
    ax_pairs.set_ylabel("y (second point of pair)")
    ax_pairs.set_title("which pairs the ratios come from")
    ax_pairs.legend(fontsize=7, loc="upper right")

    tail = g_sorted[g_sorted <= threshold + 0.5 * (threshold - g_sorted[0])]
    ax_tail.hist(tail, bins=80, color="steelblue", alpha=0.7)
    ax_tail.axvline(k_l, color="black", lw=1.5, label=f"true K_L = {k_l:.4g}")
    ax_tail.axvline(threshold, color="orange", ls="--", label=f"threshold u = {threshold:.4g}")
    if np.isfinite(c_hat):
        ax_tail.axvline(c_hat, color="red", ls="--", label=f"c_hat = {c_hat:.4g}")
    ax_tail.set_xlabel("g = |f(x) - f(y)| / |x - y|")
    ax_tail.set_title("lower tail of the ratio cloud")
    ax_tail.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lo", type=float, default=-2.0, help="x ~ Uniform(lo, hi)")
    parser.add_argument("--hi", type=float, default=2.0)
    parser.add_argument("--n", type=int, default=100, help="points in the cloud (simple_stories vocab size)")
    # The rest match lipschitz_ratio_bootstrap_eqd_sweep_gpu.py's defaults.
    parser.add_argument("--n-pairs", type=int, default=200_000)
    parser.add_argument("--n-tail-min", type=int, default=200)
    parser.add_argument("--n-tail-max", type=int, default=300)
    parser.add_argument("--n-tail-step", type=int, default=200)
    parser.add_argument("--eqd-n-boot", type=int, default=200)
    parser.add_argument("--m-plotting-points", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", type=str, default="toy_kl_x_squared.png",
                        help="8-panel pipeline diagnostic; the toy view goes to <stem>_toy.png")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(args.seed)

    x = args.lo + (args.hi - args.lo) * torch.rand(args.n, 1, device=device, generator=gen, dtype=torch.float64)
    input_dist, output_dist, ratio, pair_i, pair_j = compute_lipschitz_ratios_gpu(
        x, f(x), args.n_pairs, gen, return_pairs=True,
    )
    g_sorted = torch.sort(ratio).values

    [fit] = select_bootstrap_eqd_threshold_gpu_multi(
        [g_sorted], n_tail_min=args.n_tail_min, n_tail_max=args.n_tail_max, n_tail_step=args.n_tail_step,
        n_boot=args.eqd_n_boot, m_plotting_points=args.m_plotting_points, generator=gen, group_size=1,
    )
    n_tail, threshold = fit["best_n_tail"], fit["threshold"]
    gpd = fit_gpd_and_c_hat_gpu(g_sorted, n_tail, threshold)

    g_np = g_sorted.cpu().numpy()
    xi_pwm, sigma_pwm = gpd_pwm_fit(threshold - g_np[:n_tail])
    c_hat_pwm = threshold + sigma_pwm / xi_pwm if xi_pwm < 0 else float("nan")
    k_l = true_k_l(args.lo, args.hi)

    print(f"f(x) = x^2, x ~ U({args.lo:g}, {args.hi:g}), n={args.n}, n_pairs={g_np.size}, device={device}")
    print(f"  true K_L         = {k_l:.5f}   (true tail shape xi = {-1.0 if k_l == 0 else -0.5})")
    print(f"  min observed g   = {g_np[0]:.5f}")
    print(f"  EQD n_tail       = {n_tail}   threshold u = {threshold:.5f}")
    print(f"  MLE: xi = {gpd['shape_xi']:.4f}  sigma = {gpd['scale_sigma']:.5f}  c_hat = {gpd['c_hat']:.5f}")
    print(f"  PWM: xi = {xi_pwm:.4f}  sigma = {sigma_pwm:.5f}  c_hat = {c_hat_pwm:.5f}")

    plot_detail(
        args.plot, f"x~U({args.lo:g},{args.hi:g})", "x^2", 0, 0, g_np, n_tail, threshold,
        gpd["shape_xi"], gpd["scale_sigma"], xi_pwm, sigma_pwm,
        n_tail_min=args.n_tail_min, n_tail_max=args.n_tail_max, n_tail_step=args.n_tail_step,
        eqd_n_boot=args.eqd_n_boot, m_plotting_points=args.m_plotting_points, device=device,
        input_dist=input_dist.cpu().numpy(), output_dist=output_dist.cpu().numpy(),
    )
    toy_path = args.plot.rsplit(".", 1)[0] + "_toy.png"
    x_np = x.cpu().numpy().ravel()
    plot_toy(
        toy_path, x_np, x_np[pair_i.cpu().numpy()], x_np[pair_j.cpu().numpy()], ratio.cpu().numpy(),
        g_np, threshold, gpd["c_hat"], k_l, seed=args.seed,
    )
    print(f"  plots -> {args.plot}, {toy_path}")


if __name__ == "__main__":
    main()
