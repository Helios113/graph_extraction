"""Publication companion to plot_lipschitz_evt_paper.py: the same three-panel
EVT figure -- (a) ||x - y|| vs ||f(x) - f(y)||, (b) ratio histogram, (c) lower
tail vs fitted GPD with its finite endpoint K_L-hat -- for an analytical toy
map from proofs/dim_analysis/evt_injectivity.py, one row per dimension N
(--dims, default 384 = simple_stories' d_model; e.g. --dims 2 16 64 384 for the
full progression).

Every toy map is non-injective (default --f square: f(x) = (x_0^2, x_1, ...),
f(x) = f(R(x)) for R flipping x_0), so the true lower Lipschitz constant is 0
for every N. Pairs are drawn with evt_injectivity's samplers (--sampling,
default cube: x, y uniform on the domain, half forced across x_0 = 0). As N
grows the N - 1 identity coordinates dominate ||x - y||, the pairs concentrate
far from the origin and the ratios near 1 (concentration of measure), so the
EVT lower bound fitted to random pairs lands far above the true K_L = 0,
marked in (b).

The fit is the real sweep's own pipeline, from
scripts/lipschitz/lipschitz_ratio_bootstrap_eqd_sweep_gpu.py at its defaults:
--n-pairs (200k) ratios, bootstrap-EQD threshold selection over n_tail in
[--n-tail-min, --n-tail-max) step --n-tail-step, then the GPD fit and
K_L-hat = u + sigma / xi (none when xi >= 0). Panels are drawn by
plot_lipschitz_evt_paper.draw_evt_row, so the two figures match. Runs on CPU
(uses a GPU if present). Writes {--out}.pdf / .png.

    uv run python -m scripts.analysis.plot_lipschitz_evt_toy_paper
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from proofs.dim_analysis.evt_injectivity import MAPS, sample_pairs_cube, sample_pairs_radius, sample_pairs_volume
from scripts.analysis.plot_lipschitz_evt_paper import draw_evt_row, finish_evt_figure
from scripts.lipschitz.lipschitz_ratio_bootstrap_eqd_sweep_gpu import (
    fit_gpd_and_c_hat_gpu,
    select_bootstrap_eqd_threshold_gpu_multi,
)

SAMPLERS = {"cube": sample_pairs_cube, "radius": sample_pairs_radius, "volume": sample_pairs_volume}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--f", choices=list(MAPS), default="square")
    p.add_argument("--sampling", choices=list(SAMPLERS), default="cube")
    p.add_argument("--dims", type=int, nargs="+", default=[384], help="one row per dimension N")
    p.add_argument("--n-pairs", type=int, default=200_000)
    p.add_argument("--n-tail-min", type=int, default=200)
    p.add_argument("--n-tail-max", type=int, default=1100)
    p.add_argument("--n-tail-step", type=int, default=200)
    p.add_argument("--eqd-n-boot", type=int, default=200)
    p.add_argument("--m-plotting-points", type=int, default=50)
    p.add_argument("--n-scatter", type=int, default=20_000, help="bulk pairs drawn in the scatter")
    p.add_argument("--n-scatter-tail", type=int, default=3_000, help="max tail pairs drawn in the scatter")
    p.add_argument("--outlier-q", type=float, default=1e-5, help="display-only low-end cut for panels (b), (c)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="default figures/paper/lipschitz_evt_toy_{f}_{sampling}")
    args = p.parse_args()

    fmap = MAPS[args.f]
    if min(args.dims) < fmap.min_dim:
        p.error(f"--f {args.f} needs every --dims >= {fmap.min_dim}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    n_violations = 0
    fig, axs = plt.subplots(len(args.dims), 3, figsize=(5.5, 1.22 * len(args.dims) + 0.6))
    axs = np.atleast_2d(axs)
    for r, dim in enumerate(args.dims):
        x, y = SAMPLERS[args.sampling](args.n_pairs, fmap.half_widths(dim), rng)
        din = np.linalg.norm(x - y, axis=1)
        dout = np.linalg.norm(fmap.f(x) - fmap.f(y), axis=1)
        del x, y
        g_sorted = torch.sort(torch.from_numpy(dout / din).to(device, torch.float64)).values

        [sel] = select_bootstrap_eqd_threshold_gpu_multi(
            [g_sorted], args.n_tail_min, args.n_tail_max, args.n_tail_step, args.eqd_n_boot,
            args.m_plotting_points, gen, group_size=1,
        )
        fit = fit_gpd_and_c_hat_gpu(g_sorted, sel["best_n_tail"], sel["threshold"])
        u, xi, sigma, k_hat = sel["threshold"], fit["shape_xi"], fit["scale_sigma"], fit["c_hat"]
        g = dout / din
        print(f"N={dim:4d}  pairs {g.size}  min g {g.min():.4f}  median g {np.median(g):.4f}  "
              f"n_tail {sel['best_n_tail']}  u {u:.4f}  xi {xi:.3f}  sigma {sigma:.4f}  K_L-hat {k_hat:.4f}  "
              f"(true K_L = 0)")

        n_violations += draw_evt_row(axs[r], din, dout, u, xi, sigma, k_hat, f"$N={dim}$", rng, args.n_scatter,
                                     args.n_scatter_tail, args.outlier_q, true_kl=0.0)

    out = args.out or f"figures/paper/lipschitz_evt_toy_{args.f}_{args.sampling}"
    finish_evt_figure(fig, axs, n_violations, out, true_kl=0.0)


if __name__ == "__main__":
    main()
