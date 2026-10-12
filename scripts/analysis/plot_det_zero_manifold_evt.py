"""EVT lower-Lipschitz figure (same three panels as plot_lipschitz_evt_paper.py:
(a) ||x - y|| vs ||f(x) - f(y)||, (b) ratio histogram, (c) lower tail vs the
fitted GPD with its finite endpoint K_L-hat) for the sample sets of one
scripts/python/generate_det_zero_manifold.py file, one row per set:

  A   p: on the manifold (convex neighbour samples around p)
  B   p: ambient ball around p
  C   p: line along v_min(p), J's smallest singular direction at p
  cB  c: ambient ball around the critical (det J = 0) point c
  cC  c: line along v_min(c), the kernel direction at c

Pairs are every pair within a set (uniform sets only, not the log-spaced
ones). Distances are taken as exact by default; --fp32-floor F (e.g. 2e-6,
~2x the measured float32 output floor) instead drops pairs whose output change
is below F as rounding noise. Pairs with identical outputs (g = 0) are always
left out of the log-scale panels. The fit is the real sweep's own pipeline at its defaults
(scripts/lipschitz/lipschitz_ratio_bootstrap_eqd_sweep_gpu.py, as in
plot_lipschitz_evt_toy_paper.py): --n-pairs random pairs (default 200k, or all
when fewer), bootstrap-EQD threshold over n_tail in [--n-tail-min,
--n-tail-max) step --n-tail-step, GPD fit, K_L-hat = u + sigma / xi (none
when xi >= 0). Panels draw every pair. CPU (GPU if present).

Writes {stem}_evt.pdf / .png next to each input file.

    uv run python -m scripts.analysis.plot_det_zero_manifold_evt \\
        onnx_models/simple_stories/det_zero_manifold/add_4.add_6/row26685_row26140_manifold_c05.h5
"""
import argparse
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.analysis.plot_lipschitz_evt_paper import all_pair_dists, draw_evt_row, finish_evt_figure
from scripts.lipschitz.lipschitz_ratio_bootstrap_eqd_sweep_gpu import (
    fit_gpd_and_c_hat_gpu,
    select_bootstrap_eqd_threshold_gpu_multi,
)

ROWS = {
    "A": "$p$: on manifold\n(convex nbrs)",
    "B": "$p$: ambient\nball",
    "C": "$p$: line along\n$v_{\\min}(p)$",
    "cB": "$c$: ambient\nball",
    "cC": "$c$: kernel line\n$v_{\\min}(c)$",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="generate_det_zero_manifold.py output h5 file(s)")
    p.add_argument("--sets", nargs="+", default=list(ROWS), choices=list(ROWS), help="rows, in order")
    p.add_argument("--fp32-floor", type=float, default=0.0,
                   help="drop pairs with ||f(x) - f(y)|| below this (default 0: distances taken as exact)")
    p.add_argument("--n-pairs", type=int, default=200_000, help="pairs used for the fit (all when fewer)")
    p.add_argument("--n-tail-min", type=int, default=200)
    p.add_argument("--n-tail-max", type=int, default=1100)
    p.add_argument("--n-tail-step", type=int, default=200)
    p.add_argument("--eqd-n-boot", type=int, default=200)
    p.add_argument("--m-plotting-points", type=int, default=50)
    p.add_argument("--n-scatter", type=int, default=20_000, help="bulk pairs drawn in the scatter")
    p.add_argument("--n-scatter-tail", type=int, default=3_000, help="max tail pairs drawn in the scatter")
    p.add_argument("--outlier-q", type=float, default=1e-5, help="display-only low-end cut for panels (b), (c)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for path in args.files:
        rng = np.random.default_rng(args.seed)
        gen = torch.Generator(device=device).manual_seed(args.seed)
        with h5py.File(path, "r") as f:
            a = dict(f.attrs)
            data = {s: (f[s]["vec_in"][:], f[s]["vec_out"][:]) for s in args.sets}
        print(f"{path}: {a['sublayer_input']}.{a['sublayer_output']} pos {a['pos']} crossing {a['crossing']}, "
              f"radius {a['radius']:.3g}")

        n_violations = 0
        fig, axs = plt.subplots(len(args.sets), 3, figsize=(5.5, 1.22 * len(args.sets) + 0.6))
        axs = np.atleast_2d(axs)
        for r, s in enumerate(args.sets):
            din, dout = all_pair_dists(*data[s])
            keep = (dout >= args.fp32_floor) & (dout > 0) & (din > 0)
            din, dout = din[keep], dout[keep]
            g = dout / din
            fit_idx = rng.choice(g.size, args.n_pairs, replace=False) if g.size > args.n_pairs else slice(None)
            g_sorted = torch.sort(torch.from_numpy(g[fit_idx]).to(device, torch.float64)).values
            [sel] = select_bootstrap_eqd_threshold_gpu_multi(
                [g_sorted], args.n_tail_min, args.n_tail_max, args.n_tail_step, args.eqd_n_boot,
                args.m_plotting_points, gen, group_size=1,
            )
            fit = fit_gpd_and_c_hat_gpu(g_sorted, sel["best_n_tail"], sel["threshold"])
            u, xi, sigma, k_hat = sel["threshold"], fit["shape_xi"], fit["scale_sigma"], fit["c_hat"]
            print(f"  {s:3s} pairs {g.size} (dropped {int((~keep).sum())} below fp32 floor)  min g {g.min():.4g}  "
                  f"median g {np.median(g):.4g}  n_tail {sel['best_n_tail']}  u {u:.4g}  xi {xi:.3f}  "
                  f"sigma {sigma:.4g}  K_L-hat {k_hat:.4g}  pairs below K_L-hat {(g < k_hat).sum()}")
            n_violations += draw_evt_row(axs[r], din, dout, u, xi, sigma, k_hat, ROWS[s], rng, args.n_scatter,
                                         args.n_scatter_tail, args.outlier_q)
        finish_evt_figure(fig, axs, n_violations, os.path.splitext(path)[0] + "_evt")
        plt.close(fig)


if __name__ == "__main__":
    main()
