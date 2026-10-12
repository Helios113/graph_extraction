"""EVT lower-Lipschitz figure (same panels as lipschitz_evt.pdf) for the sample sets around p and c
(crossing 5), from the radius-1-spacing run -- the same samples as det_zero_kernel_score_c05, all
within one local spacing of their centre, where the on-manifold set is on the cloud (median u 1.13).
Fit = the sweep's own bootstrap-EQD threshold + GPD at its defaults, on all pairs within a set.

    uv run python -m scripts.analysis.plot_det_zero_evt
"""
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.analysis.plot_det_zero_manifold_paper import EVT_LABEL, PAPER_RC, STYLE
from scripts.analysis.plot_lipschitz_evt_paper import all_pair_dists, draw_evt_row, finish_evt_figure
from scripts.lipschitz.lipschitz_ratio_bootstrap_eqd_sweep_gpu import (
    fit_gpd_and_c_hat_gpu,
    select_bootstrap_eqd_threshold_gpu_multi,
)

plt.rcParams.update(PAPER_RC)

# one figure per model: (input file, output prefix)
RUNS = [
    ("onnx_models/simple_stories/det_zero_manifold/add_4.add_6/row26685_row26140_manifold_c05.h5", "figures/paper/det_zero_evt_c05"),  # simple_stories add_4.add_6 (layer-1 FFN), crossing 5
    ("onnx_models/gpt2/det_zero_manifold/add_2.add_5/row182959_row176262_manifold_c04.h5", "figures/paper/det_zero_evt_gpt2_c04"),  # GPT-2 add_2.add_5 (layer-1 FFN), crossing 4
]

for FILE, OUT in RUNS:
    rng = np.random.default_rng(0)
    gen = torch.Generator().manual_seed(0)
    fig, axs = plt.subplots(len(STYLE), 3, figsize=(5.5, 1.22 * len(STYLE) + 0.6))
    n_violations = 0
    with h5py.File(FILE, "r") as f:
        for r, g in enumerate(STYLE):  # uniform sets only
            din, dout = all_pair_dists(f[g]["vec_in"][:], f[g]["vec_out"][:])
            ok = (din > 0) & (dout > 0)
            din, dout = din[ok], dout[ok]
            ratio = dout / din
            fit_idx = rng.choice(ratio.size, 200_000, replace=False) if ratio.size > 200_000 else slice(None)
            g_sorted = torch.sort(torch.from_numpy(ratio[fit_idx]).double()).values
            [sel] = select_bootstrap_eqd_threshold_gpu_multi([g_sorted], 200, 1100, 200, 200, 50, gen, group_size=1)
            fit = fit_gpd_and_c_hat_gpu(g_sorted, sel["best_n_tail"], sel["threshold"])
            u, xi, sigma, k_hat = sel["threshold"], fit["shape_xi"], fit["scale_sigma"], fit["c_hat"]
            print(f"{g:3s} pairs {ratio.size}  min g {ratio.min():.4g}  xi {xi:.3f}  K_L-hat {k_hat:.4g}  "
                  f"pairs below {(ratio < k_hat).sum()}")
            n_violations += draw_evt_row(axs[r], din, dout, u, xi, sigma, k_hat, EVT_LABEL[g], rng, 20_000, 3_000, 1e-5)
    finish_evt_figure(fig, axs, n_violations, OUT)
