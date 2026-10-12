"""scripts/analysis/compare_gpd_estimators.py's MLE-vs-PWM comparison and
--plot diagnostic, adapted for scripts/python/generate_ambient_sweep.py's
output instead of the perturbed-sweep vocab-cloud sweep.

Ambient sweep writes one h5 per (sublayer, tau, sample) --
{src}.{dst}.ball{t}.sample{idx} (or .tau{t}. for --dist gaussian) -- holding,
for every sequence position pos, `n` points sampled as (real anchor +
isotropic uniform-ball or Gaussian noise) in input
space, and that point's real output through the {src}->{dst} subgraph
(src_vec, dst_vec, pos, anchor, multiplicity; see that script's module
docstring for the exact sampling mechanism). This is structurally the same
shape as perturbed_sweep's vocab-cloud -- N points at one position, each
with an (input, output) vector pair -- so the SAME pairwise-ratio
construction applies: draw random pairs (i, j) among that position's ambient
points and compute ||dst_vec_i - dst_vec_j|| / ||src_vec_i - src_vec_j||.

Unlike perturbed_sweep, ambient sweep has no downstream bootstrap-EQD
analysis step (no lipschitz_ratio_bootstrap_eqd_sweep_gpu.py equivalent has
been run on it yet), so there's no h5-saved threshold/n_tail/xi to compare
against or reuse -- --n-tail is required here (no default to fall back on,
unlike compare_gpd_estimators.py's --sample-idx/--pos mode which reads the
sweep's own eqd_best_n_tail).

Usage (must run on the node the ambient sweep ran on -- e.g. mauao via
srun -- --no-push output only exists on that node's local scratch):
    uv run python -m scripts.analysis.compare_gpd_estimators_ambient \\
        onnx_models/simple_stories/ambient_sweep \\
        --sublayer-input add_30 --sublayer-output add_34 --tau 1 \\
        --sample-idx 5 --pos 7 --n-tail 200 \\
        --plot ambient_detail.png --subsample-n 10000
"""

import argparse
import os

import numpy as np
import torch

from scripts.analysis.compare_gpd_estimators import plot_detail
from scripts.analysis.gpd_pwm_fit import gpd_pwm_fit
from scripts.analysis.lipschitz_ratio_eqd_fit import (
    compute_lipschitz_ratios,
    load_layer_manifest,
    load_row_range,
)
from scripts.lipschitz.batched_gpd_mle import fit_gpd_batched


def load_ambient_ratio_cloud(
    ambient_sweep_dir: str,
    sublayer_input: str,
    sublayer_output: str,
    tau: float,
    sample_idx: int,
    pos: int,
    n_pairs: int,
    seed: int,
    dist: str = "ball",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads one (sublayer pair, tau, sample, pos)'s ambient-sampled
    (src_vec, dst_vec) rows and computes the pairwise ratio cloud, same
    construction as lipschitz_ratio_eqd_fit.py's compute_lipschitz_ratios
    but over ambient sample points at a fixed position instead of the
    vocab-sweep cloud. Returns (input_dist, output_dist, ratio)."""
    layer_key = os.path.join(
        ambient_sweep_dir,
        f"{sublayer_input}.{sublayer_output}.{'tau' if dist == 'gaussian' else dist}{tau:g}.sample{sample_idx}",
    )
    manifest = load_layer_manifest(".", layer_key)
    seq = manifest["attrs"]["seq"]
    n = manifest["attrs"]["n_samples"]
    if not (0 <= pos < seq):
        raise ValueError(f"pos={pos} out of range for seq={seq} ({layer_key})")

    row_start = pos * n
    row_end = row_start + n
    rows = load_row_range(layer_key, manifest, row_start, row_end, ["src_vec", "dst_vec"])

    rng = np.random.default_rng(seed + sample_idx)
    return compute_lipschitz_ratios(rows["src_vec"], rows["dst_vec"], n_pairs=n_pairs, rng=rng)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ambient_sweep_dir", type=str, help="e.g. onnx_models/simple_stories/ambient_sweep")
    parser.add_argument("--sublayer-input", type=str, required=True)
    parser.add_argument("--sublayer-output", type=str, required=True)
    parser.add_argument("--tau", type=float, default=1.0, help="Must match a tau the ambient sweep was run with.")
    parser.add_argument("--dist", choices=["ball", "gaussian", "global"], default="ball",
                        help="Must match the --dist the ambient sweep was run with.")
    parser.add_argument("--sample-idx", type=int, required=True)
    parser.add_argument("--pos", type=int, required=True)
    parser.add_argument(
        "--n-tail", type=int, required=True,
        help="Tail size for the GPD fit. No default: unlike perturbed_sweep, ambient sweep has no downstream "
        "bootstrap-EQD analysis step that would have already picked one for this position.",
    )
    parser.add_argument(
        "--n-pairs", type=int, default=200_000,
        help="Pairs to draw from this position's ambient sample cloud (--n samples per position from the "
        "generate_ambient_sweep.py run, e.g. 1000 -- max_pairs = n*(n-1)/2 caps this automatically).",
    )
    parser.add_argument("--seed", type=int, default=1, help="Fresh seed for the pairwise ratio draw.")
    parser.add_argument(
        "--subsample-n", type=int, default=None,
        help="Shortcut for --n-pairs: draws this many pairs for EVERYTHING instead of the full --n-pairs cloud. "
        "Must stay > --n-tail-max. Use this to speed up the --plot EQD sweep/xi/c_hat panels.",
    )
    parser.add_argument("--plot", type=str, default=None, help="Save the 8-panel diagnostic figure to this path.")
    parser.add_argument("--eqd-n-boot", type=int, default=50, help="Bootstrap replicates for the --plot EQD sweep panel.")
    parser.add_argument("--n-tail-min", type=int, default=20, help="--plot EQD sweep candidate grid lower bound.")
    parser.add_argument("--n-tail-max", type=int, default=3_000, help="--plot EQD sweep candidate grid upper bound.")
    parser.add_argument("--n-tail-step", type=int, default=50, help="--plot EQD sweep candidate grid step.")
    parser.add_argument("--m-plotting-points", type=int, default=50, help="--plot EQD sweep quantile grid size.")
    parser.add_argument(
        "--device", type=str, default=None, choices=["cuda", "cpu"],
        help="Device for the GPD fits (default: cuda if available, else cpu).",
    )
    args = parser.parse_args()

    n_pairs = args.subsample_n if args.subsample_n is not None else args.n_pairs
    if args.plot is not None and n_pairs <= args.n_tail_max:
        raise SystemExit(f"--n-pairs/--subsample-n ({n_pairs}) must be > --n-tail-max ({args.n_tail_max})")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    input_dist, output_dist, ratio = load_ambient_ratio_cloud(
        args.ambient_sweep_dir, args.sublayer_input, args.sublayer_output, args.tau,
        args.sample_idx, args.pos, n_pairs=n_pairs, seed=args.seed, dist=args.dist,
    )
    g_sorted = np.sort(ratio)
    n_tail = args.n_tail
    if n_tail >= g_sorted.size:
        raise SystemExit(f"--n-tail={n_tail} >= cloud size {g_sorted.size} -- lower --n-tail or raise --n-pairs")
    threshold = float(g_sorted[n_tail])
    exceedances = threshold - g_sorted[:n_tail]

    xi_pwm, sigma_pwm = gpd_pwm_fit(exceedances)

    z_t = torch.tensor(exceedances, dtype=torch.float64, device=device).unsqueeze(0)
    xi_mle_t, sigma_mle_t, converged_t = fit_gpd_batched(z_t)
    xi_mle, sigma_mle, converged = xi_mle_t.item(), sigma_mle_t.item(), bool(converged_t.item())

    print(
        f"sample={args.sample_idx} pos={args.pos} tau={args.tau:g} n_pairs={g_sorted.size} n_tail={n_tail}\n"
        f"  xi_mle={xi_mle:.4f} sigma_mle={sigma_mle:.4f} converged={converged}\n"
        f"  xi_pwm={xi_pwm:.4f} sigma_pwm={sigma_pwm:.4f}\n"
        f"  sign_flip={(xi_mle >= 0) != (xi_pwm >= 0)}"
    )

    if args.plot is not None:
        plot_detail(
            args.plot, f"{args.sublayer_input}(ambient,tau={args.tau:g})", args.sublayer_output,
            args.sample_idx, args.pos, g_sorted, n_tail, threshold,
            xi_mle, sigma_mle, xi_pwm, sigma_pwm,
            n_tail_min=args.n_tail_min, n_tail_max=args.n_tail_max, n_tail_step=args.n_tail_step,
            eqd_n_boot=args.eqd_n_boot, m_plotting_points=args.m_plotting_points, device=device,
            input_dist=input_dist, output_dist=output_dist,
        )


if __name__ == "__main__":
    main()
