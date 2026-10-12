"""Full-sweep version of scripts/analysis/lipschitz_ratio_bootstrap_eqd_fit.py's
Algorithm 1 (Bootstrap Expected Quantile Discrepancy threshold selection) --
same sweep structure, output layout, and h5 schema as
scripts/analysis/lipschitz_ratio_eqd_fit.py, but with that script's
deterministic closed-form EQD approximation (a single statistic per
candidate threshold, no resampling) replaced by the exact bootstrap EQD
statistic:

  For each candidate threshold u (a grid of exceedance counts n_tail), draw
  B bootstrap resamples of the n_tail exceedances x_u, fit a GPD to each
  resample by MLE, and compute the mean absolute deviation between that
  fit's model quantiles and x_u^b's own empirical (linear-interpolated)
  quantiles over a grid of m plotting-position probabilities:

      d_b(u) = (1/m) sum_j | (sigma^b/xi^b) [(1-p_j)^(-xi^b) - 1]
                             - Q(p_j; x_u^b) |

  averaged over the B resamples to give d_hat_E(u) = mean_b d_b(u). The
  selected threshold u_hat minimizes d_hat_E(u) over the candidate grid.

This intentionally does NOT run Algorithm 2 (double-bootstrap uncertainty
propagation) -- see lipschitz_ratio_bootstrap_eqd_fit.py for that, scoped to
one (sublayer pair, sample, pos) at a time because of its B1*B2 cost. This
script's B-bootstraps-per-candidate cost is far cheaper (no re-running
Algorithm 1 inside an outer loop), so a full sweep here is practical, though
still much slower than lipschitz_ratio_eqd_fit.py's closed-form sweep -- see
--eqd-n-boot/--m-plotting-points for the cost knobs.

Output layout matches lipschitz_ratio_eqd_fit.py exactly (one h5 per
(sublayer pair, sample), one group per position, same dataset/attr names),
so scripts/python/plot_lipschitz_ratio_eqd_fit.py works unmodified against
this script's --out-dir.

Usage (on mauao, via srun -- reads perturbed-sweep shards straight off local
scratch, same as lipschitz_ratio_eqd_fit.py):
    uv run python -m scripts.analysis.lipschitz_ratio_bootstrap_eqd_sweep \\
        configs/simple_stories.toml --perturbed-sweep-dir onnx_models/simple_stories/perturbed_sweep
"""

import argparse
import os

import h5py
import numpy as np
import tqdm

from config import load_config
from scripts.analysis.lipschitz_ratio_bootstrap_eqd_fit import (
    fit_gpd_and_c_hat,
    select_bootstrap_eqd_threshold,
)
from scripts.analysis.lipschitz_ratio_eqd_fit import (
    compute_lipschitz_ratios,
    load_layer_manifest,
    load_row_range,
)


def run_sample(
    sublayer_input: str,
    sublayer_output: str,
    sample_idx: int,
    grid_in: np.ndarray,
    grid_out: np.ndarray,
    seq_len: int,
    vocab_size: int,
    n_pairs: int,
    n_pairs_save: int,
    n_tail_min: int,
    n_tail_max: int,
    n_tail_step: int,
    eqd_n_boot: int,
    m_plotting_points: int,
    seed: int,
    out_path: str,
) -> None:
    """grid_in, grid_out: (seq * vocab_size, d_model) vocab-sweep rows for
    this sample only, reshaped here to (seq, vocab_size, d_model). Writes
    one h5 group per position under out_path -- same schema as
    lipschitz_ratio_eqd_fit.py's run_sample, with eqd_n_tail_values/eqd_sweep
    holding the bootstrap d_hat_E(u) sweep in place of the closed-form one."""
    grid_in = grid_in.reshape(seq_len, vocab_size, -1)
    grid_out = grid_out.reshape(seq_len, vocab_size, -1)
    rng = np.random.default_rng(seed + sample_idx)

    tmp_path = out_path + ".tmp"
    with h5py.File(tmp_path, "w") as h5f:
        h5f.attrs["sublayer_input"] = sublayer_input
        h5f.attrs["sublayer_output"] = sublayer_output
        h5f.attrs["sample_idx"] = sample_idx
        h5f.attrs["n_pairs"] = n_pairs
        h5f.attrs["n_pairs_save"] = n_pairs_save
        h5f.attrs["eqd_n_boot"] = eqd_n_boot
        h5f.attrs["m_plotting_points"] = m_plotting_points

        for pos in tqdm.tqdm(range(seq_len), desc=f"sample {sample_idx} positions", leave=False):
            input_dist, output_dist, ratio = compute_lipschitz_ratios(
                grid_in[pos], grid_out[pos], n_pairs=n_pairs, rng=rng,
            )
            g_sorted = np.sort(ratio)

            n_save = min(n_pairs_save, ratio.size)
            save_idx = rng.choice(ratio.size, size=n_save, replace=False)

            grp = h5f.create_group(f"pos_{pos:04d}")
            grp.attrs["pos"] = pos
            grp.create_dataset("input_dist", data=input_dist[save_idx].astype(np.float32))
            grp.create_dataset("output_dist", data=output_dist[save_idx].astype(np.float32))
            grp.create_dataset("ratio", data=ratio[save_idx].astype(np.float32))

            try:
                eqd_result = select_bootstrap_eqd_threshold(
                    g_sorted,
                    n_tail_min=n_tail_min,
                    n_tail_max=n_tail_max,
                    n_tail_step=n_tail_step,
                    n_boot=eqd_n_boot,
                    m_plotting_points=m_plotting_points,
                    rng=rng,
                )
            except ValueError as e:
                grp.attrs["fit_error"] = str(e)
                continue

            gpd_result = fit_gpd_and_c_hat(g_sorted, eqd_result["best_n_tail"], eqd_result["threshold"])

            grp.create_dataset("eqd_n_tail_values", data=eqd_result["n_tail_values"])
            grp.create_dataset("eqd_sweep", data=eqd_result["d_hat_e"].astype(np.float32))
            grp.attrs["eqd_best_n_tail"] = eqd_result["best_n_tail"]
            grp.attrs["eqd_best_eqd"] = eqd_result["best_d_hat_e"]
            grp.attrs["threshold"] = eqd_result["threshold"]
            grp.attrs["shape_xi"] = gpd_result["shape_xi"]
            grp.attrs["scale_sigma"] = gpd_result["scale_sigma"]
            grp.attrs["g_min_observed"] = float(g_sorted[0])
            grp.attrs["c_hat"] = gpd_result["c_hat"]

        h5f.attrs["complete"] = True

    os.replace(tmp_path, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=str, help="Path to a toml ModelConfig")
    parser.add_argument(
        "--perturbed-sweep-dir",
        type=str,
        default="onnx_models/smoll/perturbed_sweep",
        help="Output directory produced by scripts/python/generate_perturbed_sweep.py "
        "(one sharded output per layer)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to write per-(sample, layer-pair) h5 result files to "
        "(default: <perturbed-sweep-dir>/../lipschitz_ratio_bootstrap_eqd_sweep, on NFS).",
    )
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="Number of base samples to process (default: all samples in the sweep)",
    )
    parser.add_argument(
        "--n-pairs", type=int, default=200_000,
        help="Number of random point pairs to sample per (sample, pos) vocab-sweep cloud, "
        "for the threshold/GPD fit (same meaning as lipschitz_ratio_eqd_fit.py's --n-pairs)",
    )
    parser.add_argument(
        "--n-pairs-save", type=int, default=10_000,
        help="Number of the --n-pairs ratios (randomly subsampled) to write to the output "
        "h5's scatter datasets",
    )
    parser.add_argument("--n-tail-min", type=int, default=20, help="Smallest n_tail in the EQD sweep")
    parser.add_argument(
        "--n-tail-max", type=int, default=2_000,
        help="Largest n_tail (exclusive) in the EQD sweep -- kept far below "
        "lipschitz_ratio_eqd_fit.py's default of 20,000 since each candidate now costs "
        "--eqd-n-boot GPD refits instead of one closed-form fit",
    )
    parser.add_argument("--n-tail-step", type=int, default=50, help="Step size in the EQD n_tail sweep")
    parser.add_argument(
        "--eqd-n-boot", type=int, default=50,
        help="B: number of bootstrap resamples per candidate threshold in the bootstrap EQD "
        "statistic. Total GPD fits per position ~= B * n_tail_values, so this is the primary "
        "runtime knob.",
    )
    parser.add_argument(
        "--m-plotting-points", type=int, default=50,
        help="m: number of equally-spaced plotting-position probabilities used in the mean "
        "absolute deviation d_b(u)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed for pair subsampling")
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute even if a sample's output file already exists",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.perturbed_sweep_dir.rstrip("/")), "lipschitz_ratio_bootstrap_eqd_sweep",
    )
    os.makedirs(out_dir, exist_ok=True)

    sublayer_pairs = [(s.input, s.output) for s in cfg.sublayers]
    print(f"sublayer pairs: {sublayer_pairs}")

    for sublayer_input, sublayer_output in sublayer_pairs:
        print(f"\n=== {sublayer_input} -> {sublayer_output} ===")

        in_layer_key = os.path.join(args.perturbed_sweep_dir, sublayer_input)
        out_layer_key = os.path.join(args.perturbed_sweep_dir, sublayer_output)
        if not os.path.exists(in_layer_key + ".manifest.json"):
            print(f"skipping {sublayer_input} -> {sublayer_output}: "
                  f"no perturbed-sweep output for {sublayer_input!r}")
            continue
        in_manifest = load_layer_manifest(".", in_layer_key)
        out_manifest = load_layer_manifest(".", out_layer_key)

        n_samples_total = in_manifest["attrs"]["n_samples"]
        seq_len = in_manifest["attrs"]["seq"]
        n_rows_total = in_manifest["n_rows"]
        vocab_size = n_rows_total // (n_samples_total * seq_len)

        n_samples = args.n_samples or n_samples_total
        n_samples = min(n_samples, n_samples_total)

        pair_dir = os.path.join(out_dir, f"{sublayer_input}.{sublayer_output}")
        os.makedirs(pair_dir, exist_ok=True)

        rows_per_sample = seq_len * vocab_size
        for s in tqdm.tqdm(range(n_samples), desc=f"{sublayer_input}.{sublayer_output}"):
            out_path = os.path.join(pair_dir, f"sample_{s:05d}.h5")
            if os.path.exists(out_path) and not args.overwrite:
                continue

            row_start, row_end = s * rows_per_sample, (s + 1) * rows_per_sample
            in_vec = load_row_range(in_layer_key, in_manifest, row_start, row_end, ["vec"])["vec"]
            out_vec = load_row_range(out_layer_key, out_manifest, row_start, row_end, ["vec"])["vec"]

            run_sample(
                sublayer_input,
                sublayer_output,
                s,
                in_vec,
                out_vec,
                seq_len=seq_len,
                vocab_size=vocab_size,
                n_pairs=args.n_pairs,
                n_pairs_save=args.n_pairs_save,
                n_tail_min=args.n_tail_min,
                n_tail_max=args.n_tail_max,
                n_tail_step=args.n_tail_step,
                eqd_n_boot=args.eqd_n_boot,
                m_plotting_points=args.m_plotting_points,
                seed=args.seed,
                out_path=out_path,
            )

    print(f"\nSaved bootstrap-EQD lipschitz ratio fits under {out_dir}")


if __name__ == "__main__":
    main()
