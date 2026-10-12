"""Benchmark: how batch size (number of (sample, pos) positions' bootstrap-EQD
candidate grids fit in one fit_gpd_batched call) affects wall-clock time,
convergence, and GPU memory -- using real co-Lipschitz ratio data from
generate_perturbed_sweep.py's shards, not synthetic data.

Builds each position's (n_candidates * n_boot, max_n_tail) exceedance-resample
batch the same way scripts/analysis/bootstrap_eqd_gpu.py's
bootstrap_eqd_statistic_batched does, concatenates N positions' batches
together, and times one fit_gpd_batched call over the concatenated batch --
answering "does batching more positions together in one GPU call actually
reduce time-per-position, and where does GPU memory become the limit."

Usage (on mauao, via srun -- reads perturbed-sweep shards straight off local
scratch, same as lipschitz_ratio_eqd_fit.py):
    uv run python -m scripts.analysis.bench_batched_gpd_scale \\
        --perturbed-sweep-dir onnx_models/simple_stories/perturbed_sweep \\
        --sublayer-input add_4 --sublayer-output add_6

sbatch version: scripts/batch_scripts/bench_batched_gpd_scale_mauao_sbatch.sh
"""

import argparse
import functools
import time

import numpy as np
import torch

from scripts.lipschitz.batched_gpd_mle import fit_gpd_batched
from scripts.analysis.lipschitz_ratio_bootstrap_eqd_fit import load_ratio_cloud

# sbatch's output redirection buffers stdout by default, so plain print()
# calls (e.g. during the ~64-position real-data loading loop, each a
# multi-second shard read) don't show up in the .out log until the buffer
# fills or the process exits -- making a genuinely-progressing job look
# stuck. flush=True on every print forces each line out immediately.
print = functools.partial(print, flush=True)


def build_position_batch(
    perturbed_sweep_dir: str,
    sublayer_input: str,
    sublayer_output: str,
    sample_idx: int,
    pos: int,
    n_tail_values: np.ndarray,
    n_boot: int,
    n_pairs: int,
    device: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One (sample, pos)'s full bootstrap-EQD candidate grid, flattened to
    (n_candidates * n_boot, max_n_tail) -- same construction as
    bootstrap_eqd_gpu.py's select_bootstrap_eqd_threshold_gpu /
    bootstrap_eqd_statistic_batched, pulled out here standalone so this
    script doesn't need the full pipeline wired up yet."""
    max_n_tail = int(n_tail_values.max())
    n_candidates = n_tail_values.size

    ratio = load_ratio_cloud(
        perturbed_sweep_dir, sublayer_input, sublayer_output, sample_idx, pos, n_pairs=n_pairs, seed=0,
    )
    g_sorted = torch.tensor(np.sort(ratio), dtype=torch.float64, device=device)

    n_tail_t = torch.tensor(n_tail_values, device=device, dtype=torch.long)
    u = g_sorted[n_tail_t]
    k_idx = torch.arange(max_n_tail, device=device)
    mask = k_idx.view(1, -1) < n_tail_t.view(-1, 1)
    g_head = g_sorted[:max_n_tail].view(1, -1).expand(n_candidates, -1)
    exceedances = torch.where(mask, u.view(-1, 1) - g_head, torch.zeros(1, device=device, dtype=g_sorted.dtype))

    n_tail_per_candidate = mask.sum(dim=-1)
    resample_idx = (
        torch.rand(n_candidates, n_boot, max_n_tail, device=device, generator=generator)
        * n_tail_per_candidate.view(n_candidates, 1, 1)
    ).long().clamp(max=max_n_tail - 1)
    resample_mask = (
        torch.arange(max_n_tail, device=device).view(1, 1, -1) < n_tail_per_candidate.view(n_candidates, 1, 1)
    ).expand(n_candidates, n_boot, max_n_tail)

    exc_expanded = exceedances.unsqueeze(1).expand(n_candidates, n_boot, max_n_tail)
    resamples = torch.gather(exc_expanded, -1, resample_idx)
    resamples = torch.where(resample_mask, resamples, torch.zeros_like(resamples))

    flat_shape = (n_candidates * n_boot, max_n_tail)
    return resamples.reshape(flat_shape), resample_mask.reshape(flat_shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--perturbed-sweep-dir", type=str, default="onnx_models/simple_stories/perturbed_sweep",
    )
    parser.add_argument("--sublayer-input", type=str, default="add_4")
    parser.add_argument("--sublayer-output", type=str, default="add_6")
    parser.add_argument("--n-pairs", type=int, default=200_000)
    parser.add_argument("--n-tail-min", type=int, default=20)
    parser.add_argument("--n-tail-max", type=int, default=2_000)
    parser.add_argument("--n-tail-step", type=int, default=50)
    parser.add_argument("--n-boot", type=int, default=50)
    parser.add_argument("--n-iters", type=int, default=10, help="fit_gpd_batched's Newton iteration budget")
    parser.add_argument(
        "--n-positions", type=int, nargs="+", default=[1, 4, 16, 64],
        help="batch sizes to try, in units of (sample, pos) positions",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cpu":
        print("WARNING: no GPU available, this benchmark is meaningless on CPU")

    n_tail_values = np.arange(args.n_tail_min, args.n_tail_max, args.n_tail_step)
    print(f"n_candidates={n_tail_values.size}, n_boot={args.n_boot}, "
          f"rows_per_position={n_tail_values.size * args.n_boot}, max_n_tail={int(n_tail_values.max())}")

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    max_n_positions = max(args.n_positions)
    print(f"\nloading {max_n_positions} real (sample, pos) ratio clouds from "
          f"{args.sublayer_input}->{args.sublayer_output}...")
    position_batches = []
    load_t0 = time.time()
    for i in range(max_n_positions):
        sample_idx, pos = i % 64, i % 8
        pos_t0 = time.time()
        z, rm = build_position_batch(
            args.perturbed_sweep_dir, args.sublayer_input, args.sublayer_output,
            sample_idx, pos, n_tail_values, args.n_boot, args.n_pairs, device, gen,
        )
        position_batches.append((z, rm))
        print(f"  [{i+1}/{max_n_positions}] sample={sample_idx} pos={pos} "
              f"loaded+resampled in {time.time()-pos_t0:.2f}s "
              f"(elapsed {time.time()-load_t0:.1f}s)")
    print(f"done loading {max_n_positions} positions in {time.time()-load_t0:.1f}s.\n")

    print(f"{'n_positions':>12} {'n_rows':>8} {'time(s)':>10} {'ms/pos':>10} "
          f"{'converged':>10} {'peak_mem(GB)':>13}")
    for n_positions in sorted(args.n_positions):
        print(f"  running n_positions={n_positions}...")
        zs = [b[0] for b in position_batches[:n_positions]]
        rms = [b[1] for b in position_batches[:n_positions]]
        z_all = torch.cat(zs, dim=0)
        rm_all = torch.cat(rms, dim=0)

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()
        xi, sigma, conv = fit_gpd_batched(z_all, rm_all, n_iters=args.n_iters)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0

        mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else float("nan")
        print(f"{n_positions:>12} {z_all.shape[0]:>8} {dt:>10.3f} {dt/n_positions*1000:>10.1f} "
              f"{conv.float().mean().item():>9.1%} {mem:>13.2f}")


if __name__ == "__main__":
    main()
