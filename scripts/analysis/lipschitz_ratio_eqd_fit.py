"""Per-(sample, position, sublayer-pair) local co-Lipschitz tail fit.

For every sublayer pair (input -> output) in the config, every base sample,
and every sequence position, this loads that (sample, pos) vocab-sweep
cloud's `vec` from both layers (`scripts/python/generate_perturbed_sweep.py`'s
output), computes the pairwise distance ratio

    g = ||f(x) - f(y)|| / ||x - y||

over a random subsample of point pairs (`--n-pairs`, default 200k -- a local
estimate of the map's co-Lipschitz / lower Lipschitz constant, see the
derivation in playground.ipynb; true exhaustive all-C(vocab,2)-pairs
pairing was measured impractically slow at tiny_llama/gpt2 vocab sizes, so
this raises random-pair coverage instead), then:

  1. runs the Expected Quantile Discrepancy (Bader, Yan & Zhang 2018) sweep
     over candidate tail sizes to auto-select an exceedance threshold u,
  2. fits a GPD to the exceedances at that threshold,
  3. saves a `--n-pairs-save`-sized random subsample of the ratio scatter
     (kept much smaller than --n-pairs so output size doesn't scale with
     fit quality) plus the EQD sweep curve and GPD fit parameters to one
     small h5 file on NFS.

This is a CPU-only, post-hoc analysis step meant to run on the same node the
perturbed sweep itself ran on (mauao, via srun -- see
scripts/batch_scripts/lipschitz_ratio_eqd_fit_mauao.sh), reading shards
straight off that node's local scratch (`$NEW_JAC_SCRATCH`). No NFS/Hetzner
pull -- shards not found on local scratch are treated as missing (the sweep
should be re-run or relayed via scripts/helper/sync_node_scratch.sh first;
see CLAUDE.md's Hetzner ban). Its own output is small (one `--n-pairs-save`
-sized ratio scatter plus a handful of scalars per (sample, pos,
layer-pair)) and is written directly under `--out-dir` on NFS as plain h5
files, not through `ShardedLayerWriter`.

Usage (on mauao, via srun -- see scripts/batch_scripts/lipschitz_ratio_eqd_fit_mauao.sh):
    uv run python -m scripts.analysis.lipschitz_ratio_eqd_fit \\
        configs/simple_stories.toml --perturbed-sweep-dir onnx_models/simple_stories/perturbed_sweep
"""

import argparse
import json
import os

import h5py
import numpy as np
import tqdm
from scipy.stats import genpareto

from config import load_config
from execute.sharded_writer import _scratch_root


def load_layer_manifest(project_root: str, layer_key: str) -> dict:
    manifest_path = os.path.join(project_root, layer_key + ".manifest.json")
    with open(manifest_path) as f:
        return json.load(f)


def local_shard_path(layer_key: str, shard_idx: int) -> str:
    """Path a shard was (or would be) written to on this node's local
    scratch by ShardedLayerWriter, regardless of whether it's since been
    pushed and/or deleted from scratch -- see ShardedLayerWriter._scratch_dir."""
    return os.path.join(_scratch_root(), layer_key, f"shard_{shard_idx:05d}.h5")


def load_row_range(
    layer_key: str, manifest: dict, row_start: int, row_end: int, dataset_names: list[str],
) -> dict[str, np.ndarray]:
    """Reads only rows [row_start, row_end) of `layer_key`'s sharded output
    directly from this node's local scratch, i.e. only the shards that
    overlap that range -- not every shard for the whole layer.

    No NFS/Hetzner pull: this is meant to run via srun on the same node
    (mauao) the sweep itself ran on (see module docstring), so every shard
    should already be sitting on local scratch. Raises with a clear message
    if a needed shard isn't there (e.g. scratch was cleaned up, or this is
    running on the wrong node) instead of silently reaching for a remote
    transport.

    Relies on the same invariant `load_shard_rows` does (shards cover
    disjoint, contiguous, sorted row ranges) plus generate_perturbed_sweep's
    row layout (rows unravelled from (sample_idx, pos, vocab_id) in that
    order), which is what lets a caller ask for one sample's contiguous row
    range in the first place.
    """
    out: dict[str, list[np.ndarray]] = {name: [] for name in dataset_names}
    for shard in sorted(manifest["shards"], key=lambda s: s["shard_idx"]):
        s_start, s_end = shard["row_start"], shard["row_end"]
        if s_end <= row_start or s_start >= row_end:
            continue
        local_path = local_shard_path(layer_key, shard["shard_idx"])
        if not os.path.exists(local_path):
            raise FileNotFoundError(
                f"{local_path!r} not found on local scratch (node scratch root "
                f"{_scratch_root()!r}) -- this script must run on the same node the "
                f"perturbed sweep ran on (mauao), with a matching $NEW_JAC_SCRATCH, and "
                "does not pull from NFS staging or Hetzner. If the sweep's scratch copy "
                "was cleaned up, relay it back with scripts/helper/sync_node_scratch.sh "
                "or re-run generate_perturbed_sweep.py."
            )
        lo, hi = max(row_start, s_start) - s_start, min(row_end, s_end) - s_start
        with h5py.File(local_path, "r") as h5f:
            for name in dataset_names:
                out[name].append(h5f[name][lo:hi])
    return {name: np.concatenate(chunks, axis=0) for name, chunks in out.items()}


def eqd_statistic(exceedances: np.ndarray) -> float:
    """Expected Quantile Discrepancy (Bader, Yan & Zhang 2018): weighted L1
    distance between empirical and fitted-GPD exceedance quantiles, weight
    w_i = min(p_i, 1 - p_i) so tail quantiles count most. Lower is better.

    Uses the exact scipy MLE fit -- this is the slow, precise version, called
    once per position on the EQD-selected threshold's exceedances (see
    run_sample), not in the O(n_tail_values) search loop -- see
    gpd_moments_fit / eqd_statistic_moments for that."""
    k = exceedances.size
    exc_sorted = np.sort(exceedances)
    xi_k, _loc_k, sigma_k = genpareto.fit(exc_sorted, floc=0)

    ranks = np.arange(1, k + 1)
    p = ranks / (k + 1)
    theoretical_q = genpareto.ppf(p, xi_k, loc=0, scale=sigma_k)

    weights = np.minimum(p, 1 - p)
    return float(np.sum(weights * np.abs(exc_sorted - theoretical_q)) / k)


def gpd_moments_fit(mean: np.ndarray, var: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form method-of-moments GPD(xi, sigma) estimator (loc=0), from
    matching the GPD's theoretical mean sigma/(1-xi) and variance
    sigma^2/((1-xi)^2*(1-2xi)) to the sample mean/variance and solving the
    2x2 system:

        xi    = (mean^2/var - 1) / 2
        sigma = mean * (mean^2/var + 1) / 2

    Fully vectorized (mean/var are arrays, one entry per candidate) -- used
    only to rank candidate thresholds in the EQD sweep, not as the final
    reported fit (see eqd_statistic / fit_gpd_at_threshold for the exact
    scipy MLE used once the threshold is chosen). Standard starting-value
    estimator for GPD MLE (e.g. in the ismev/extRemes R packages); accurate
    enough to rank candidates despite being less precise than MLE -- see
    module-level notes on the EQD speedup this enables."""
    ratio = np.where(var > 0, mean**2 / np.where(var > 0, var, 1.0), np.nan)
    xi = 0.5 * (ratio - 1)
    sigma = 0.5 * mean * (ratio + 1)
    return xi, sigma


def gpd_ppf_closed_form(p: np.ndarray, xi: float, sigma: float) -> np.ndarray:
    """Closed-form GPD(xi, sigma, loc=0) quantile function
    ppf(p) = sigma/xi * ((1-p)^(-xi) - 1), xi -> 0 limit -sigma*log(1-p) --
    numerically identical to scipy.stats.genpareto.ppf but avoids its
    generic per-call dispatch overhead in the sweep's hot loop."""
    if abs(xi) < 1e-10:
        return -sigma * np.log(1 - p)
    return (sigma / xi) * ((1 - p) ** (-xi) - 1)


def eqd_statistic_moments(exc_sorted: np.ndarray, xi: float, sigma: float) -> float:
    """eqd_statistic's discrepancy formula, but taking an already-fitted
    (xi, sigma) (from gpd_moments_fit) instead of calling genpareto.fit
    itself -- the piece of eqd_statistic that's still O(k) per candidate
    even after removing the optimizer call."""
    k = exc_sorted.size
    if not (sigma > 0) or not np.isfinite(xi):
        return np.inf
    ranks = np.arange(1, k + 1)
    p = ranks / (k + 1)
    theoretical_q = gpd_ppf_closed_form(p, xi, sigma)
    weights = np.minimum(p, 1 - p)
    return float(np.sum(weights * np.abs(exc_sorted - theoretical_q)) / k)


def select_eqd_threshold(
    g_sorted: np.ndarray, n_tail_min: int, n_tail_max: int, n_tail_step: int,
) -> dict:
    n_tail_max = min(n_tail_max, g_sorted.size - 1)
    n_tail_values = np.arange(n_tail_min, n_tail_max, n_tail_step)
    if n_tail_values.size == 0:
        raise ValueError(
            f"empty n_tail sweep range [{n_tail_min}, {n_tail_max}) for {g_sorted.size} points"
        )

    # Cumulative sums give every candidate's prefix mean/var in one pass,
    # instead of re-summing g_sorted[:n_tail] from scratch per candidate.
    # exceedances = u_n - g_sorted[:n_tail] is a per-candidate constant
    # shift of g_sorted[:n_tail], so var(exceedances) == var(g_sorted[:n_tail]).
    csum = np.cumsum(g_sorted)
    csum2 = np.cumsum(g_sorted**2)
    n = n_tail_values.astype(np.float64)
    prefix_mean = csum[n_tail_values - 1] / n
    prefix_var = csum2[n_tail_values - 1] / n - prefix_mean**2
    u = g_sorted[n_tail_values]
    exc_mean = u - prefix_mean  # mean(u_n - g_sorted[:n_tail])

    xi_sweep, sigma_sweep = gpd_moments_fit(exc_mean, prefix_var)

    eqd_sweep = np.full(n_tail_values.shape, np.nan)
    for i, n_tail in enumerate(n_tail_values):
        u_n = g_sorted[n_tail]
        exc_n = np.sort(u_n - g_sorted[:n_tail])
        eqd_sweep[i] = eqd_statistic_moments(exc_n, xi_sweep[i], sigma_sweep[i])

    best_idx = int(np.argmin(eqd_sweep))
    best_n_tail = int(n_tail_values[best_idx])
    threshold = float(g_sorted[best_n_tail])

    return {
        "n_tail_values": n_tail_values,
        "eqd_sweep": eqd_sweep,
        "best_n_tail": best_n_tail,
        "threshold": threshold,
        "best_eqd": float(eqd_sweep[best_idx]),
    }


def fit_gpd_at_threshold(g_sorted: np.ndarray, n_tail: int, threshold: float) -> dict:
    """Fits a GPD to the n_tail smallest values of g (as exceedances below
    `threshold`) and, if the shape is negative, derives the finite lower
    endpoint estimate for g itself (see estimate_colipschitz_gpd.py's
    fit_gpd_tail for the same derivation applied to sigma_min instead of
    a Lipschitz ratio)."""
    exceedances = threshold - g_sorted[:n_tail]
    shape, _loc, scale = genpareto.fit(exceedances, floc=0)

    result = {
        "shape_xi": float(shape),
        "scale_sigma": float(scale),
        "g_min_observed": float(g_sorted[0]),
    }
    if shape < 0:
        result["c_hat"] = float(threshold + scale / shape)
    else:
        result["c_hat"] = np.nan
    return result


def compute_lipschitz_ratios(
    vec_in: np.ndarray, vec_out: np.ndarray, n_pairs: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """vec_in, vec_out: (n_points, d) arrays of the same points' input/output
    layer vectors (row-aligned). Draws `n_pairs` random unordered index pairs
    and returns (input_dist, output_dist, ratio) for each."""
    n_points = vec_in.shape[0]
    max_pairs = n_points * (n_points - 1) // 2
    n_pairs = min(n_pairs, max_pairs)

    # Rejection-free unordered pair sampling: draw i < j directly rather than
    # sampling two indices and discarding i == j collisions, which would
    # waste draws as n_pairs approaches max_pairs.
    i = rng.integers(0, n_points, size=n_pairs)
    j = rng.integers(0, n_points, size=n_pairs)
    same = i == j
    while np.any(same):
        j[same] = rng.integers(0, n_points, size=int(same.sum()))
        same = i == j

    input_dist = np.linalg.norm(vec_in[i] - vec_in[j], axis=-1)
    output_dist = np.linalg.norm(vec_out[i] - vec_out[j], axis=-1)
    ratio = output_dist / input_dist
    return input_dist, output_dist, ratio


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
    seed: int,
    out_path: str,
) -> None:
    """grid_in, grid_out: (seq * vocab_size, d_model) vocab-sweep rows for
    this sample only (one sample's contiguous row range), reshaped here to
    (seq, vocab_size, d_model). Writes one h5 group per position under
    out_path.

    `n_pairs` random pairs drive the EQD/GPD fit (want this large -- default
    200k -- so the tail the fit sees is densely populated); only a
    `n_pairs_save`-sized random subsample of that same array is written to
    disk, so saved scatter size doesn't scale with fit quality (see
    --n-pairs/--n-pairs-save help text)."""
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

        for pos in tqdm.tqdm(range(seq_len), desc=f"sample {sample_idx} positions", leave=False):
            input_dist, output_dist, ratio = compute_lipschitz_ratios(
                grid_in[pos], grid_out[pos], n_pairs=n_pairs, rng=rng,
            )
            g_sorted = np.sort(ratio)

            # Subsample only what's written to disk -- the fit above already
            # ran on the full n_pairs array. min() guards n_pairs_save >
            # actual pair count (e.g. a small vocab where n_pairs was
            # clipped in compute_lipschitz_ratios).
            n_save = min(n_pairs_save, ratio.size)
            save_idx = rng.choice(ratio.size, size=n_save, replace=False)

            grp = h5f.create_group(f"pos_{pos:04d}")
            grp.attrs["pos"] = pos
            grp.create_dataset("input_dist", data=input_dist[save_idx].astype(np.float32))
            grp.create_dataset("output_dist", data=output_dist[save_idx].astype(np.float32))
            grp.create_dataset("ratio", data=ratio[save_idx].astype(np.float32))

            try:
                eqd_result = select_eqd_threshold(g_sorted, n_tail_min, n_tail_max, n_tail_step)
            except ValueError as e:
                grp.attrs["fit_error"] = str(e)
                continue

            gpd_result = fit_gpd_at_threshold(
                g_sorted, eqd_result["best_n_tail"], eqd_result["threshold"],
            )

            grp.create_dataset("eqd_n_tail_values", data=eqd_result["n_tail_values"])
            grp.create_dataset("eqd_sweep", data=eqd_result["eqd_sweep"].astype(np.float32))
            grp.attrs["eqd_best_n_tail"] = eqd_result["best_n_tail"]
            grp.attrs["eqd_best_eqd"] = eqd_result["best_eqd"]
            grp.attrs["threshold"] = eqd_result["threshold"]
            grp.attrs["shape_xi"] = gpd_result["shape_xi"]
            grp.attrs["scale_sigma"] = gpd_result["scale_sigma"]
            grp.attrs["g_min_observed"] = gpd_result["g_min_observed"]
            grp.attrs["c_hat"] = gpd_result["c_hat"]

        h5f.attrs["complete"] = True

    os.replace(tmp_path, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "(default: <perturbed-sweep-dir>/../lipschitz_ratio_eqd_fit, on NFS). "
        "One small h5 file per (sublayer pair, sample) is written directly "
        "here -- no ShardedLayerWriter/push, this output is small enough to "
        "write straight to NFS.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of base samples to process (default: all samples in the sweep)",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=200_000,
        help="Number of random point pairs to sample per (sample, pos) vocab-sweep cloud, "
        "for the threshold/GPD fit. True exhaustive pairing (all C(vocab_size, 2) pairs) "
        "was measured impractically slow at these vocab sizes (tiny_llama/gpt2: "
        "~500M-1.3B pairs/position) -- this raises random-pair coverage well past "
        "--n-tail-max instead, so the tail the EQD/GPD fit sees is densely populated "
        "without the O(vocab^2) blowup. See --n-pairs-save for what's actually written "
        "to disk (kept much smaller so output size doesn't scale with fit quality).",
    )
    parser.add_argument(
        "--n-pairs-save",
        type=int,
        default=10_000,
        help="Number of the --n-pairs ratios (randomly subsampled) to write to the output "
        "h5's scatter datasets (input_dist/output_dist/ratio) -- kept much smaller than "
        "--n-pairs so output size doesn't scale with fit quality (e.g. 200k --n-pairs x 44 "
        "sublayer pairs x 64 samples x 4 positions would otherwise be tens of GB on NFS).",
    )
    parser.add_argument(
        "--n-tail-min", type=int, default=20, help="Smallest n_tail in the EQD sweep",
    )
    parser.add_argument(
        "--n-tail-max", type=int, default=20_000, help="Largest n_tail (exclusive) in the EQD sweep",
    )
    parser.add_argument(
        "--n-tail-step", type=int, default=100, help="Step size in the EQD n_tail sweep",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed for pair subsampling")
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute even if a sample's output file already exists",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.perturbed_sweep_dir.rstrip("/")), "lipschitz_ratio_eqd_fit",
    )
    os.makedirs(out_dir, exist_ok=True)

    sublayer_pairs = [(s.input, s.output) for s in cfg.sublayers]
    print(f"sublayer pairs: {sublayer_pairs}")

    for sublayer_input, sublayer_output in sublayer_pairs:
        print(f"\n=== {sublayer_input} -> {sublayer_output} ===")

        in_layer_key = os.path.join(args.perturbed_sweep_dir, sublayer_input)
        out_layer_key = os.path.join(args.perturbed_sweep_dir, sublayer_output)
        if not os.path.exists(in_layer_key + ".manifest.json"):
            # sublayer_input is the very first sublayer's `input` (e.g.
            # "input_ids"), which is never itself a `sublayer.output` --
            # generate_perturbed_sweep.py only writes one output per
            # cfg.sublayers *output* name, so there's no sweep grid for it.
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
                seed=args.seed,
                out_path=out_path,
            )

    print(f"\nSaved lipschitz ratio EQD/GPD fits under {out_dir}")


if __name__ == "__main__":
    main()
