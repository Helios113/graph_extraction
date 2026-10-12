"""Drives scripts.analysis.lipschitz_ratio_bootstrap_eqd_sweep_gpu over a
full scripts.python.generate_perturbed_sweep output in --target-batch-gb-sized
batches of sublayer PAIRS, relaying each batch's input data from
--storage-node's local scratch to --compute-node's local scratch before
running the bootstrap-EQD fit, so the compute node never needs more than
one batch's input sweep data resident at once. Same structural pattern as
scripts.python.run_layer_batched_tangent_sweep_direct, adapted for this
sweep's own batch unit (see below) and completeness check.

This exists for the same reason that script does: generate_perturbed_sweep.py's
output can be large (~tens of GB per layer at gpt2/tiny_llama scale), and
you may want to run the bootstrap-EQD GPU fit on a different node than the
one the sweep itself ran on and stored its output on (e.g. mauao's A40s are
busy running something else, or ruapehu's H100s are simply faster/free --
this project's Hetzner ban means the only inter-node transport is the NFS
staging relay scripts/helper/sync_node_scratch.sh uses, not a direct copy).

Batch unit: SUBLAYER PAIRS, not individual src layers (unlike
run_layer_batched_tangent_sweep_direct's per-layer batching) -- this sweep's
own driver (lipschitz_ratio_bootstrap_eqd_sweep_gpu.py) already processes
one (input, output) sublayer pair fully before moving to the next, and many
layers are each other's neighbors (e.g. add_4 is one pair's output and the
next pair's input), so batching by individual layer risks relaying the same
layer twice across adjacent batches unless carefully deduplicated. A batch
here is a list of (src, dst) pairs whose combined (src UNION dst) layer
sizes stay under --target-batch-gb; --sublayers on the underlying sweep
script restricts one invocation to exactly that batch's pairs.

Per batch:
  1. relay that batch's layers (the union of every pair's src and dst,
     deduplicated -- a layer can be both a pair's dst and the next pair's
     src) storage_node -> compute_node (scripts/helper/sync_node_scratch.sh,
     one relay chain per layer, run concurrently)
  2. run scripts.analysis.lipschitz_ratio_bootstrap_eqd_sweep_gpu on
     compute_node, restricted to that batch's --sublayers (a new sbatch
     launcher pinned to hopper/ruapehu or ampere/mauao)
  3. reclaim (rm -rf) that batch's input layer dirs from compute_node's
     scratch, once the job has finished -- but only layers not needed by a
     LATER batch too (see _layers_needed_later), so a layer shared across
     batch boundaries isn't relayed twice

This only submits/polls sbatch jobs from taranaki -- no compute runs here.
Layer sizing is measured directly on storage_node via one short sbatch `du`
job, same as the tangent-sweep driver's _layer_sizes_gb.

--compute-node/--storage-node can also be the same node (e.g. both
"mauao") -- relay-in already no-ops in this case, and the reclaim step is
explicitly skipped (reclaiming "relayed-in" input that's actually
storage_node's only copy would delete real data) -- same guard as the
tangent-sweep driver.

Idempotent to re-running after a crash: a batch whose sublayer pairs'
sample h5 files are all already present under out_dir is skipped entirely
(lipschitz_ratio_bootstrap_eqd_sweep_gpu.py's own per-sample resume, see
_batch_complete below, matches what that script already checks on startup
via its own os.path.exists skip -- this just lets the DRIVER skip
relaying+launching a whole batch's job up front too). Does NOT resume a
half-relayed or half-run batch transparently -- the whole batch is
reissued, and the underlying sweep script's own per-sample skip avoids
redundant GPU work for samples that already finished within a re-run batch.

Usage:
    uv run python -m scripts.python.run_lipschitz_ratio_bootstrap_eqd_sweep_gpu \\
        configs/gpt2.toml \\
        --target-batch-gb 250 \\
        --compute-node ruapehu \\
        --storage-node mauao \\
        --job-time-limit 12:00:00
"""

import argparse
import json
import os
import subprocess
import time

from autoencoder.data import sweep_dir_for
from config import load_config

REPO_ROOT = "/nfs-share/pa511/code_bases/new_jac"
NEW_JAC_SCRATCH_TEMPLATE = "/local/scratch/{user}/new_jac"


def _partition_for(node: str) -> str:
    return {"ruapehu": "hopper", "mauao": "ampere"}.get(node, "interactive")


def _squeue_state(job_id: str) -> str | None:
    out = subprocess.run(
        ["squeue", "-j", job_id, "-h", "-o", "%T"], capture_output=True, text=True,
    ).stdout.strip()
    return out or None


def _wait_for_jobs(job_ids: list[str], poll_every: float = 15.0) -> None:
    pending = set(job_ids)
    while pending:
        time.sleep(poll_every)
        pending = {j for j in pending if _squeue_state(j) is not None}
    print(f"[run_lipschitz_ratio_bootstrap_eqd_sweep_gpu] jobs finished (left the queue): {job_ids}", flush=True)


def _job_succeeded(job_id: str) -> bool:
    out = subprocess.run(
        ["sacct", "-j", job_id, "-n", "-o", "State", "--parsable2"],
        capture_output=True, text=True,
    ).stdout.strip()
    state = out.splitlines()[0] if out else ""
    return state.startswith("COMPLETED")


def _sbatch(cmd: list[str]) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if out.returncode != 0:
        raise RuntimeError(f"sbatch failed: {' '.join(cmd)}\n{out.stderr}")
    job_id = out.stdout.strip()
    print(f"[run_lipschitz_ratio_bootstrap_eqd_sweep_gpu] submitted: {' '.join(cmd)} -> job {job_id}", flush=True)
    return job_id


def _layer_sizes_gb(node: str, subpaths: list[str], time_limit: str = "00:15:00") -> dict[str, float]:
    """du -sb each of `subpaths` (relative to that node's $NEW_JAC_SCRATCH)
    via a short sbatch job on `node` -- same as
    run_layer_batched_tangent_sweep_direct.py's _layer_sizes_gb (taranaki/NFS
    can't see node-local scratch, and the sweep's manifests on NFS don't
    carry on-disk byte sizes, only row counts)."""
    script = f"""#!/bin/bash
#SBATCH --job-name=layer_sizes
#SBATCH --partition={_partition_for(node)}
#SBATCH --nodelist={node}
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time={time_limit}
#SBATCH --output={REPO_ROOT}/logs/layer_sizes_%j.out
set -euo pipefail
cd {REPO_ROOT}
export NEW_JAC_SCRATCH="{NEW_JAC_SCRATCH_TEMPLATE.format(user=os.environ["USER"])}"
uv run python3 - <<'PYEOF'
import json, os, subprocess
scratch = os.environ["NEW_JAC_SCRATCH"]
subpaths = {subpaths!r}
sizes = {{}}
for sp in subpaths:
    full = os.path.join(scratch, sp)
    if not os.path.exists(full):
        continue
    out = subprocess.run(["du", "-sb", full], capture_output=True, text=True, check=True)
    sizes[sp] = int(out.stdout.split()[0]) / 1024**3
print("LAYER_SIZES_JSON=" + json.dumps(sizes))
PYEOF
"""
    script_path = os.path.join(REPO_ROOT, "logs", f"_layer_sizes_driver_{os.getpid()}.sh")
    os.makedirs(os.path.join(REPO_ROOT, "logs"), exist_ok=True)
    with open(script_path, "w") as f:
        f.write(script)
    try:
        job_id = _sbatch(["sbatch", "--parsable", script_path])
        _wait_for_jobs([job_id])
        assert _job_succeeded(job_id), f"layer_sizes job {job_id} did not complete -- see logs/layer_sizes_{job_id}.out"
        out_path = os.path.join(REPO_ROOT, "logs", f"layer_sizes_{job_id}.out")
        with open(out_path) as f:
            for line in f:
                if line.startswith("LAYER_SIZES_JSON="):
                    return json.loads(line[len("LAYER_SIZES_JSON="):])
        raise RuntimeError(f"layer_sizes job {job_id} produced no LAYER_SIZES_JSON line -- see {out_path}")
    finally:
        os.remove(script_path)


def _make_pair_batches(
    sublayer_pairs: list[tuple[str, str]], sizes_gb: dict[str, float], target_batch_gb: float,
) -> list[list[tuple[str, str]]]:
    """Greedy fill by (src, dst) PAIR, not individual layer -- a pair's
    "size" is its src+dst layers' combined GB (deduplicated against layers
    already in the current batch, since e.g. add_4 as one pair's dst and
    the next pair's src shouldn't be double-counted within the same
    batch)."""
    batches: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    cur_layers: set[str] = set()
    cur_gb = 0.0
    for src, dst in sublayer_pairs:
        added_layers = {l for l in (src, dst) if l not in cur_layers}
        added_gb = sum(sizes_gb[l] for l in added_layers)
        if cur and cur_gb + added_gb > target_batch_gb:
            batches.append(cur)
            cur, cur_layers, cur_gb = [], set(), 0.0
            added_layers = {l for l in (src, dst) if l not in cur_layers}
            added_gb = sum(sizes_gb[l] for l in added_layers)
        cur.append((src, dst))
        cur_layers |= added_layers
        cur_gb += added_gb
    if cur:
        batches.append(cur)
    return batches


def _layers_needed_later(batches: list[list[tuple[str, str]]], from_batch_idx: int, layer: str) -> bool:
    """True if `layer` appears (as either src or dst) in any batch AFTER
    from_batch_idx -- guards reclaim from deleting a layer's relayed-in copy
    that a later batch in this same driver run still needs."""
    for b in batches[from_batch_idx + 1 :]:
        for src, dst in b:
            if layer in (src, dst):
                return True
    return False


def _relay(from_node: str, to_node: str, subpath: str, time_limit: str | None = None) -> str:
    cmd = ["scripts/helper/sync_node_scratch.sh", "--from", from_node, "--to", to_node]
    if time_limit:
        cmd += ["--time", time_limit]
    cmd.append(subpath)
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, check=True)
    job2 = None
    for line in out.stdout.splitlines():
        if "submitted nfs_to_scratch" in line:
            job2 = line.strip().split()[-1]
    if job2 is None:
        raise RuntimeError(f"could not parse nfs_to_scratch job id from sync_node_scratch.sh output:\n{out.stdout}")
    print(f"[run_lipschitz_ratio_bootstrap_eqd_sweep_gpu] relay {from_node}->{to_node} {subpath}: "
          f"nfs_to_scratch job {job2}", flush=True)
    return job2


def _rm_rf_on_node(node: str, subpath: str, time_limit: str = "00:10:00") -> None:
    script = f"""#!/bin/bash
#SBATCH --job-name=reclaim_batch
#SBATCH --partition={_partition_for(node)}
#SBATCH --nodelist={node}
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time={time_limit}
#SBATCH --output={REPO_ROOT}/logs/reclaim_batch_%j.out
set -euo pipefail
cd {REPO_ROOT}
NEW_JAC_SCRATCH="{NEW_JAC_SCRATCH_TEMPLATE.format(user=os.environ["USER"])}"
TARGET="${{NEW_JAC_SCRATCH}}/{subpath}"
echo "removing ${{TARGET}} on $(hostname)"
rm -rf "${{TARGET}}"
echo "done"
"""
    script_path = os.path.join(REPO_ROOT, "logs", f"_reclaim_driver_{os.getpid()}.sh")
    with open(script_path, "w") as f:
        f.write(script)
    try:
        job_id = _sbatch(["sbatch", "--parsable", script_path])
        _wait_for_jobs([job_id])
        assert _job_succeeded(job_id), f"reclaim job {job_id} did not complete -- see logs/reclaim_batch_{job_id}.out"
    finally:
        os.remove(script_path)


def _batch_complete(
    out_dir: str, sample_indices: list[int], batch_sublayers: list[tuple[str, str]],
) -> bool:
    """True if every (sublayer pair, sample) in this batch already has an
    output h5 under out_dir -- read directly from taranaki over NFS.
    Matches lipschitz_ratio_bootstrap_eqd_sweep_gpu.py's own per-sample
    os.path.exists skip, just checked up front so the DRIVER can skip
    relaying+launching a whole batch's job."""
    for src, dst in batch_sublayers:
        pair_dir = os.path.join(REPO_ROOT, out_dir, f"{src}.{dst}")
        for s in sample_indices:
            if not os.path.exists(os.path.join(pair_dir, f"sample_{s:05d}.h5")):
                return False
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=str, help="Path to a toml ModelConfig")
    p.add_argument("--n-samples", type=int, default=None,
                    help="forwarded to lipschitz_ratio_bootstrap_eqd_sweep_gpu.py's --n-samples "
                         "for every batch -- also used here to compute _batch_complete's sample "
                         "range (default: cfg's full n_samples, read from the first available "
                         "layer manifest)")
    p.add_argument("--target-batch-gb", type=float, default=250.0,
                    help="group sublayer PAIRS into batches whose combined (deduplicated) input "
                         "layer data stays under this many GB on disk")
    p.add_argument("--compute-node", default="ruapehu", help="node the bootstrap-EQD GPU sweep runs on")
    p.add_argument("--storage-node", default="mauao", help="node the full perturbed sweep lives on")
    p.add_argument("--job-time-limit", default="12:00:00",
                    help="--time for the lipschitz_ratio_bootstrap_eqd_sweep_gpu sbatch job (12h partition cap)")
    p.add_argument("--relay-time-limit", default=None,
                    help="--time override for relay sbatch jobs (default: sync_node_scratch.sh's own default)")
    p.add_argument("--sublayers", nargs="*", default=None,
                    help="forwarded restriction -- only run these src.dst sublayer pairs "
                         "(default: every cfg.sublayers entry)")
    p.add_argument("--n-pairs", type=int, default=200_000, help="forwarded --n-pairs")
    p.add_argument("--n-pairs-save", type=int, default=10_000, help="forwarded --n-pairs-save")
    p.add_argument("--n-tail-min", type=int, default=20, help="forwarded --n-tail-min")
    p.add_argument("--n-tail-max", type=int, default=2_000, help="forwarded --n-tail-max")
    p.add_argument("--n-tail-step", type=int, default=50, help="forwarded --n-tail-step")
    p.add_argument("--eqd-n-boot", type=int, default=200, help="forwarded --eqd-n-boot")
    p.add_argument("--m-plotting-points", type=int, default=50, help="forwarded --m-plotting-points")
    p.add_argument("--group-size", type=int, default=16, help="forwarded --group-size")
    p.add_argument("--seed", type=int, default=0, help="forwarded --seed")
    p.add_argument("--out-dir", default=None,
                    help="forwarded --out-dir (default: onnx_models/{cfg.name}/lipschitz_ratio_bootstrap_eqd_sweep_gpu)")
    p.add_argument("--skip-reclaim", action="store_true",
                    help="leave each batch's input data on the compute node after the sweep runs (debugging only)")
    p.add_argument("--dry-run", action="store_true",
                    help="print the planned batches and exit without submitting anything")
    args = p.parse_args()

    cfg = load_config(args.config)
    sweep_dir = sweep_dir_for(cfg)
    all_sublayer_pairs = [(s.input, s.output) for s in cfg.sublayers]
    if args.sublayers is not None:
        wanted = set(args.sublayers)
        all_sublayer_pairs = [(src, dst) for src, dst in all_sublayer_pairs if f"{src}.{dst}" in wanted]
        missing = wanted - {f"{src}.{dst}" for src, dst in all_sublayer_pairs}
        if missing:
            raise SystemExit(f"--sublayers requested pairs not in {args.config}: {sorted(missing)}")

    out_dir = args.out_dir or os.path.join("onnx_models", cfg.name, "lipschitz_ratio_bootstrap_eqd_sweep_gpu")

    all_layers = sorted({l for src, dst in all_sublayer_pairs for l in (src, dst)})
    layer_subpath_for = {l: os.path.join(sweep_dir, l) for l in all_layers}

    print(f"measuring layer sizes on {args.storage_node} ...", flush=True)
    sizes_by_subpath = _layer_sizes_gb(args.storage_node, list(layer_subpath_for.values()))
    sizes_gb = {l: sizes_by_subpath[sp] for l, sp in layer_subpath_for.items() if sp in sizes_by_subpath}
    # A pair whose input is the sweep's very first layer (e.g. "input_ids")
    # has no perturbed-sweep output for that layer -- lipschitz_ratio_bootstrap_eqd_sweep_gpu.py
    # already skips such pairs internally (no manifest to read), so drop
    # them here too rather than failing the missing-layer check below.
    all_sublayer_pairs = [(src, dst) for src, dst in all_sublayer_pairs if src in sizes_gb]
    missing = [l for l in all_layers if l not in sizes_gb and any(l == dst for _s, dst in all_sublayer_pairs)]
    if missing:
        raise RuntimeError(f"output layers missing from {args.storage_node}'s scratch (never generated there?): {missing}")
    for l in sorted(sizes_gb):
        print(f"  {l}: {sizes_gb[l]:.1f} GB", flush=True)

    batches = _make_pair_batches(all_sublayer_pairs, sizes_gb, args.target_batch_gb)
    print(f"\n{len(batches)} batch(es) at --target-batch-gb={args.target_batch_gb}:", flush=True)
    for i, b in enumerate(batches):
        batch_layers = {l for src, dst in b for l in (src, dst)}
        print(f"  batch {i + 1}: {len(b)} pair(s), {sum(sizes_gb[l] for l in batch_layers):.1f} GB -- {b}", flush=True)

    if args.dry_run:
        return

    n_samples_for_check = args.n_samples
    if n_samples_for_check is None:
        # Need SOME sample count to check batch completeness up front --
        # read it off the first pair's input-layer manifest the same way
        # lipschitz_ratio_bootstrap_eqd_sweep_gpu.py itself does per pair,
        # rather than duplicating manifest-loading logic here for every pair.
        first_manifest_path = os.path.join(REPO_ROOT, sweep_dir, f"{all_sublayer_pairs[0][0]}.manifest.json")
        with open(first_manifest_path) as f:
            n_samples_for_check = json.load(f)["attrs"]["n_samples"]
    sample_indices = list(range(n_samples_for_check))

    for i, batch_pairs in enumerate(batches):
        print(f"\n=== batch {i + 1}/{len(batches)}: {batch_pairs} ===", flush=True)

        if _batch_complete(out_dir, sample_indices, batch_pairs):
            print(f"batch {i + 1}/{len(batches)}: already complete, skipping", flush=True)
            continue

        batch_layers = sorted({l for src, dst in batch_pairs for l in (src, dst)})
        layer_subpaths = [layer_subpath_for[l] for l in batch_layers]

        present_sizes_gb = _layer_sizes_gb(args.compute_node, layer_subpaths)
        to_relay = [
            sp for sp in layer_subpaths
            if abs(present_sizes_gb.get(sp, -1) - sizes_by_subpath[sp]) > 0.01
        ]
        already_present = [sp for sp in layer_subpaths if sp not in to_relay]
        if already_present:
            print(f"already present on {args.compute_node}, skipping relay for: {already_present}", flush=True)

        if to_relay:
            print(f"relaying {len(to_relay)} layer(s) {args.storage_node} -> {args.compute_node} ...", flush=True)
            relay_in_jobs = [
                _relay(args.storage_node, args.compute_node, sp, time_limit=args.relay_time_limit)
                for sp in to_relay
            ]
            _wait_for_jobs(relay_in_jobs)
            failed = [j for j in relay_in_jobs if not _job_succeeded(j)]
            if failed:
                raise RuntimeError(f"batch {i + 1}: relay-in job(s) {failed} did not complete -- see logs/nfs_to_scratch_<job>.out")

        print(f"running bootstrap-EQD sweep for batch {i + 1}/{len(batches)} on {args.compute_node} ...", flush=True)
        gen_job = _submit_sweep_job(
            args.config, batch_pairs, args.compute_node, out_dir, args,
        )
        _wait_for_jobs([gen_job])
        if not _job_succeeded(gen_job):
            raise RuntimeError(
                f"batch {i + 1}: sweep job {gen_job} did not complete -- see "
                f"logs/lipschitz_ratio_bootstrap_eqd_sweep_gpu_{gen_job}.out"
            )

        if args.skip_reclaim:
            print(f"batch {i + 1}/{len(batches)}: --skip-reclaim, leaving input data on {args.compute_node}", flush=True)
        elif args.compute_node == args.storage_node:
            print(f"compute_node == storage_node ({args.compute_node}), skipping input-data reclaim "
                  "(it's storage_node's only copy, not a relayed-in duplicate)", flush=True)
        else:
            print(f"reclaiming batch {i + 1}/{len(batches)}'s input data on {args.compute_node} ...", flush=True)
            for l, sp in zip(batch_layers, layer_subpaths):
                if _layers_needed_later(batches, i, l):
                    print(f"  {l}: needed by a later batch, not reclaiming", flush=True)
                    continue
                _rm_rf_on_node(args.compute_node, sp)

        print(f"=== batch {i + 1}/{len(batches)} done ===", flush=True)

    print("\nall batches complete", flush=True)


_SWEEP_LAUNCHER = os.path.join(
    REPO_ROOT, "scripts/batch_scripts/lipschitz_ratio_bootstrap_eqd_sweep_gpu_mauao_sbatch.sh",
)


def _submit_sweep_job(
    config_path: str, batch_pairs: list[tuple[str, str]], node: str, out_dir: str, args: argparse.Namespace,
) -> str:
    """Submits scripts/batch_scripts/lipschitz_ratio_bootstrap_eqd_sweep_gpu_mauao_sbatch.sh
    with this batch's args, pinned to `node` via --partition/--nodelist/--gres
    overrides -- the launcher's own name says "mauao" but it's a thin
    wrapper (config, --perturbed-sweep-dir, --out-dir, passthrough args),
    nothing mauao-specific, so overriding --nodelist/--partition/--gres here
    retargets it to any node with a GPU, same pattern as the tangent-sweep
    driver's _submit_generation_job pinning a differently-named launcher."""
    gpu_gres = {"mauao": "gpu:a40:1", "ruapehu": "gpu:h100:1"}.get(node, "gpu:1")
    cmd = [
        "sbatch", "--parsable",
        "--partition", _partition_for(node), "--nodelist", node, "--gres", gpu_gres,
        "--time", args.job_time_limit,
        _SWEEP_LAUNCHER,
        config_path,
        "--sublayers", *[f"{src}.{dst}" for src, dst in batch_pairs],
        "--out-dir", out_dir,
        "--n-pairs", str(args.n_pairs),
        "--n-pairs-save", str(args.n_pairs_save),
        "--n-tail-min", str(args.n_tail_min),
        "--n-tail-max", str(args.n_tail_max),
        "--n-tail-step", str(args.n_tail_step),
        "--eqd-n-boot", str(args.eqd_n_boot),
        "--m-plotting-points", str(args.m_plotting_points),
        "--group-size", str(args.group_size),
        "--seed", str(args.seed),
    ]
    if args.n_samples is not None:
        cmd += ["--n-samples", str(args.n_samples)]
    return _sbatch(cmd)


if __name__ == "__main__":
    main()
