"""Drives scripts.python.generate_tangent_sweep_direct over a full
scripts.python.generate_perturbed_sweep output in --target-batch-gb-sized
batches of (src) layers, relaying each batch's input data from mauao's
local scratch to ruapehu's local scratch before running the fused
basis+sampling step, so the compute node never needs more than one batch's
input sweep data resident at once (ruapehu's ~1.2TB node-local scratch,
shared with other users, can't hold the full multi-TB sweep -- see
CLAUDE.md's local-scratch guidance).

This is the scripts.python.run_layer_batched_tangent_bases equivalent for
generate_tangent_sweep_direct.py, with the same structural simplification
that script has over run_layer_batched_training: generate_tangent_sweep_direct's
output is ShardedLayerWriter-backed (--backend nfs), so each
(sublayer, radius, sample) shard is already pushed to NFS staging by the
job itself as it finishes -- no separate relay-out step. It differs from
run_layer_batched_tangent_bases in one respect: batching here only exists
to bound how much of the (large, ~37GB/layer for gpt2) perturbed-sweep
INPUT sits on ruapehu's scratch at once -- generate_tangent_sweep_direct's
whole point is that its OUTPUT is tiny (~23MB per (sublayer, radius,
sample) at --n=1000, vs. generate_tangent_bases.py's ~92GB dense basis per
(layer, sample) at gpt2 scale -- see that script's docstring), so there is
no output-size batching concern to speak of, unlike the now-irrelevant
--target-batch-gb motivation in the tangent-bases driver.

Per batch:
  1. relay that batch's SRC layers storage_node -> compute_node
     (scripts/helper/sync_node_scratch.sh, one relay chain per layer, run
     concurrently)
  2. run scripts.python.generate_tangent_sweep_direct on compute_node,
     restricted to that batch's --layers (a new sbatch launcher pinned to
     hopper/ruapehu)
  3. reclaim (rm -rf) that batch's input layer dirs from compute_node's
     scratch, once the job has pushed its output

This only submits/polls sbatch jobs from taranaki -- no compute runs here.
Layer sizing is measured directly on storage_node (the manifests on NFS
carry row counts/schema but not on-disk byte sizes -- see _layer_sizes_gb),
via one short sbatch `du` job.

--compute-node/--storage-node can also be the same node (e.g. both
"mauao") -- relay-in already no-ops in this case (present_sizes_gb already
matches storage_node's size), and the reclaim step is explicitly skipped
(see the compute_node == storage_node check below): reclaiming "relayed-in"
input that's actually storage_node's only copy would delete real data.
`sync_node_scratch.sh` itself refuses a --from/--to on the same node, which
is what motivated this guard (see the identical fix in
run_layer_batched_training.py).

Idempotent to re-running after a crash: a batch whose (sublayer, radius,
sample) outputs are already complete (per each layer_key's NFS-visible
.manifest.json "complete" field) is skipped entirely, so a driver restart
resumes from wherever it left off. Does NOT resume a half-relayed or
half-generated batch transparently -- the whole batch is reissued
(ShardedLayerWriter's own per-shard resume avoids redundant GPU work for
shards that already exist).
"""

import argparse
import json
import os
import subprocess
import time

from autoencoder.data import sweep_dir_for
from config import load_config
from execute.sharded_writer import discover_layer_names

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
    print(f"[run_layer_batched_tangent_sweep_direct] jobs finished (left the queue): {job_ids}", flush=True)


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
    print(f"[run_layer_batched_tangent_sweep_direct] submitted: {' '.join(cmd)} -> job {job_id}", flush=True)
    return job_id


def _layer_sizes_gb(node: str, subpaths: list[str], time_limit: str = "00:15:00") -> dict[str, float]:
    """du -sb each of `subpaths` (relative to that node's $NEW_JAC_SCRATCH)
    via a short sbatch job on `node` -- taranaki/NFS can't see node-local
    scratch, and the sweep's manifests (on NFS) don't carry on-disk byte
    sizes, only row counts."""
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


def _make_batches(layers: list[str], sizes_gb: dict[str, float], target_batch_gb: float) -> list[list[str]]:
    """Greedy fill, same as run_layer_batched_tangent_bases._make_batches --
    bounds how much perturbed-sweep INPUT sits on compute_node's scratch at
    once (this script's output is tiny, so only the input matters here)."""
    batches: list[list[str]] = []
    cur: list[str] = []
    cur_gb = 0.0
    for layer in layers:
        size = sizes_gb[layer]
        if cur and cur_gb + size > target_batch_gb:
            batches.append(cur)
            cur, cur_gb = [], 0.0
        cur.append(layer)
        cur_gb += size
    if cur:
        batches.append(cur)
    return batches


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
    print(f"[run_layer_batched_tangent_sweep_direct] relay {from_node}->{to_node} {subpath}: nfs_to_scratch job {job2}", flush=True)
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
    out_dir: str, sample_indices: list[int], batch_sublayers: list[tuple[str, str]], radii: list[float],
) -> bool:
    """True if every (sublayer, radius, sample) in this batch already has a
    "complete" ShardedLayerWriter manifest under `out_dir` -- read directly
    from taranaki over NFS, same as run_layer_batched_tangent_bases'
    _batch_complete."""
    for s in sample_indices:
        for src, dst in batch_sublayers:
            for radius in radii:
                manifest_path = os.path.join(
                    REPO_ROOT, out_dir, f"{src}.{dst}.radius{radius:g}.sample{s}.manifest.json"
                )
                if not os.path.exists(manifest_path):
                    return False
                with open(manifest_path) as f:
                    if not json.load(f).get("complete", False):
                        return False
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=str, help="Path to a toml ModelConfig")
    p.add_argument("--sample-idx", type=int, nargs="+", default=[0],
                   help="forwarded to generate_tangent_sweep_direct.py's --sample-idx for every batch")
    p.add_argument("--target-batch-gb", type=float, default=250.0,
                   help="group SRC layers into batches whose perturbed-sweep INPUT stays under this many GB "
                        "on disk (output is tiny regardless of batch size, see module docstring)")
    p.add_argument("--compute-node", default="ruapehu", help="node tangent-sweep generation runs on")
    p.add_argument("--storage-node", default="mauao", help="node the full perturbed sweep lives on")
    p.add_argument("--job-time-limit", default="12:00:00", help="--time for the generate_tangent_sweep_direct sbatch job (12h partition cap)")
    p.add_argument("--relay-time-limit", default=None, help="--time override for relay sbatch jobs (default: sync_node_scratch.sh's own default)")
    p.add_argument("--sublayers", nargs="*", default=None,
                   help="forwarded to generate_tangent_sweep_direct.py -- only run these src.dst sublayers "
                        "(default: every cfg.sublayers entry whose input is in this batch's layers)")
    p.add_argument("--knn-k", type=int, default=None, help="forwarded to generate_tangent_sweep_direct.py (default: cfg.d_model + 1)")
    p.add_argument("--basis-method", choices=["full", "dual"], default="dual",
                   help="forwarded to generate_tangent_sweep_direct.py -- 'dual' (default here) so --knn-k can go below cfg.d_model")
    p.add_argument("--n", type=int, default=1000, help="forwarded to generate_tangent_sweep_direct.py's --n (samples per sublayer/position)")
    p.add_argument("--radius", type=float, nargs="+", default=[0.1], help="forwarded to generate_tangent_sweep_direct.py's --radius")
    p.add_argument("--seed", type=int, default=0, help="forwarded to generate_tangent_sweep_direct.py")
    p.add_argument("--out-dir", default=None, help="forwarded to generate_tangent_sweep_direct.py's --out-dir (default: onnx_models/{cfg.name}/tangent_sweep_direct)")
    p.add_argument("--skip-reclaim", action="store_true", help="leave each batch's input data on the compute node after generation (debugging only)")
    p.add_argument("--dry-run", action="store_true", help="print the planned batches and exit without submitting anything")
    args = p.parse_args()

    cfg = load_config(args.config)
    sweep_dir = sweep_dir_for(cfg)
    full_layers = discover_layer_names(sweep_dir)
    print(f"{len(full_layers)} layers in {sweep_dir}: {full_layers}", flush=True)

    out_dir = args.out_dir or os.path.join("onnx_models", cfg.name, "tangent_sweep_direct")
    layer_subpath_for = {l: os.path.join("onnx_models", cfg.name, "perturbed_sweep", l) for l in full_layers}

    print(f"measuring layer sizes on {args.storage_node} ...", flush=True)
    sizes_by_subpath = _layer_sizes_gb(args.storage_node, list(layer_subpath_for.values()))
    sizes_gb = {l: sizes_by_subpath[sp] for l, sp in layer_subpath_for.items() if sp in sizes_by_subpath}
    missing = [l for l in full_layers if l not in sizes_gb]
    if missing:
        raise RuntimeError(f"layers missing from {args.storage_node}'s scratch (never generated there?): {missing}")
    for l in full_layers:
        print(f"  {l}: {sizes_gb[l]:.1f} GB", flush=True)

    batches = _make_batches(full_layers, sizes_gb, args.target_batch_gb)
    print(f"\n{len(batches)} batch(es) at --target-batch-gb={args.target_batch_gb}:", flush=True)
    for i, b in enumerate(batches):
        print(f"  batch {i + 1}: {len(b)} layer(s), {sum(sizes_gb[l] for l in b):.1f} GB -- {b}", flush=True)

    if args.dry_run:
        return

    for i, batch_layers in enumerate(batches):
        print(f"\n=== batch {i + 1}/{len(batches)}: {batch_layers} ===", flush=True)

        batch_sublayers = [(s.input, s.output) for s in cfg.sublayers if s.input in batch_layers]
        if args.sublayers is not None:
            wanted = set(args.sublayers)
            batch_sublayers = [(src, dst) for src, dst in batch_sublayers if f"{src}.{dst}" in wanted]
        if not batch_sublayers:
            print(f"batch {i + 1}/{len(batches)}: no sublayers to run, skipping", flush=True)
            continue

        if _batch_complete(out_dir, args.sample_idx, batch_sublayers, args.radius):
            print(f"batch {i + 1}/{len(batches)}: tangent sweep already complete, skipping", flush=True)
            continue

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

        print(f"generating tangent sweep for batch {i + 1}/{len(batches)} on {args.compute_node} ...", flush=True)
        gen_job = _submit_generation_job(
            args.config, batch_layers, batch_sublayers, args.sample_idx, args.compute_node, out_dir,
            knn_k=args.knn_k, basis_method=args.basis_method, n=args.n, radii=args.radius,
            seed=args.seed, time_limit=args.job_time_limit,
        )
        _wait_for_jobs([gen_job])
        if not _job_succeeded(gen_job):
            raise RuntimeError(f"batch {i + 1}: generation job {gen_job} did not complete -- see logs/tangent_sweep_direct_ruapehu_{gen_job}.out")

        if args.skip_reclaim:
            print(f"batch {i + 1}/{len(batches)}: --skip-reclaim, leaving input data on {args.compute_node}", flush=True)
        elif args.compute_node == args.storage_node:
            print(f"compute_node == storage_node ({args.compute_node}), skipping input-data reclaim "
                  "(it's storage_node's only copy, not a relayed-in duplicate)", flush=True)
        else:
            print(f"reclaiming batch {i + 1}/{len(batches)}'s input data on {args.compute_node} ...", flush=True)
            for sp in layer_subpaths:
                _rm_rf_on_node(args.compute_node, sp)

        print(f"=== batch {i + 1}/{len(batches)} done ===", flush=True)

    print("\nall batches complete", flush=True)


_GEN_LAUNCHER = os.path.join(
    REPO_ROOT, "scripts/batch_scripts/generate_tangent_sweep_direct_ruapehu_driven.sh",
)


def _submit_generation_job(
    config_path: str, layers: list[str], sublayers: list[tuple[str, str]], sample_idx: list[int], node: str,
    out_dir: str, knn_k: int | None, basis_method: str, n: int, radii: list[float], seed: int, time_limit: str,
) -> str:
    """Submits scripts/batch_scripts/generate_tangent_sweep_direct_ruapehu_driven.sh
    with this batch's args, pinned to `node` via --partition/--nodelist
    overrides, same pattern as run_layer_batched_tangent_bases._submit_generation_job."""
    cmd = [
        "sbatch", "--parsable",
        "--partition", _partition_for(node), "--nodelist", node, "--time", time_limit,
        _GEN_LAUNCHER,
        config_path,
        "--sample-idx", *[str(s) for s in sample_idx],
        "--layers", *layers,
        "--sublayers", *[f"{src}.{dst}" for src, dst in sublayers],
        "--out-dir", out_dir,
        "--basis-method", basis_method,
        "--n", str(n),
        "--radius", *[str(r) for r in radii],
        "--seed", str(seed),
        # sweep data was relayed onto `node`'s scratch via scripts/helper/sync_node_scratch.sh
        # (NFS staging + node_relay), not pushed to Hetzner -- see this batch's relay-in step.
        "--backend", "nfs",
    ]
    if knn_k is not None:
        cmd += ["--knn-k", str(knn_k)]
    return _sbatch(cmd)


if __name__ == "__main__":
    main()
