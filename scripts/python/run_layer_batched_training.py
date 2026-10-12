"""Drives scripts.python.train_batched_manifold over a full
scripts.python.generate_perturbed_sweep output in --target-batch-gb-sized
batches of layers, relaying each batch's input data from mauao's local
scratch to ruapehu's local scratch before training and each batch's finished
checkpoints back to mauao afterward, so training never needs more than one
batch's input data plus in-progress checkpoints resident on ruapehu at once
(ruapehu's ~1.2TB node-local scratch, shared with other users, can't hold the
full multi-TB sweep -- see CLAUDE.md's local-scratch guidance and this
script's task description).

Per batch:
  1. relay that batch's layers mauao -> ruapehu (scripts/helper/sync_node_scratch.sh,
     one relay chain per layer, run concurrently)
  2. train scripts.python.train_batched_manifold on ruapehu, restricted to that
     batch's --layers (a new sbatch launcher pinned to hopper/ruapehu,
     --dependency=afterok on every relay-in job)
  3. relay that batch's finished checkpoints ruapehu -> mauao
     (sync_node_scratch.sh again, --dependency=afterok on the training job)
  4. reclaim (rm -rf) that batch's input layer dirs from ruapehu's scratch,
     once step 3 confirms the checkpoints are safely on mauao

This only submits/polls sbatch jobs from taranaki -- no compute runs here.
Layer sizing is measured directly on mauao (the manifests on NFS carry row
counts/schema but not on-disk byte sizes -- see _layer_sizes_gb), via one
short sbatch `du` job.

--compute-node/--storage-node can also both be "mauao" (training runs on
the same a40 node the sweep data already lives on) -- relay-in already
no-ops in this case (present_sizes_gb already matches storage_node's size,
since it's the same node), and relay-out/reclaim are explicitly skipped
(see the compute_node == storage_node checks below): relaying checkpoints
to yourself is pointless work, and reclaiming "relayed-in" input that's
actually storage_node's only copy would delete real data. `sync_node_scratch.sh`
itself refuses a --from/--to on the same node, which is what motivated
these guards.

Idempotent to re-running after a crash: a batch whose checkpoints are
already complete on mauao (per autoencoder.batched_autoencoder.load_batched's
expected_indices check) is skipped entirely, so a driver restart resumes
from wherever it left off rather than re-relaying/re-training finished
batches. Does NOT resume a half-relayed or half-trained batch transparently
-- the whole batch is reissued (train_batched_manifold.py's own per-row
checkpoint skip, see its "done" check, avoids redundant GPU work for rows
that already have a checkpoint file).
"""

import argparse
import json
import os
import subprocess
import time

from autoencoder.data import ae_checkpoints_dir_for, sweep_dir_for
from config import load_config
from execute.sharded_writer import discover_layer_names

REPO_ROOT = "/nfs-share/pa511/code_bases/new_jac"
NEW_JAC_SCRATCH_TEMPLATE = "/local/scratch/{user}/new_jac"


def _partition_for(node: str) -> str:
    return {"ruapehu": "hopper", "mauao": "ampere"}.get(node, "interactive")


def _squeue_state(job_id: str) -> str | None:
    """None once the job has left the queue entirely (completed or failed --
    squeue -j on a finished job returns nothing, not a terminal state), same
    poll this session already used for other sbatch chains."""
    out = subprocess.run(
        ["squeue", "-j", job_id, "-h", "-o", "%T"], capture_output=True, text=True,
    ).stdout.strip()
    return out or None


def _wait_for_jobs(job_ids: list[str], poll_every: float = 15.0) -> None:
    pending = set(job_ids)
    while pending:
        time.sleep(poll_every)
        pending = {j for j in pending if _squeue_state(j) is not None}
    print(f"[run_layer_batched_training] jobs finished (left the queue): {job_ids}", flush=True)


def _job_succeeded(job_id: str) -> bool:
    """sacct is the only way to distinguish afterok-satisfied (COMPLETED)
    from a job that left the queue via FAILED/CANCELLED/TIMEOUT -- squeue
    only sees jobs still queued/running."""
    out = subprocess.run(
        ["sacct", "-j", job_id, "-n", "-o", "State", "--parsable2"],
        capture_output=True, text=True,
    ).stdout.strip()
    # sacct prints one line per job step; the parent job's line comes first
    state = out.splitlines()[0] if out else ""
    return state.startswith("COMPLETED")


def _sbatch(cmd: list[str]) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if out.returncode != 0:
        raise RuntimeError(f"sbatch failed: {' '.join(cmd)}\n{out.stderr}")
    job_id = out.stdout.strip()
    print(f"[run_layer_batched_training] submitted: {' '.join(cmd)} -> job {job_id}", flush=True)
    return job_id


def _layer_sizes_gb(node: str, subpaths: list[str], time_limit: str = "00:15:00") -> dict[str, float]:
    """du -sb each of `subpaths` (relative to that node's $NEW_JAC_SCRATCH)
    via a short sbatch job on `node` -- taranaki/NFS can't see node-local
    scratch, and the sweep's manifests (on NFS) don't carry on-disk byte
    sizes, only row counts. Returns {subpath: size_gb}; a subpath that
    doesn't exist on `node` yet is omitted (caller decides what that means)."""
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
    """Greedy fill: walk `layers` in order, closing a batch (starting a new
    one) whenever adding the next layer would push it over target_batch_gb.
    Order-preserving (not a bin-packing optimum) so each batch is a
    contiguous slice of the layer list, which keeps checkpoint indices
    within a batch contiguous too (easier to reason about / verify)."""
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
    """Submits sync_node_scratch.sh's two-job chain for one subpath, returns
    the SECOND job's id (the one that actually lands the data on `to_node`
    -- callers should wait on/depend on this one, not the staging leg)."""
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
    print(f"[run_layer_batched_training] relay {from_node}->{to_node} {subpath}: nfs_to_scratch job {job2}", flush=True)
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
    config_path: str, sample_indices: list[int], batch_layers: list[str], full_layers: list[str],
    seq: int, storage_node: str,
) -> bool:
    """True if every (sample, layer, pos) row this batch is responsible for
    already has a checkpoint file under each sample's ae_checkpoints_dir_for
    directory on `storage_node`. Checkpoints live on scratch, never NFS
    (CLAUDE.md), and taranaki (where this driver runs) doesn't mount any
    node's local scratch -- so this check has to run ON storage_node, via a
    short sbatch job, same as _layer_sizes_gb."""
    expected = sorted({full_layers.index(layer) * seq + pos for layer in batch_layers for pos in range(seq)})

    payload = f"""
import glob, json, os, re
from autoencoder.data import ae_checkpoints_dir_for
from config import load_config

cfg = load_config({config_path!r})
expected = set({expected!r})
complete = True
for s in {sample_indices!r}:
    out_dir = ae_checkpoints_dir_for(cfg, s)
    by_index = set()
    if os.path.isdir(out_dir):
        for path in glob.glob(os.path.join(out_dir, "model_*.pt")):
            m = re.match(r"model_(\\d+)\\.pt$", os.path.basename(path))
            if m:
                by_index.add(int(m.group(1)))
    if not expected.issubset(by_index):
        complete = False
print("BATCH_COMPLETE=" + json.dumps(complete))
"""
    script = f"""#!/bin/bash
#SBATCH --job-name=batch_complete_check
#SBATCH --partition={_partition_for(storage_node)}
#SBATCH --nodelist={storage_node}
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output={REPO_ROOT}/logs/batch_complete_check_%j.out
set -euo pipefail
cd {REPO_ROOT}
export NEW_JAC_SCRATCH="{NEW_JAC_SCRATCH_TEMPLATE.format(user=os.environ["USER"])}"
uv run python3 - <<'PYEOF'
{payload}
PYEOF
"""
    script_path = os.path.join(REPO_ROOT, "logs", f"_batch_complete_driver_{os.getpid()}.sh")
    os.makedirs(os.path.join(REPO_ROOT, "logs"), exist_ok=True)
    with open(script_path, "w") as f:
        f.write(script)
    try:
        job_id = _sbatch(["sbatch", "--parsable", script_path])
        _wait_for_jobs([job_id])
        if not _job_succeeded(job_id):
            return False  # treat a failed check job as "not confirmed complete", not fatal
        out_path = os.path.join(REPO_ROOT, "logs", f"batch_complete_check_{job_id}.out")
        with open(out_path) as f:
            for line in f:
                if line.startswith("BATCH_COMPLETE="):
                    return json.loads(line[len("BATCH_COMPLETE="):])
        return False
    finally:
        os.remove(script_path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=str, help="Path to a toml ModelConfig")
    p.add_argument("--sample-idx", type=int, nargs="+", default=[0],
                   help="forwarded to train_batched_manifold.py's --sample-idx for every batch")
    p.add_argument("--target-batch-gb", type=float, default=250.0,
                   help="group layers into batches that each stay under this many GB on disk")
    p.add_argument("--compute-node", default="ruapehu", help="node training runs on")
    p.add_argument("--storage-node", default="mauao", help="node the full sweep input + accumulated checkpoints live on")
    p.add_argument("--train-time-limit", default="12:00:00", help="--time for the training sbatch job (12h partition cap)")
    p.add_argument("--relay-time-limit", default=None, help="--time override for relay sbatch jobs (default: sync_node_scratch.sh's own default)")
    p.add_argument("--n-models-per-chunk", type=int, default=200, help="forwarded to train_batched_manifold.py")
    p.add_argument("--epochs", type=int, default=200, help="forwarded to train_batched_manifold.py")
    p.add_argument("--checkpoint-every", type=int, default=50, help="forwarded to train_batched_manifold.py")
    p.add_argument("--lr", type=float, default=None, help="forwarded to train_batched_manifold.py's --lr (default: that script's own default)")
    p.add_argument("--skip-reclaim", action="store_true", help="leave each batch's input data on the compute node after training (debugging only)")
    p.add_argument("--dry-run", action="store_true", help="print the planned batches and exit without submitting anything")
    args = p.parse_args()

    cfg = load_config(args.config)
    sweep_dir = sweep_dir_for(cfg)
    full_layers = discover_layer_names(sweep_dir)
    print(f"{len(full_layers)} layers in {sweep_dir}: {full_layers}", flush=True)

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

    ckpt_subpath = os.path.relpath(ae_checkpoints_dir_for(cfg, args.sample_idx[0]), NEW_JAC_SCRATCH_TEMPLATE.format(user=os.environ["USER"]))
    ckpt_root_subpath = os.path.dirname(ckpt_subpath)  # {...}/ae_checkpoints -- shared root across every --sample-idx

    for i, batch_layers in enumerate(batches):
        print(f"\n=== batch {i + 1}/{len(batches)}: {batch_layers} ===", flush=True)

        if _batch_complete(args.config, args.sample_idx, batch_layers, full_layers, cfg.seq, args.storage_node):
            print(f"batch {i + 1}/{len(batches)}: checkpoints already complete on {args.storage_node}, skipping", flush=True)
            continue

        layer_subpaths = [layer_subpath_for[l] for l in batch_layers]

        # Skip subpaths a previous (e.g. crashed/cancelled) run already relayed onto
        # compute_node and never reclaimed -- compare against storage_node's size
        # (already measured above) rather than just presence, so a partial leftover
        # copy from an interrupted relay still gets re-relayed instead of silently
        # treated as complete.
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

        print(f"training batch {i + 1}/{len(batches)} on {args.compute_node} ...", flush=True)
        train_job = _submit_training_job(
            args.config, batch_layers, args.sample_idx, args.compute_node,
            n_models_per_chunk=args.n_models_per_chunk, epochs=args.epochs,
            checkpoint_every=args.checkpoint_every, time_limit=args.train_time_limit,
            lr=args.lr,
        )
        _wait_for_jobs([train_job])
        if not _job_succeeded(train_job):
            raise RuntimeError(f"batch {i + 1}: training job {train_job} did not complete -- see logs/train_batched_manifold_ruapehu_{train_job}.out")

        if args.compute_node == args.storage_node:
            print(f"compute_node == storage_node ({args.compute_node}), skipping checkpoint relay-out "
                  "(already on the right node)", flush=True)
        else:
            print(f"relaying checkpoints {args.compute_node} -> {args.storage_node} ...", flush=True)
            relay_out_job = _relay(args.compute_node, args.storage_node, ckpt_root_subpath, time_limit=args.relay_time_limit)
            _wait_for_jobs([relay_out_job])
            if not _job_succeeded(relay_out_job):
                raise RuntimeError(f"batch {i + 1}: relay-out job {relay_out_job} did not complete -- see logs/nfs_to_scratch_<job>.out")

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


_TRAIN_LAUNCHER = os.path.join(
    REPO_ROOT, "scripts/batch_scripts/tiny_llama/train_batched_manifold_ruapehu.sh",
)


def _submit_training_job(
    config_path: str, layers: list[str], sample_idx: list[int], node: str,
    n_models_per_chunk: int, epochs: int, checkpoint_every: int, time_limit: str,
    lr: float | None = None,
) -> str:
    """Submits scripts/batch_scripts/tiny_llama/train_batched_manifold_ruapehu.sh
    (a thin `exec uv run python -m scripts.python.train_batched_manifold "$@"`
    wrapper -- see that file) with this batch's args, pinned to `node` via
    --partition/--nodelist overrides (the launcher's own #SBATCH directives
    default to ruapehu/hopper, but --nodelist/--partition on the sbatch
    command line take precedence, so --compute-node still works if ever
    pointed somewhere else)."""
    cmd = [
        "sbatch", "--parsable",
        "--partition", _partition_for(node), "--nodelist", node, "--time", time_limit,
        _TRAIN_LAUNCHER,
        config_path,
        "--sample-idx", *[str(s) for s in sample_idx],
        "--layers", *layers,
        "--n-models-per-chunk", str(n_models_per_chunk),
        "--epochs", str(epochs),
        "--checkpoint-every", str(checkpoint_every),
        # sweep data was relayed onto `node`'s scratch via scripts/helper/sync_node_scratch.sh
        # (NFS staging + node_relay), not pushed to Hetzner -- see this batch's relay-in step.
        "--backend", "nfs",
    ]
    if lr is not None:
        cmd += ["--lr", str(lr)]
    return _sbatch(cmd)


if __name__ == "__main__":
    main()
