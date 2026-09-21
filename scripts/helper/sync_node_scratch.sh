#!/bin/bash
# One-command relay to move a subpath of node-local scratch from one SLURM
# node to another. Node-local scratch (/local/scratch) is not shared between
# nodes -- NFS (/nfs-share) is the only filesystem every node mounts -- so
# this submits two dependent sbatch jobs:
#   1. scratch_to_nfs.sh on --from, staging SUBPATH under
#      /nfs-share/datasets/$USER/
#   2. nfs_to_scratch.sh on --to, gated on job 1 succeeding
#      (--dependency=afterok), pulling it into --to's local scratch
#
# No daemon, proxy, or direct node-to-node network path is involved or
# needed -- both hops are queued SLURM batch jobs, and the relay point is
# the shared filesystem. This machine (taranaki) only needs `sbatch`, which
# it already has.
#
# Usage:
#   scripts/helper/sync_node_scratch.sh --from ruapehu --to mauao \
#     onnx_models/gpt2/ambient_sweep
#
# Defaults each job to that node's dedicated partition (hopper for ruapehu,
# ampere for mauao) rather than one shared partition name -- both have
# OverSubscribe=NO (no artificial per-node job-count cap, unlike cpu-only's
# OverSubscribe=FORCE:2 -- see notes in scripts/helper/bench_scratch_sync.sh)
# and a 12hr cap. The priority partition covers both nodes under one name
# and has no cap either, but its QOS only allows cls_admins, not this
# account's cls_phd (confirmed by an actual rejected submission, not just
# `sbatch --test-only`, which doesn't validate partition QOS). interactive
# also covers both nodes and works for cls_phd, but caps at 1hr. Pass
# --partition to override any of this for a specific run.
#
# Prints both job IDs and exits immediately (does not block on completion --
# use `squeue -u $USER` or check logs/scratch_to_nfs_<job1>.out /
# logs/nfs_to_scratch_<job2>.out).

set -euo pipefail

usage() {
    echo "Usage: $0 --from <node> --to <node> [--partition <partition>] [--time <HH:MM:SS>] <subpath under \$NEW_JAC_SCRATCH>" >&2
    exit 1
}

FROM=""
TO=""
PARTITION=""
TIME_LIMIT=""
SUBPATH=""

while [ $# -gt 0 ]; do
    case "$1" in
        --from) FROM="$2"; shift 2 ;;
        --to) TO="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --time) TIME_LIMIT="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) SUBPATH="$1"; shift ;;
    esac
done

[ -n "${FROM}" ] && [ -n "${TO}" ] && [ -n "${SUBPATH}" ] || usage

if [ "${FROM}" = "${TO}" ]; then
    echo "error: --from and --to are both ${FROM} -- nothing to relay" >&2
    exit 1
fi

# Each node's own dedicated partition, used when --partition isn't given.
_default_partition() {
    case "$1" in
        ruapehu) echo "hopper" ;;
        mauao) echo "ampere" ;;
        *) echo "interactive" ;;  # unknown node: fall back to the partition every node shares
    esac
}

cd /nfs-share/pa511/code_bases/new_jac
mkdir -p logs

TIME_ARGS=()
[ -n "${TIME_LIMIT}" ] && TIME_ARGS=(--time="${TIME_LIMIT}")

FROM_PARTITION="${PARTITION:-$(_default_partition "${FROM}")}"
TO_PARTITION="${PARTITION:-$(_default_partition "${TO}")}"

echo "Staging ${FROM}:\$NEW_JAC_SCRATCH/${SUBPATH} -> NFS (partition ${FROM_PARTITION}) ..."
JOB1=$(sbatch --parsable --partition="${FROM_PARTITION}" --nodelist="${FROM}" "${TIME_ARGS[@]}" \
    scripts/batch_scripts/scratch_to_nfs.sh "${SUBPATH}")
echo "  submitted scratch_to_nfs on ${FROM}: job ${JOB1}"

echo "Queuing NFS -> ${TO}:\$NEW_JAC_SCRATCH/${SUBPATH} (partition ${TO_PARTITION}, after job ${JOB1} succeeds) ..."
JOB2=$(sbatch --parsable --partition="${TO_PARTITION}" --nodelist="${TO}" "${TIME_ARGS[@]}" \
    --dependency=afterok:"${JOB1}" \
    scripts/batch_scripts/nfs_to_scratch.sh "${SUBPATH}")
echo "  submitted nfs_to_scratch on ${TO}: job ${JOB2}"

echo
echo "Track with: squeue -u ${USER}"
echo "Logs: logs/scratch_to_nfs_${JOB1}.out  logs/nfs_to_scratch_${JOB2}.out"
