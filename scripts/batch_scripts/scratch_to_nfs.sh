#!/bin/bash
# Rsyncs a subpath of this node's local scratch ($NEW_JAC_SCRATCH) up to the
# NFS staging area, so it becomes visible to every other node in the
# cluster. Node-local scratch (/local/scratch) is NOT shared between nodes
# -- NFS is the only filesystem all of taranaki/mauao/ruapehu can see, so
# moving data between two compute nodes' scratch disks is a two-hop relay
# through NFS: this script does the "up" hop, scripts/batch_scripts/nfs_to_scratch.sh
# does the "down" hop on the receiving node. See
# scripts/helper/sync_node_scratch.sh, which chains the two as dependent
# sbatch jobs for a one-command node-to-node move.
#
# Submit via sbatch, pinned to the source node, e.g.:
#   sbatch --partition=interactive --nodelist=ruapehu \
#     scripts/batch_scripts/scratch_to_nfs.sh onnx_models/gpt2/ambient_sweep
#
# Positional arg: SUBPATH, relative to $NEW_JAC_SCRATCH (and mirrored at the
# same relative path under the NFS staging dir). Required -- syncing all of
# scratch in one job defeats the point of moving only what's needed.

#SBATCH --job-name=scratch_to_nfs
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/scratch_to_nfs_%j.out

set -euo pipefail

cd /nfs-share/pa511/code_bases/new_jac
mkdir -p logs

SUBPATH="${1:?Usage: sbatch scratch_to_nfs.sh <subpath under \$NEW_JAC_SCRATCH>}"

export NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac"
STAGING_ROOT="/nfs-share/datasets/${USER}"

SRC="${NEW_JAC_SCRATCH}/${SUBPATH}"
DEST="${STAGING_ROOT}/${SUBPATH}"

if [ ! -e "${SRC}" ]; then
    echo "error: ${SRC} does not exist on $(hostname)" >&2
    exit 1
fi

echo "Job ${SLURM_JOB_ID} on $(hostname): ${SRC} -> ${DEST}"
mkdir -p "$(dirname "${DEST}")"

# -a: archive (recursive, preserves perms/times). Trailing slashes on both
# sides when SRC is a directory so DEST ends up with the same relative
# layout under scratch_staging/ rather than a nested SUBPATH/SUBPATH/.
if [ -d "${SRC}" ]; then
    mkdir -p "${DEST}"
    rsync -a --info=progress2 "${SRC}/" "${DEST}/"
else
    rsync -a --info=progress2 "${SRC}" "${DEST}"
fi

echo "done: staged at ${DEST}"
