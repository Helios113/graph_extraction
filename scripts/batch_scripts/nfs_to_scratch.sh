#!/bin/bash
# Rsyncs a subpath of the NFS staging area down into this node's local
# scratch ($NEW_JAC_SCRATCH). This is the "down" hop of the two-hop relay --
# see scripts/batch_scripts/scratch_to_nfs.sh for the "up" hop and
# scripts/helper/sync_node_scratch.sh for the orchestrator that chains both
# as dependent sbatch jobs to move data directly between two nodes' scratch
# disks (which can't see each other -- only NFS is shared by every node).
#
# Submit via sbatch, pinned to the destination node, e.g.:
#   sbatch --partition=interactive --nodelist=mauao \
#     scripts/batch_scripts/nfs_to_scratch.sh onnx_models/gpt2/ambient_sweep
#
# Positional arg: SUBPATH, relative to the NFS staging dir (and mirrored at
# the same relative path under $NEW_JAC_SCRATCH). Required, same reasoning
# as scratch_to_nfs.sh.

#SBATCH --job-name=nfs_to_scratch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/nfs_to_scratch_%j.out

set -euo pipefail

cd /nfs-share/pa511/code_bases/new_jac
mkdir -p logs

SUBPATH="${1:?Usage: sbatch nfs_to_scratch.sh <subpath under /nfs-share/datasets/\$USER/>}"

export NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac"
STAGING_ROOT="/nfs-share/datasets/${USER}"

SRC="${STAGING_ROOT}/${SUBPATH}"
DEST="${NEW_JAC_SCRATCH}/${SUBPATH}"

if [ ! -e "${SRC}" ]; then
    echo "error: ${SRC} does not exist on NFS" >&2
    exit 1
fi

echo "Job ${SLURM_JOB_ID} on $(hostname): ${SRC} -> ${DEST}"
mkdir -p "$(dirname "${DEST}")"

if [ -d "${SRC}" ]; then
    mkdir -p "${DEST}"
    rsync -a --info=progress2 "${SRC}/" "${DEST}/"
else
    rsync -a --info=progress2 "${SRC}" "${DEST}"
fi

# NFS staging is a transient relay point, not storage (see node_relay.py's
# STAGING_ROOT, a fixed-size per-user ZFS quota) -- only reached once rsync
# above has exited 0 (set -e would have already killed this script
# otherwise), so DEST is a confirmed full copy before SRC is removed.
echo "cleaning up NFS staging copy at ${SRC}"
rm -rf "${SRC}"

echo "done: $(hostname) scratch now has ${DEST}"
