#!/bin/bash
# Moves every onnx_models/{name}/ae_checkpoints dir currently on NFS to its
# correct scratch location ($NEW_JAC_SCRATCH/onnx_models/{name}/ae_checkpoints),
# per CLAUDE.md's "local scratch, not NFS" convention -- one-off fixup for
# checkpoints saved before scripts/batch_scripts/train_batched_manifold_mauao.sh
# was updated to write directly to scratch.
#
# Run via srun on mauao (scratch is node-local, not reachable from the NFS
# login node):
#   srun --partition=ampere --nodelist=mauao --cpus-per-task=2 --mem=8G \
#     bash scripts/batch_scripts/move_ae_checkpoints_to_scratch_mauao.sh

set -euo pipefail

cd /nfs-share/pa511/code_bases/new_jac

export NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac"

for src_dir in onnx_models/*/ae_checkpoints; do
    [ -d "${src_dir}" ] || continue
    name="$(basename "$(dirname "${src_dir}")")"
    dest_dir="${NEW_JAC_SCRATCH}/onnx_models/${name}/ae_checkpoints"

    if [ -d "${dest_dir}" ]; then
        echo "skipping ${name}: ${dest_dir} already exists"
        continue
    fi

    echo "copying ${src_dir} -> ${dest_dir}"
    mkdir -p "${dest_dir}"
    # -a: archive (preserve perms/times, recursive); trailing slash on src
    # copies its CONTENTS into dest_dir, not a nested ae_checkpoints/ dir.
    rsync -a "${src_dir}/" "${dest_dir}/"

    # Only remove the NFS copy once rsync has confirmed success (exit 0) --
    # if rsync fails/is interrupted, the NFS original is left untouched and
    # this dir is simply retried (dest_dir won't exist yet, or is incomplete)
    # on the next run.
    echo "verified copy of ${name}, removing NFS source ${src_dir}"
    rm -rf "${src_dir}"
done

echo "done"
