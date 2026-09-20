#!/bin/bash
#SBATCH --job-name=gen_base_seqs
#SBATCH --partition=ampere
#SBATCH --nodelist=mauao
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/gen_base_seqs_%j.out

set -euo pipefail

mkdir -p logs

cd /nfs-share/pa511/code_bases/new_jac

CONFIG="${1:-configs/tiny_llama.toml}"
OVERWRITE=0
if [[ "${2:-}" == "--overwrite" ]]; then
    OVERWRITE=1
fi
NAME="$(basename "${CONFIG}" .toml)"
OUT_KEY="data/${NAME}"

export NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac"

EXTRA_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
    echo "Overwrite requested: clearing progress/manifest for ${OUT_KEY} and any local scratch shards"
    rm -f "${OUT_KEY}.progress.json" "${OUT_KEY}.manifest.json"
    rm -rf "${NEW_JAC_SCRATCH}/${OUT_KEY}"
    # Regenerated content will conflict with whatever this out-key's shards
    # already hold on the remote store (same shard names, different bytes)
    # -- overwrite-remote makes that push win instead of the non-interactive
    # push deferring it via upload.py apply-overrides.
    EXTRA_ARGS+=(--overwrite-remote)
fi

echo "Job ${SLURM_JOB_ID} running on $(hostname), config ${CONFIG}, scratch ${NEW_JAC_SCRATCH}"

exec uv run python -m scripts.python.generate_base_sequences "${CONFIG}" --out-key "${OUT_KEY}" "${EXTRA_ARGS[@]}"
