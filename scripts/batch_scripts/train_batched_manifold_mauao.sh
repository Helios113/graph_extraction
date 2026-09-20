#!/bin/bash
#SBATCH --job-name=train_batched_manifold
#SBATCH --partition=ampere
#SBATCH --nodelist=mauao
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_batched_manifold_%j.out

set -euo pipefail

mkdir -p logs

cd /nfs-share/pa511/code_bases/new_jac

CONFIG="${1:-configs/gpt2.toml}"
NAME="$(basename "${CONFIG}" .toml)"
SAMPLE_IDX="${2:-0}"

export NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac"
# Must match autoencoder.data.ae_checkpoints_dir_for(cfg, sample_idx)
OUT_DIR="${NEW_JAC_SCRATCH}/onnx_models/${NAME}/ae_checkpoints/sample${SAMPLE_IDX}"

echo "Job ${SLURM_JOB_ID} running on $(hostname), config ${CONFIG}, sample-idx ${SAMPLE_IDX}, scratch ${NEW_JAC_SCRATCH}, out-dir ${OUT_DIR}"

exec uv run python -m scripts.python.train_batched_manifold \
    "${CONFIG}" \
    --sample-idx "${SAMPLE_IDX}" \
    --out-dir "${OUT_DIR}"
