#!/bin/bash
#SBATCH --job-name=ae_sampled_sweep
#SBATCH --partition=ampere
#SBATCH --nodelist=mauao
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ae_sampled_sweep_%j.out

set -euo pipefail

mkdir -p logs

cd /nfs-share/pa511/code_bases/new_jac

CONFIG="${1:-configs/gpt2.toml}"
NAME="$(basename "${CONFIG}" .toml)"

export NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac"
AE_CHECKPOINTS_DIR="${2:-${NEW_JAC_SCRATCH}/onnx_models/${NAME}/ae_checkpoints}"

echo "Job ${SLURM_JOB_ID} running on $(hostname), config ${CONFIG}, ae dir ${AE_CHECKPOINTS_DIR}, scratch ${NEW_JAC_SCRATCH}"

exec uv run python -m scripts.python.run_autoencoder_sampled_sweep \
    "${CONFIG}" \
    --ae-checkpoints-dir "${AE_CHECKPOINTS_DIR}" \
    --out-dir "${NEW_JAC_SCRATCH}/onnx_models/${NAME}/ae_sampled_sweep" \
    --n 1000
