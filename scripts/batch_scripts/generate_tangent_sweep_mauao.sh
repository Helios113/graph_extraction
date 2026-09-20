#!/bin/bash
#SBATCH --job-name=tangent_sweep
#SBATCH --partition=ampere
#SBATCH --nodelist=mauao
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/tangent_sweep_%j.out

set -euo pipefail

mkdir -p logs

cd /nfs-share/pa511/code_bases/new_jac

CONFIG="${1:-configs/gpt2.toml}"
NAME="$(basename "${CONFIG}" .toml)"

export NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac"
TANGENT_BASES_DIR="${2:-${NEW_JAC_SCRATCH}/onnx_models/${NAME}/tangent_bases}"

echo "Job ${SLURM_JOB_ID} running on $(hostname), config ${CONFIG}, tangent bases ${TANGENT_BASES_DIR}, scratch ${NEW_JAC_SCRATCH}"

exec uv run python -m scripts.python.generate_tangent_sweep \
    "${CONFIG}" \
    --tangent-bases-dir "${TANGENT_BASES_DIR}" \
    --out-dir "onnx_models/${NAME}/tangent_sweep" \
    --n 1000
