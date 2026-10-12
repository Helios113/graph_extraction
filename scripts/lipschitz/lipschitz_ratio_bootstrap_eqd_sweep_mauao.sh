#!/bin/bash
#SBATCH --job-name=lipschitz_ratio_bootstrap_eqd_sweep_gpu
#SBATCH --partition=ampere
#SBATCH --nodelist=mauao
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lipschitz_ratio_bootstrap_eqd_sweep_gpu_%j.out

set -euo pipefail

mkdir -p logs

cd /nfs-share/pa511/code_bases/new_jac

CONFIG="${1:-configs/simple_stories.toml}"
shift || true
NAME="$(basename "${CONFIG}" .toml)"

export NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac"

echo "Job ${SLURM_JOB_ID} running on $(hostname), config ${CONFIG}, scratch ${NEW_JAC_SCRATCH}"

exec uv run python -m scripts.lipschitz.lipschitz_ratio_bootstrap_eqd_sweep_gpu \
    "${CONFIG}" \
    --perturbed-sweep-dir "onnx_models/${NAME}/perturbed_sweep" \
    --out-dir "onnx_models/${NAME}/lipschitz_ratio_bootstrap_eqd_sweep_gpu" \
    "$@"
