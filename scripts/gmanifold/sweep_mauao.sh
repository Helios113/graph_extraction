#!/bin/bash
#SBATCH --job-name=gmanifold_sweep
#SBATCH --partition=ampere
#SBATCH --nodelist=mauao
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gmanifold_sweep_%j.out

set -euo pipefail

mkdir -p logs

cd /nfs-share/pa511/code_bases/new_jac

CONFIG="${1:-configs/simple_stories.toml}"
NAME="$(basename "${CONFIG}" .toml)"

export NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac"

echo "Job ${SLURM_JOB_ID} running on $(hostname), config ${CONFIG}, scratch ${NEW_JAC_SCRATCH}"

exec uv run python -m scripts.gmanifold.sweep "${CONFIG}" \
    --out-dir "onnx_models/${NAME}/gmanifold_sweep" \
    --locs embedding add_6 add_8 \
    --propagate add_6 add_8
