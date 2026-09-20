#!/bin/bash
#SBATCH --job-name=compute_base_jac_stats
#SBATCH --partition=hopper
#SBATCH --nodelist=ruapehu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/compute_base_jac_stats_%j.out

set -euo pipefail

mkdir -p logs

cd /nfs-share/pa511/code_bases/new_jac

CONFIG="${1:-configs/smoll_jac.toml}"

echo "Job ${SLURM_JOB_ID} running on $(hostname), config ${CONFIG}"

exec uv run python -m scripts.python.compute_base_jacobian_stats "${CONFIG}"
