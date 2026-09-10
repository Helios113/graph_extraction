#!/bin/bash
#SBATCH --job-name=find_max_batch_size
#SBATCH --partition=hopper
#SBATCH --nodelist=ruapehu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/find_max_batch_size_%j.out

set -euo pipefail

mkdir -p logs

cd /nfs-share/pa511/code_bases/new_jac

CONFIG="${1:-configs/smoll_jac.toml}"

echo "Job ${SLURM_JOB_ID} running on $(hostname), config ${CONFIG}"

exec uv run python -m scripts.find_max_batch_size "${CONFIG}"
