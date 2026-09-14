#!/bin/bash
#SBATCH --job-name=gen_perturbed_sweep
#SBATCH --partition=hopper
#SBATCH --nodelist=ruapehu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gen_perturbed_sweep_%j.out

set -euo pipefail

mkdir -p logs

cd /nfs-share/pa511/code_bases/new_jac

CONFIG="${1:-configs/smoll_jac.toml}"

echo "Job ${SLURM_JOB_ID} running on $(hostname), config ${CONFIG}"

exec uv run python -m scripts.generate_perturbed_sweep_last configs/simple_stories.toml --base-seqs-h5 data/simple_stories.h5 --out onnx_models/simple_stories/perturbed_sweep_small.h5
