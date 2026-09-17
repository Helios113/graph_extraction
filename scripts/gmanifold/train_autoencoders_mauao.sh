#!/bin/bash
#SBATCH --job-name=gmanifold_train_autoencoders
#SBATCH --partition=ampere
#SBATCH --nodelist=mauao
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gmanifold_train_autoencoders_%j.out

set -euo pipefail

mkdir -p logs

cd /nfs-share/pa511/code_bases/new_jac

NAME="${1:-simple_stories}"

export NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac"

echo "Job ${SLURM_JOB_ID} running on $(hostname), model ${NAME}, scratch ${NEW_JAC_SCRATCH}"

exec uv run python -m scripts.gmanifold.train_autoencoders \
    --activations-dir "onnx_models/${NAME}/perturbed_sweep" \
    --out-dir "onnx_models/${NAME}/gmanifold_autoencoders" \
    --latent-dim 16
