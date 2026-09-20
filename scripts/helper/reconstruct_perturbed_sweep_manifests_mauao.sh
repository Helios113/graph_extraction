#!/bin/bash
#SBATCH --job-name=reconstruct_perturbed_sweep_manifests
#SBATCH --partition=ampere
#SBATCH --nodelist=mauao
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/reconstruct_perturbed_sweep_manifests_%j.out

set -euo pipefail

mkdir -p logs

cd /nfs-share/pa511/code_bases/new_jac

CONFIG="${1:-configs/tiny_llama.toml}"
NAME="$(basename "${CONFIG}" .toml)"

export NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac"

echo "Job ${SLURM_JOB_ID} running on $(hostname), model ${NAME}, scratch ${NEW_JAC_SCRATCH}"

exec uv run python -m scripts.helper.reconstruct_perturbed_sweep_manifests \
    --activations-dir "onnx_models/${NAME}/perturbed_sweep"
