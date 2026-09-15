#!/bin/bash
#SBATCH --job-name=gen_perturbed_sweep
#SBATCH --partition=ampere
#SBATCH --nodelist=mauao
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gen_perturbed_sweep_%j.out

set -euo pipefail

mkdir -p logs

cd /nfs-share/pa511/code_bases/new_jac

CONFIG="${1:-configs/tiny_llama.toml}"
NAME="$(basename "${CONFIG}" .toml)"

export NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac"

echo "Job ${SLURM_JOB_ID} running on $(hostname), config ${CONFIG}, scratch ${NEW_JAC_SCRATCH}"

exec uv run python -m scripts.generate_perturbed_sweep "${CONFIG}" --base-seqs-h5 "data/${NAME}" --out-dir "onnx_models/${NAME}/perturbed_sweep" --n-vocab-sample 20000
