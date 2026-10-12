#!/bin/bash
#SBATCH --job-name=ae_sampled_sweep_gpt2
#SBATCH --partition=ampere
#SBATCH --nodelist=mauao
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ae_sampled_sweep_gpt2_%j.out
#
# Background/batch counterpart to the gpt2 ae_sampled_sweep srun command in
# SCRIPTS.md -- submit with sbatch and walk away.
#
# Usage:
#   sbatch scripts/batch_scripts/run_sampled_sweep_gpt2_mauao_sbatch.sh

set -euo pipefail

mkdir -p logs

cd /nfs-share/pa511/code_bases/new_jac

export NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac"

echo "Job ${SLURM_JOB_ID} running on $(hostname), scratch ${NEW_JAC_SCRATCH}"

exec uv run python -m scripts.python.run_autoencoder_sampled_sweep configs/gpt2.toml \
    --sample-idx $(seq 0 63 | tr '\n' ' ') \
    --out-dir onnx_models/gpt2/ae_sampled_sweep \
    --n 1000 \
    --alpha 0.3 1.5 3.0 \
    --tau 1.0 \
    --noise 0.0 0.5 1.0
