#!/bin/bash
#SBATCH --job-name=gen_base_seqs
#SBATCH --partition=ampere
#SBATCH --nodelist=mauao
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/gen_base_seqs_%j.out

set -euo pipefail

mkdir -p logs

cd /nfs-share/pa511/code_bases/new_jac

export NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac"

echo "Job ${SLURM_JOB_ID} running on $(hostname), scratch ${NEW_JAC_SCRATCH}"

exec uv run python -m scripts.generate_base_sequences configs/tiny_llama.toml --out-key data/simple_stories
