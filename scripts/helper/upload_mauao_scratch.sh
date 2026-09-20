#!/bin/bash
#SBATCH --job-name=upload_scratch
#SBATCH --partition=ampere
#SBATCH --nodelist=mauao
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/upload_mauao_scratch_%j.out

set -euo pipefail

mkdir -p logs

cd /nfs-share/pa511/code_bases/new_jac

export NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac"

echo "Job ${SLURM_JOB_ID} running on $(hostname), pushing ${NEW_JAC_SCRATCH} (local files kept)"

exec uv run python upload.py push "${NEW_JAC_SCRATCH}"
