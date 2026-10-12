#!/bin/bash
#SBATCH --job-name=bench_batched_gpd_scale
#SBATCH --partition=ampere
#SBATCH --nodelist=mauao
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/bench_batched_gpd_scale_%j.out

# Benchmarks scripts/analysis/batched_gpd_mle.py's fit_gpd_batched at
# increasing batch sizes (more (sample, pos) positions' bootstrap-EQD
# candidate grids concatenated into one GPU call) using REAL co-Lipschitz
# ratio data read from generate_perturbed_sweep.py's shards on mauao local
# scratch -- answers whether batching more positions together in one call
# reduces time-per-position and where GPU memory becomes the limit. See
# scripts/analysis/bench_batched_gpd_scale.py's module docstring.
#
# Usage:
#   sbatch scripts/batch_scripts/bench_batched_gpd_scale_mauao_sbatch.sh \
#       [sublayer_input] [sublayer_output] [extra bench_batched_gpd_scale.py args...]
#
# Example (default add_4->add_6 on simple_stories):
#   sbatch scripts/batch_scripts/bench_batched_gpd_scale_mauao_sbatch.sh
#
# Example, custom sublayer pair and batch sizes:
#   sbatch scripts/batch_scripts/bench_batched_gpd_scale_mauao_sbatch.sh \
#       add_10 add_13 --n-positions 1 8 32 128

set -euo pipefail

mkdir -p logs

cd /nfs-share/pa511/code_bases/new_jac

SUBLAYER_INPUT="${1:-add_4}"
SUBLAYER_OUTPUT="${2:-add_6}"
if [ "$#" -ge 2 ]; then shift 2; fi

export NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac"

echo "Job ${SLURM_JOB_ID} running on $(hostname), ${SUBLAYER_INPUT}->${SUBLAYER_OUTPUT}, scratch ${NEW_JAC_SCRATCH}"

exec uv run python -m scripts.analysis.bench_batched_gpd_scale \
    --perturbed-sweep-dir onnx_models/simple_stories/perturbed_sweep \
    --sublayer-input "${SUBLAYER_INPUT}" \
    --sublayer-output "${SUBLAYER_OUTPUT}" \
    "$@"
