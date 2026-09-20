#!/bin/bash
# ruapehu/hopper counterpart of train_batched_manifold.sh, for
# scripts/python/run_layer_batched_training.py's per-layer-batch training
# step (GPU compute on ruapehu, full sweep storage on mauao -- see that
# script's docstring). Unlike train_batched_manifold.sh (which hardcodes
# CONFIG/SAMPLE_INDICES for a single full-dataset run), every
# train_batched_manifold.py arg is passed straight through as this script's
# own args, since the driver already knows exactly which config/--layers/
# --sample-idx/etc a given batch needs and there's no single fixed set of
# samples/layers to bake in here.
#
# Usage (typically invoked by run_layer_batched_training.py, not by hand):
#   sbatch --time=12:00:00 scripts/batch_scripts/tiny_llama/train_batched_manifold_ruapehu.sh \
#     configs/tiny_llama.toml --sample-idx 0 --layers embedding add_4 --n-models-per-chunk 200 \
#     --epochs 200 --checkpoint-every 50

#SBATCH --job-name=train_batched_manifold_ruapehu
#SBATCH --partition=hopper
#SBATCH --nodelist=ruapehu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_batched_manifold_ruapehu_%j.out

set -euo pipefail

mkdir -p logs

cd /nfs-share/pa511/code_bases/new_jac

export NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac"

echo "Job ${SLURM_JOB_ID} running on $(hostname), scratch ${NEW_JAC_SCRATCH}, args: $*"

exec uv run python -m scripts.python.train_batched_manifold "$@"
