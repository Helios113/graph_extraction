#!/bin/bash
# ruapehu/hopper counterpart of generate_tangent_bases_ruapehu.sh, for
# scripts/python/run_layer_batched_tangent_bases.py's per-layer-batch driver
# (GPU compute on ruapehu, full sweep storage on mauao -- see that script's
# docstring). Unlike generate_tangent_bases_ruapehu.sh (which hardcodes
# --knn-k/--basis-method/--backend for a single full-sweep run), every
# generate_tangent_bases.py arg is passed straight through as this script's
# own args, since the driver already knows exactly which config/--layers/
# --sample-idx/etc a given batch needs and there's no single fixed set to
# bake in here.
#
# Usage (typically invoked by run_layer_batched_tangent_bases.py, not by hand):
#   sbatch --time=12:00:00 scripts/batch_scripts/generate_tangent_bases_ruapehu_driven.sh \
#     configs/gpt2.toml --sample-idx 0 --layers add_10 add_11 --knn-k 160 \
#     --basis-method dual --out-dir onnx_models/gpt2/tangent_bases
#
# --gres=gpu:1 (no type pinned) rather than a specific GPU model, since the
# driver overrides --partition/--nodelist on the sbatch command line to
# point this at either ruapehu (h100) or mauao (a40) via --compute-node --
# same reasoning as scripts/batch_scripts/tiny_llama/train_batched_manifold_ruapehu.sh.

#SBATCH --job-name=tangent_bases_ruapehu
#SBATCH --partition=hopper
#SBATCH --nodelist=ruapehu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/tangent_bases_ruapehu_%j.out

set -euo pipefail

mkdir -p logs

cd /nfs-share/pa511/code_bases/new_jac

export NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac"

echo "Job ${SLURM_JOB_ID} running on $(hostname), scratch ${NEW_JAC_SCRATCH}, args: $*"

exec uv run python -m scripts.python.generate_tangent_bases "$@"
