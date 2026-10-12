#!/bin/bash
# ruapehu/hopper counterpart of generate_tangent_bases_ruapehu_driven.sh, for
# scripts.python.run_layer_batched_tangent_sweep_direct's per-layer-batch
# driver -- runs scripts.python.generate_tangent_sweep_direct (the fused
# basis+sampling script that never materializes a dense per-vocab-point
# basis, see that script's docstring for why generate_tangent_bases.py's
# dense-basis approach OOMs at gpt2 scale). Every
# generate_tangent_sweep_direct.py arg is passed straight through as this
# script's own args, since the driver already knows exactly which
# config/--layers/--sample-idx/--radius/etc a given batch needs.
#
# --gres=gpu:1 (no type pinned) rather than a specific GPU model, since the
# driver overrides --partition/--nodelist on the sbatch command line to
# point this at either ruapehu (h100) or mauao (a40) via --compute-node --
# same reasoning as scripts/batch_scripts/tiny_llama/train_batched_manifold_ruapehu.sh.
#
# Usage (typically invoked by run_layer_batched_tangent_sweep_direct.py, not
# by hand):
#   sbatch --time=12:00:00 scripts/batch_scripts/generate_tangent_sweep_direct_ruapehu_driven.sh \
#     configs/gpt2.toml --sample-idx 0 --layers add_10 add_11 --knn-k 160 \
#     --basis-method dual --n 1000 --radius 0.1 \
#     --out-dir onnx_models/gpt2/tangent_sweep_direct

#SBATCH --job-name=tangent_sweep_direct_ruapehu
#SBATCH --partition=hopper
#SBATCH --nodelist=ruapehu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/tangent_sweep_direct_ruapehu_%j.out

set -euo pipefail

mkdir -p logs

cd /nfs-share/pa511/code_bases/new_jac

export NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac"

echo "Job ${SLURM_JOB_ID} running on $(hostname), scratch ${NEW_JAC_SCRATCH}, args: $*"

exec uv run python -m scripts.python.generate_tangent_sweep_direct "$@"
