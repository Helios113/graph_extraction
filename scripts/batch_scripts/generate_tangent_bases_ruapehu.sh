#!/bin/bash
# ruapehu/hopper counterpart of generate_tangent_bases_mauao.sh. GPU compute
# runs here, but the perturbed-sweep input this script reads
# (onnx_models/${NAME}/perturbed_sweep) is generated on mauao -- run
# scripts/helper/sync_node_scratch.sh from taranaki FIRST to relay it onto
# ruapehu's local scratch (same two-hop mauao-scratch -> NFS staging ->
# ruapehu-scratch pattern train_batched_manifold's pipeline uses, see
# README.md's tiny_llama launch commands), e.g.:
#   scripts/helper/sync_node_scratch.sh --from mauao --to ruapehu \
#     onnx_models/gpt2/perturbed_sweep
# Without that, generate_tangent_bases' cloud loader will raise
# FileNotFoundError the first time it needs a shard neither on ruapehu's
# scratch nor already pushed to NFS staging.
#
# Usage: sync_node_scratch.sh (above), then:
#   sbatch scripts/batch_scripts/generate_tangent_bases_ruapehu.sh configs/gpt2.toml
#SBATCH --job-name=tangent_bases
#SBATCH --partition=hopper
#SBATCH --nodelist=ruapehu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/tangent_bases_%j.out

set -euo pipefail

mkdir -p logs

cd /nfs-share/pa511/code_bases/new_jac

CONFIG="${1:-configs/gpt2.toml}"
NAME="$(basename "${CONFIG}" .toml)"

export NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac"

echo "Job ${SLURM_JOB_ID} running on $(hostname), config ${CONFIG}, scratch ${NEW_JAC_SCRATCH}"

exec uv run python -m scripts.python.generate_tangent_bases \
    "${CONFIG}" \
    --out-dir "onnx_models/${NAME}/tangent_bases" \
    --knn-k 160 \
    --basis-method dual \
    --backend nfs
