#!/bin/bash
# Runs scripts/analysis/lipschitz_ratio_eqd_fit.py on mauao via srun (not
# sbatch) -- this is a CPU-only, foreground analysis job meant to read
# generate_perturbed_sweep.py's shards straight off mauao's local scratch
# (see that script's module docstring: no NFS/Hetzner pull, must run on the
# same node the sweep ran on). Output (small per-sample h5 files) is written
# straight to NFS under onnx_models/${NAME}/lipschitz_ratio_eqd_fit.
#
# Usage:
#   scripts/batch_scripts/lipschitz_ratio_eqd_fit_mauao.sh [config] [extra lipschitz_ratio_eqd_fit.py args...]
#
# Examples:
#   scripts/batch_scripts/lipschitz_ratio_eqd_fit_mauao.sh configs/simple_stories.toml
#   scripts/batch_scripts/lipschitz_ratio_eqd_fit_mauao.sh configs/tiny_llama.toml --n-samples 8 --n-pairs 5000

set -euo pipefail

cd /nfs-share/pa511/code_bases/new_jac

CONFIG="${1:-configs/tiny_llama.toml}"
shift || true
NAME="$(basename "${CONFIG}" .toml)"

exec srun --partition=interactive --nodelist=mauao --time=01:00:00 \
    --cpus-per-task=8 --mem=32G \
    --export=ALL,NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac" \
    uv run python -m scripts.analysis.lipschitz_ratio_eqd_fit \
        "${CONFIG}" \
        --perturbed-sweep-dir "onnx_models/${NAME}/perturbed_sweep" \
        --out-dir "onnx_models/${NAME}/lipschitz_ratio_eqd_fit" \
        "$@"
