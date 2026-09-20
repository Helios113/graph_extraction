#!/bin/bash
#SBATCH --job-name=ae_sample_diagnostics
#SBATCH --partition=ampere
#SBATCH --nodelist=mauao
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ae_sample_diagnostics_%j.out

set -euo pipefail

mkdir -p logs

cd /nfs-share/pa511/code_bases/new_jac

CONFIG="${1:-configs/gpt2.toml}"
SUBLAYER="${2:?usage: sbatch plot_autoencoder_sample_diagnostics_mauao.sh CONFIG SUBLAYER POS [SAMPLE_IDX] [AE_CHECKPOINTS_DIR]}"
POS="${3:?usage: sbatch plot_autoencoder_sample_diagnostics_mauao.sh CONFIG SUBLAYER POS [SAMPLE_IDX] [AE_CHECKPOINTS_DIR]}"
SAMPLE_IDX="${4:-0}"  # must match the --sample-idx train_batched_manifold_mauao.sh was run with
NAME="$(basename "${CONFIG}" .toml)"

# Everything this script READS lives on node-local scratch, not NFS
# (CLAUDE.md's "local scratch, not NFS" convention): the trained autoencoder
# checkpoints (autoencoder.data.ae_checkpoints_dir_for(cfg, sample_idx)'s
# default, already written there by train_batched_manifold_mauao.sh) and the
# sweep data / base_seqs.h5 / ONNX subgraphs load_sweep_data and
# construct.construct pull from $NEW_JAC_SCRATCH first (see
# execute/sharded_writer.py's _scratch_root and
# execute/remote_storage.py's ensure_local). The output PNG is a one-shot
# diagnostic figure meant to be looked at, not repeated/bulk output, so it's
# written straight to NFS instead -- no separate copy-off-scratch step needed
# to view it.
export NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac"
AE_CHECKPOINTS_DIR="${5:-}"  # optional override; default: ae_checkpoints_dir_for(cfg, sample_idx)
OUT_DIR="onnx_models/${NAME}/ae_sample_diagnostics"
mkdir -p "${OUT_DIR}"
OUT="${OUT_DIR}/${SUBLAYER}.pos${POS}.sample${SAMPLE_IDX}.png"

echo "Job ${SLURM_JOB_ID} running on $(hostname), config ${CONFIG}, sublayer ${SUBLAYER}, pos ${POS}, sample-idx ${SAMPLE_IDX}, scratch ${NEW_JAC_SCRATCH}"

exec uv run python -m scripts.python.plot_autoencoder_sample_diagnostics \
    "${CONFIG}" \
    ${AE_CHECKPOINTS_DIR:+--ae-checkpoints-dir "${AE_CHECKPOINTS_DIR}"} \
    --sublayer "${SUBLAYER}" \
    --pos "${POS}" \
    --sample-idx "${SAMPLE_IDX}" \
    --out "${OUT}"
