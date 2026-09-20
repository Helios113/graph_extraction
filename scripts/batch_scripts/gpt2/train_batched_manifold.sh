#!/bin/bash
#SBATCH --job-name=train_batched_manifold_gpt2
#SBATCH --partition=ampere
#SBATCH --nodelist=mauao
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_batched_manifold_gpt2_%j.out

set -euo pipefail

mkdir -p logs

cd /nfs-share/pa511/code_bases/new_jac

CONFIG="configs/gpt2.toml"
NAME="gpt2"
export NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac"

# Edit SAMPLE_INDICES to run a subset of samples; SAMPLES_PER_CHUNK controls
# how many samples are batched together per GPU pass -- raise it on hardware
# with more headroom, 1 (sequential) is always safe. See CLAUDE.md's
# model_chunk chunking convention -- sample is the same kind of knob.
N_SAMPLES="$(uv run python -c 'from autoencoder.data import n_samples_for, sweep_dir_for; from config import load_config; from execute.sharded_writer import discover_layer_names; cfg=load_config("'"${CONFIG}"'"); d=sweep_dir_for(cfg); print(n_samples_for(d, discover_layer_names(d)[0]))')"
SAMPLE_INDICES="$(seq 0 $((N_SAMPLES - 1)))"
SAMPLES_PER_CHUNK="1"

echo "Job ${SLURM_JOB_ID} running on $(hostname), config ${CONFIG}, samples [${SAMPLE_INDICES}], scratch ${NEW_JAC_SCRATCH}"

# --out-dir is the root; each sample's checkpoints land under
# {out-dir}/sample{idx} (default when unset: autoencoder.data.ae_checkpoints_dir_for
# per sample, i.e. $NEW_JAC_SCRATCH/onnx_models/${NAME}/ae_checkpoints/sample{idx}).
# --n-models-per-chunk caps how many (sample, layer, pos) models train together
# in one batched GPU call -- multiple --sample-idx values can span one chunk.
exec uv run python -m scripts.python.train_batched_manifold \
    "${CONFIG}" \
    --sample-idx ${SAMPLE_INDICES} \
    --n-models-per-chunk 200 \
    --epochs 200 --batch-size 96 --lr 2e-3 --lam-geom 0.1
