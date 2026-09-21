#!/bin/bash
#SBATCH --job-name=ae_sampled_sweep_tiny_llama
#SBATCH --partition=ampere
#SBATCH --nodelist=mauao
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ae_sampled_sweep_tiny_llama_%j.out

set -euo pipefail

mkdir -p logs

cd /nfs-share/pa511/code_bases/new_jac

CONFIG="configs/tiny_llama.toml"
NAME="tiny_llama"
export NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac"

# Edit SAMPLE_INDICES to run a subset of samples; SAMPLES_PER_CHUNK controls
# how many samples are batched together per GPU pass -- raise it on hardware
# with more headroom, 1 (sequential) is always safe. See CLAUDE.md's
# model_chunk chunking convention -- sample is the same kind of knob.
N_SAMPLES="$(uv run python -c 'from autoencoder.data import n_samples_for, sweep_dir_for; from config import load_config; from execute.sharded_writer import discover_layer_names; cfg=load_config("'"${CONFIG}"'"); d=sweep_dir_for(cfg); print(n_samples_for(d, discover_layer_names(d)[0]))')"
SAMPLE_INDICES="$(seq 0 $((N_SAMPLES - 1)))"
SAMPLES_PER_CHUNK="1"

# Edit these to change the sampled-sweep parameters.
ALPHA="0.3"
NOISE=""
TAU="1.0"
RADIUS="global"

echo "Job ${SLURM_JOB_ID} running on $(hostname), config ${CONFIG}, samples [${SAMPLE_INDICES}], samples-per-chunk ${SAMPLES_PER_CHUNK}, alpha [${ALPHA}], tau ${TAU}, radius ${RADIUS}, noise [${NOISE}], scratch ${NEW_JAC_SCRATCH}"

exec uv run python -m scripts.python.run_autoencoder_sampled_sweep \
    "${CONFIG}" \
    --sample-idx ${SAMPLE_INDICES} \
    --samples-per-chunk "${SAMPLES_PER_CHUNK}" \
    --out-dir "onnx_models/${NAME}/ae_sampled_sweep" \
    --n 1000 --alpha ${ALPHA} --tau "${TAU}" --radius "${RADIUS}" \
    ${NOISE:+--noise ${NOISE}}
