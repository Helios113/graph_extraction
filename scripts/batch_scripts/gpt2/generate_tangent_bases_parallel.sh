#!/bin/bash
# Splits generate_tangent_bases's --sample-idx range across N_JOBS concurrent
# sbatch jobs (one GPU each), instead of running every sample sequentially in
# one job -- each (sample, layer) pair is fully independent (see
# scripts.python.generate_tangent_bases's docstring), so this is pure wall-
# clock parallelism, not a change in what gets computed. mauao/ampere has 8
# A40s; check `sinfo -N -o "%N %P %G %C"` for current headroom before raising
# N_JOBS, and squeue -u $USER to confirm none of the launched jobs are stuck
# pending on GPU availability.
#
# Usage: scripts/batch_scripts/gpt2/generate_tangent_bases_parallel.sh [N_JOBS]

set -euo pipefail

cd /nfs-share/pa511/code_bases/new_jac
mkdir -p logs

CONFIG="configs/gpt2.toml"
NAME="gpt2"
N_JOBS="${1:-4}"

N_SAMPLES="$(uv run python -c 'from autoencoder.data import n_samples_for, sweep_dir_for; from config import load_config; from execute.sharded_writer import discover_layer_names; cfg=load_config("'"${CONFIG}"'"); d=sweep_dir_for(cfg); print(n_samples_for(d, discover_layer_names(d)[0]))')"
echo "N_SAMPLES=${N_SAMPLES}, splitting across N_JOBS=${N_JOBS}"

for ((j = 0; j < N_JOBS; j++)); do
    # Contiguous, near-equal sample ranges -- ceil-divided so the last job
    # gets any remainder rather than overrunning N_SAMPLES.
    CHUNK=$(( (N_SAMPLES + N_JOBS - 1) / N_JOBS ))
    START=$((j * CHUNK))
    END=$((START + CHUNK - 1))
    if [ "${END}" -ge "${N_SAMPLES}" ]; then END=$((N_SAMPLES - 1)); fi
    if [ "${START}" -gt "${END}" ]; then continue; fi
    SAMPLE_INDICES="$(seq "${START}" "${END}")"

    sbatch --job-name="tangent_bases_${NAME}_${j}" --partition=ampere --nodelist=mauao \
        --gres=gpu:a40:1 --cpus-per-task=8 --mem=64G --time=12:00:00 \
        --output="logs/tangent_bases_${NAME}_job${j}_%j.out" \
        --wrap="uv run python -m scripts.python.generate_tangent_bases ${CONFIG} \
            --sample-idx ${SAMPLE_INDICES} \
            --samples-per-chunk 1 \
            --rows-per-block 1500 \
            --out-dir onnx_models/${NAME}/tangent_bases \
            --knn-k 160 \
            --basis-method dual \
            --backend nfs"
    echo "  submitted job ${j}: samples [${SAMPLE_INDICES}]"
done
