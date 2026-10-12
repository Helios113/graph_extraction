#!/bin/bash
#SBATCH --job-name=kernel_tangent_overlap_simple_stories
#SBATCH --partition=ampere
#SBATCH --nodelist=mauao
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/kernel_tangent_overlap_simple_stories_%j.out

set -euo pipefail

mkdir -p logs

cd /nfs-share/pa511/code_bases/new_jac

CONFIG="configs/simple_stories.toml"
NAME="simple_stories"
export NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac"

# mauao: the perturbed sweep's shards live on mauao's local scratch. ~6 min
# per sublayer per sample on an A40. Re-running resumes per sublayer.
#
# Edit SUBLAYERS ({in}.{out}) to restrict; empty = every sublayer except
# input_ids.embedding. SAMPLES picks which base samples' perturbed points.
SUBLAYERS=""
SAMPLES="0"
OUT_DIR="onnx_models/${NAME}/kernel_tangent_overlap_samples_${SAMPLES// /_}"

echo "Job ${SLURM_JOB_ID} running on $(hostname), config ${CONFIG}, sublayers [${SUBLAYERS:-all}], samples [${SAMPLES}], out ${OUT_DIR}"

uv run python -m scripts.python.compute_kernel_tangent_overlap \
    "${CONFIG}" \
    --perturbed-sweep-dir "onnx_models/${NAME}/perturbed_sweep" \
    --out-dir "${OUT_DIR}" \
    --samples ${SAMPLES} \
    ${SUBLAYERS:+--sublayers ${SUBLAYERS}}

mkdir -p figures
for SIDE in in out; do
    uv run python -m scripts.analysis.plot_kernel_tangent_overlap "${OUT_DIR}" --side "${SIDE}" \
        --out "figures/kernel_tangent_overlap_${NAME}_samples_${SAMPLES// /_}_${SIDE}"
done
