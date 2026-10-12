#!/bin/bash
#SBATCH --job-name=perturbed_jac_det_simple_stories
#SBATCH --partition=ampere
#SBATCH --nodelist=mauao
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/perturbed_jac_det_simple_stories_%j.out

set -euo pipefail

mkdir -p logs

cd /nfs-share/pa511/code_bases/new_jac

CONFIG="configs/simple_stories.toml"
NAME="simple_stories"
export NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac"

# mauao: the perturbed sweep's shards live on mauao's local scratch (not in
# NFS staging), so this has to run there unless they're relayed first with
# scripts/helper/sync_node_scratch.sh.
#
# Edit SUBLAYERS to restrict to a subset ({in}.{out}); empty = every
# sublayer except input_ids.embedding. Re-running resumes per sublayer.
SUBLAYERS="add_4.add_6"
# Edit SAMPLES to restrict to perturbed rows of these base sample_idx values;
# empty = every sample. Output goes to a per-selection --out-dir, since
# resume progress is recorded against the filtered row count.
SAMPLES="0"
OUT_DIR="onnx_models/${NAME}/perturbed_jacobian_det${SAMPLES:+_samples_${SAMPLES// /_}}"

echo "Job ${SLURM_JOB_ID} running on $(hostname), config ${CONFIG}, sublayers [${SUBLAYERS:-all}], samples [${SAMPLES:-all}], out ${OUT_DIR}, scratch ${NEW_JAC_SCRATCH}"

exec uv run python -m scripts.python.compute_perturbed_jacobian_det \
    "${CONFIG}" \
    --perturbed-sweep-dir "onnx_models/${NAME}/perturbed_sweep" \
    --out-dir "${OUT_DIR}" \
    ${SUBLAYERS:+--sublayers ${SUBLAYERS}} \
    ${SAMPLES:+--samples ${SAMPLES}}
