#!/bin/bash
#SBATCH --job-name=ae_sampled_sweep
#SBATCH --partition=ampere
#SBATCH --nodelist=mauao
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ae_sampled_sweep_%j.out

set -euo pipefail

mkdir -p logs

cd /nfs-share/pa511/code_bases/new_jac

usage() {
  echo "Usage: sbatch $0 [config] [sample_idx...] [-- --alpha A [A ...]] [--tau T] [--auto-tau] [--noise N [N ...]] [--n N] [--samples-per-chunk K]" >&2
  echo "  config: path to toml (default configs/gpt2.toml), must come first" >&2
  echo "  sample_idx...: one or more sample indices (default: 0), space-separated, before any --flag" >&2
  echo 'Example: sbatch scripts/batch_scripts/run_sampled_sweep_mauao.sh configs/gpt2.toml $(seq 0 63) --alpha 0.3 1.5 --tau 1.0 --noise 0.5 1.0' >&2
}

CONFIG="configs/gpt2.toml"
SAMPLE_IDX=()
ALPHA=(0.3)
NOISE=()
TAU="1.0"
AUTO_TAU=""
SAMPLES_PER_CHUNK="1"
N="1000"

# First non-flag arg (if any) is the config; subsequent non-flag args (until
# the first --flag) are sample indices. Everything from the first --flag
# onward is parsed as ordinary long options.
if [[ $# -gt 0 && "$1" != --* ]]; then
  CONFIG="$1"; shift
fi
while [[ $# -gt 0 && "$1" != --* ]]; do
  SAMPLE_IDX+=("$1"); shift
done
[[ ${#SAMPLE_IDX[@]} -eq 0 ]] && SAMPLE_IDX=(0)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --alpha)
      shift; ALPHA=()
      while [[ $# -gt 0 && "$1" != --* ]]; do ALPHA+=("$1"); shift; done
      ;;
    --noise)
      shift; NOISE=()
      while [[ $# -gt 0 && "$1" != --* ]]; do NOISE+=("$1"); shift; done
      ;;
    --tau) TAU="$2"; shift 2 ;;
    --auto-tau) AUTO_TAU="1"; shift ;;
    --n) N="$2"; shift 2 ;;
    --samples-per-chunk) SAMPLES_PER_CHUNK="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

NAME="$(basename "${CONFIG}" .toml)"
export NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac"
# No --ae-checkpoints-dir: with multiple --sample-idx values, each resolves to
# its own autoencoder.data.ae_checkpoints_dir_for(cfg, sample_idx) directory
# (see run_autoencoder_sampled_sweep.py's --ae-checkpoints-dir help).

echo "Job ${SLURM_JOB_ID} running on $(hostname), config ${CONFIG}, scratch ${NEW_JAC_SCRATCH}, sample-idx [${SAMPLE_IDX[*]}], alpha [${ALPHA[*]}], tau ${TAU}, noise [${NOISE[*]}]"

exec uv run python -m scripts.python.run_autoencoder_sampled_sweep \
    "${CONFIG}" \
    --sample-idx "${SAMPLE_IDX[@]}" \
    --samples-per-chunk "${SAMPLES_PER_CHUNK}" \
    --out-dir "onnx_models/${NAME}/ae_sampled_sweep" \
    --n "${N}" \
    --alpha "${ALPHA[@]}" \
    --tau "${TAU}" \
    ${AUTO_TAU:+--auto-tau} \
    ${NOISE:+--noise "${NOISE[@]}"}
