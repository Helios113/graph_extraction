#!/bin/bash
#SBATCH --job-name=lipschitz_ratio_bootstrap_eqd_sweep_gpu_driver
#SBATCH --partition=ampere
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=mauao
#SBATCH --gres=0
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH --output=logs/run_lipschitz_ratio_bootstrap_eqd_sweep_gpu_%j.out

# Runs scripts/python/run_lipschitz_ratio_bootstrap_eqd_sweep_gpu.py -- the
# batch/relay/compute/reclaim DRIVER for the GPU bootstrap-EQD sweep (see
# that script's module docstring). This job itself does no GPU work and
# needs almost no resources -- it only submits and polls the actual compute
# jobs (each pinned to --compute-node via its own --partition/--nodelist/
# --gres), same as how run_layer_batched_tangent_sweep_direct.py's driver
# runs (SCRIPTS.md's tangent-sweep-direct example uses the same
# --wrap="uv run python -m ..." pattern from taranaki, unwrapped here into
# its own script since the driver can run for many hours across all
# batches and a stray terminal disconnect shouldn't kill it).
#
# --time=12:00:00 bounds THIS job (the driver/poller), not the individual
# compute jobs it submits -- each of those gets its own --job-time-limit
# (forwarded to --time on that job specifically). If the full sweep across
# all batches could plausibly exceed 12h end to end, raise this job's
# --time accordingly (interactive partition caps at 1h -- see below).
#
# Usage:
#   sbatch scripts/batch_scripts/run_lipschitz_ratio_bootstrap_eqd_sweep_gpu_sbatch.sh \
#       <config> [extra run_lipschitz_ratio_bootstrap_eqd_sweep_gpu.py args...]
#
# Example (gpt2, relay mauao -> ruapehu, compute on ruapehu):
#   sbatch scripts/batch_scripts/run_lipschitz_ratio_bootstrap_eqd_sweep_gpu_sbatch.sh \
#       configs/gpt2.toml --target-batch-gb 250 --compute-node ruapehu --storage-node mauao
#
# Example, compute directly on mauao (no relay, storage_node == compute_node):
#   sbatch scripts/batch_scripts/run_lipschitz_ratio_bootstrap_eqd_sweep_gpu_sbatch.sh \
#       configs/simple_stories.toml --compute-node mauao --storage-node mauao
#
# NOTE: this job's own --partition=cpu-only (2-day cap, so --time=12:00:00
# fits comfortably) was picked because the driver process itself needs no
# GPU -- it only submits/polls the actual compute jobs, which separately
# request their own --gres via --compute-node. Submitting the driver itself
# to a GPU partition (ampere/hopper) would tie up a GPU-node CPU slot for no
# reason. cpu-only does have a FORCE:2 concurrent-job cap on this account
# (see CLAUDE.md/memory notes on cluster partition scheduling) -- if that's
# already saturated by other work, override --partition on the sbatch
# command line, e.g.:
#   sbatch --partition=interactive --time=01:00:00 scripts/batch_scripts/run_lipschitz_ratio_bootstrap_eqd_sweep_gpu_sbatch.sh ...
# (interactive has no such cap but is capped at 1h itself -- only viable for
# a sweep whose batches all finish within that hour).

set -euo pipefail

mkdir -p logs

cd /nfs-share/pa511/code_bases/new_jac

CONFIG="${1:?usage: sbatch ... <config> [extra args...]}"
shift

echo "Driver job ${SLURM_JOB_ID} running on $(hostname), config ${CONFIG}"

exec uv run python -m scripts.lipschitz.run_lipschitz_ratio_bootstrap_eqd_sweep_gpu \
    "${CONFIG}" \
    "$@"
