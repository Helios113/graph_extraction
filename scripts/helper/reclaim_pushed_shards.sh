#!/bin/bash
# sbatch wrapper for scripts/helper/reclaim_pushed_shards.py -- deletes
# already-pushed (Hetzner or NFS/mauao) shard files from the target node's
# local scratch to free space, verifying each shard's backing copy is
# actually present before removing the local one. Never touches an
# unpushed/in-progress shard.
#
# Usage:
#   sbatch --nodelist=ruapehu scripts/helper/reclaim_pushed_shards.sh [--dry-run] [--target-free-gb 400] [--backend nfs]
#
# Defaults to hopper/ruapehu if no --nodelist/--partition given at submit
# time; override both for mauao, e.g.:
#   sbatch --partition=ampere --nodelist=mauao scripts/helper/reclaim_pushed_shards.sh ...

#SBATCH --job-name=reclaim_shards
#SBATCH --partition=hopper
#SBATCH --nodelist=ruapehu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/reclaim_shards_%j.out

set -euo pipefail

cd /nfs-share/pa511/code_bases/new_jac
mkdir -p logs

export NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac"

echo "Job ${SLURM_JOB_ID} reclaiming pushed shards on $(hostname) ..."
exec uv run python -m scripts.helper.reclaim_pushed_shards "$@"
