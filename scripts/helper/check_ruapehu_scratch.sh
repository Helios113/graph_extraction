#!/bin/bash
# Reports free space on ruapehu's node-local scratch disk
# (/local/scratch, shared across every user on the node -- not a per-user
# quota) plus this user's own footprint under $NEW_JAC_SCRATCH. Exits
# non-zero if free space has dropped below --threshold-gb, so it can gate a
# sweep script or be polled from a launcher before starting a large job.
#
# Submit via sbatch (needs to run ON ruapehu -- df/du against
# /local/scratch from taranaki would report the wrong filesystem or fail,
# since it's node-local):
#   sbatch scripts/helper/check_ruapehu_scratch.sh [--threshold-gb 300]
#
# For mauao instead, override both directives:
#   sbatch --partition=ampere --nodelist=mauao scripts/helper/check_ruapehu_scratch.sh

#SBATCH --job-name=check_scratch
#SBATCH --partition=hopper
#SBATCH --nodelist=ruapehu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/check_scratch_%j.out

set -euo pipefail

cd /nfs-share/pa511/code_bases/new_jac
mkdir -p logs

THRESHOLD_GB=300

while [ $# -gt 0 ]; do
    case "$1" in
        --threshold-gb) THRESHOLD_GB="$2"; shift 2 ;;
        -h|--help) echo "Usage: sbatch $0 [--threshold-gb N]" >&2; exit 1 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

NEW_JAC_SCRATCH="/local/scratch/${USER}/new_jac"

read -r SIZE_B USED_B AVAIL_B <<< "$(df --output=size,used,avail -B1 /local/scratch | tail -1)"
AVAIL_GB=$((AVAIL_B / 1024 / 1024 / 1024))
SIZE_GB=$((SIZE_B / 1024 / 1024 / 1024))
USED_GB=$((USED_B / 1024 / 1024 / 1024))

echo "=== $(hostname) /local/scratch (node-wide, shared across all users) ==="
echo "  total: ${SIZE_GB} GB   used: ${USED_GB} GB   available: ${AVAIL_GB} GB"

if [ -d "${NEW_JAC_SCRATCH}" ]; then
    OWN_KB=$(du -sk "${NEW_JAC_SCRATCH}" 2>/dev/null | cut -f1)
    OWN_GB=$((OWN_KB / 1024 / 1024))
    echo
    echo "=== ${USER}'s own usage under ${NEW_JAC_SCRATCH} ==="
    echo "  ${OWN_GB} GB"
    echo
    echo "  by layer (top 15):"
    du -sh "${NEW_JAC_SCRATCH}"/onnx_models/*/*/* 2>/dev/null | sort -rh | head -15 | sed 's/^/    /'
else
    echo
    echo "  (no ${NEW_JAC_SCRATCH} yet)"
fi

echo
if [ "${AVAIL_GB}" -lt "${THRESHOLD_GB}" ]; then
    echo "WARNING: only ${AVAIL_GB} GB free, below threshold of ${THRESHOLD_GB} GB."
    echo "Run scripts/helper/reclaim_pushed_shards.sh to offload already-pushed shards to mauao and free space."
    exit 1
fi
echo "OK: ${AVAIL_GB} GB free (threshold: ${THRESHOLD_GB} GB)."
