#!/bin/bash
# One-shot throughput benchmark for the scratch<->NFS relay
# (scripts/batch_scripts/scratch_to_nfs.sh / nfs_to_scratch.sh). Generates a
# random file of SIZE_MB on --from's local scratch, times the upload leg to
# NFS staging, then times the download leg into --to's local scratch, and
# reports MB/s for each. Cleans up the test file on all three locations
# (source scratch, NFS staging, dest scratch) when done.
#
# Submits two independent single-node sbatch jobs chained with
# --dependency=afterok (same pattern as scripts/helper/sync_node_scratch.sh)
# rather than one two-node job step -- a single job spanning both --from and
# --to at once needs the backfill scheduler to find a simultaneous opening
# on BOTH nodes, which on a busy cluster with long-running array jobs queued
# on each node can get reserved hours out. Two separate single-node jobs
# each only need one node's next opening, which schedules far sooner.
#
# Usage:
#   scripts/helper/bench_scratch_sync.sh --from ruapehu --to mauao [--size-mb 2048]
#
# Prints job IDs and exits immediately; check logs/bench_upload_<job1>.out
# and logs/bench_download_<job2>.out (the latter has the full summary,
# since it runs after the upload leg completes).

set -euo pipefail

usage() {
    echo "Usage: $0 --from <node> --to <node> [--size-mb N] [--partition <p>]" >&2
    exit 1
}

FROM=""
TO=""
SIZE_MB=2048
PARTITION=""

while [ $# -gt 0 ]; do
    case "$1" in
        --from) FROM="$2"; shift 2 ;;
        --to) TO="$2"; shift 2 ;;
        --size-mb) SIZE_MB="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) usage ;;
    esac
done

[ -n "${FROM}" ] && [ -n "${TO}" ] || usage
if [ "${FROM}" = "${TO}" ]; then
    echo "error: --from and --to are both ${FROM}" >&2
    exit 1
fi

# Each node's own dedicated partition (OverSubscribe=NO, 12hr cap, works for
# this account's QOS) when --partition isn't given -- see
# scripts/helper/sync_node_scratch.sh for why not one shared partition name.
_default_partition() {
    case "$1" in
        ruapehu) echo "hopper" ;;
        mauao) echo "ampere" ;;
        *) echo "interactive" ;;
    esac
}
FROM_PARTITION="${PARTITION:-$(_default_partition "${FROM}")}"
TO_PARTITION="${PARTITION:-$(_default_partition "${TO}")}"

REPO=/nfs-share/pa511/code_bases/new_jac
cd "${REPO}"
mkdir -p logs

TAG="bench_$(date +%s)_$$"
SUBPATH="onnx_models/_${TAG}"

UPLOAD_SCRIPT="$(mktemp)"
cat > "${UPLOAD_SCRIPT}" <<EOF
#!/bin/bash
#SBATCH --job-name=bench_upload
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=${REPO}/logs/bench_upload_%j.out
set -euo pipefail
cd ${REPO}
NEW_JAC_SCRATCH="/local/scratch/\${USER}/new_jac"
SRC_DIR="\${NEW_JAC_SCRATCH}/${SUBPATH}"
STAGE_DIR="${REPO}/scratch_staging/${SUBPATH}"

echo "=== upload leg: \$(hostname) scratch -> NFS staging, ${SIZE_MB} MB ==="
mkdir -p "\${SRC_DIR}"
dd if=/dev/urandom of="\${SRC_DIR}/testfile.bin" bs=1M count=${SIZE_MB} status=none
sync
echo "generated: \$(du -sh "\${SRC_DIR}/testfile.bin" | cut -f1)"

mkdir -p "\${STAGE_DIR}"
T0=\$(date +%s.%N)
rsync -a "\${SRC_DIR}/" "\${STAGE_DIR}/"
T1=\$(date +%s.%N)
SECS=\$(echo "\${T1} - \${T0}" | bc)
RATE=\$(echo "scale=2; ${SIZE_MB} / \${SECS}" | bc)
echo "upload: \${SECS}s -> \${RATE} MB/s"

rm -rf "\${SRC_DIR}"
echo "UPLOAD_SECS=\${SECS}"
echo "UPLOAD_RATE=\${RATE}"
EOF

echo "Submitting upload leg on ${FROM} (${SIZE_MB} MB, partition ${FROM_PARTITION}) ..."
JOB1=$(sbatch --parsable --partition="${FROM_PARTITION}" --nodelist="${FROM}" "${UPLOAD_SCRIPT}")
echo "  job ${JOB1}: logs/bench_upload_${JOB1}.out"

DOWNLOAD_SCRIPT="$(mktemp)"
cat > "${DOWNLOAD_SCRIPT}" <<EOF
#!/bin/bash
#SBATCH --job-name=bench_download
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=${REPO}/logs/bench_download_%j.out
set -euo pipefail
cd ${REPO}
NEW_JAC_SCRATCH="/local/scratch/\${USER}/new_jac"
STAGE_DIR="${REPO}/scratch_staging/${SUBPATH}"
DEST_DIR="\${NEW_JAC_SCRATCH}/${SUBPATH}"

echo "=== download leg: NFS staging -> \$(hostname) scratch ==="
mkdir -p "\${DEST_DIR}"
T0=\$(date +%s.%N)
rsync -a "\${STAGE_DIR}/" "\${DEST_DIR}/"
T1=\$(date +%s.%N)
SECS=\$(echo "\${T1} - \${T0}" | bc)
RATE=\$(echo "scale=2; ${SIZE_MB} / \${SECS}" | bc)
echo "download: \${SECS}s -> \${RATE} MB/s"

rm -rf "\${DEST_DIR}" "\${STAGE_DIR}"
echo "DOWNLOAD_SECS=\${SECS}"
echo "DOWNLOAD_RATE=\${RATE}"

echo
echo "=== Summary (${SIZE_MB} MB, ${FROM} -> ${TO}) ==="
echo "  See logs/bench_upload_${JOB1}.out for the upload leg."
echo "  download: \${SECS}s -> \${RATE} MB/s"
EOF

echo "Queuing download leg on ${TO} (partition ${TO_PARTITION}, after job ${JOB1} succeeds) ..."
JOB2=$(sbatch --parsable --partition="${TO_PARTITION}" --nodelist="${TO}" \
    --dependency=afterok:"${JOB1}" "${DOWNLOAD_SCRIPT}")
echo "  job ${JOB2}: logs/bench_download_${JOB2}.out"

rm -f "${UPLOAD_SCRIPT}" "${DOWNLOAD_SCRIPT}"

echo
echo "Track with: squeue -u ${USER}"
