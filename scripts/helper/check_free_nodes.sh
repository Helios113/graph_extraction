#!/bin/bash
# Show per-node state and free GPU/CPU capacity across the cluster.
set -euo pipefail

echo "=== Per-node state (deduped across partitions) ==="
sinfo -N -h -o "%N %T %G %C" | sort -u -k1,1 | column -t

echo
echo "=== Summary ==="
sinfo -N -h -o "%N %T" | sort -u -k1,1 | awk '{print $2}' | sort | uniq -c
