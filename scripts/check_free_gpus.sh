#!/bin/bash
# Show free GPU counts per node, computed as (total GRES GPUs) - (GPUs held by running jobs).
set -euo pipefail

sinfo -N -h -o "%N %G" | sort -u -k1,1 | awk '
{
    node = $1
    gres = $2
    total = gres
    sub(/\(.*/, "", total)      # drop trailing "(S:0-1)"
    sub(/^gpu:[^:]*:/, "", total)  # drop "gpu:h100:" prefix, leaving count
    node_total[node] = total + 0
    node_gpu_type[node] = gres
    order[++n] = node
}
END {
    while (("squeue -h -t RUNNING -o \"%N|%b\"" | getline line) > 0) {
        split(line, parts, "|")
        node = parts[1]
        gres_req = parts[2]
        if (node == "" || gres_req !~ /gpu/) continue
        cnt = gres_req
        sub(/^gres\//, "", cnt)
        sub(/^gpu:/, "", cnt)
        if (cnt ~ /:/) {
            sub(/^[^:]*:/, "", cnt)   # "h100:1" -> "1"
        }
        cnt = cnt + 0
        if (cnt == 0) cnt = 1
        node_used[node] += cnt
    }
    printf "%-12s %-8s %-8s %-8s  %s\n", "NODE", "TOTAL", "USED", "FREE", "GPU_TYPE"
    free_sum = 0
    total_sum = 0
    for (i = 1; i <= n; i++) {
        node = order[i]
        used = node_used[node] + 0
        free = node_total[node] - used
        if (free < 0) free = 0
        printf "%-12s %-8s %-8s %-8s  %s\n", node, node_total[node], used, free, node_gpu_type[node]
        free_sum += free
        total_sum += node_total[node]
    }
    print "---"
    print "Total free GPUs: " free_sum " / " total_sum
}
'
