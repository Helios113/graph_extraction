#!/usr/bin/env bash
# Plots the ratio-histogram + EQD-sweep detail figure (via
# plot_lipschitz_ratio_eqd_fit.py --mode detail) for every (sample, pos) in
# a sublayer-pair directory whose fit is flagged as suspect: xi >= 0, or
# c_hat < 0 (see list_flagged_positions.py).
#
# Usage:
#   scripts/analysis/plot_flagged_positions.sh \
#       onnx_models/simple_stories/lipschitz_ratio_bootstrap_eqd_sweep_gpu/embedding.add_4 \
#       out_dir/flagged
#
# Writes one PNG per flagged position to <out_dir>/sample_<S>_pos_<P>.png.

set -euo pipefail

pair_dir="${1:?usage: plot_flagged_positions.sh <pair_dir> <out_dir>}"
out_dir="${2:?usage: plot_flagged_positions.sh <pair_dir> <out_dir>}"

mkdir -p "$out_dir"

uv run python scripts/analysis/list_flagged_positions.py "$pair_dir" | \
while read -r sample_idx pos; do
    out_path="$out_dir/sample_${sample_idx}_pos_${pos}.png"
    echo "plotting sample=$sample_idx pos=$pos -> $out_path"
    uv run python -m scripts.python.plot_lipschitz_ratio_eqd_fit \
        "$pair_dir" --mode detail --sample-idx "$sample_idx" --pos "$pos" --out "$out_path"
done
