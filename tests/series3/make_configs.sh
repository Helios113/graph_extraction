#!/bin/bash
#SBATCH --partition=turing
#SBATCH --nodelist=ngongotaha
#SBATCH --time=01:00:00
#SBATCH --output=/nfs-share/pa511/code_bases/new_jac/logs/mkcfg_%j.out
# Generates a config TOML per HF model into tests/series3/configs/ (construct/make_config.py).
cd /nfs-share/pa511/code_bases/new_jac
export CUDA_VISIBLE_DEVICES=
for m in "$@"; do
  echo "### $m"; uv run python -m construct.make_config $m --out tests/series3/configs/$(basename $m | tr 'A-Z.' 'a-z_').toml 2>&1 | grep -E "^wrote|Error|no ONNX|Traceback|not distinct|no ModuleList|no attention" | cut -c1-400
done
