#!/bin/bash
#SBATCH --partition=hopper
#SBATCH --nodelist=ruapehu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/nfs-share/pa511/code_bases/new_jac/logs/%j_series5.out
export CUDA_CACHE_MAXSIZE=4294967296  # ORT CUDA kernels JIT from PTX on H100
cd /nfs-share/pa511/code_bases/new_jac
uv run --with plotly --with kaleido python tests/series5/run_series5.py "$@"
