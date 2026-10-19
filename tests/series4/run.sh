#!/bin/bash
#SBATCH --partition=hopper
#SBATCH --nodelist=ruapehu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/nfs-share/pa511/code_bases/new_jac/logs/%j_series4.out
cd /nfs-share/pa511/code_bases/new_jac
export NVIDIA_TF32_OVERRIDE=0 CUDA_CACHE_MAXSIZE=4294967296  # ORT CUDA kernels JIT from PTX on H100
uv run --with plotly --with kaleido python tests/series4/run_series4.py "$@"
