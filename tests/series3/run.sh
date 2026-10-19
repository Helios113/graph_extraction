#!/bin/bash
#SBATCH --partition=hopper
#SBATCH --nodelist=ruapehu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/nfs-share/pa511/code_bases/new_jac/logs/%j_series3.out
cd /nfs-share/pa511/code_bases/new_jac
export NVIDIA_TF32_OVERRIDE=0 SERIES_EP=cuda CUDA_CACHE_MAXSIZE=4294967296
uv run --with plotly --with kaleido python tests/series3/run_series3.py "$@"
