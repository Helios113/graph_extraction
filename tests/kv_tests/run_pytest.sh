#!/bin/bash
#SBATCH --partition=hopper
#SBATCH --nodelist=ruapehu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/nfs-share/pa511/code_bases/new_jac/logs/%j_series4.out
export CUDA_CACHE_MAXSIZE=4294967296  # ORT CUDA kernels are sm_86 SASS + PTX; on H100 they are JIT-compiled from PTX
cd /nfs-share/pa511/code_bases/new_jac
uv run --with plotly --with kaleido pytest tests/kv_tests -q --junitxml=results/series4/junit.xml -p no:cacheprovider --durations=0
