#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --nodelist=mauao
#SBATCH --time=01:00:00
#SBATCH --output=/nfs-share/pa511/code_bases/new_jac/logs/%j_series1.out
cd /nfs-share/pa511/code_bases/new_jac
export NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=""
uv run --with plotly --with kaleido python tests/series1/run_series1.py
