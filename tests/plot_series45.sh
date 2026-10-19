#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --nodelist=mauao
#SBATCH --time=00:20:00
#SBATCH --output=/nfs-share/pa511/code_bases/new_jac/logs/%j_plot_series45.out
cd /nfs-share/pa511/code_bases/new_jac
export CUDA_VISIBLE_DEVICES=""
uv run --with plotly --with kaleido python tests/plot_series45.py
