#!/bin/bash
#SBATCH -p ampere
#SBATCH -w mauao
#SBATCH --gres=gpu:1
#SBATCH -t 600

cd /nfs-share/pa511/code_bases/new_jac
.venv/bin/python -m pytest tests/kv_tests/test_kv_pullback_jacobian.py -q
