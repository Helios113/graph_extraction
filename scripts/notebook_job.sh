#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=hopper
#SBATCH --nodelist=ruapehu
#SBATCH --output=/nfs-share/pa511/code_bases/new_jac/jupyter_%j.log
#SBATCH --error=/nfs-share/pa511/code_bases/new_jac/jupyter_%j.err
#SBATCH --export=ALL

TARANAKI_PORT=9000
RUAPEHU_PORT=8888

# 1. Identify the login node you submitted from
LOGIN_NODE=$SLURM_SUBMIT_HOST

# 2. Define a secure, custom socket path on the compute node
export SSH_AUTH_SOCK="/tmp/ssh-agent-$SLURM_JOB_ID.sock"

# 3. Find your active agent socket on the login node and tunnel it here
# This runs a background SSH tunnel from the compute node back to the login node
ssh -o StrictHostKeyChecking=no -f -N \
    -L "$SSH_AUTH_SOCK:$(ssh $LOGIN_NODE 'echo $SSH_AUTH_SOCK')" \
    $LOGIN_NODE
# 2. Start Jupyter
# uv run jupyter lab --no-browser --port=${RUAPEHU_PORT}
