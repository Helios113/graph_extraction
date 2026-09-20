#!/bin/bash
#SBATCH --job-name=test_conn
#SBATCH --partition=hopper
#SBATCH --nodelist=ruapehu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/jupyter_%j.out

set -euo pipefail

mkdir -p logs

PORT=8888
echo "Job ${SLURM_JOB_ID} running on $(hostname), port ${PORT}"

# Reverse-tunnel the server port from this compute node to taranaki, so a
# client on taranaki (or your laptop, chained through taranaki) can reach
# 127.0.0.1:${PORT} and land on this server. Bound to localhost only — never
# exposed on the LAN, reachable solely through this SSH tunnel.
ssh -i /nfs-share/pa511/.ssh/id_ed25519_cluster -N -f \
    -R "${PORT}:localhost:${PORT}" taranaki

exec uv run --with jupyter jupyter lab \
    --no-browser \
    --ip=127.0.0.1 \
    --port="${PORT}" \
    --ServerApp.token='' \
    --ServerApp.password=''
