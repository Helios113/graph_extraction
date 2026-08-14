notebook-headless gpus="1":
    sbatch --gres=gpu:h100:{{gpus}} scripts/notebook_job.sh


# Forward local port to remote port on taranaki (default: remote 29500, local 8080)
remote_tunnel REMOTE_PORT="29500" LOCAL_PORT="8080":
    ssh -i /nfs-share/pa511/.ssh/id_ed25519_cluster -N -f -R {{REMOTE_PORT}}:localhost:{{LOCAL_PORT}} taranaki


run file:
    uv run python -B {{file}}

# local_tunnel REMOTE_PORT="29500" LOCAL_PORT="8080":
#    ssh -N -f -L {{LOCAL_PORT}}:localhost:{{REMOTE_PORT}} taranaki
