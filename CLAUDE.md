# Python tooling

Use uv for every Python command in this project.

- Run scripts and tools with `uv run`, never bare `python` or `python3`.
- Add and remove dependencies with `uv add` and `uv remove`, never `pip install`.
- Sync and lock the environment with `uv sync` and `uv lock`.
- Run a one-off tool without adding it to the project: `uvx ruff check`.
- For a standalone script, use `uv run script.py` and add its dependencies with
  `uv add --script script.py <package>`.

# Working rules

- Never do things I have not told you to, such as adding an extension.
- Do not go beyond what I ask for.
- Run things in a non-blocking way; don't wait for batched jobs to finish.
- Always run on a free node, unless told otherwise. If a job fits on ngog, run it there first.
- run sbatch jobs in a non-blocking way please
- ALWAYS PUT THE id of the job as the first element of the log name

# Cluster

All nodes are SLURM-only (no direct SSH). Node-local scratch (`/local/scratch/$USER`) is writable
only on compute nodes, not on the login node.

| Node (alias) | GPUs | CPUs | RAM | Batch partition (12h) | Notes |
|---|---|---|---|---|---|
| ngongotaha (ngog) | 7x RTX 2080 Ti | 56 | 257 GB | `turing` (default) | **Try first** if the job fits in 11 GB VRAM per GPU. |
| mauao | 8x A40 | 88 | 773 GB | `ampere` | Next choice; 48 GB VRAM per GPU. |
| ruapehu | 2x H100 + 1x H100 single + 1x H100 PCIe | 192 | 773 GB | `hopper` | Largest memory; use when it doesn't fit elsewhere. |
| gpu-vm | 2x V100 | 48 | 193 GB | none | Not usable: it has no batch partition. |

Run on a specific node with `sbatch --partition=<partition> --nodelist=<node> ...`.

- Never use the `interactive` or `cpu-only` partitions.
- `priority` is rejected for this account (QOS `cls_phd`), and `sbatch --test-only` will not catch that.
- Scripts passed to `srun`/`sbatch` must live on NFS, not in a local `/tmp` path.


# Plotting
Use plotly for plotting please. Always save a html and a png render.


Keep test findings written out in the coressponding series directory