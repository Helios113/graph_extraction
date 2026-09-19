## Setup

1. Get uv: https://docs.astral.sh/uv/getting-started/installation/

2. Download the wheel at 

3. Point `[tool.uv.sources]` in `pyproject.toml` at your wheel:

   ```toml
   [tool.uv.sources]
   onnxruntime-training = { path = "/path/to/onnxruntime_training-1.20.0+cu128-cp311-cp311-linux_x86_64.whl" }
   ```

4. Install the environment:

   ```
   uv sync  
   ```

<!--## Commands

All commands are run with `uv run` (see `CLAUDE.md`) and take a single TOML
config path, e.g. `configs/qwen_jac.toml` or `configs/qwen_jac_sublayer.toml`.
Pass `--force` to rebuild the base model / graphs even if cached artifacts
already exist.

`justfile` also has a couple of cluster-related helpers:

- `just run <file>` — run a script with `uv run python -B`.
- `just notebook-headless [gpus]` — submit a headless Jupyter job via `sbatch`.
- `just remote_tunnel [remote_port] [local_port]` — SSH-tunnel a remote port
  on `taranaki` to a local port.-->



# preview what would be deleted
uv run scripts/clean_corrupt_shards.py --dry-run

# actually delete everything
uv run scripts/clean_corrupt_shards.py

# only default scratch root ($NEW_JAC_SCRATCH or /local/scratch/<user>/new_jac) is scanned;
# override if needed:
uv run scripts/clean_corrupt_shards.py --scratch-root /local/scratch/pa511/new_jac


# mauao, CPU-only interactive shell
srun --partition=ampere --nodelist=mauao --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash

# ruapehu, CPU-only interactive shell
srun --partition=hopper --nodelist=ruapehu --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash



Drop the per-step float() sync


HYPERPARAM SEARCH SWEEP --- THIS THIS THIS


So concretely: GPT-2 and SimpleStories each have one extra "bonus" sublayer beyond 2 × n_blocks because their embedding tap (input_ids → add / input_ids → embedding) is a distinct sublayer in addition to the 2 × n_blocks block-level sublayers, whereas TinyLlama's input_ids → embedding tap is itself sublayer #1, and every subsequent sublayer pairs up block-by-block with no extra standalone entry — giving TinyLlama exactly 2 × n_blocks with no +1, while GPT-2/SimpleStories get 2 × n_blocks + 1.

That's the actual mechanism: it comes down to whether the config's very first [[sublayers]] entry is counted as one of the 2×n_blocks pairs or as an extra one tacked on before them — a config-authoring detail (how each configs/*.toml was hand-written), not a difference in the underlying model architectures' layer counts.