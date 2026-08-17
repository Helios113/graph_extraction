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
