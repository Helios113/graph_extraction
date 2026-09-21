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
uv run scripts/helper/clean_corrupt_shards.py --dry-run

# actually delete everything
uv run scripts/helper/clean_corrupt_shards.py

# only default scratch root ($NEW_JAC_SCRATCH or /local/scratch/<user>/new_jac) is scanned;
# override if needed:
uv run scripts/helper/clean_corrupt_shards.py --scratch-root /local/scratch/pa511/new_jac


# mauao, CPU-only interactive shell
srun --partition=ampere --nodelist=mauao --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash

# ruapehu, CPU-only interactive shell
srun --partition=hopper --nodelist=ruapehu --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash



Drop the per-step float() sync


HYPERPARAM SEARCH SWEEP --- THIS THIS THIS


So concretely: GPT-2 and SimpleStories each have one extra "bonus" sublayer beyond 2 × n_blocks because their embedding tap (input_ids → add / input_ids → embedding) is a distinct sublayer in addition to the 2 × n_blocks block-level sublayers, whereas TinyLlama's input_ids → embedding tap is itself sublayer #1, and every subsequent sublayer pairs up block-by-block with no extra standalone entry — giving TinyLlama exactly 2 × n_blocks with no +1, while GPT-2/SimpleStories get 2 × n_blocks + 1.

That's the actual mechanism: it comes down to whether the config's very first [[sublayers]] entry is counted as one of the 2×n_blocks pairs or as an extra one tacked on before them — a config-authoring detail (how each configs/*.toml was hand-written), not a difference in the underlying model architectures' layer counts.


# tiny_llama: full pipeline, every script, every argument, all 64 samples (0-63)
#
# tiny_llama has 64 samples (autoencoder.data.n_samples_for(sweep_dir, layers[0])
# == 64 as of this writing -- re-check if the sweep was regenerated with a
# different --n-samples). All --sample-idx below is
# `$(seq 0 63 | tr '\n' ' ')` -- NOT bare `$(seq 0 63)`: plain `seq` prints
# one value per line, and inside a double-quoted --wrap="..." string those
# embedded newlines break the --wrap value (everything after the first
# newline gets lost, so the job silently only sees --sample-idx 0). `tr '\n'
# ' '` joins seq's output into one space-separated line first, matching what
# --sample-idx (nargs="+") actually needs. Swap in a smaller range to test on
# fewer samples first, but keep the `| tr '\n' ' '` if you do.
#
# Every script that writes sweep/layer output (all but generate_perturbed_sweep.py,
# see below) takes --backend {hetzner,nfs}, default "nfs" -- pushes go straight
# to NFS staging (execute/node_relay.py) instead of the old Hetzner/SFTP store
# (execute/remote_storage.py, kept only as a dormant fallback -- don't pass
# --backend hetzner for new work). --out-dir (and --tangent-bases-dir) must be
# project-relative ("onnx_models/...", NOT "${NEW_JAC_SCRATCH}/onnx_models/..."):
# ShardedLayerWriter joins it onto $NEW_JAC_SCRATCH for the scratch write path
# and separately uses it AS-IS for the pushed virtual_path and the NFS-relative
# manifest -- an absolute --out-dir silently breaks both. See CLAUDE.md's
# "Data transport" and "Where output lives" sections for the full mechanics.
#
# --layers/--sublayers are intentionally OMITTED below, not passed empty --
# these are nargs="*" flags, so `--layers` with no value parses to `[]`, not
# "use the default": for --layers that's harmless (`args.layers or
# discover_layer_names(...)` treats [] same as None) but for --sublayers it's
# not (`args.sublayers is not None` -- [] passes that check and filters every
# sublayer out). Separately, tiny_llama's default --layers
# (discover_layer_names(sweep_dir), the perturbed_sweep manifest names) is
# missing "input_ids" (it's never any sublayer's *output*, only the first
# one's input) -- so the default already runs 44 of 45 sublayers, not 45;
# pass --layers explicitly (every cfg.sublayers[i].input, including
# "input_ids") if you need that first sublayer included too. --knn-k is also
# omitted (generate_tangent_bases.py computes cfg.d_model + 1 by default);
# there's no "all values" equivalent for a single optional int.
#
# Prerequisite (unchanged, not covered here): scripts.python.generate_perturbed_sweep
# must have already produced onnx_models/tiny_llama/perturbed_sweep -- it has
# no --backend flag yet and still pushes to Hetzner; see
# scripts/batch_scripts/generate_perturbed_sweep_mauao.sh.

# 1) tangent_bases -- per-(sample, layer) local tangent bases, input for
# step 3. Now pushes via --backend nfs like the rest of this family (pass
# --no-push if you only need it on this node for an immediately-following
# step 3 run and don't want it relayed/kept elsewhere).
sbatch --job-name=tangent_bases_tiny_llama --partition=ampere --nodelist=mauao \
  --gres=gpu:a40:1 --cpus-per-task=8 --mem=64G --time=12:00:00 \
  --output=logs/tangent_bases_tiny_llama_%j.out \
  --wrap="uv run python -m scripts.python.generate_tangent_bases configs/tiny_llama.toml \
    --sample-idx $(seq 0 63 | tr '\n' ' ') \
    --samples-per-chunk 1 \
    --query-batch 4096 \
    --rows-per-block 500 \
    --flush-every 20 \
    --out-dir onnx_models/tiny_llama/tangent_bases \
    --backend nfs"


sbatch --job-name=tangent_bases_gpt2 --partition=ampere --nodelist=mauao \
  --gres=gpu:a40:1 --cpus-per-task=8 --mem=64G --time=12:00:00 \
  --output=logs/tangent_bases_gpt2_%j.out \
  --wrap="uv run python -m scripts.python.generate_tangent_bases configs/gpt2.toml \
    --sample-idx $(seq 0 63 | tr '\n' ' ') \
    --samples-per-chunk 1 \
    --query-batch 4096 \
    --rows-per-block 500 \
    --flush-every 20 \
    --out-dir onnx_models/gpt2/tangent_bases \
    --backend nfs"



sbatch --job-name=tangent_bases_gpt2 --partition=hopper --nodelist=ruapehu \
  --gres=gpu:h100:1 --cpus-per-task=8 --mem=64G --time=12:00:00 \
  --output=logs/tangent_bases_gpt2_%j.out \
  --wrap="uv run python -m scripts.python.generate_tangent_bases configs/gpt2.toml \
    --sample-idx $(seq 0 63 | tr '\n' ' ') \
    --samples-per-chunk 1 \
    --query-batch 4096 \
    --rows-per-block 500 \
    --flush-every 20 \
    --out-dir onnx_models/gpt2/tangent_bases \
    --backend nfs"


# 2) ambient_sweep -- Gaussian-noise sampling in ambient space, one or more taus
sbatch --job-name=ambient_sweep_tiny_llama --partition=ampere --nodelist=mauao \
  --gres=gpu:a40:1 --cpus-per-task=8 --mem=64G --time=12:00:00 \
  --output=logs/ambient_sweep_tiny_llama_%j.out \
  --wrap="uv run python -m scripts.python.generate_ambient_sweep configs/tiny_llama.toml \
    --sample-idx $(seq 0 63 | tr '\n' ' ') \
    --samples-per-chunk 1 \
    --n 1000 \
    --tau 0.25 0.5 1.0 2.0 4.0 \
    --seed 0 \
    --out-dir onnx_models/tiny_llama/ambient_sweep \
    --flush-every 10 \
    --upload-workers 4 \
    --backend nfs"

# 3) tangent_sweep -- local tangent-ball sampling, one or more radii; requires
# step 1's tangent_bases already present on this node's scratch
sbatch --job-name=tangent_sweep_tiny_llama --partition=ampere --nodelist=mauao \
  --gres=gpu:a40:1 --cpus-per-task=8 --mem=64G --time=12:00:00 \
  --output=logs/tangent_sweep_tiny_llama_%j.out \
  --wrap="uv run python -m scripts.python.generate_tangent_sweep configs/tiny_llama.toml \
    --tangent-bases-dir onnx_models/tiny_llama/tangent_bases \
    --sample-idx $(seq 0 63 | tr '\n' ' ') \
    --samples-per-chunk 1 \
    --n 1000 \
    --radius 0.025 0.05 0.1 0.2 0.4 \
    --seed 0 \
    --out-dir onnx_models/tiny_llama/tangent_sweep \
    --flush-every 10 \
    --upload-workers 4 \
    --backend nfs"

# 4) manifold training -- one batched_autoencoder per sublayer/pos, trained
# jointly across every --sample-idx given (driven via run_layer_batched_training.py,
# which batches layers by on-disk size and relays mauao -> ruapehu -> mauao
# per batch; see that script's module docstring for the full batch/relay/
# reclaim loop). network-dims/backend are not passed here -- the driver's
# training launcher always forwards --backend nfs (its relay-in step already
# lands the batch on ruapehu's scratch via NFS staging), and network-dims
# defaults to configs/tiny_llama.toml's own [1088, 128].
sbatch --job-name=layer_batched_driver --partition=hopper --nodelist=ruapehu --time=12:00:00 \
  --cpus-per-task=1 --mem=2G --output=logs/run_layer_batched_training_%j.out \
  --wrap="uv run python -m scripts.python.run_layer_batched_training configs/tiny_llama.toml \
    --sample-idx $(seq 0 63 | tr '\n' ' ') \
    --target-batch-gb 250 \
    --compute-node ruapehu \
    --storage-node mauao \
    --train-time-limit 12:00:00 \
    --n-models-per-chunk 200 \
    --epochs 12 \
    --checkpoint-every 3 \
    --lr 3e-4"


sbatch --job-name=layer_batched_driver --partition=hopper --nodelist=ruapehu --time=12:00:00 \
  --cpus-per-task=1 --mem=2G --output=logs/run_layer_batched_training_%j.out \
  --wrap="uv run python -m scripts.python.run_layer_batched_training configs/gpt2.toml \
    --sample-idx $(seq 0 63 | tr '\n' ' ') \
    --target-batch-gb 250 \
    --compute-node ruapehu \
    --storage-node mauao \
    --train-time-limit 12:00:00 \
    --n-models-per-chunk 200 \
    --epochs 12 \
    --checkpoint-every 3 \
    --lr 3e-4"

# equivalent single-batch direct call (skips run_layer_batched_training.py's
# batching/relay orchestration -- only use if the full sweep already fits on
# one node's scratch, e.g. after step 4's driver has relayed it there):
sbatch --job-name=train_batched_manifold_tiny_llama --partition=hopper --nodelist=ruapehu \
  --gres=gpu:a40:1 --cpus-per-task=8 --mem=64G --time=12:00:00 \
  --output=logs/train_batched_manifold_tiny_llama_%j.out \
  --wrap="uv run python -m scripts.python.train_batched_manifold configs/tiny_llama.toml \
    --sample-idx $(seq 0 63 | tr '\n' ' ') \
    --network-dims 1088 128 \
    --k 32 \
    --knn-chunk 512 \
    --knn-model-chunk 8 \
    --epochs 200 \
    --batch-size 96 \
    --lr 3e-4 \
    --lam-geom 0.1 \
    --val-frac 0.1 \
    --seed 0 \
    --n-models-per-chunk 200 \
    --checkpoint-every 50 \
    --backend nfs"

# 5) sampled_sweep -- AE-manifold sampling (one or more alpha levels, plus
# optional noise-calibration variants) against step 4's trained checkpoints;
# omit --ae-checkpoints-dir for multi-sample use so each --sample-idx value
# resolves its own checkpoint dir (autoencoder.data.ae_checkpoints_dir_for)
sbatch --job-name=sampled_sweep_tiny_llama --partition=ampere --nodelist=mauao \
  --gres=gpu:a40:1 --cpus-per-task=8 --mem=64G --time=12:00:00 \
  --output=logs/sampled_sweep_tiny_llama_%j.out \
  --wrap="uv run python -m scripts.python.run_autoencoder_sampled_sweep configs/tiny_llama.toml \
    --knn-chunk 512 \
    --knn-model-chunk 8 \
    --sample-idx $(seq 0 63 | tr '\n' ' ') \
    --samples-per-chunk 1 \
    --n 1000 \
    --alpha 0.3 \
    --radius global \
    --tau 1.0 \
    --noise 0.5 1.0 \
    --seed 0 \
    --jacobian-chunk 256 \
    --jacobian-model-chunk 8 \
    --support-chunk 2048 \
    --support-model-chunk 4 \
    --out-dir onnx_models/tiny_llama/ae_sampled_sweep \
    --flush-every 10 \
    --upload-workers 4 \
    --backend nfs"

# relay each finished step's output from mauao's scratch onto ruapehu's
# scratch (two-hop rsync through NFS staging, see
# scripts/helper/sync_node_scratch.sh) -- run once the sbatch job above has
# completed for the layer(s) you want to move; --backend nfs already pushed
# them onto NFS staging as each sublayer finished, so this only needs the
# "down" hop, but sync_node_scratch.sh's "up" hop is a no-op re-sync if
# you'd rather just run the whole two-hop command uniformly. tangent_bases
# (step 1) isn't relayed here since step 3 reads it on the SAME node (mauao)
# right after -- it now pushes to NFS staging too (unless run with
# --no-push), so it CAN be relayed the same way if you ever need
# generate_tangent_sweep run on a different node than generate_tangent_bases:
scripts/helper/sync_node_scratch.sh --from mauao --to ruapehu \
  onnx_models/tiny_llama/ambient_sweep
scripts/helper/sync_node_scratch.sh --from mauao --to ruapehu \
  onnx_models/tiny_llama/tangent_sweep
scripts/helper/sync_node_scratch.sh --from mauao --to ruapehu \
  onnx_models/tiny_llama/ae_sampled_sweep