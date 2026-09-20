# Python tooling

Use uv for every Python command in this project.

- Run scripts and tools with `uv run`, never bare `python` or `python3`.
- Add and remove dependencies with `uv add` and `uv remove`, never `pip install`.
- Sync and lock the environment with `uv sync` and `uv lock`.
- Run a one-off tool without adding it to the project: `uvx ruff check`.
- For a standalone script, use `uv run script.py` and add its dependencies with
  `uv add --script script.py <package>`.

# Long-running GPU scripts under scripts/

Scripts that sweep over all sublayers of a config (e.g. `configs/smoll_jac.toml`)
and can run for hours must be resilient to a crash/interrupt partway through,
per the pattern established across `scripts/python/compute_base_jacobian_stats.py`,
`scripts/python/generate_perturbed_sweep.py`, and `scripts/python/generate_tangent_manifold_distances.py`:

- **One output h5 file per sublayer**, written to an `--out-dir` directory,
  named `{sublayer_input}.{sublayer_output}.h5` -- not one combined file for
  all sublayers. If a run dies on sublayer 4 of 6, sublayers 1-3's files are
  already complete and usable.
- **Flush periodically within a sublayer**, not just at the end. Process rows
  in chunks/blocks, and call `h5f.flush()` (or `out_grp.file.flush()`) every
  `--flush-every` chunks/blocks (and always on the last one). A `--flush-every`
  CLI arg controls the cadence; keep it independent from any batching
  parameter used for GPU throughput (e.g. `--rows-per-block`) since they're
  tuned for different things (crash-recovery granularity vs. GPU utilization).
- **Push each layer's file to the remote store (Hetzner, via
  `execute/remote_storage.push`) as soon as that layer finishes**, not once at
  the very end after all sublayers complete. See `execute/remote_storage.py`
  (thin wrapper around the repo-root `upload.py` SFTP store) -- call
  `push(layer_h5_path)` right after that layer's `with h5py.File(...)` block
  closes, inside the per-sublayer loop. Gate it behind a `--no-push` flag.
- Use `execute.remote_storage.ensure_local` to pull required input h5 files
  from the remote store if they're not already present locally, before the
  main sublayer loop starts.

This means a job that dies 2 hours into a 3-hour run has already pushed every
completed sublayer's results to Hetzner, and only the in-progress sublayer's
partial work (still flushed to local disk, just not pushed) needs re-running.

# Chunk the model dim too, not just the point dim, in batched GPU code

`autoencoder/batched_autoencoder.py`'s `batched_autoencoder` and
`autoencoder/sampling.py`'s `batched_*` functions operate on tensors shaped
`(n_models, n_points, ...)`. Any GPU op here -- `cdist`, the batched MLP's
`torch.bmm` (via `encode`/`decode`/`project`), forward-mode-AD jacobians --
scales with `n_models * n_points` together, not just `n_points`. A function
chunked only along the point dim (a `chunk` kwarg with no matching
`model_chunk`) is silently unsafe: it works at small `n_models` (e.g. a
100-model chunk during training, where GPU memory happens to have headroom)
and OOMs later at the exact same point-chunk size once more models are
loaded into one batch at once (e.g. `fit_stats()`/sampling after training,
which reload every (sublayer, pos) model together rather than training's
`--n-models-per-chunk`-limited groups).

When adding or calling a batched op here:
- Give it both a `chunk` (point dim) and a `model_chunk` (model dim) kwarg,
  and actually loop over both -- see `knn()` in `batched_autoencoder.py` for
  the reference pattern, or `chunked_apply`/`_recon_geom` in
  `scripts/python/validate_autoencoder.py` for encode/decode/project over a full
  points pool.
- Never call `model.encode(x)` / `model.decode(z)` / `model.project(x)`
  directly on a full `(n_models, N, D)` states/val pool -- go through
  `batched_autoencoder.chunked_apply` (or an equivalent chunked loop)
  instead, the same way `train_batched_manifold.py`'s `chunked_recon_loss`
  avoids calling `bmnfld.project(x)` unchunked.
- Default `model_chunk` to something bounded (e.g. 8), not `None`/all models
  -- `None` is only safe for call sites that already guarantee a small
  `n_models` (like the per-chunk training loop), and it's easy to add a new
  caller (fit_stats, validate, sampling) that doesn't.
- When you see an OOM crash log from one of these scripts, the traceback
  frame plus the "Tried to allocate N GiB" size is usually enough to compute
  which dim was left unchunked (n_models * n_points * d_model * 4 bytes, or
  n_models * chunk * n_train_rows * 4 bytes for a cdist) -- check that first
  before assuming it's a batch-size or `--n-models-per-chunk` tuning issue.

# Where output lives: local scratch, not NFS

Any output a mauao job writes repeatedly or writes a lot of -- sweep shards,
model checkpoints, periodic training checkpoints, anything written more than
once per job -- goes under node-local scratch (`$NEW_JAC_SCRATCH`, i.e.
`/local/scratch/$USER/new_jac`), never directly under the NFS-mounted repo
path (`onnx_models/...` relative to `/nfs-share/pa511/code_bases/new_jac`).
NFS is a network filesystem; writing to it repeatedly (e.g. a checkpoint
every 50 epochs) is slower and adds needless network I/O compared to local
disk. `--out-dir`-style flags on mauao launcher `.sh` scripts should default
to `${NEW_JAC_SCRATCH}/onnx_models/${NAME}/...`, not `onnx_models/${NAME}/...`.
Only sweep shard data gets pushed off scratch, and only to the Hetzner remote
store via `execute/remote_storage.push` (see above) -- not to NFS. Other
scratch-resident outputs (e.g. autoencoder checkpoints) currently stay on
scratch only, with no NFS or remote copy, unless a task says otherwise.



<!-- # Execution & Editing Guidelines -->
<!-- 
## File Modification Constraints & Workflow

- **Scoped Edit Permission:** You are strictly permitted to edit, create, or modify files located within `/nfs-share/pa511/code_bases/new_jac/execute/**` and `/nfs-share/pa511/code_bases/new_jac/lapac/**` only.
- **Read-Only / Advisory Mode Outside Scope:** For all other directories, never apply code changes directly. Do not create, modify, or delete files outside the allowed path, and do not execute automated edit tools/commands on them.
- **Workflow for Issues & Refactoring Outside Allowed Scope:**
  1. **Diagnosis:** Clearly identify and explain the root cause of the issue or the reason for the requested change.
  2. **Proposed Solution:** Outline the fix conceptually.
  3. **Code Snippets:** Provide the exact code snippets or diffs in the chat output for manual review and application, specifying the relevant file paths and line numbers. -->