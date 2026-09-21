# Python tooling

Use uv for every Python command in this project.

- Run scripts and tools with `uv run`, never bare `python` or `python3`.
- Add and remove dependencies with `uv add` and `uv remove`, never `pip install`.
- Sync and lock the environment with `uv sync` and `uv lock`.
- Run a one-off tool without adding it to the project: `uvx ruff check`.
- For a standalone script, use `uv run script.py` and add its dependencies with
  `uv add --script script.py <package>`.

# Script reference: read SCRIPTS.md before re-deriving a command

`SCRIPTS.md` at the repo root indexes every script under `scripts/python/`,
`scripts/helper/`, and `scripts/batch_scripts/` (including the
gpt2/tiny_llama/simple_stories batch-script variants) -- one-line purpose,
key CLI flags with defaults, gotchas, and example `uv run` invocations,
plus a "Common workflows" section chaining scripts into end-to-end
pipelines. Check it first for "how do I run X" instead of opening and
re-reading a script's argparse/docstring from scratch. Keep it in sync: any
new script, renamed flag, or changed default belongs in `SCRIPTS.md` too.

# Data transport: NFS relay, not Hetzner -- Hetzner is banned

This project moved off the Hetzner/SFTP remote store (`execute/remote_storage.py`)
onto a same-cluster NFS staging relay (`execute/node_relay.py`) as the
default transport for sweep/layer output. **Do not use `execute/remote_storage.py`,
`backend="hetzner"`, or `--backend hetzner` in any new or edited script, one-off
command, or invocation -- ask before reaching for Hetzner at all, even to read
old data pushed there before the migration.** If you're picking up work here,
the short version:

- `ShardedLayerWriter`'s (and `load_shard_rows`'/`load_sweep_data`'s)
  `backend` kwarg defaults to `"nfs"` everywhere now -- don't pass
  `backend="hetzner"` or add a new script that defaults to it, and don't add
  a `--backend` CLI choice that includes `"hetzner"` to a new script.
  `execute/remote_storage.py` still exists in the tree for old scripts that
  already call it, but it's banned for anything new: don't call it directly,
  don't route a new read/write through it, and don't add a fallback path to
  it "just in case" NFS staging is missing something -- surface that as an
  error instead and regenerate/relay the data via NFS.
- The mechanics (push-per-sublayer, `--out-dir`/`layer_key` must be
  project-relative, moving a finished layer between two nodes' scratch via
  `scripts/helper/sync_node_scratch.sh`) are covered in full in "Long-running
  GPU scripts" and "Where output lives" below -- read those before writing
  or editing a sweep script, since the project-relative-`--out-dir`
  requirement is easy to get wrong (it silently breaks the push/manifest
  path rather than erroring).
- `generate_ambient_sweep.py --tau` and `generate_tangent_sweep.py --radius`
  now both take one or more values in a single job (each value gets its own
  output file, sharing that (sample, sublayer)'s already-loaded
  cloud/bases/context across values) -- see README.md's tiny_llama launch
  commands for the current end-to-end example, including the
  `sync_node_scratch.sh --from mauao --to ruapehu` relay step.

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
- **Push each layer's file to NFS staging (via `execute/node_relay.push`,
  `--backend nfs`) as soon as that layer finishes**, not once at the very end
  after all sublayers complete. `ShardedLayerWriter` (in
  `execute/sharded_writer.py`) does this automatically when constructed with
  its default `backend="nfs"` -- it relays scratch -> `/nfs-share/datasets/$USER/`
  staging -> (later) another node's scratch, entirely over the filesystem
  every node already mounts, no SFTP or off-cluster transport involved. Gate
  it behind a `--no-push` flag. Don't reach for `execute/remote_storage.py`
  (Hetzner/SFTP) -- this project doesn't push there anymore; it's kept in
  the tree only as a dormant fallback transport, not something new scripts
  should default to or document as the primary path.
- Use `execute.node_relay.ensure_local` to pull required input h5 files from
  NFS staging if they're not already present locally, before the main
  sublayer loop starts.
- To move a finished layer's directory from one compute node's local scratch
  to another's (e.g. sweep data generated on mauao, needed for training on
  ruapehu), use `scripts/helper/sync_node_scratch.sh --from <node> --to
  <node> <subpath>` -- it chains two dependent sbatch jobs
  (`scratch_to_nfs.sh` then `nfs_to_scratch.sh`) through the same NFS staging
  area `node_relay` uses, so a layer already pushed via `--backend nfs`
  during generation is often already most of the way there.

This means a job that dies 2 hours into a 3-hour run has already pushed every
completed sublayer's results to NFS staging, and only the in-progress
sublayer's partial work (still flushed to local disk, just not pushed) needs
re-running.

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

Any output a mauao/ruapehu job writes repeatedly or writes a lot of -- sweep
shards, model checkpoints, periodic training checkpoints, anything written
more than once per job -- goes under node-local scratch (`$NEW_JAC_SCRATCH`,
i.e. `/local/scratch/$USER/new_jac`), never directly under the NFS-mounted
repo path (`onnx_models/...` relative to `/nfs-share/pa511/code_bases/new_jac`).
NFS is a network filesystem; writing to it repeatedly (e.g. a checkpoint
every 50 epochs) is slower and adds needless network I/O compared to local
disk.

This splits into two different `--out-dir` conventions depending on whether
the flag feeds a `ShardedLayerWriter` (sweep/layer output) or is a plain
scratch-only output path (checkpoints):

- **`ShardedLayerWriter`-backed output (`--out-dir` becomes a `layer_key`,
  e.g. `generate_perturbed_sweep.py`, `generate_ambient_sweep.py`,
  `generate_tangent_sweep.py`)**: `--out-dir` must be **project-relative**
  (`onnx_models/${NAME}/...`, NOT `${NEW_JAC_SCRATCH}/onnx_models/${NAME}/...`).
  `ShardedLayerWriter` itself joins `layer_key` onto `$NEW_JAC_SCRATCH` for
  the actual scratch write path, and separately uses `layer_key` as-is both
  for the pushed shard's virtual path (so it lands under
  `onnx_models/${NAME}/...` in NFS staging, not some nested
  `.../local/scratch/...` path) and for the small manifest/progress JSON,
  which lives on NFS at that same project-relative path. Passing an already
  scratch-absolute `--out-dir` here double-applies the scratch root on the
  push/manifest side and breaks both. See `execute/sharded_writer.py`'s
  `layer_key` docs and CLAUDE.md's push guidance above.
- **Scratch-only output with no push (e.g. `autoencoder checkpoints`,
  `train_batched_manifold.py --out-dir`,
  `autoencoder.data.ae_checkpoints_dir_for`)**: `--out-dir` (or its default)
  should be scratch-absolute, `${NEW_JAC_SCRATCH}/onnx_models/${NAME}/...`.
  These never get pushed anywhere (no manifest, no virtual_path), so there's
  no project-relative convention to match -- baking in the scratch root is
  correct here, and stays scratch-only unless a task says otherwise.



<!-- # Execution & Editing Guidelines -->
<!-- 
## File Modification Constraints & Workflow

- **Scoped Edit Permission:** You are strictly permitted to edit, create, or modify files located within `/nfs-share/pa511/code_bases/new_jac/execute/**` and `/nfs-share/pa511/code_bases/new_jac/lapac/**` only.
- **Read-Only / Advisory Mode Outside Scope:** For all other directories, never apply code changes directly. Do not create, modify, or delete files outside the allowed path, and do not execute automated edit tools/commands on them.
- **Workflow for Issues & Refactoring Outside Allowed Scope:**
  1. **Diagnosis:** Clearly identify and explain the root cause of the issue or the reason for the requested change.
  2. **Proposed Solution:** Outline the fix conceptually.
  3. **Code Snippets:** Provide the exact code snippets or diffs in the chat output for manual review and application, specifying the relevant file paths and line numbers. -->