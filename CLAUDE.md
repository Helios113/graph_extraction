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
per the pattern established across `scripts/compute_base_jacobian_stats.py`,
`scripts/generate_perturbed_sweep.py`, and `scripts/generate_tangent_manifold_distances.py`:

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



<!-- # Execution & Editing Guidelines -->
<!-- 
## File Modification Constraints & Workflow

- **Scoped Edit Permission:** You are strictly permitted to edit, create, or modify files located within `/nfs-share/pa511/code_bases/new_jac/execute/**` and `/nfs-share/pa511/code_bases/new_jac/lapac/**` only.
- **Read-Only / Advisory Mode Outside Scope:** For all other directories, never apply code changes directly. Do not create, modify, or delete files outside the allowed path, and do not execute automated edit tools/commands on them.
- **Workflow for Issues & Refactoring Outside Allowed Scope:**
  1. **Diagnosis:** Clearly identify and explain the root cause of the issue or the reason for the requested change.
  2. **Proposed Solution:** Outline the fix conceptually.
  3. **Code Snippets:** Provide the exact code snippets or diffs in the chat output for manual review and application, specifying the relevant file paths and line numbers. -->