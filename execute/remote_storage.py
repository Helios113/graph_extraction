"""Thin wrapper around the repo-root `upload.py` DVC-style SFTP store, for
scripts that need to pull an input h5 that may only exist on the remote, or
push an output h5 once it's finished writing.

`upload.py` lives at the repo root (not as an installed package) since it's
also used standalone as a CLI (`uv run python upload.py push ...`), so this
module imports it by path the same way `scripts/` modules import `config`.
"""

import json
import os
from concurrent.futures import Future, ProcessPoolExecutor

import upload


def remote_exists(virtual_path: str, config: dict | None = None) -> bool:
    """Cheap check (no download) for whether `virtual_path` has a pointer in
    the remote index -- i.e. whether a prior push actually landed, as opposed
    to trusting a local progress/manifest record that claims it did (e.g. if
    the pushing process was killed between push() returning and the
    progress write, or the remote store was reset after the push happened).
    Does not verify the pointer's blob is still in the content-addressed
    cache, only that the index entry exists."""
    config = config or upload.load_config()
    sftp = upload.open_sftp(config)
    try:
        remote_idx = upload.remote_index_path(config, virtual_path)
        sftp.stat(remote_idx)
        return True
    except FileNotFoundError:
        return False
    finally:
        sftp.close()


def ensure_local(path: str, config: dict | None = None, dest_dir: str | None = None) -> str:
    """Make sure `path` exists locally, pulling it from the remote store if
    not. Returns the local path actually used (== `path` when `dest_dir` is
    None). No-op if already present at that location.

    `dest_dir`: pull into this directory instead of under
    `upload.PROJECT_ROOT` (NFS) -- e.g. node-local scratch disk, so a caller
    that only needs a file transiently doesn't grow shared NFS storage for
    it. `path` is still treated as project-relative (or made so via
    os.path.relpath against PROJECT_ROOT if it's already an absolute
    NFS-rooted path, e.g. from a caller that built it the old way) for
    pointer lookup/registration -- only where the pulled bytes land changes.
    A copy already sitting on NFS (e.g. left over from a run that pulled
    there before this parameter existed, or from a different caller that
    doesn't pass dest_dir) is reused as-is rather than pulled a second time
    onto `dest_dir` -- NFS is checked first since that data's already been
    paid for over SFTP.
    """
    virtual_path = os.path.relpath(path, upload.PROJECT_ROOT) if os.path.isabs(path) else path
    nfs_path = os.path.join(upload.PROJECT_ROOT, virtual_path)
    local_path = os.path.join(dest_dir, virtual_path) if dest_dir is not None else path

    if os.path.exists(local_path):
        return local_path
    if dest_dir is not None and os.path.exists(nfs_path):
        return nfs_path

    config = config or upload.load_config()
    ptr_path = upload._ptr_path_for(os.path.join(upload.PROJECT_ROOT, virtual_path))
    if not os.path.exists(ptr_path):
        print(f"[remote_storage] no local pointer for {virtual_path!r}, checking remote index...")
        sftp = upload.open_sftp(config)
        try:
            remote_idx = upload.remote_index_path(config, virtual_path)
            with sftp.open(remote_idx) as f:
                ptr = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{virtual_path!r} not found locally or in the remote index ({remote_idx})"
            )
        finally:
            sftp.close()
        os.makedirs(os.path.dirname(ptr_path), exist_ok=True)
        with open(ptr_path, "w") as f:
            json.dump(ptr, f, indent=2)

    print(f"[remote_storage] pulling {virtual_path!r} from remote store" + (f" into {dest_dir!r}" if dest_dir else "") + "...")
    upload._board = upload._Board(1) if upload._IS_TTY else upload._LogBoard()
    ptr = upload.read_ptr(virtual_path)
    upload._pull_file(ptr, config, dest_dir=dest_dir)
    return local_path


def push(
    path: str,
    config: dict | None = None,
    auto_delete: bool = False,
    virtual_path: str | None = None,
    force: bool = False,
) -> None:
    """Push a local file to the remote store (content-addressed, dedup'd by
    hash -- safe to call even if an identical version is already there).

    `virtual_path`: project-relative path to record in the pointer/remote
    index instead of `path`'s real location relative to the project root --
    for files written outside the repo (e.g. node-local scratch disk) that
    should still show up in `upload.py ls` / be pullable at the path they'd
    occupy if they lived in the project, e.g.
    "onnx_models/smoll/tangent_manifold_distances/embedding.add_6/shard_00000.h5".

    `force`: overwrite the remote version unconditionally instead of
    prompting/deferring on a content conflict -- see upload.cmd_push. Use
    when the caller knows the local version should win, e.g. a script
    re-run with --overwrite-remote after its prior remote output is known
    to be stale.
    """
    config = config or upload.load_config()
    print(f"[remote_storage] pushing {path!r} to remote store"
          + (f" as {virtual_path!r}" if virtual_path else "") + "...")
    upload.cmd_push(path, config, auto_delete=auto_delete, virtual_path=virtual_path, force=force)


def _push_worker(path: str, auto_delete: bool, virtual_path: str | None, force: bool) -> None:
    """Runs in a separate OS process (see AsyncPusher) -- `upload.cmd_push`
    stashes SFTP progress state in the `upload` module's `_board` global, so
    concurrent pushes must not share a process/interpreter, only a process
    pool. Re-loads its own config rather than pickling one across the
    process boundary."""
    push(path, config=None, auto_delete=auto_delete, virtual_path=virtual_path, force=force)


class AsyncPusher:
    """Shared, bounded-concurrency uploader for shard files produced by
    `ShardedLayerWriter`. Backed by a `ProcessPoolExecutor` (not threads --
    `upload.cmd_push` relies on a module-global `_board` for progress
    reporting, which concurrent threads in one interpreter would stomp on;
    separate processes each get their own copy).

    One `AsyncPusher` is meant to be shared across every layer's writer in a
    run, so shards from several layers can be uploading at once -- each
    writer separately caps how many of its *own* shards may be in flight
    (see `ShardedLayerWriter`), so this pool's `max_workers` bounds total
    upload concurrency across layers, not concurrency within a layer.
    """

    def __init__(self, max_workers: int = 4):
        self._pool = ProcessPoolExecutor(max_workers=max_workers)

    def submit(
        self, path: str, virtual_path: str | None = None, auto_delete: bool = True, force: bool = False,
    ) -> Future:
        """Non-blocking: hands `path` off to the process pool and returns
        immediately with a Future. Blocks only if every worker process is
        already busy (ProcessPoolExecutor queues in that case, which is the
        desired backpressure)."""
        return self._pool.submit(_push_worker, path, auto_delete, virtual_path, force)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)
