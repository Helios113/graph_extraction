"""Direct node-to-node data relay via NFS staging, as an alternative
transport to the Hetzner SFTP store in `execute/remote_storage.py` --
mirrors that module's `push`/`ensure_local`/`AsyncPusher` API so
`execute/sharded_writer.py` can swap between the two with a `backend` kwarg.

Where remote_storage.py goes over SFTP to a remote box (durable, dedup'd,
but capped by Hetzner upload bandwidth), this module goes local-scratch ->
NFS -> local-scratch, entirely through the filesystem every node already
mounts. No content-addressing, no remote index, no SFTP -- just an rsync
through /nfs-share/datasets/<user>/ (kept out of the repo tree under
/nfs-share/pa511/... since it holds bulk relay data, not project files, and
that tree has a tighter quota). Roughly 5-10x the throughput of the
Hetzner leg (measured ~450 MB/s NFS-side vs. Hetzner's SFTP bottleneck),
since it never leaves the local network.

Use this when compute and storage are split across two nodes on the SAME
cluster (e.g. compute on ruapehu, large local-scratch storage on mauao) and
durability/offsite backup isn't the point -- just fast, cheap movement. Keep
using remote_storage.py (Hetzner) for anything that needs to survive both
nodes' scratch disks being wiped, or that needs to leave the cluster.

This module does its own rsync inline in the caller's process (no `sbatch`
involved) -- it assumes the caller is itself already running as a SLURM job
on the source (for `push`) or destination (for `ensure_local`) node, same as
every existing ShardedLayerWriter/load_shard_rows caller in scripts/. It is
NOT what backs `scripts/helper/sync_node_scratch.sh`, which submits its own
sbatch jobs and is meant for one-off moves triggered from taranaki instead
of from inside a running compute job.
"""

import os
import shutil
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGING_ROOT = os.path.join("/nfs-share/datasets", os.environ.get("USER", "pa511"))


def _rsync(src: str, dst: str) -> None:
    """`src`/`dst` are file paths (not dirs) -- always a single-file
    rsync, since shards are already individual h5 files. Uses rsync (not
    shutil.copy) for the same reason scripts/batch_scripts/*.sh do: resumable
    and consistent with the rest of this relay's transfers, even though a
    plain copy would suffice for one file."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp_dst = dst + f".tmp.{os.getpid()}"
    subprocess.run(
        ["rsync", "-a", src, tmp_dst],
        check=True,
    )
    os.replace(tmp_dst, dst)  # atomic on the same filesystem -- see remote_storage's sftp_put note


def _staging_path(virtual_path: str) -> str:
    return os.path.join(STAGING_ROOT, virtual_path)


def remote_exists(virtual_path: str) -> bool:
    """Cheap existence check against NFS staging -- the equivalent of
    remote_storage.remote_exists, but staging is just a directory, so this
    is a stat, not an SFTP round-trip."""
    return os.path.exists(_staging_path(virtual_path))


def push(
    path: str,
    virtual_path: str | None = None,
    auto_delete: bool = False,
    **_ignored,
) -> None:
    """Copy `path` (on this node's local scratch) to NFS staging at
    `virtual_path` (or `path`'s project-relative location if not given, same
    convention as remote_storage.push). Runs inline (no sbatch) -- the
    caller must already be executing on the node that has `path` locally.

    `**_ignored` absorbs remote_storage.push's `config`/`force` kwargs so
    ShardedLayerWriter can call either backend the same way without
    branching on which kwargs are valid.
    """
    virtual_path = virtual_path or os.path.relpath(path, PROJECT_ROOT)
    dst = _staging_path(virtual_path)
    print(f"[node_relay] pushing {path!r} -> NFS staging {dst!r} ...")
    _rsync(path, dst)
    if auto_delete:
        os.remove(path)


def ensure_local(path: str, dest_dir: str | None = None, **_ignored) -> str:
    """Make sure `path` exists locally, pulling it from NFS staging if not.
    Mirrors remote_storage.ensure_local's signature/semantics (including
    treating `path` as project-relative for staging lookup, and reusing an
    already-local NFS copy before re-pulling onto `dest_dir`), but sources
    from STAGING_ROOT instead of the Hetzner remote index.
    """
    virtual_path = os.path.relpath(path, PROJECT_ROOT) if os.path.isabs(path) else path
    nfs_path = os.path.join(PROJECT_ROOT, virtual_path)
    local_path = os.path.join(dest_dir, virtual_path) if dest_dir is not None else path

    if os.path.exists(local_path):
        return local_path
    if dest_dir is not None and os.path.exists(nfs_path):
        return nfs_path

    staged = _staging_path(virtual_path)
    if not os.path.exists(staged):
        raise FileNotFoundError(
            f"{virtual_path!r} not found locally or in NFS staging ({staged}) -- "
            "was it pushed with node_relay.push (or AsyncPusher backend='nfs')?"
        )

    print(f"[node_relay] pulling {virtual_path!r} from NFS staging" + (f" into {dest_dir!r}" if dest_dir else "") + "...")
    _rsync(staged, local_path)
    return local_path


def _push_worker(path: str, virtual_path: str | None, auto_delete: bool) -> None:
    push(path, virtual_path=virtual_path, auto_delete=auto_delete)


class AsyncPusher:
    """Drop-in equivalent of remote_storage.AsyncPusher, backed by
    node_relay.push instead of an SFTP upload. Uses a thread pool rather
    than remote_storage's process pool -- node_relay has no module-global
    progress-board state to isolate across workers (an rsync subprocess is
    already its own process), so threads are enough and avoid the
    pickling/import overhead of spinning up worker processes per shard.
    """

    def __init__(self, max_workers: int = 4):
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def submit(
        self, path: str, virtual_path: str | None = None, auto_delete: bool = True, **_ignored,
    ) -> Future:
        return self._pool.submit(_push_worker, path, virtual_path, auto_delete)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)


def _require_rsync() -> None:
    if shutil.which("rsync") is None:
        raise RuntimeError("node_relay backend requires `rsync` on PATH")


_require_rsync()
