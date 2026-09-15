"""Thin wrapper around the repo-root `upload.py` DVC-style SFTP store, for
scripts that need to pull an input h5 that may only exist on the remote, or
push an output h5 once it's finished writing.

`upload.py` lives at the repo root (not as an installed package) since it's
also used standalone as a CLI (`uv run python upload.py push ...`), so this
module imports it by path the same way `scripts/` modules import `config`.
"""

import json
import os

import upload


def ensure_local(path: str, config: dict | None = None) -> str:
    """Make sure `path` exists locally, pulling it from the remote store if
    not. Returns `path` unchanged (for chaining). No-op if already present.
    """
    if os.path.exists(path):
        return path

    config = config or upload.load_config()
    ptr_path = upload._ptr_path_for(os.path.join(upload.PROJECT_ROOT, path))
    if not os.path.exists(ptr_path):
        print(f"[remote_storage] no local pointer for {path!r}, checking remote index...")
        sftp = upload.open_sftp(config)
        try:
            remote_idx = upload.remote_index_path(config, os.path.relpath(path, upload.PROJECT_ROOT))
            with sftp.open(remote_idx) as f:
                ptr = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{path!r} not found locally or in the remote index ({remote_idx})"
            )
        finally:
            sftp.close()
        os.makedirs(os.path.dirname(ptr_path), exist_ok=True)
        with open(ptr_path, "w") as f:
            json.dump(ptr, f, indent=2)

    print(f"[remote_storage] pulling {path!r} from remote store...")
    upload._board = upload._Board(1) if upload._IS_TTY else upload._LogBoard()
    ptr = upload.read_ptr(os.path.relpath(path, upload.PROJECT_ROOT))
    upload._pull_file(ptr, config)
    return path


def push(
    path: str,
    config: dict | None = None,
    auto_delete: bool = True,
    virtual_path: str | None = None,
) -> None:
    """Push a local file to the remote store (content-addressed, dedup'd by
    hash -- safe to call even if an identical version is already there).

    `virtual_path`: project-relative path to record in the pointer/remote
    index instead of `path`'s real location relative to the project root --
    for files written outside the repo (e.g. node-local scratch disk) that
    should still show up in `upload.py ls` / be pullable at the path they'd
    occupy if they lived in the project, e.g.
    "onnx_models/smoll/tangent_manifold_distances/embedding.add_6/shard_00000.h5".
    """
    config = config or upload.load_config()
    print(f"[remote_storage] pushing {path!r} to remote store"
          + (f" as {virtual_path!r}" if virtual_path else "") + "...")
    upload.cmd_push(path, config, auto_delete=auto_delete, virtual_path=virtual_path)
