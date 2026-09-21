"""Frees space on the current node's local scratch by deleting shard files
that ShardedLayerWriter has already confirmed pushed (either backend --
Hetzner or NFS/mauao, see execute/node_relay.py) and re-verifying the pushed
copy is actually still there before removing the local one. Never touches an
unpushed/in-progress shard.

Run ON the node whose scratch you want reclaimed (ruapehu or mauao) --
scans NFS for every layer's `.progress.json` under onnx_models/, and for
each shard marked `pushed: true`, checks whether a local copy still exists
under $NEW_JAC_SCRATCH. If so, verifies the shard's backing copy (Hetzner
.ptr, or a file under execute.node_relay.STAGING_ROOT) is actually present,
then deletes the local file.

Usage:
  uv run python -m scripts.helper.reclaim_pushed_shards [--dry-run]
      [--target-free-gb 400] [--backend hetzner|nfs|both]

Without --target-free-gb, reclaims every eligible shard found. With it,
stops early once /local/scratch has that much free (oldest-pushed-first --
see --sort), so a routine sweep doesn't delete more than needed.
"""

import argparse
import glob
import json
import os
import shutil

REPO_ROOT = "/nfs-share/pa511/code_bases/new_jac"


def _scratch_root() -> str:
    root = os.environ.get("NEW_JAC_SCRATCH")
    if root:
        return root
    import getpass
    return os.path.join("/local/scratch", getpass.getuser(), "new_jac")


def _free_gb(path: str) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / 1024**3


def _hetzner_ptr_exists(virtual_path: str) -> bool:
    ptr_path = os.path.join(REPO_ROOT, ".ptrs", virtual_path + ".ptr")
    return os.path.exists(ptr_path)


def _nfs_staged_exists(virtual_path: str) -> bool:
    from execute.node_relay import STAGING_ROOT
    return os.path.exists(os.path.join(STAGING_ROOT, virtual_path))


def _find_progress_files() -> list[str]:
    return sorted(glob.glob(os.path.join(REPO_ROOT, "onnx_models", "**", "*.progress.json"), recursive=True))


def _eligible_shards(backend_filter: str) -> list[dict]:
    """Every (progress_path, shard_entry, local_path, mtime) for a shard
    that's marked pushed and still present on local scratch, sorted oldest
    (by local mtime) first -- so a capped sweep reclaims the
    longest-untouched data first."""
    scratch_root = _scratch_root()
    candidates = []

    for progress_path in _find_progress_files():
        with open(progress_path) as f:
            progress = json.load(f)
        layer_key = progress["layer_key"]

        for shard in progress.get("completed_shards", []):
            if not shard.get("pushed"):
                continue
            shard_idx = shard["shard_idx"]
            local_path = os.path.join(scratch_root, layer_key, f"shard_{shard_idx:05d}.h5")
            if not os.path.exists(local_path):
                continue

            virtual_path = shard["virtual_path"]
            backed_by = []
            if _hetzner_ptr_exists(virtual_path):
                backed_by.append("hetzner")
            if _nfs_staged_exists(virtual_path):
                backed_by.append("nfs")

            if backend_filter != "both" and backend_filter not in backed_by:
                continue
            if not backed_by:
                # pushed=True in progress.json but neither backend actually has
                # a copy right now (e.g. scratch_staging/ was cleaned up
                # separately) -- NOT safe to delete, skip it.
                continue

            candidates.append({
                "progress_path": progress_path,
                "layer_key": layer_key,
                "shard_idx": shard_idx,
                "local_path": local_path,
                "virtual_path": virtual_path,
                "backed_by": backed_by,
                "size_bytes": os.path.getsize(local_path),
                "mtime": os.path.getmtime(local_path),
            })

    candidates.sort(key=lambda c: c["mtime"])
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="List what would be deleted without deleting.")
    parser.add_argument("--target-free-gb", type=float, default=None,
                         help="Stop once local scratch has at least this much free (oldest-pushed shards first). Default: reclaim everything eligible.")
    parser.add_argument("--backend", choices=["hetzner", "nfs", "both"], default="both",
                         help="Only reclaim shards backed by this transport (default: either).")
    args = parser.parse_args()

    scratch_root = _scratch_root()
    candidates = _eligible_shards(args.backend)

    if not candidates:
        print("No eligible (pushed + locally present + backing-verified) shards found.")
        return

    total_bytes = sum(c["size_bytes"] for c in candidates)
    print(f"Found {len(candidates)} eligible shard(s), {total_bytes / 1024**3:.1f} GB total, "
          f"on {scratch_root}.")

    freed_bytes = 0
    deleted = 0
    for c in candidates:
        if args.target_free_gb is not None:
            current_free = _free_gb(scratch_root)
            if current_free >= args.target_free_gb:
                print(f"Reached target free space ({current_free:.1f} GB >= {args.target_free_gb} GB), stopping.")
                break

        size_gb = c["size_bytes"] / 1024**3
        backed = "+".join(c["backed_by"])
        if args.dry_run:
            print(f"  [dry-run] would delete {c['local_path']} ({size_gb:.2f} GB, backed by {backed})")
        else:
            os.remove(c["local_path"])
            print(f"  deleted {c['local_path']} ({size_gb:.2f} GB, backed by {backed})")
            freed_bytes += c["size_bytes"]
            deleted += 1

    if args.dry_run:
        print(f"\nDry run: {len(candidates)} shard(s), {total_bytes / 1024**3:.1f} GB would be reclaimed.")
    else:
        print(f"\nReclaimed {deleted} shard(s), {freed_bytes / 1024**3:.1f} GB freed. "
              f"Free space now: {_free_gb(scratch_root):.1f} GB.")


if __name__ == "__main__":
    main()
