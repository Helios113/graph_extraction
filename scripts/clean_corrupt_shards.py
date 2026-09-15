"""Wipe node-local scratch of all in-progress shard state left behind by
ShardedLayerWriter (see execute/sharded_writer.py) -- both corrupt and valid
files alike. Use this when you want a hard reset (e.g. after a desync between
layers' next_row values leaves in-progress shards you don't trust) rather
than relying on ShardedLayerWriter's own corrupt-shard auto-recovery, which
only discards files that actually fail to open.

Only ever touches local scratch (never NFS -- shard files are only ever
written to local scratch, per CLAUDE.md). Removing a layer's in-progress
shard just means that layer's writer starts over from its last *completed*
shard on next run (completed_shards/progress.json live on NFS and are
untouched by this script) -- no already-uploaded data is lost.

Usage:
    uv run scripts/clean_corrupt_shards.py                # wipe default scratch root
    uv run scripts/clean_corrupt_shards.py --dry-run
    uv run scripts/clean_corrupt_shards.py --scratch-root /local/scratch/pa511/new_jac
    uv run scripts/clean_corrupt_shards.py --only-corrupt  # old behavior: corrupt files only
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execute.sharded_writer import _scratch_root  # noqa: E402


def find_shard_files(scratch_root: str) -> list[str]:
    pattern = os.path.join(scratch_root, "**", "shard_*.h5")
    return sorted(glob.glob(pattern, recursive=True))


def is_corrupt(path: str) -> str | None:
    """Returns an error message if `path` fails to open as HDF5, else None."""
    try:
        with h5py.File(path, "r"):
            pass
    except OSError as e:
        return str(e)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--scratch-root", default=None,
        help="Root to wipe (default: $NEW_JAC_SCRATCH or /local/scratch/<user>/new_jac)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be deleted without deleting it",
    )
    parser.add_argument(
        "--only-corrupt", action="store_true",
        help="Only delete shards that fail to open (old, conservative behavior) "
             "instead of removing every in-progress shard",
    )
    args = parser.parse_args()

    scratch_root = args.scratch_root or _scratch_root()
    if not os.path.isdir(scratch_root):
        print(f"scratch root {scratch_root!r} does not exist -- nothing to clean")
        return

    shard_paths = find_shard_files(scratch_root)
    print(f"scanning {len(shard_paths)} shard file(s) under {scratch_root}")

    n_removed = 0
    n_bytes_freed = 0
    for path in shard_paths:
        if args.only_corrupt:
            err = is_corrupt(path)
            if err is None:
                continue
            reason = err
        else:
            reason = "removing (--only-corrupt not set)"

        n_removed += 1
        size = os.path.getsize(path)
        if args.dry_run:
            print(f"[would delete] {path} ({size} bytes) -- {reason}")
        else:
            os.remove(path)
            n_bytes_freed += size
            print(f"[deleted] {path} ({size} bytes) -- {reason}")

    verb = "found" if args.dry_run else "deleted"
    print(f"{verb} {n_removed} shard file(s)"
          + ("" if args.dry_run else f", freed {n_bytes_freed / 1e6:.1f} MB"))


if __name__ == "__main__":
    main()
