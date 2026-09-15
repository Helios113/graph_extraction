#!/usr/bin/env python3
"""
Data remote storage tool -- DVC-style content-addressed storage over SFTP.

Remote layout:
  <remote_path>/
    cache/<hash[:2]>/<hash[2:]>  -- content-addressed data blobs
    index/<filename>.ptr          -- pointer files (source of truth for ls)

Local layout:
  .ptrs/<filename>.ptr            -- mirrors of pointer files, tracked by git

Commands:
  push <file|dir>  Hash, upload, write pointers, optionally delete local
  pull <name>      Download file described by .ptrs/<name>.ptr
  ls               Browse remote index interactively
"""

import configparser
from datetime import datetime, timezone
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from shutil import get_terminal_size

import paramiko
import xxhash


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(PROJECT_ROOT, "upload.cfg")
PTRS_DIR = os.path.join(PROJECT_ROOT, ".ptrs")
HETZNER_PORT = 23
DEFAULT_WORKERS = 4

_print_lock = threading.Lock()


def tprint(*args, **kwargs):
    with _print_lock:
        print(*args, flush=True, **kwargs)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config():
    cfg = configparser.ConfigParser()
    if not cfg.read(CONFIG_FILE):
        sys.exit(f"Config file not found: {CONFIG_FILE}")
    try:
        return {
            "host": cfg["remote"]["host"],
            "username": cfg["remote"]["username"],
            "password": cfg["remote"]["password"],
            "remote_path": cfg["remote"]["remote_path"].rstrip("/"),
            "workers": cfg["remote"].getint("workers", DEFAULT_WORKERS),
        }
    except KeyError as e:
        sys.exit(f"Missing config key: {e}")


# ---------------------------------------------------------------------------
# SFTP helpers
# ---------------------------------------------------------------------------

def connect(config):
    transport = paramiko.Transport((config["host"], HETZNER_PORT))
    # SSH-layer compression costs CPU for ~0 benefit on h5 shards (float32
    # arrays -- already high-entropy) and was measured as the dominant
    # bottleneck: ~6 MB/s with compression vs. ~36 MB/s without on this link.
    transport.use_compression(False)
    transport.connect(username=config["username"], password=config["password"])
    return transport


def open_sftp(config, max_retries=5):
    import random
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = (2 ** attempt) + random.uniform(0, 1)
                tprint(f"[sftp] connection attempt {attempt + 1}/{max_retries}, retrying in {delay:.1f}s...")
                time.sleep(delay)
            return paramiko.SFTPClient.from_transport(connect(config))
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            tprint(f"[sftp] connect failed ({e}), will retry...")


def sftp_mkdir_p(sftp, remote_dir):
    parts = [p for p in remote_dir.split("/") if p]
    if remote_dir.startswith("/"):
        paths = ["/" + "/".join(parts[:i+1]) for i in range(len(parts))]
    else:
        paths = ["/".join(parts[:i+1]) for i in range(len(parts))]
    for path in paths:
        try:
            sftp.stat(path)
        except FileNotFoundError:
            try:
                sftp.mkdir(path)
            except OSError:
                sftp.stat(path)  # only ok if another thread already created it


# ---------------------------------------------------------------------------
# Progress display
#
# n_slots lines are printed upfront. Each worker thread owns one slot (row).
# ANSI cursor movement rewrites that row in place — strictly one line per file.
# ---------------------------------------------------------------------------

_IS_TTY = sys.stdout.isatty()


class _Board:
    """ANSI in-place progress board for interactive terminals."""
    def __init__(self, n_slots):
        self.n_slots = n_slots
        self._tid_slot = {}
        self._next = 0
        sys.stdout.write("\n" * n_slots)
        sys.stdout.flush()

    def _slot(self):
        tid = threading.get_ident()
        with _print_lock:
            if tid not in self._tid_slot:
                self._tid_slot[tid] = self._next
                self._next += 1
        return self._tid_slot[tid]

    def _write(self, slot, text):
        rows_up = self.n_slots - slot
        width = get_terminal_size((80, 20)).columns
        line = f"\r{text}"[:width].ljust(width)
        with _print_lock:
            sys.stdout.write(f"\x1b[{rows_up}A{line}\x1b[{rows_up}B")
            sys.stdout.flush()

    def make_callback(self, label, total):
        label = label[:24]
        slot = self._slot()
        t0 = time.monotonic()

        def callback(transferred, _total):
            pct = transferred / total * 100 if total else 0
            elapsed = time.monotonic() - t0
            rate = transferred / elapsed if elapsed > 0.1 else 0
            remaining = (total - transferred) / rate if rate > 0 else 0
            eta = f"{int(remaining // 60)}m{int(remaining % 60):02d}s" if remaining > 0 else "--:--"
            width = get_terminal_size((80, 20)).columns
            bar_w = max(width - 46, 8)
            filled = int(bar_w * pct / 100)
            bar = "#" * filled + "-" * (bar_w - filled)
            self._write(slot, f"{label:<24} [{bar}] {pct:5.1f}% ETA {eta:>7}")

        return callback

    def finish(self, label, size):
        slot = self._tid_slot.get(threading.get_ident())
        if slot is None:
            return
        self._write(slot, f"{label:<24}  done  {size / 1_048_576:.1f} MB")


class _LogBoard:
    """Plain-text progress for non-TTY (SLURM logs). Prints at most once per 10%."""
    def make_callback(self, label, total):
        label = label[:40]
        last_pct = [-1]

        def callback(transferred, _total):
            pct = int(transferred / total * 100) if total else 0
            bucket = pct // 10 * 10
            if bucket > last_pct[0]:
                last_pct[0] = bucket
                tprint(f"[{label}] {bucket}%")

        return callback

    def finish(self, label, size):
        tprint(f"[{label[:40]}] done  {size / 1_048_576:.1f} MB")


_board: _Board | _LogBoard | None = None


def sftp_put(sftp, local_path, remote_path):
    """Upload to a temp path in the same remote directory, then atomically
    rename onto `remote_path`. Content-addressed blobs are checked for
    existence by `sftp.stat` alone (not hash-verified) elsewhere in this
    module, so a plain sftp.put that dies mid-transfer would otherwise leave
    a truncated file at the final path that looks "already uploaded" to
    every future push of the same content, forever. The temp+rename means a
    crash never leaves anything visible at `remote_path` at all -- the next
    attempt just re-uploads."""
    size = os.path.getsize(local_path)
    name = os.path.basename(local_path)
    tmp_remote = f"{remote_path}.tmp.{os.getpid()}.{threading.get_ident()}"
    try:
        sftp.put(local_path, tmp_remote, callback=_board.make_callback(name, size))
        sftp.posix_rename(tmp_remote, remote_path)
    except BaseException:
        # Best-effort cleanup only -- if the connection itself died (the
        # common case: network drop / process killed mid-transfer), a
        # cleanup call on the same dead sftp client can hang indefinitely
        # (paramiko sets no socket timeout by default), so cap the attempt.
        if sftp.get_channel() is not None and sftp.get_channel().active:
            try:
                sftp.remove(tmp_remote)
            except Exception:
                pass
        raise
    _board.finish(name, size)


def sftp_write_atomic(sftp, remote_path, data: bytes):
    """Write `data` to `remote_path` via temp-file-then-rename, so a crash
    mid-write never leaves truncated/corrupt content visible at the final
    path (unlike a plain sftp.open(remote_path, "w"))."""
    tmp_remote = f"{remote_path}.tmp.{os.getpid()}.{threading.get_ident()}"
    try:
        with sftp.open(tmp_remote, "w") as f:
            f.write(data)
        sftp.posix_rename(tmp_remote, remote_path)
    except BaseException:
        if sftp.get_channel() is not None and sftp.get_channel().active:
            try:
                sftp.remove(tmp_remote)
            except Exception:
                pass
        raise


def sftp_get(sftp, remote_path, local_path):
    size = sftp.stat(remote_path).st_size
    name = os.path.basename(local_path)
    sftp.get(remote_path, local_path, callback=_board.make_callback(name, size))
    _board.finish(name, size)


# ---------------------------------------------------------------------------
# Hashing  (xxh64: ~3-5x faster than MD5, sufficient for dedup/integrity)
# ---------------------------------------------------------------------------

def hash_file(path, chunk=1 << 23):
    h = xxhash.xxh3_128()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Pointer files
# ---------------------------------------------------------------------------

def ptr_name(filename):
    return filename if filename.endswith(".ptr") else filename + ".ptr"


def _ptr_path_for(local_path, virtual_path=None):
    rel = virtual_path if virtual_path is not None else os.path.relpath(local_path, PROJECT_ROOT)
    return os.path.join(PTRS_DIR, ptr_name(rel))


def _read_h5_meta(local_path):
    """Return the /meta attrs dict from an HDF5 file, or None if not HDF5."""
    if not local_path.endswith(".h5"):
        return None
    try:
        import h5py
        import numpy as np
        def _scalar(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            if hasattr(v, "item"):
                return v.item()
            return v
        with h5py.File(local_path, "r") as f:
            if "meta" not in f:
                return None
            return {k: _scalar(v) for k, v in f["meta"].attrs.items()}
    except Exception:
        return None


def write_ptr(local_path, digest, size, virtual_path=None):
    """`virtual_path` (project-relative, e.g. "onnx_models/foo/bar/shard_00000.h5")
    lets a file that physically lives outside PROJECT_ROOT (e.g. node-local
    scratch disk) register in the NFS-side .ptrs/ index and remote index at
    the path it would occupy if it were checked into the project -- so
    `pull`/`ls`/`clean` keep working against a consistent project-relative
    namespace regardless of where a shard was actually written."""
    rel = virtual_path if virtual_path is not None else os.path.relpath(local_path, PROJECT_ROOT)
    ptr_path = _ptr_path_for(local_path, virtual_path)
    os.makedirs(os.path.dirname(ptr_path), exist_ok=True)
    data = {
        "filename": os.path.basename(local_path),
        "hash": digest,
        "size": size,
        "mtime": os.path.getmtime(local_path),
        "uploaded_at": datetime.now(timezone.utc).timestamp(),
        "original_path": rel,
    }
    h5_meta = _read_h5_meta(local_path)
    if h5_meta is not None:
        data["h5_meta"] = h5_meta
    with open(ptr_path, "w") as f:
        json.dump(data, f, indent=2)
    return ptr_path, data


def load_ptr(local_path, virtual_path=None):
    """Return cached ptr data if mtime matches, else None."""
    ptr_path = _ptr_path_for(local_path, virtual_path)
    if not os.path.exists(ptr_path):
        return None
    with open(ptr_path) as f:
        data = json.load(f)
    if data.get("mtime") == os.path.getmtime(local_path):
        return data
    return None


def read_ptr(name):
    ptr_path = os.path.join(PTRS_DIR, ptr_name(name))
    if not os.path.exists(ptr_path):
        sys.exit(f"Pointer not found: {ptr_path}")
    with open(ptr_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Remote path helpers
# ---------------------------------------------------------------------------

def remote_cache_path(config, digest):
    return f"{config['remote_path']}/cache/{digest[:2]}/{digest[2:]}"


def remote_index_path(config, original_path):
    return f"{config['remote_path']}/index/{ptr_name(original_path)}"


# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------

OVERRIDE_FLAG = "_override"


def _mark_override(local_path, digest, size, virtual_path=None):
    """Write a ptr flagged for later force-push (used when conflict can't be resolved interactively)."""
    rel = virtual_path if virtual_path is not None else os.path.relpath(local_path, PROJECT_ROOT)
    ptr_path = _ptr_path_for(local_path, virtual_path)
    os.makedirs(os.path.dirname(ptr_path), exist_ok=True)
    data = {
        "filename": os.path.basename(local_path),
        "hash": digest,
        "size": size,
        "mtime": os.path.getmtime(local_path),
        "uploaded_at": None,
        "original_path": rel,
        OVERRIDE_FLAG: True,
    }
    with open(ptr_path, "w") as f:
        json.dump(data, f, indent=2)
    return ptr_path


def _push_file(local_path, config, force=False, virtual_path=None):
    """Run in a worker thread. Returns (local_path, ok: bool).

    `virtual_path`: see write_ptr -- the project-relative path to record in
    the pointer/remote index, when it differs from local_path's actual
    location relative to PROJECT_ROOT (e.g. a shard written to node-local
    scratch disk)."""
    import random
    name = os.path.basename(local_path)
    size = os.path.getsize(local_path)

    cached = load_ptr(local_path, virtual_path)
    if cached:
        digest = cached["hash"]
        tprint(f"[{name}] mtime unchanged, reusing hash {digest[:8]}…")
    else:
        tprint(f"[{name}] hashing ({size / 1_048_576:.1f} MB)...")
        digest = hash_file(local_path)
        tprint(f"[{name}] hash: {digest}")

    # stagger connection starts to avoid hammering the server
    time.sleep(random.uniform(0, 2))

    sftp = open_sftp(config)
    try:
        original_path = virtual_path if virtual_path is not None else os.path.relpath(local_path, PROJECT_ROOT)
        remote_idx = remote_index_path(config, original_path)
        sftp_mkdir_p(sftp, os.path.dirname(remote_idx))

        if not force:
            try:
                with sftp.open(remote_idx) as f:
                    raw = f.read()
                try:
                    existing = json.loads(raw)
                except json.JSONDecodeError:
                    # A prior push crashed mid-write to remote_idx (writing it
                    # isn't atomic pre-fix) and left truncated/corrupt JSON.
                    # Treat it like a missing pointer rather than crashing --
                    # this push's own write below will overwrite it cleanly.
                    tprint(f"[{name}] WARNING: remote index at {remote_idx!r} is corrupt "
                           f"(likely a prior interrupted push); overwriting.")
                    raise FileNotFoundError
                if existing["hash"] == digest:
                    tprint(f"[{name}] hash matches remote, skipping upload.")
                    write_ptr(local_path, digest, size, virtual_path)
                    return local_path, True
                else:
                    remote_mtime = existing.get("mtime")
                    local_mtime  = os.path.getmtime(local_path)
                    remote_ts = datetime.fromtimestamp(remote_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if remote_mtime else "unknown"
                    local_ts  = datetime.fromtimestamp(local_mtime,  timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                    with _print_lock:
                        print(f"\n[{name}] CONFLICT: remote and local have different content.")
                        print(f"  Remote version: {existing['hash'][:12]}…  modified {remote_ts}")
                        print(f"  Local  version: {digest[:12]}…  modified {local_ts}")
                        try:
                            ans1 = input("  Overwrite remote with local version? [y/N] ").strip().lower()
                            if ans1 != "y":
                                print("  Skipped.")
                                return local_path, False
                            ans2 = input(f"  Are you sure? This will replace the remote version from {remote_ts}. [y/N] ").strip().lower()
                            if ans2 != "y":
                                print("  Skipped.")
                                return local_path, False
                        except EOFError:
                            ptr_path = _mark_override(local_path, digest, size, virtual_path)
                            print(f"  Non-interactive: marked for later override ({ptr_path})")
                            return local_path, False
            except FileNotFoundError:
                pass  # new file, no conflict

        # upload blob (content-addressed — different hash = new blob, old one preserved)
        cache_remote = remote_cache_path(config, digest)
        sftp_mkdir_p(sftp, os.path.dirname(cache_remote))
        try:
            sftp.stat(cache_remote)
            tprint(f"[{name}] blob exists in cache, skipping upload.")
        except FileNotFoundError:
            sftp_put(sftp, local_path, cache_remote)

        # write pointer locally (clears any override flag)
        ptr_path, ptr_data = write_ptr(local_path, digest, size, virtual_path)
        tprint(f"[{name}] pointer: {ptr_path}")

        # mirror pointer to remote index
        sftp_write_atomic(sftp, remote_idx, json.dumps(ptr_data, indent=2).encode())

    finally:
        sftp.close()

    return local_path, True


def cmd_push(local_path, config, auto_delete=False, virtual_path=None):
    """`virtual_path` (single-file push only -- ignored for a directory,
    since a dir push derives each file's project-relative path by walking)
    lets `local_path` be a file outside PROJECT_ROOT that should register in
    the pointer/remote index at the given project-relative path instead."""
    if not os.path.exists(local_path):
        sys.exit(f"Path does not exist: {local_path}")

    is_dir = os.path.isdir(local_path)
    if is_dir and virtual_path is not None:
        sys.exit("virtual_path is only supported when pushing a single file, not a directory")
    files = (
        [os.path.join(dp, f) for dp, _, fnames in os.walk(local_path)
         for f in fnames if not f.startswith("._") and not f.startswith(".")]
        if is_dir else [local_path]
    )
    if not files:
        sys.exit("Directory is empty (or only hidden files).")

    workers = min(config["workers"], len(files))
    print(f"Starting {workers} parallel worker(s)...")

    global _board
    _board = _Board(workers) if _IS_TTY else _LogBoard()

    uploaded, skipped = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_push_file, f, config, virtual_path=virtual_path): f for f in files}
        for fut in as_completed(futures):
            path, ok = fut.result()
            (uploaded if ok else skipped).append(path)

    print(f"\nPush complete: {len(uploaded)} uploaded, {len(skipped)} skipped.")

    if uploaded:
        if auto_delete:
            ans = "y"
        elif not sys.stdin.isatty():
            print(f"Not deleting {len(uploaded)} local file(s): stdin is not a TTY "
                  f"(pass auto_delete=True to delete without prompting).")
            ans = "n"
        else:
            ans = input(f"Delete {len(uploaded)} local file(s)? [y/N] ").strip().lower()
        if ans == "y":
            for f in uploaded:
                os.remove(f)
            if is_dir:
                for dp, _, _ in os.walk(local_path, topdown=False):
                    try:
                        os.rmdir(dp)
                    except OSError:
                        pass
            print("Deleted.")


# ---------------------------------------------------------------------------
# Apply overrides
# ---------------------------------------------------------------------------

def cmd_apply_overrides(config):
    if not os.path.exists(PTRS_DIR):
        sys.exit("No .ptrs/ directory found.")

    pending = []
    for dirpath, _, fnames in os.walk(PTRS_DIR):
        for fname in fnames:
            if not fname.endswith(".ptr"):
                continue
            ptr_path = os.path.join(dirpath, fname)
            with open(ptr_path) as f:
                ptr = json.load(f)
            if ptr.get(OVERRIDE_FLAG):
                local = os.path.join(PROJECT_ROOT, ptr["original_path"])
                pending.append((ptr_path, ptr, local))

    if not pending:
        print("No pending overrides found.")
        return

    print(f"Pending overrides ({len(pending)}):")
    for _, ptr, local in pending:
        exists = "present" if os.path.exists(local) else "MISSING locally"
        print(f"  {ptr['original_path']}  ({exists})")

    ans1 = input("\nForce-push all and overwrite remote versions? [y/N] ").strip().lower()
    if ans1 != "y":
        print("Aborted.")
        return
    ans2 = input("Are you sure? Remote versions will be permanently replaced. [y/N] ").strip().lower()
    if ans2 != "y":
        print("Aborted.")
        return

    workers = min(config["workers"], len(pending))
    global _board
    _board = _Board(workers) if _IS_TTY else _LogBoard()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_push_file, local, config, force=True): (ptr_path, ptr, local)
                   for ptr_path, ptr, local in pending if os.path.exists(local)}
        for fut in as_completed(futures):
            ptr_path, ptr, local = futures[fut]
            _, ok = fut.result()
            if ok:
                print(f"  Override applied: {ptr['original_path']}")
            else:
                print(f"  Failed: {ptr['original_path']}")

    missing = [(ptr_path, ptr, local) for ptr_path, ptr, local in pending if not os.path.exists(local)]
    for ptr_path, ptr, local in missing:
        print(f"  Skipped (local file missing): {ptr['original_path']}")

    print("Done.")


# ---------------------------------------------------------------------------
# Pull
# ---------------------------------------------------------------------------

def _pull_file(ptr, config):
    """Run in a worker thread."""
    import random
    digest = ptr["hash"]
    dest = os.path.join(PROJECT_ROOT, ptr["original_path"])
    name = ptr["filename"]

    if os.path.exists(dest):
        if hash_file(dest) == digest:
            tprint(f"[{name}] already present, skipping.")
            return dest, False
        tprint(f"[{name}] WARNING: local file differs, overwriting.")

    # stagger connection starts to avoid hammering the server
    time.sleep(random.uniform(0, 2))

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    sftp = open_sftp(config)
    try:
        sftp_get(sftp, remote_cache_path(config, digest), dest)
    finally:
        sftp.close()

    # update local pointer so mtime cache is valid on next push
    write_ptr(dest, digest, ptr["size"])

    return dest, True


def cmd_pull(name, config):
    global _board

    if name is None:
        # no name given — sync all pointers from remote index into .ptrs/
        print("Fetching remote index...")
        ptrs = _fetch_ptrs(config)
        if not ptrs:
            print("Remote index is empty.")
            return
        new, updated = 0, 0
        for p in ptrs:
            ptr_path = _ptr_path_for(os.path.join(PROJECT_ROOT, p["original_path"]))
            existed = os.path.exists(ptr_path)
            os.makedirs(os.path.dirname(ptr_path), exist_ok=True)
            with open(ptr_path, "w") as f:
                json.dump(p, f, indent=2)
            if existed:
                updated += 1
            else:
                new += 1
        print(f"Synced {len(ptrs)} pointer(s) to .ptrs/  ({new} new, {updated} updated).")
        return

    ptr = read_ptr(name)
    print(f"Downloading {ptr['filename']} ({ptr['size'] / 1_048_576:.1f} MB)...")
    _board = _Board(1) if _IS_TTY else _LogBoard()
    dest, pulled = _pull_file(ptr, config)
    if pulled:
        print(f"\nPulled to: {dest}")


# ---------------------------------------------------------------------------
# meta -- print /meta attrs stored in a pointer file
# ---------------------------------------------------------------------------

def cmd_meta(name, config):
    """Print h5_meta stored in the local ptr, fetching from remote index if absent."""
    ptr_path = os.path.join(PTRS_DIR, ptr_name(name))
    if os.path.exists(ptr_path):
        with open(ptr_path) as f:
            ptr = json.load(f)
    else:
        # not in local .ptrs/ — fetch from remote index
        print(f"Local ptr not found, checking remote index for {name!r}...")
        sftp = open_sftp(config)
        try:
            remote_idx = remote_index_path(config, name)
            with sftp.open(remote_idx) as f:
                ptr = json.load(f)
        except FileNotFoundError:
            sys.exit(f"Not found locally or remotely: {name!r}")
        finally:
            sftp.close()

    meta = ptr.get("h5_meta")
    if meta is None:
        print(f"No h5_meta in pointer for {ptr.get('original_path', name)!r}.")
        print("Re-push the file to embed metadata in the pointer.")
        return

    print(f"h5_meta for {ptr['original_path']}  ({ptr['size'] / 1_048_576:.1f} MB):")
    for k, v in sorted(meta.items()):
        print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# ls -- interactive directory browser
# ---------------------------------------------------------------------------

def _fetch_ptrs(config):
    sftp = open_sftp(config)
    try:
        index_dir = f"{config['remote_path']}/index"
        ptrs = []

        def _walk(remote_dir):
            try:
                entries = sftp.listdir_attr(remote_dir)
            except FileNotFoundError:
                return
            import stat as stat_mod
            for entry in sorted(entries, key=lambda e: e.filename):
                remote_path = f"{remote_dir}/{entry.filename}"
                if stat_mod.S_ISDIR(entry.st_mode):
                    _walk(remote_path)
                elif entry.filename.endswith(".ptr"):
                    with sftp.open(remote_path) as f:
                        raw = f.read()
                    if b"Mac OS X" in raw:
                        try:
                            sftp.remove(remote_path)
                            print(f"  [info] removed macOS metadata ptr: {remote_path!r}")
                        except OSError as e:
                            print(f"  [warn] failed to remove macOS metadata ptr {remote_path!r}: {e}")
                        continue
                    if not raw.strip():
                        continue
                    try:
                        ptrs.append(json.loads(raw))
                    except json.JSONDecodeError:
                        print(f"  [warn] skipping corrupt ptr: {remote_path!r} contents: {raw!r}")

        _walk(index_dir)
        return ptrs
    finally:
        sftp.close()


def _build_tree(ptrs):
    """Map each directory path -> {subdirs: set, files: [ptr]}."""
    tree = {}
    for p in ptrs:
        parts = p["original_path"].replace("\\", "/").split("/")
        for depth in range(len(parts)):
            dir_path = "/".join(parts[:depth]) or "."
            if dir_path not in tree:
                tree[dir_path] = {"subdirs": set(), "files": []}
            if depth < len(parts) - 1:
                child = "/".join(parts[:depth + 1])
                tree[dir_path]["subdirs"].add(child)
            else:
                tree[dir_path]["files"].append(p)
    if "." not in tree:
        tree["."] = {"subdirs": set(), "files": []}
    return tree


def cmd_ls(config):
    from browser import browse

    print("Fetching remote index...")
    ptrs = _fetch_ptrs(config)
    if not ptrs:
        print("Remote index is empty (no files pushed yet).")
        return

    tree = _build_tree(ptrs)
    to_pull, to_delete = browse(tree)

    if to_delete:
        print(f"\nFiles marked for remote deletion ({len(to_delete)}):")
        for p in to_delete:
            print(f"  {p['original_path']}  ({p['size'] / 1_048_576:.1f} MB)")
        ans1 = input("\nPermanently delete these from remote? [y/N] ").strip().lower()
        if ans1 != "y":
            print("Aborted.")
            return
        ans2 = input("Are you sure? This cannot be undone. [y/N] ").strip().lower()
        if ans2 != "y":
            print("Aborted.")
            return
        sftp = open_sftp(config)
        try:
            for p in to_delete:
                cache_path = remote_cache_path(config, p["hash"])
                idx_path   = remote_index_path(config, p["original_path"])
                for rpath in (cache_path, idx_path):
                    try:
                        sftp.remove(rpath)
                        print(f"  Deleted {rpath}")
                    except FileNotFoundError:
                        print(f"  Already gone: {rpath}")
                local_ptr = _ptr_path_for(os.path.join(PROJECT_ROOT, p["original_path"]))
                if os.path.exists(local_ptr):
                    os.remove(local_ptr)
                    print(f"  Removed local ptr {local_ptr}")
        finally:
            sftp.close()
        print("Deletion complete.")
        return

    if not to_pull:
        return

    for p in to_pull:
        ptr_path = _ptr_path_for(os.path.join(PROJECT_ROOT, p["original_path"]))
        os.makedirs(os.path.dirname(ptr_path), exist_ok=True)
        with open(ptr_path, "w") as f:
            json.dump(p, f, indent=2)

    workers = min(config["workers"], len(to_pull))
    print(f"Pulling {len(to_pull)} file(s) with {workers} worker(s)...")

    global _board
    _board = _Board(workers) if _IS_TTY else _LogBoard()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_pull_file, p, config) for p in to_pull]
        for fut in as_completed(futures):
            dest, pulled = fut.result()
            if pulled:
                tprint(f"Pulled: {dest}")

    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

USAGE = """Usage:
  upload.py push [-y] <file|dir>   Upload, write pointers, optionally delete local (-y auto-deletes)
  upload.py pull [<name>]          Download file described by .ptrs/<name>[.ptr], or sync all pointers from remote
  upload.py meta <name>            Print /meta attrs stored in the pointer for an H5 file
  upload.py ls                     Browse remote index, pull or delete selected files/folders
  upload.py clean                  Remove local files that are safely backed up remotely
  upload.py apply-overrides        Force-push files marked as conflicting during non-interactive push
  upload.py purge                  Delete remote cache blobs not referenced by any index pointer
"""


def cmd_clean(config):
    if not os.path.exists(PTRS_DIR):
        sys.exit("No .ptrs/ directory found.")

    ptr_files = [f for f in os.listdir(PTRS_DIR) if f.endswith(".ptr")]
    if not ptr_files:
        print("No pointers found.")
        return

    # fetch remote index once
    print("Fetching remote index...")
    remote_ptrs = {p["filename"]: p for p in _fetch_ptrs(config)}

    to_delete = []
    for fname in sorted(ptr_files):
        with open(os.path.join(PTRS_DIR, fname)) as f:
            ptr = json.load(f)

        name     = ptr["filename"]
        local    = os.path.join(PROJECT_ROOT, ptr["original_path"])
        local_hash  = ptr["hash"]
        local_mtime = ptr.get("mtime")

        if not os.path.exists(local):
            continue  # already gone

        # verify pointer is fresh (mtime matches local file)
        actual_mtime = os.path.getmtime(local)
        if local_mtime is None or abs(actual_mtime - local_mtime) > 1:
            local_ts = datetime.fromtimestamp(actual_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            ptr_ts   = datetime.fromtimestamp(local_mtime,  timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if local_mtime else "unknown"
            print(f"  SKIP {name}: local file mtime ({local_ts}) doesn't match pointer ({ptr_ts}) — run push first.")
            continue

        # verify remote has the same hash
        remote = remote_ptrs.get(name)
        if not remote:
            print(f"  SKIP {name}: not found in remote index.")
            continue

        if remote["hash"] != local_hash:
            remote_ts = datetime.fromtimestamp(remote.get("mtime", 0), timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            local_ts  = datetime.fromtimestamp(local_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            print(f"  SKIP {name}: hash mismatch — local ({local_hash[:12]}… {local_ts}) vs remote ({remote['hash'][:12]}… {remote_ts}).")
            continue

        size_mb = ptr["size"] / 1_048_576
        mtime_ts = datetime.fromtimestamp(local_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        to_delete.append((local, name, size_mb, mtime_ts))

    if not to_delete:
        print("Nothing to clean — no locally present files are verified on remote.")
        return

    print(f"\nFiles safe to delete ({len(to_delete)}):")
    total_mb = 0.0
    for local, name, size_mb, mtime_ts in to_delete:
        print(f"  {name:<44} {size_mb:8.1f} MB   backed up {mtime_ts}")
        total_mb += size_mb
    print(f"\n  Total: {total_mb:.1f} MB")

    ans = input("\nDelete all? [y/N] ").strip().lower()
    if ans != "y":
        print("Aborted.")
        return

    for local, _, _, _ in to_delete:
        os.remove(local)
        print(f"  Deleted {local}")

    print("Done.")


def cmd_purge(config):
    """Delete remote cache blobs not referenced by any index pointer."""
    print("Fetching remote index...")
    ptrs = _fetch_ptrs(config)
    referenced = {p["hash"] for p in ptrs}
    print(f"  {len(referenced)} hashes referenced by index.")

    sftp = open_sftp(config)
    try:
        cache_root = f"{config['remote_path']}/cache"
        try:
            prefix_dirs = sftp.listdir(cache_root)
        except FileNotFoundError:
            print("Cache directory does not exist — nothing to purge.")
            return

        orphans = []
        for prefix in sorted(prefix_dirs):
            prefix_path = f"{cache_root}/{prefix}"
            try:
                blobs = sftp.listdir(prefix_path)
            except IOError:
                continue
            for blob in blobs:
                full_hash = prefix + blob
                if full_hash not in referenced:
                    blob_path = f"{prefix_path}/{blob}"
                    size = sftp.stat(blob_path).st_size
                    orphans.append((blob_path, full_hash, size))

        if not orphans:
            print("No orphaned blobs found.")
            return

        total_mb = sum(s for _, _, s in orphans) / 1_048_576
        print(f"\nOrphaned blobs ({len(orphans)}, {total_mb:.1f} MB total):")
        for path, h, size in orphans:
            print(f"  {h}  {size / 1_048_576:8.1f} MB  {path}")

        ans1 = input("\nDelete all orphaned blobs? [y/N] ").strip().lower()
        if ans1 != "y":
            print("Aborted.")
            return
        ans2 = input("Are you sure? This cannot be undone. [y/N] ").strip().lower()
        if ans2 != "y":
            print("Aborted.")
            return

        for path, h, _ in orphans:
            sftp.remove(path)
            print(f"  Deleted {path}")
        print(f"Purge complete: {len(orphans)} blobs removed ({total_mb:.1f} MB freed).")
    finally:
        sftp.close()


def main():
    if len(sys.argv) < 2:
        sys.exit(USAGE)

    cmd = sys.argv[1]
    config = load_config()

    if cmd == "push":
        if len(sys.argv) < 3:
            sys.exit("Usage: upload.py push [-y] <file|dir>")
        args = sys.argv[2:]
        auto_delete = "-y" in args
        args = [a for a in args if a != "-y"]
        if not args:
            sys.exit("Usage: upload.py push [-y] <file|dir>")
        cmd_push(args[0], config, auto_delete=auto_delete)

    elif cmd == "pull":
        if len(sys.argv) > 3:
            sys.exit("Usage: upload.py pull [<name>]")
        cmd_pull(sys.argv[2] if len(sys.argv) == 3 else None, config)

    elif cmd == "meta":
        if len(sys.argv) != 3:
            sys.exit("Usage: upload.py meta <name>")
        cmd_meta(sys.argv[2], config)

    elif cmd == "ls":
        cmd_ls(config)

    elif cmd == "clean":
        cmd_clean(config)

    elif cmd == "apply-overrides":
        cmd_apply_overrides(config)

    elif cmd == "purge":
        cmd_purge(config)

    else:
        sys.exit(USAGE)


if __name__ == "__main__":
    main()
