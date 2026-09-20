#!/usr/bin/env python3
"""
Stress test for the upload.py / execute.remote_storage push-pull cycle.

Targets an ISOLATED remote subpath (config["remote_path"] + "/_stress_test")
on the real Hetzner box so we exercise real SFTP/network semantics without
touching real project data under iclr/cache, iclr/index.

What it does:
  1. Generates synthetic h5 files of varying sizes (small -> multi-chunk).
  2. Fault-injects mid-transfer aborts (simulating a crash/network drop)
     during sftp.put and sftp.get, via a wrapped progress callback that
     kills the underlying socket at a random point.
  3. After each simulated crash, checks:
       - Does a plain retry (re-running push/pull) succeed and produce a
         file whose hash matches the original?
       - Is there a corrupted/truncated blob left at the content-addressed
         remote cache path that a *future* push of identical content would
         mistake for "already uploaded" (existence-only check)?
       - Does a corrupt/truncated remote index pointer break cmd_meta /
         _fetch_ptrs / conflict-check on push, or degrade gracefully?
  4. Runs many iterations across file sizes and both push+pull directions,
     plus a concurrent multi-worker push interruption test.
  5. Cleans up the _stress_test remote subpath at the end (best-effort).

Usage:
  uv run python scripts/stress_test_remote_storage.py [--iterations N] [--keep-remote]

Run on a compute node with a writable node-local scratch disk (this repo's
convention is /local/scratch/<user>/new_jac, see execute/sharded_writer.py),
e.g. via SLURM: `srun --partition=interactive --nodelist=mauao ... uv run
python scripts/stress_test_remote_storage.py`. Most login/dev nodes don't
have a writable /local/scratch.

Note for anyone extending this: upload.sftp_put/sftp_write_atomic build
their own progress callback internally and don't accept an external one, so
fault-injecting through them (rather than by calling raw sftp.put/sftp.open
directly, as phases 1/3/4 below do) requires monkeypatching
upload._board.make_callback to splice the kill into whatever callback it
returns, e.g.:

    orig = upload._board.make_callback
    def spliced(label, total):
        inner = orig(label, total)
        def cb(transferred, tot):
            killer(transferred, tot)  # may raise, aborting the transfer
            inner(transferred, tot)
        return cb
    upload._board.make_callback = spliced
"""

import argparse
import getpass
import json
import os
import random
import shutil
import socket
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = "/nfs-share/pa511/code_bases/new_jac"
sys.path.insert(0, PROJECT_ROOT)

import h5py
import numpy as np
import paramiko

import upload


def _scratch_root() -> str:
    """Node-local scratch dir, same convention as execute/sharded_writer.py's
    _scratch_root(): /local/scratch/<user>/new_jac, overridable via
    $NEW_JAC_SCRATCH. Files here live outside PROJECT_ROOT, so they're
    registered with upload.py via `virtual_path` (see cmd_push calls below)
    rather than relying on os.path.relpath(PROJECT_ROOT)."""
    root = os.environ.get("NEW_JAC_SCRATCH")
    if root:
        return root
    return os.path.join("/local/scratch", getpass.getuser(), "new_jac")


class Fault(Exception):
    """Raised inside a fault-injected callback to simulate a hard interrupt."""


class Result:
    def __init__(self):
        self.cases = []  # list of dict

    def record(self, name, ok, detail=""):
        self.cases.append({"name": name, "ok": ok, "detail": detail})
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))

    def summary(self):
        n = len(self.cases)
        failed = [c for c in self.cases if not c["ok"]]
        print("\n" + "=" * 70)
        print(f"SUMMARY: {n - len(failed)}/{n} passed")
        if failed:
            print("FAILURES:")
            for c in failed:
                print(f"  - {c['name']}: {c['detail']}")
        print("=" * 70)
        return len(failed) == 0


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def make_h5(path, n_rows, seed):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=rng.standard_normal((n_rows, 256), dtype=np.float32))
        meta = f.create_group("meta")
        meta.attrs["n_rows"] = n_rows
        meta.attrs["seed"] = seed
    return path


# ---------------------------------------------------------------------------
# Fault injection: kill the transport mid-transfer at a random point
# ---------------------------------------------------------------------------

class TransferKiller:
    """Wraps a paramiko callback; raises Fault once cumulative bytes
    transferred crosses a randomly chosen fraction of total size, and closes
    the underlying transport socket to simulate a hard connection drop
    (not a clean cancellation)."""

    def __init__(self, transport, min_frac=0.1, max_frac=0.9, rng=None):
        self.transport = transport
        self.frac = (rng or random).uniform(min_frac, max_frac)
        self.fired = False

    def __call__(self, transferred, total):
        if self.fired:
            return
        if total and transferred / total >= self.frac:
            self.fired = True
            try:
                sock = self.transport.sock
                if sock is not None:
                    sock.close()
            except Exception:
                pass
            try:
                self.transport.close()
            except Exception:
                pass
            raise Fault(f"simulated crash at {transferred}/{total} bytes ({self.frac:.0%} target)")


def crash_during_put(sftp, transport, local_path, remote_path, rng=None):
    size = os.path.getsize(local_path)
    killer = TransferKiller(transport, rng=rng)
    try:
        sftp.put(local_path, remote_path, callback=killer)
        return False  # never fired -- file too small / race, transferred cleanly
    except (Fault, EOFError, OSError, paramiko.SSHException):
        return True


def crash_during_get(sftp, transport, remote_path, local_path, rng=None):
    size = sftp.stat(remote_path).st_size
    killer = TransferKiller(transport, rng=rng)
    try:
        sftp.get(remote_path, local_path, callback=killer)
        return False
    except (Fault, EOFError, OSError, paramiko.SSHException):
        return True


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def fresh_sftp(config, max_retries=5):
    """Like upload.open_sftp, but also hands back the transport (the stress
    test needs it to kill the connection mid-transfer). Hetzner's SFTP
    occasionally drops the negotiation (EOFError/SSHException) under the
    connection churn this harness generates -- retry instead of dying."""
    import random as _random
    last_exc = None
    for attempt in range(max_retries):
        if attempt > 0:
            time.sleep((2 ** attempt) + _random.uniform(0, 1))
        try:
            transport = upload.connect(config)
            sftp = paramiko.SFTPClient.from_transport(transport)
            return sftp, transport
        except Exception as e:
            last_exc = e
    raise last_exc


def remote_exists(sftp, path):
    try:
        sftp.stat(path)
        return True
    except FileNotFoundError:
        return False


def remote_read_bytes(sftp, path):
    with sftp.open(path) as f:
        return f.read()


def test_put_interrupt_leaves_no_valid_blob_masquerading(stress_config, tmpdir, result, rng, n_rows=4000):
    """Core fear: an interrupted push leaves a *partial* blob at the
    content-addressed cache path, and a later push of the same content
    sees `sftp.stat(cache_remote)` succeed and skips upload -- permanent
    silent corruption, since existence is checked, not content."""
    name = "put_interrupt_leaves_no_valid_blob_masquerading"
    local_path = os.path.join(tmpdir, "crash_put.h5")
    make_h5(local_path, n_rows, seed=1)
    digest = upload.hash_file(local_path)
    size = os.path.getsize(local_path)
    cache_remote = upload.remote_cache_path(stress_config, digest)

    sftp, transport = fresh_sftp(stress_config)
    try:
        upload.sftp_mkdir_p(sftp, os.path.dirname(cache_remote))
        crashed = crash_during_put(sftp, transport, local_path, cache_remote, rng=rng)
    finally:
        try:
            sftp.close()
        except Exception:
            pass

    if not crashed:
        result.record(name, True, "transfer completed before fault fired (file too small at chosen frac); skipping")
        # clean up any partial artifact
        try:
            sftp2, _ = fresh_sftp(stress_config)
            if remote_exists(sftp2, cache_remote):
                sftp2.remove(cache_remote)
            sftp2.close()
        except Exception:
            pass
        return

    # Reconnect fresh (old transport is dead) and inspect remote state.
    sftp2, transport2 = fresh_sftp(stress_config)
    try:
        exists = remote_exists(sftp2, cache_remote)
        if not exists:
            result.record(name, True, "no partial blob left after crash (paramiko rolled back or never flushed)")
            return

        remote_size = sftp2.stat(cache_remote).st_size
        if remote_size == size:
            # Rare: crash fired right at the tail after full bytes were already
            # sent but before callback/ack; verify content hash too.
            data = remote_read_bytes(sftp2, cache_remote)
            import xxhash
            actual = xxhash.xxh3_128(data).hexdigest()
            if actual == digest:
                result.record(name, True, "partial blob present but happens to be byte-complete and hash-valid")
                return
            result.record(name, False,
                          f"CORRUPTION: remote blob has correct size ({remote_size}B) but WRONG HASH "
                          f"({actual[:12]} != {digest[:12]}) -- would be silently treated as valid on future push/pull")
            return

        # This is the dangerous case: partial blob, wrong size.
        # Does the *existence check* in _push_file treat this as "already uploaded"?
        # Simulate exactly what _push_file does at cache-exists-check time.
        try:
            sftp2.stat(cache_remote)
            masquerades_as_uploaded = True
        except FileNotFoundError:
            masquerades_as_uploaded = False

        if masquerades_as_uploaded:
            result.record(name, False,
                          f"CORRUPTION CONFIRMED: interrupted put left a {remote_size}B partial blob "
                          f"(expected {size}B) at content-addressed path {cache_remote!r}. "
                          f"_push_file's existence-only check (sftp.stat, not hash verify) will treat any "
                          f"FUTURE push of this exact content as already-uploaded and skip re-uploading -- "
                          f"the corrupt truncated blob becomes permanent, and any consumer that pulls it "
                          f"will get a truncated/corrupt h5 file with no error until h5py tries to parse it.")
        else:
            result.record(name, False, "partial blob left on remote but somehow not stat-able (unexpected)")
    finally:
        try:
            sftp2.remove(cache_remote)
        except Exception:
            pass
        try:
            sftp2.close()
        except Exception:
            pass


def test_put_interrupt_then_retry_recovers(stress_config, tmpdir, result, rng, n_rows=4000):
    """After an interrupted push, does simply re-running push (the natural
    recovery action) succeed and produce a correct remote blob?"""
    name = "put_interrupt_then_retry_recovers"
    local_path = os.path.join(tmpdir, "crash_retry.h5")
    virtual_path = f"_stress_test_scratch/{os.path.basename(tmpdir)}/crash_retry.h5"
    make_h5(local_path, n_rows, seed=2)
    digest = upload.hash_file(local_path)
    cache_remote = upload.remote_cache_path(stress_config, digest)

    sftp, transport = fresh_sftp(stress_config)
    try:
        upload.sftp_mkdir_p(sftp, os.path.dirname(cache_remote))
        crashed = crash_during_put(sftp, transport, local_path, cache_remote, rng=rng)
    finally:
        try:
            sftp.close()
        except Exception:
            pass

    try:
        # Natural recovery: call cmd_push again (as a user/script would after a crash).
        upload._board = upload._LogBoard()
        upload.cmd_push(local_path, stress_config, auto_delete=False, virtual_path=virtual_path)

        sftp3, _ = fresh_sftp(stress_config)
        try:
            remote_size = sftp3.stat(cache_remote).st_size
            data = remote_read_bytes(sftp3, cache_remote)
            import xxhash
            actual = xxhash.xxh3_128(data).hexdigest()
            if actual == digest:
                result.record(name, True, f"retry after crash={crashed} produced valid blob")
            else:
                result.record(name, False,
                              f"retry after crash left WRONG HASH blob ({actual[:12]} != {digest[:12]}) -- "
                              f"push_file's existence-only check likely skipped re-upload of a corrupt blob "
                              f"from the first (crashed) attempt")
        finally:
            sftp3.close()
    finally:
        try:
            sftp4, _ = fresh_sftp(stress_config)
            sftp4.remove(cache_remote)
            idx = upload.remote_index_path(stress_config, virtual_path)
            try:
                sftp4.remove(idx)
            except FileNotFoundError:
                pass
            sftp4.close()
        except Exception:
            pass
        ptr_path = upload._ptr_path_for(local_path, virtual_path)
        if os.path.exists(ptr_path):
            os.remove(ptr_path)


def test_index_write_interrupt_leaves_corrupt_json(stress_config, tmpdir, result, rng):
    """The pointer-index write (sftp.open(remote_idx, 'w')) is not atomic.
    If interrupted mid-write, does it leave truncated/corrupt JSON, and does
    that break cmd_push's conflict-check (json.load) or cmd_meta on the
    NEXT push attempt for the same file?"""
    name = "index_write_interrupt_leaves_corrupt_json"
    local_path = os.path.join(tmpdir, "crash_index.h5")
    virtual_path = f"_stress_test_scratch/{os.path.basename(tmpdir)}/crash_index.h5"
    make_h5(local_path, 500, seed=3)
    digest = upload.hash_file(local_path)
    size = os.path.getsize(local_path)
    remote_idx = upload.remote_index_path(stress_config, virtual_path)

    ptr_path, ptr_data = upload.write_ptr(local_path, digest, size, virtual_path)
    payload = json.dumps(ptr_data, indent=2).encode()

    sftp, transport = fresh_sftp(stress_config)
    try:
        upload.sftp_mkdir_p(sftp, os.path.dirname(remote_idx))
        killer = TransferKiller(transport, rng=rng)
        crashed = False
        try:
            with sftp.open(remote_idx, "w") as f:
                # write in small pieces so a mid-write kill is possible
                chunk = max(1, len(payload) // 10)
                sent = 0
                for i in range(0, len(payload), chunk):
                    piece = payload[i:i + chunk]
                    f.write(piece)
                    sent += len(piece)
                    killer(sent, len(payload))
        except (Fault, EOFError, OSError, paramiko.SSHException):
            crashed = True
    finally:
        try:
            sftp.close()
        except Exception:
            pass

    if os.path.exists(ptr_path):
        os.remove(ptr_path)

    if not crashed:
        result.record(name, True, "index write completed before fault fired; skipping")
        try:
            sftp2, _ = fresh_sftp(stress_config)
            sftp2.remove(remote_idx)
            sftp2.close()
        except Exception:
            pass
        return

    sftp2, _ = fresh_sftp(stress_config)
    try:
        if not remote_exists(sftp2, remote_idx):
            result.record(name, True, "no index file left after interrupted write")
            return
        raw = remote_read_bytes(sftp2, remote_idx)
        try:
            json.loads(raw)
            result.record(name, True, "index file present and valid JSON despite interruption")
            return
        except json.JSONDecodeError:
            pass

        # Corrupt JSON confirmed on remote. Does a subsequent push's
        # conflict-check crash on it? _push_file does `json.load(f)` at the
        # conflict-check step with no except for JSONDecodeError.
        make_h5(local_path, 501, seed=4)  # different content -> triggers conflict path
        try:
            upload._board = upload._LogBoard()
            with patch_stdin_noninteractive():
                path, ok = upload._push_file(local_path, stress_config, virtual_path=virtual_path)
            result.record(name, False,
                          f"CORRUPTION CONFIRMED: interrupted index write left invalid JSON at {remote_idx!r}. "
                          f"A subsequent push of the same original_path did NOT crash (got ok={ok}) -- "
                          f"check whether this silently clobbered the corrupt pointer without surfacing the "
                          f"earlier corruption, or investigate the traceback captured separately.")
        except json.JSONDecodeError as e:
            result.record(name, False,
                          f"CORRUPTION CONFIRMED: interrupted index write left invalid JSON at {remote_idx!r} "
                          f"({raw[:80]!r}...). A subsequent push's conflict-check does `json.load(f)` on the "
                          f"remote index with NO except for JSONDecodeError, so it crashes uncaught: {e!r}. "
                          f"This means one bad interrupted push permanently blocks all future pushes of the "
                          f"same file until someone manually fixes the remote index.")
        except Exception as e:
            result.record(name, False, f"push after corrupt index raised unexpected {type(e).__name__}: {e}")
    finally:
        try:
            sftp2.remove(remote_idx)
        except Exception:
            pass
        try:
            cache_remote = upload.remote_cache_path(stress_config, upload.hash_file(local_path))
            sftp2.remove(cache_remote)
        except Exception:
            pass
        sftp2.close()
        ptr_path2 = upload._ptr_path_for(local_path, virtual_path)
        if os.path.exists(ptr_path2):
            os.remove(ptr_path2)


class patch_stdin_noninteractive:
    """Context manager: make sys.stdin.isatty() report False so _push_file's
    conflict path takes the EOFError/non-interactive branch instead of
    blocking on input()."""
    def __enter__(self):
        self._orig = sys.stdin
        class FakeStdin:
            def isatty(self):
                return False
            def readline(self):
                raise EOFError
        sys.stdin = FakeStdin()
        return self
    def __exit__(self, *a):
        sys.stdin = self._orig


def test_get_interrupt_leaves_partial_local_file(stress_config, tmpdir, result, rng, n_rows=4000):
    """A crash during pull (ensure_local / cmd_pull) can leave a partial
    local file. Does anything detect this on next use, or would a
    downstream h5py.File(...) open silently get corrupt data?"""
    name = "get_interrupt_leaves_partial_local_file_detected_on_retry"
    remote_local_src = os.path.join(tmpdir, "crash_get_src.h5")
    make_h5(remote_local_src, n_rows, seed=5)
    digest = upload.hash_file(remote_local_src)
    size = os.path.getsize(remote_local_src)
    cache_remote = upload.remote_cache_path(stress_config, digest)

    sftp, _ = fresh_sftp(stress_config)
    try:
        upload.sftp_mkdir_p(sftp, os.path.dirname(cache_remote))
        sftp.put(remote_local_src, cache_remote)
    finally:
        sftp.close()

    # _pull_file always writes to PROJECT_ROOT/ptr["original_path"] and calls
    # write_ptr(dest, ...) with no virtual_path, so dest must be inside
    # PROJECT_ROOT for that call to resolve a sane relative path.
    dest_rel = f"_stress_test_scratch/{os.path.basename(tmpdir)}/crash_get_dest.h5"
    dest = os.path.join(upload.PROJECT_ROOT, dest_rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    ptr = {"hash": digest, "original_path": dest_rel, "filename": "crash_get_dest.h5", "size": size}

    sftp2, transport2 = fresh_sftp(stress_config)
    try:
        crashed = crash_during_get(sftp2, transport2, cache_remote, dest, rng=rng)
    finally:
        try:
            sftp2.close()
        except Exception:
            pass

    try:
        if not crashed:
            result.record(name, True, "transfer completed before fault fired; skipping")
            return

        if not os.path.exists(dest):
            result.record(name, True, "no partial file left locally after interrupted get")
            return

        partial_size = os.path.getsize(dest)
        if partial_size == size and upload.hash_file(dest) == digest:
            result.record(name, True, "partial file happens to be byte-complete and hash-valid")
            return

        # Does _pull_file's own hash check (on a *retry*) detect and fix this?
        sftp3, _ = fresh_sftp(stress_config)
        try:
            dest2, pulled = upload._pull_file(ptr, stress_config)
        finally:
            sftp3.close()

        final_hash = upload.hash_file(dest)
        if final_hash == digest:
            result.record(name, True,
                          f"partial file ({partial_size}B of {size}B) left after crash, but retrying "
                          f"_pull_file correctly detected the hash mismatch and re-downloaded (pulled={pulled})")
        else:
            result.record(name, False,
                          f"CORRUPTION: partial file ({partial_size}B of {size}B) left after crash, and "
                          f"retry did not repair it (final hash {final_hash[:12]} != {digest[:12]})")
    finally:
        if os.path.exists(dest):
            os.remove(dest)
        ptr_path = upload._ptr_path_for(dest)
        if os.path.exists(ptr_path):
            os.remove(ptr_path)
        parent = os.path.dirname(dest)
        try:
            os.rmdir(parent)
        except OSError:
            pass
        try:
            sftp4, _ = fresh_sftp(stress_config)
            sftp4.remove(cache_remote)
            sftp4.close()
        except Exception:
            pass


def test_concurrent_push_interrupt_one_worker(stress_config, tmpdir, result, n_files=6):
    """Simulate cmd_push's ThreadPoolExecutor with multiple files, where one
    worker's connection is killed mid-transfer. Do the OTHER workers'
    files still complete correctly (no cross-worker corruption from the
    shared _board / thread pool)?

    Files are node-local scratch shards pushed individually with
    virtual_path, matching how execute/sharded_writer.py actually pushes
    (one shard file at a time, not a directory push)."""
    name = "concurrent_push_one_worker_interrupted_others_unaffected"
    run_id = os.path.basename(tmpdir)
    paths = []
    vpaths = {}
    digests = {}
    for i in range(n_files):
        p = os.path.join(tmpdir, f"concurrent_{i}.h5")
        make_h5(p, 300 + i * 50, seed=100 + i)
        paths.append(p)
        vpaths[p] = f"_stress_test_scratch/{run_id}/concurrent_{i}.h5"
        digests[p] = upload.hash_file(p)

    victim = paths[0]

    orig_sftp_put = upload.sftp_put
    fire_lock = threading.Lock()
    fired = {"done": False}

    def patched_sftp_put(sftp, local_path, remote_path):
        if local_path == victim and not fired["done"]:
            with fire_lock:
                if not fired["done"]:
                    fired["done"] = True
                    size = os.path.getsize(local_path)
                    killer = TransferKiller(sftp.get_channel().get_transport(), min_frac=0.2, max_frac=0.6)
                    try:
                        sftp.put(local_path, remote_path, callback=killer)
                        return
                    except (Fault, EOFError, OSError, paramiko.SSHException):
                        raise
        return orig_sftp_put(sftp, local_path, remote_path)

    upload.sftp_put = patched_sftp_put
    try:
        upload._board = upload._LogBoard()
        with ThreadPoolExecutor(max_workers=len(paths)) as pool:
            futures = [pool.submit(upload._push_file, p, stress_config, virtual_path=vpaths[p]) for p in paths]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception:
                    pass  # victim worker's exception is expected; other workers checked below
    finally:
        upload.sftp_put = orig_sftp_put

    # Verify: for every file EXCEPT the victim, remote blob should be correct.
    sftp, _ = fresh_sftp(stress_config)
    try:
        problems = []
        for p in paths:
            if p == victim:
                continue
            cache_remote = upload.remote_cache_path(stress_config, digests[p])
            if not remote_exists(sftp, cache_remote):
                problems.append(f"{os.path.basename(p)}: missing from remote after concurrent push")
                continue
            data = remote_read_bytes(sftp, cache_remote)
            import xxhash
            actual = xxhash.xxh3_128(data).hexdigest()
            if actual != digests[p]:
                problems.append(f"{os.path.basename(p)}: remote blob hash mismatch (cross-worker corruption?)")

        if problems:
            result.record(name, False, "; ".join(problems))
        else:
            result.record(name, True, f"{n_files - 1} non-victim files all uploaded correctly despite one worker crashing")
    finally:
        for p in paths:
            try:
                cache_remote = upload.remote_cache_path(stress_config, digests[p])
                sftp.remove(cache_remote)
            except Exception:
                pass
            try:
                idx = upload.remote_index_path(stress_config, vpaths[p])
                sftp.remove(idx)
            except Exception:
                pass
            ptr_path = upload._ptr_path_for(p, vpaths[p])
            if os.path.exists(ptr_path):
                os.remove(ptr_path)
        sftp.close()


def cleanup_stress_subtree(config, result):
    sftp, _ = fresh_sftp(config)
    try:
        root = config["remote_path"]
        def rm_tree(path):
            try:
                for entry in sftp.listdir_attr(path):
                    full = f"{path}/{entry.filename}"
                    import stat as stat_mod
                    if stat_mod.S_ISDIR(entry.st_mode):
                        rm_tree(full)
                        try:
                            sftp.rmdir(full)
                        except Exception:
                            pass
                    else:
                        try:
                            sftp.remove(full)
                        except Exception:
                            pass
            except FileNotFoundError:
                pass
        rm_tree(root)
        try:
            sftp.rmdir(root)
        except Exception:
            pass
        print(f"[cleanup] removed remote stress-test subtree {root!r}")
    finally:
        sftp.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=5, help="repetitions per fault-injection case")
    ap.add_argument("--keep-remote", action="store_true", help="don't delete the _stress_test remote subtree at the end")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    base_config = upload.load_config()
    stress_config = dict(base_config)
    stress_config["remote_path"] = base_config["remote_path"].rstrip("/") + "/_stress_test"
    stress_config["workers"] = base_config["workers"]

    print(f"Target remote path (isolated): {stress_config['remote_path']}")
    print(f"Host: {stress_config['host']}")
    print(f"Iterations per case: {args.iterations}\n")

    result = Result()
    rng = random.Random(args.seed)

    # Node-local scratch disk, same convention as execute/sharded_writer.py
    # (never NFS for in-progress shard data). Files written here are outside
    # PROJECT_ROOT, so every push/pull in this harness uses `virtual_path`
    # to register them at a _stress_test_scratch/<run_id>/... project-relative
    # path, matching how ShardedLayerWriter registers node-local shards.
    scratch_root = os.path.join(_scratch_root(), "stress_test")
    os.makedirs(scratch_root, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="run_", dir=scratch_root)
    print(f"Local scratch dir: {tmpdir}\n")

    try:
        print("=== Phase 1: put() interrupted mid-transfer -- blob masquerade check ===")
        for i in range(args.iterations):
            test_put_interrupt_leaves_no_valid_blob_masquerading(stress_config, tmpdir, result, rng, n_rows=6000 + i * 500)

        print("\n=== Phase 2: put() interrupted, then natural retry via cmd_push ===")
        for i in range(args.iterations):
            test_put_interrupt_then_retry_recovers(stress_config, tmpdir, result, rng, n_rows=6000 + i * 500)

        print("\n=== Phase 3: remote index write interrupted -- corrupt JSON check ===")
        for i in range(args.iterations):
            test_index_write_interrupt_leaves_corrupt_json(stress_config, tmpdir, result, rng)

        print("\n=== Phase 4: get() interrupted mid-transfer -- partial local file check ===")
        for i in range(args.iterations):
            test_get_interrupt_leaves_partial_local_file(stress_config, tmpdir, result, rng, n_rows=6000 + i * 500)

        print("\n=== Phase 5: concurrent multi-worker push, one worker crashes ===")
        for i in range(min(args.iterations, 3)):
            test_concurrent_push_interrupt_one_worker(stress_config, tmpdir, result)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        shutil.rmtree(scratch_root, ignore_errors=True)
        if not args.keep_remote:
            try:
                cleanup_stress_subtree(stress_config, result)
            except Exception as e:
                print(f"[cleanup] WARNING: failed to clean up remote stress subtree: {e}")
                print(f"          Manually check/remove: {stress_config['remote_path']}")

    ok = result.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
