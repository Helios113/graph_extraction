import contextlib
import os
import time
from pathlib import Path


def _strip_last_quantifier(path: str) -> str:
    return ".".join(path.split(".")[:-1])


def remove_stale(lock_path: Path, stale_seconds: float = 600.0) -> None:
    """Removes `lock_path` if it exists and is older than `stale_seconds`.

    A lock holder killed with SIGKILL (OOM, scancel, node crash) never reaches the cleanup
    in `exclusive_lock`'s `finally`, leaving the lock file orphaned forever, since PIDs
    aren't comparable across nodes here. A lock file whose mtime is older than
    `stale_seconds` is assumed abandoned by a crashed holder, so it's safe to remove --
    races against other callers doing the same are fine since the removal itself is a
    no-op if another one already won it."""
    try:
        age = time.time() - lock_path.stat().st_mtime
    except FileNotFoundError:
        return
    if age > stale_seconds:
        print(
            f"[remove_stale] {lock_path} is {age:.0f}s old (> {stale_seconds:.0f}s) "
            "-- assuming its holder crashed without cleaning up, removing it",
            flush=True,
        )
        lock_path.unlink(missing_ok=True)


@contextlib.contextmanager
def exclusive_lock(lock_path: Path, poll_seconds: float = 2.0, stale_seconds: float = 600.0):
    """Waits until `lock_path` can be created exclusively (os.O_CREAT|O_EXCL is atomic even
    on network filesystems, unlike flock -- this repo's onnx_models/ can be reached from
    several cluster nodes at once), then holds it for the block, deleting it on exit (even
    on error, so a crash doesn't wedge future runs forever).

    Calls `remove_stale` before every acquisition attempt -- both the initial one and each
    one after losing a race against a waiter that just took the lock -- so an abandoned
    lock never blocks the wait for the full `stale_seconds` on top of however long it had
    already been sitting there."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        remove_stale(lock_path, stale_seconds)
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            time.sleep(poll_seconds)
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


@contextlib.contextmanager
def chdir(path: Path):
    """Temporarily changes the process cwd. Every path passed to the wrapped block must be
    absolute -- callers are responsible for resolving any relative path first."""
    prev = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(prev)
