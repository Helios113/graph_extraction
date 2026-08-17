import contextlib
import os
import time
from pathlib import Path


def _strip_last_quantifier(path: str) -> str:
    return ".".join(path.split(".")[:-1])


@contextlib.contextmanager
def exclusive_lock(lock_path: Path, poll_seconds: float = 2.0):
    """Waits until `lock_path` can be created exclusively (os.O_CREAT|O_EXCL is atomic even
    on network filesystems, unlike flock -- this repo's onnx_models/ can be reached from
    several cluster nodes at once), then holds it for the block, deleting it on exit (even
    on error, so a crash doesn't wedge future runs forever)."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
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
