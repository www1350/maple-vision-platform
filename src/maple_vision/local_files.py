"""Bounded reads and atomic create-only publication of local evidence."""

import os
import stat
import tempfile
from pathlib import Path


def read_regular(path: Path, limit: int, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an existing regular file, not a symlink")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > limit:
            raise ValueError(f"{label} exceeds its size limit or is not a regular file")
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{label} exceeds its size limit")
    return data


def publish_new(path: Path, data: bytes, limit: int, label: str) -> None:
    if len(data) > limit:
        raise ValueError(f"{label} exceeds its size limit")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{label} already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".maple-write-", dir=path.parent)
    stage = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(stage, path)
    finally:
        stage.unlink(missing_ok=True)
