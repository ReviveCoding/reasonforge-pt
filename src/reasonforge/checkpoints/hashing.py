"""Deterministic hashing and atomic small-file publication helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for *path* without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON in the stable form used for checkpoint identities."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Durably replace a file using a temporary sibling and ``os.replace``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value))


def hash_inventory(root: Path, relative_paths: Iterable[str]) -> dict[str, str]:
    """Hash a declared set of relative paths, rejecting path traversal."""

    resolved_root = root.resolve()
    inventory: dict[str, str] = {}
    for name in sorted(set(relative_paths)):
        candidate = (root / name).resolve()
        if candidate == resolved_root or resolved_root not in candidate.parents:
            raise ValueError(f"artifact path escapes checkpoint root: {name!r}")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        inventory[name] = sha256_file(candidate)
    return inventory
