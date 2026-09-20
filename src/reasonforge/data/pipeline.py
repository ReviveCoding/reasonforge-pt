"""Small-footprint streaming preparation into canonical Parquet."""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from reasonforge.data.schema import CanonicalRow

_PUBLICATION_LOCK = threading.Lock()


def validate_unique_ids(rows: Iterable[CanonicalRow]) -> list[CanonicalRow]:
    materialized = list(rows)
    ids = [row.example_id for row in materialized]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate example_id")
    return materialized


def write_canonical_parquet(rows: Iterable[CanonicalRow], output: Path) -> Path:
    validated = validate_unique_ids(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([row.parquet_record() for row in validated])
    temporary = output.with_name(f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        # Explicit streams matter on Windows: implicit Arrow path readers may
        # retain a file handle long enough to make atomic replacement fail.
        with temporary.open("wb") as sink:
            pq.write_table(table, sink, compression="zstd")
            sink.flush()
            os.fsync(sink.fileno())
        with temporary.open("rb") as source:
            checked_rows = pq.read_metadata(source).num_rows
        if checked_rows != len(validated):
            raise RuntimeError("Parquet read-back row-count mismatch")
        # Windows may reject two simultaneous replaces of the same destination.
        # Publication is rare relative to Arrow construction, so serializing the
        # final metadata operation is cheap and preserves deterministic semantics.
        with _PUBLICATION_LOCK:
            os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def read_canonical_parquet(path: Path) -> list[CanonicalRow]:
    with path.open("rb") as source:
        rows = pq.read_table(source).to_pylist()
    parsed: list[CanonicalRow] = []
    for row in rows:
        row["metadata"] = json.loads(row["metadata"])
        parsed.append(CanonicalRow.model_validate(row))
    return parsed
