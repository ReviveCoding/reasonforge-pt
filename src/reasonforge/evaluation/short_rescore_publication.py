"""Additive short-path publication; frozen scorer functions remain byte-identical."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from types import FunctionType
from typing import Any

from reasonforge.checkpoints.hashing import atomic_write_json, canonical_json_bytes, sha256_file
from reasonforge.checkpoints.models import utc_now

from .protected_outputs import (
    DeterministicGraderResult,
    ProtectedOutputIntegrityError,
    ProtectedOutputWriter,
    _validate_component,
    _validate_sha256,
    _validate_text,
    verify_deterministic_rescore,
)


def publish_short_rescore(
    output_directory: Path | str,
    *,
    staging_root: Path,
    rescore_id: str,
    grader_id: str,
    grader_version: str,
    grader_config_sha256: str,
    results: Iterable[DeterministicGraderResult],
) -> Path:
    """Use the frozen result contract, with staging under a short same-volume root.

    Publication is one directory rename. Failed staging stays in this dedicated root;
    no cleanup or overwrite is attempted, and the recovery lock remains fail-closed.
    """
    _validate_component("rescore_id", rescore_id)
    _validate_component("grader_id", grader_id)
    _validate_text("grader_version", grader_version)
    _validate_sha256("grader_config_sha256", grader_config_sha256)
    source = Path(output_directory)
    verification = ProtectedOutputWriter(source).verify(require_complete=True)
    materialized = list(results)
    seen: set[str] = set()
    for result in materialized:
        result.validate()
        if result.grader_id != grader_id or result.grader_version != grader_version:
            raise ProtectedOutputIntegrityError("grader result identity mismatch")
        if result.grader_kind != "DETERMINISTIC":
            raise ProtectedOutputIntegrityError("only deterministic grader results are accepted")
        if result.grader_config_sha256 != grader_config_sha256:
            raise ProtectedOutputIntegrityError("grader result configuration mismatch")
        if result.output_set_id != verification.manifest.identity.output_set_id:
            raise ProtectedOutputIntegrityError("grader result output-set mismatch")
        if result.example_id in seen:
            raise ProtectedOutputIntegrityError("duplicate grader result")
        if verification.record_sha256.get(result.example_id) != result.source_record_sha256:
            raise ProtectedOutputIntegrityError("grader result is not bound to source record")
        seen.add(result.example_id)
    if seen != set(verification.manifest.expected_example_ids):
        raise ProtectedOutputIntegrityError("grader results incomplete")
    destination = source / "rescores" / rescore_id
    if destination.exists():
        raise FileExistsError("rescore already exists")
    staging_root.mkdir(parents=True, exist_ok=True)
    if staging_root.stat().st_dev != source.stat().st_dev:
        raise ProtectedOutputIntegrityError("rescore staging must share the source filesystem")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="s-", dir=staging_root))
    results_path = staging / "results.jsonl"
    with results_path.open("wb") as handle:
        for result in sorted(materialized, key=lambda item: item.example_id):
            handle.write(canonical_json_bytes(result.to_dict()))
        handle.flush()
        os.fsync(handle.fileno())
    manifest = {
        "schema_version": 1,
        "status": "COMPLETE",
        "rescore_id": rescore_id,
        "output_set_id": verification.manifest.identity.output_set_id,
        "source_manifest_sha256": sha256_file(source / "manifest.json"),
        "source_shard_sha256": {
            shard.filename: shard.sha256 for shard in verification.manifest.shards
        },
        "grader_id": grader_id,
        "grader_version": grader_version,
        "grader_kind": "DETERMINISTIC",
        "grader_config_sha256": grader_config_sha256,
        "expected_example_ids": verification.manifest.expected_example_ids,
        "result_count": len(materialized),
        "results_sha256": sha256_file(results_path),
        "created_at": utc_now(),
    }
    atomic_write_json(staging / "manifest.json", manifest)
    atomic_write_json(
        staging / "RESCORE_COMPLETE.json",
        {
            "status": "COMPLETE",
            "rescore_id": rescore_id,
            "manifest_sha256": sha256_file(staging / "manifest.json"),
        },
    )
    verify_deterministic_rescore(staging, source)
    # rename (rather than replace) also refuses an occupied destination on Windows.
    os.rename(staging, destination)
    verify_deterministic_rescore(destination, source)
    return destination


def scorer_with_short_publication(scorer: FunctionType, staging_root: Path) -> Callable[..., Path]:
    """Bind only the publisher dependency without mutating module globals or scorer code."""

    def publisher(*args: Any, **kwargs: Any) -> Path:
        return publish_short_rescore(*args, staging_root=staging_root, **kwargs)

    namespace = dict(scorer.__globals__)
    if "publish_deterministic_rescore" not in namespace:
        raise ProtectedOutputIntegrityError("scorer lacks the frozen publication dependency")
    namespace["publish_deterministic_rescore"] = publisher
    bound = FunctionType(
        scorer.__code__, namespace, scorer.__name__, scorer.__defaults__, scorer.__closure__
    )
    bound.__kwdefaults__ = scorer.__kwdefaults__
    return bound
