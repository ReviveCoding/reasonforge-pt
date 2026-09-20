"""Explicitly bounded acquisition for Hugging Face dataset sources."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any, cast

from reasonforge.config import file_sha256
from reasonforge.data.registry import SourceRecord

DatasetLoader = Callable[..., Iterable[Mapping[str, Any]]]
RevisionResolver = Callable[[str, str], str]


class AcquisitionError(RuntimeError):
    """A bounded acquisition could not be completed without violating policy."""


@dataclass(frozen=True, slots=True)
class AcquisitionManifest:
    """Identity and bounds for one locally materialized source slice."""

    schema_version: str
    created_at: str
    source_id: str
    repository_id: str
    requested_revision: str
    resolved_revision: str
    split: str
    config_name: str | None
    requested_max_rows: int
    rows_acquired: int
    source_role: str
    protection: str
    governance_status: str
    training_allowed: bool
    provider_fingerprint: str | None
    content_fingerprint: str
    fingerprint_algorithm: str
    rows_file: str
    rows_file_sha256: str


def _default_revision_resolver(repository_id: str, revision: str) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - exercised only without train dependencies.
        raise AcquisitionError(
            "bounded Hugging Face acquisition requires the 'train' dependencies"
        ) from exc
    info = HfApi().dataset_info(repo_id=repository_id, revision=revision)
    if not info.sha:
        raise AcquisitionError(f"Hugging Face did not resolve a SHA for {repository_id!r}")
    return str(info.sha)


def _default_dataset_loader(**kwargs: Any) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without train dependencies.
        raise AcquisitionError(
            "bounded Hugging Face acquisition requires the 'train' dependencies"
        ) from exc
    loaded = load_dataset(**kwargs)
    if not hasattr(loaded, "__iter__"):
        raise AcquisitionError("streaming load did not return an iterable dataset")
    return cast(Iterable[Mapping[str, Any]], loaded)


def _resolved_revision_matches(requested: str, resolved: str) -> bool:
    requested_lower = requested.casefold()
    resolved_lower = resolved.casefold()
    return resolved_lower == requested_lower or resolved_lower.startswith(requested_lower)


def _canonical_row(row: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            dict(row),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AcquisitionError(f"dataset row is not canonical JSON data: {exc}") from exc
    return (text + "\n").encode("utf-8")


def _provider_fingerprint(stream: object) -> str | None:
    value = getattr(stream, "_fingerprint", None)
    return str(value) if value is not None else None


def acquire_huggingface_source(
    source: SourceRecord,
    output_dir: Path,
    *,
    max_rows: int,
    split: str = "train",
    config_name: str | None = None,
    loader: DatasetLoader | None = None,
    revision_resolver: RevisionResolver | None = None,
    now: Callable[[], datetime] | None = None,
) -> AcquisitionManifest:
    """Stream at most ``max_rows`` and atomically publish rows plus identity metadata.

    ``max_rows`` deliberately has no default. The Hugging Face loader is always
    called with ``streaming=True`` and the immutable resolved commit, so this
    helper cannot silently materialize the full remote corpus.
    """
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows <= 0:
        raise ValueError("max_rows must be an explicit positive integer")
    if not split.strip():
        raise ValueError("split must be non-empty")
    source.require_acquisition_allowed()
    if "huggingface" not in source.revision_kind:
        raise AcquisitionError(
            f"source {source.source_id!r} is not registered as a Hugging Face revision"
        )

    if output_dir.exists() and not output_dir.is_dir():
        raise AcquisitionError(f"output path exists and is not a directory: {output_dir}")
    rows_path = output_dir / "rows.jsonl"
    manifest_path = output_dir / "ACQUISITION_MANIFEST.json"
    rows_temporary = output_dir / ".rows.jsonl.tmp"
    manifest_temporary = output_dir / ".ACQUISITION_MANIFEST.json.tmp"
    collisions = [
        path
        for path in (rows_path, manifest_path, rows_temporary, manifest_temporary)
        if path.exists()
    ]
    if collisions:
        raise AcquisitionError(f"refusing to overwrite acquisition artifacts: {collisions}")

    resolve = revision_resolver or _default_revision_resolver
    resolved_revision = resolve(source.name, source.revision)
    if not _resolved_revision_matches(source.revision, resolved_revision):
        raise AcquisitionError(
            f"resolved revision mismatch for {source.source_id!r}: "
            f"requested {source.revision}, resolved {resolved_revision}"
        )

    load = loader or _default_dataset_loader
    load_args: dict[str, Any] = {
        "path": source.name,
        "split": split,
        "revision": resolved_revision,
        "streaming": True,
    }
    if config_name is not None:
        load_args["name"] = config_name
    stream = load(**load_args)

    output_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    rows_acquired = 0
    try:
        with rows_temporary.open("wb") as handle:
            for row in islice(iter(stream), max_rows):
                if not isinstance(row, Mapping):
                    raise AcquisitionError("stream yielded a non-mapping dataset row")
                encoded = _canonical_row(row)
                handle.write(encoded)
                digest.update(encoded)
                rows_acquired += 1
            handle.flush()
        rows_temporary.replace(rows_path)

        clock = now or (lambda: datetime.now(UTC))
        manifest = AcquisitionManifest(
            schema_version="1.0.0",
            created_at=clock().astimezone(UTC).isoformat(),
            source_id=source.source_id,
            repository_id=source.name,
            requested_revision=source.revision,
            resolved_revision=resolved_revision,
            split=split,
            config_name=config_name,
            requested_max_rows=max_rows,
            rows_acquired=rows_acquired,
            source_role=source.role.value,
            protection=source.protection,
            governance_status=source.governance_status,
            training_allowed=source.training_allowed,
            provider_fingerprint=_provider_fingerprint(stream),
            content_fingerprint=digest.hexdigest(),
            fingerprint_algorithm="sha256(canonical-jsonl-utf8)",
            rows_file=rows_path.name,
            rows_file_sha256=file_sha256(rows_path),
        )
        manifest_temporary.write_text(
            json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # The manifest is the completion marker and is published last.
        manifest_temporary.replace(manifest_path)
        return manifest
    except Exception:
        for temporary in (rows_temporary, manifest_temporary):
            if temporary.exists():
                temporary.unlink()
        raise
