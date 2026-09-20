"""Durable protected outputs and immutable deterministic rescoring.

This module stores already-generated text. It deliberately performs no model loading or
inference. Completed output shards are content-addressed so corrected graders can rescore
the exact same generations without regenerating them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from reasonforge.checkpoints.hashing import (
    atomic_write_json,
    canonical_json_bytes,
    sha256_file,
)
from reasonforge.checkpoints.models import utc_now

_SHA256_LENGTH = 64
_MANIFEST_NAME = "manifest.json"
_COMPLETE_NAME = "PROTECTED_OUTPUT_COMPLETE.json"


class ProtectedOutputIntegrityError(RuntimeError):
    """Raised when protected evidence is incomplete, inconsistent, or modified."""


def _validate_component(name: str, value: str) -> None:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ProtectedOutputIntegrityError(f"{name} must be a non-empty path component")


def _validate_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ProtectedOutputIntegrityError(f"{name} must be a non-empty string")


def _validate_sha256(name: str, value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtectedOutputIntegrityError(f"{name} must be a lowercase SHA-256 digest")


def _ensure_json(value: object, *, context: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ProtectedOutputIntegrityError(f"{context} must be finite JSON data") from error


def _require_exact_keys(value: Mapping[str, Any], fields: Iterable[str], *, context: str) -> None:
    expected = set(fields)
    actual = set(value)
    if actual != expected:
        raise ProtectedOutputIntegrityError(
            f"{context} fields differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _record_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


@dataclass(frozen=True, slots=True)
class ProtectedOutputIdentity:
    """Frozen identities shared by every record in an output set."""

    output_set_id: str
    protocol_id: str
    model_id: str
    model_checkpoint_sha256: str
    tokenizer_id: str
    generation_config_id: str
    generation_config_sha256: str
    protected_dataset_fingerprint: str

    def validate(self) -> None:
        _validate_component("output_set_id", self.output_set_id)
        _validate_component("model_id", self.model_id)
        for name in ("protocol_id", "tokenizer_id", "generation_config_id"):
            _validate_text(name, getattr(self, name))
        for name in (
            "model_checkpoint_sha256",
            "generation_config_sha256",
            "protected_dataset_fingerprint",
        ):
            _validate_sha256(name, getattr(self, name))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProtectedOutputIdentity:
        _require_exact_keys(value, cls.__dataclass_fields__, context="output identity")
        try:
            identity = cls(**{name: value[name] for name in cls.__dataclass_fields__})
        except (KeyError, TypeError) as error:
            raise ProtectedOutputIntegrityError(f"invalid output identity: {error}") from error
        identity.validate()
        return identity


@dataclass(frozen=True, slots=True)
class ProtectedOutputRecord:
    """One immutable model generation for one protected example."""

    output_set_id: str
    protocol_id: str
    model_id: str
    model_checkpoint_sha256: str
    generation_config_id: str
    generation_config_sha256: str
    example_id: str
    benchmark_id: str
    benchmark_revision: str
    prompt_template_id: str
    output_text: str
    finish_reason: str
    created_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for name in (
            "output_set_id",
            "protocol_id",
            "model_id",
            "generation_config_id",
            "example_id",
            "benchmark_id",
            "benchmark_revision",
            "prompt_template_id",
            "finish_reason",
            "created_at",
        ):
            _validate_text(name, getattr(self, name))
        for name in ("model_checkpoint_sha256", "generation_config_sha256"):
            _validate_sha256(name, getattr(self, name))
        if not isinstance(self.output_text, str):
            raise ProtectedOutputIntegrityError("output_text must be a string")
        if not isinstance(self.metadata, dict):
            raise ProtectedOutputIntegrityError("metadata must be an object")
        _ensure_json(self.metadata, context="record metadata")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProtectedOutputRecord:
        _require_exact_keys(value, cls.__dataclass_fields__, context="protected output record")
        try:
            record = cls(**{name: value[name] for name in cls.__dataclass_fields__})
        except (KeyError, TypeError) as error:
            raise ProtectedOutputIntegrityError(
                f"invalid protected output record: {error}"
            ) from error
        record.validate()
        return record


@dataclass(frozen=True, slots=True)
class ProtectedOutputShard:
    filename: str
    status: str
    sha256: str
    byte_size: int
    record_count: int
    example_ids: list[str]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProtectedOutputShard:
        _require_exact_keys(value, cls.__dataclass_fields__, context="shard manifest")
        try:
            shard = cls(**{name: value[name] for name in cls.__dataclass_fields__})
        except (KeyError, TypeError) as error:
            raise ProtectedOutputIntegrityError(f"invalid shard manifest: {error}") from error
        if not isinstance(shard.filename, str) or not shard.filename:
            raise ProtectedOutputIntegrityError("shard filename must be a non-empty string")
        if shard.status not in {"IN_PROGRESS", "COMPLETE"}:
            raise ProtectedOutputIntegrityError("invalid shard completion status")
        _validate_sha256("shard sha256", shard.sha256)
        if (
            not isinstance(shard.byte_size, int)
            or isinstance(shard.byte_size, bool)
            or not isinstance(shard.record_count, int)
            or isinstance(shard.record_count, bool)
            or shard.byte_size <= 0
            or shard.record_count <= 0
        ):
            raise ProtectedOutputIntegrityError("shard size and record count must be positive")
        if (
            not isinstance(shard.example_ids, list)
            or any(not isinstance(item, str) or not item for item in shard.example_ids)
            or len(shard.example_ids) != shard.record_count
            or len(set(shard.example_ids)) != len(shard.example_ids)
        ):
            raise ProtectedOutputIntegrityError("shard example inventory is invalid")
        return shard


@dataclass(frozen=True, slots=True)
class ProtectedOutputManifest:
    identity: ProtectedOutputIdentity
    expected_example_ids: list[str]
    shard_max_records: int
    shards: list[ProtectedOutputShard] = field(default_factory=list)
    status: str = "IN_PROGRESS"
    created_at: str = field(default_factory=utc_now)
    completed_at: str | None = None
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProtectedOutputManifest:
        _require_exact_keys(value, cls.__dataclass_fields__, context="protected manifest")
        try:
            if not isinstance(value["expected_example_ids"], list):
                raise TypeError("expected_example_ids is not a list")
            if not isinstance(value["shards"], list):
                raise TypeError("shards is not a list")
            identity = ProtectedOutputIdentity.from_dict(value["identity"])
            shards = [ProtectedOutputShard.from_dict(item) for item in value["shards"]]
            manifest = cls(
                identity=identity,
                expected_example_ids=list(value["expected_example_ids"]),
                shard_max_records=int(value["shard_max_records"]),
                shards=shards,
                status=str(value["status"]),
                created_at=str(value["created_at"]),
                completed_at=value.get("completed_at"),
                schema_version=int(value.get("schema_version", 1)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ProtectedOutputIntegrityError(f"invalid protected manifest: {error}") from error
        manifest.validate()
        return manifest

    def validate(self) -> None:
        self.identity.validate()
        if not self.expected_example_ids or any(
            not isinstance(item, str) or not item for item in self.expected_example_ids
        ):
            raise ProtectedOutputIntegrityError("expected example IDs must be non-empty strings")
        if len(set(self.expected_example_ids)) != len(self.expected_example_ids):
            raise ProtectedOutputIntegrityError("expected example IDs contain duplicates")
        if self.expected_example_ids != sorted(self.expected_example_ids):
            raise ProtectedOutputIntegrityError("expected example IDs must be sorted")
        if (
            not isinstance(self.shard_max_records, int)
            or isinstance(self.shard_max_records, bool)
            or self.shard_max_records <= 0
        ):
            raise ProtectedOutputIntegrityError("shard_max_records must be positive")
        if self.schema_version != 1:
            raise ProtectedOutputIntegrityError("unsupported protected manifest schema version")
        _validate_text("created_at", self.created_at)
        if self.completed_at is not None and not isinstance(self.completed_at, str):
            raise ProtectedOutputIntegrityError("completed_at must be a string or null")
        if self.status not in {"IN_PROGRESS", "COMPLETE"}:
            raise ProtectedOutputIntegrityError("invalid output-set status")
        if self.status == "COMPLETE" and not self.completed_at:
            raise ProtectedOutputIntegrityError("complete manifest is missing completed_at")
        if self.status == "IN_PROGRESS" and self.completed_at is not None:
            raise ProtectedOutputIntegrityError("in-progress manifest has completed_at")
        incomplete = [
            index for index, shard in enumerate(self.shards) if shard.status == "IN_PROGRESS"
        ]
        if len(incomplete) > 1 or (incomplete and incomplete[0] != len(self.shards) - 1):
            raise ProtectedOutputIntegrityError("only the final shard may be in progress")
        if self.status == "COMPLETE" and incomplete:
            raise ProtectedOutputIntegrityError("complete output set contains incomplete shard")
        for index, shard in enumerate(self.shards):
            if shard.record_count > self.shard_max_records:
                raise ProtectedOutputIntegrityError("shard exceeds shard_max_records")
            if shard.status == "IN_PROGRESS" and shard.record_count >= self.shard_max_records:
                raise ProtectedOutputIntegrityError("full shard cannot remain in progress")
            if index < len(self.shards) - 1 and shard.record_count != self.shard_max_records:
                raise ProtectedOutputIntegrityError("non-final shard is not full")
        expected_filenames = [f"shard_{index:03d}.jsonl" for index in range(len(self.shards))]
        if [shard.filename for shard in self.shards] != expected_filenames:
            raise ProtectedOutputIntegrityError("shard filenames are not contiguous")


@dataclass(frozen=True, slots=True)
class ProtectedOutputVerification:
    manifest: ProtectedOutputManifest
    records: tuple[ProtectedOutputRecord, ...]
    record_sha256: dict[str, str]
    missing_example_ids: tuple[str, ...]


class ProtectedOutputWriter:
    """Append-only JSONL writer with hash-bound, incrementally durable manifests."""

    def __init__(self, output_directory: Path | str) -> None:
        self.output_directory = Path(output_directory)

    @classmethod
    def create(
        cls,
        root: Path | str,
        identity: ProtectedOutputIdentity,
        expected_example_ids: Iterable[str],
        *,
        shard_max_records: int = 100,
    ) -> ProtectedOutputWriter:
        identity.validate()
        directory = Path(root) / identity.model_id / identity.output_set_id
        if directory.exists():
            raise FileExistsError(f"protected output set already exists: {directory}")
        expected = sorted(expected_example_ids)
        manifest = ProtectedOutputManifest(
            identity=identity,
            expected_example_ids=expected,
            shard_max_records=shard_max_records,
        )
        manifest.validate()
        directory.mkdir(parents=True)
        atomic_write_json(directory / _MANIFEST_NAME, manifest.to_dict())
        return cls(directory)

    @classmethod
    def open(cls, output_directory: Path | str) -> ProtectedOutputWriter:
        writer = cls(output_directory)
        writer.verify()
        return writer

    def append(self, record: ProtectedOutputRecord) -> None:
        verification = self.verify()
        manifest = verification.manifest
        if manifest.status == "COMPLETE":
            raise ProtectedOutputIntegrityError("cannot append to a complete output set")
        record.validate()
        self._assert_record_identity(record, manifest.identity)
        if record.example_id not in manifest.expected_example_ids:
            raise ProtectedOutputIntegrityError(f"unexpected example ID: {record.example_id}")
        if record.example_id in verification.record_sha256:
            raise ProtectedOutputIntegrityError(f"duplicate example ID: {record.example_id}")

        shards = list(manifest.shards)
        if shards and shards[-1].status == "IN_PROGRESS":
            shard_index = len(shards) - 1
            shard = shards[-1]
        else:
            shard_index = len(shards)
            shard = None
        filename = f"shard_{shard_index:03d}.jsonl"
        path = self.output_directory / filename
        line = canonical_json_bytes(record.to_dict())
        with path.open("ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

        records, _ = self._read_shard(path)
        status = "COMPLETE" if len(records) >= manifest.shard_max_records else "IN_PROGRESS"
        updated = ProtectedOutputShard(
            filename=filename,
            status=status,
            sha256=sha256_file(path),
            byte_size=path.stat().st_size,
            record_count=len(records),
            example_ids=[item.example_id for item in records],
        )
        if shard is None:
            shards.append(updated)
        else:
            shards[-1] = updated
        atomic_write_json(
            self.output_directory / _MANIFEST_NAME,
            replace(manifest, shards=shards).to_dict(),
        )

    def complete(self) -> ProtectedOutputVerification:
        verification = self.verify()
        manifest = verification.manifest
        if manifest.status == "COMPLETE":
            return self.verify(require_complete=True)
        if verification.missing_example_ids:
            raise ProtectedOutputIntegrityError(
                "cannot complete output set; missing example IDs: "
                f"{list(verification.missing_example_ids)}"
            )
        completed_at = utc_now()
        complete_manifest = replace(
            manifest,
            shards=[replace(shard, status="COMPLETE") for shard in manifest.shards],
            status="COMPLETE",
            completed_at=completed_at,
        )
        atomic_write_json(self.output_directory / _MANIFEST_NAME, complete_manifest.to_dict())
        atomic_write_json(
            self.output_directory / _COMPLETE_NAME,
            {
                "status": "COMPLETE",
                "output_set_id": manifest.identity.output_set_id,
                "manifest_sha256": sha256_file(self.output_directory / _MANIFEST_NAME),
                "completed_at": completed_at,
            },
        )
        return self.verify(require_complete=True)

    def verify(self, *, require_complete: bool = False) -> ProtectedOutputVerification:
        manifest_path = self.output_directory / _MANIFEST_NAME
        if not manifest_path.is_file():
            raise ProtectedOutputIntegrityError("protected output manifest is missing")
        try:
            raw_manifest = json.loads(manifest_path.read_text("utf-8"))
            manifest = ProtectedOutputManifest.from_dict(raw_manifest)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ProtectedOutputIntegrityError(
                f"cannot read protected manifest: {error}"
            ) from error

        declared = {shard.filename for shard in manifest.shards}
        discovered = {path.name for path in self.output_directory.glob("shard_*.jsonl")}
        if declared != discovered:
            missing = sorted(declared - discovered)
            unregistered = sorted(discovered - declared)
            raise ProtectedOutputIntegrityError(
                f"shard inventory mismatch; missing={missing}, unregistered={unregistered}"
            )

        records: list[ProtectedOutputRecord] = []
        hashes: dict[str, str] = {}
        seen: set[str] = set()
        for shard in manifest.shards:
            path = self.output_directory / shard.filename
            if sha256_file(path) != shard.sha256 or path.stat().st_size != shard.byte_size:
                raise ProtectedOutputIntegrityError(
                    f"shard hash or size mismatch: {shard.filename}"
                )
            shard_records, shard_hashes = self._read_shard(path)
            ids = [record.example_id for record in shard_records]
            if len(shard_records) != shard.record_count or ids != shard.example_ids:
                raise ProtectedOutputIntegrityError(
                    f"shard record inventory mismatch: {shard.filename}"
                )
            for record in shard_records:
                self._assert_record_identity(record, manifest.identity)
                if record.example_id in seen:
                    raise ProtectedOutputIntegrityError(
                        f"duplicate example ID across shards: {record.example_id}"
                    )
                if record.example_id not in manifest.expected_example_ids:
                    raise ProtectedOutputIntegrityError(
                        f"unexpected example ID in shard: {record.example_id}"
                    )
                seen.add(record.example_id)
            records.extend(shard_records)
            hashes.update(shard_hashes)

        missing_ids = tuple(sorted(set(manifest.expected_example_ids) - seen))
        if manifest.status == "COMPLETE":
            if missing_ids:
                raise ProtectedOutputIntegrityError(
                    f"complete output set is missing example IDs: {list(missing_ids)}"
                )
            sentinel_path = self.output_directory / _COMPLETE_NAME
            if not sentinel_path.is_file():
                raise ProtectedOutputIntegrityError(
                    "complete output set has no completion sentinel"
                )
            try:
                sentinel = json.loads(sentinel_path.read_text("utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ProtectedOutputIntegrityError(
                    f"invalid completion sentinel: {error}"
                ) from error
            if (
                sentinel.get("status") != "COMPLETE"
                or sentinel.get("output_set_id") != manifest.identity.output_set_id
                or sentinel.get("manifest_sha256") != sha256_file(manifest_path)
                or sentinel.get("completed_at") != manifest.completed_at
            ):
                raise ProtectedOutputIntegrityError("completion sentinel does not bind manifest")
        elif (self.output_directory / _COMPLETE_NAME).exists():
            raise ProtectedOutputIntegrityError("in-progress output set has a completion sentinel")
        if require_complete and manifest.status != "COMPLETE":
            raise ProtectedOutputIntegrityError(
                f"protected output set is incomplete; missing example IDs: {list(missing_ids)}"
            )
        return ProtectedOutputVerification(manifest, tuple(records), hashes, missing_ids)

    @staticmethod
    def _read_shard(
        path: Path,
    ) -> tuple[list[ProtectedOutputRecord], dict[str, str]]:
        records: list[ProtectedOutputRecord] = []
        hashes: dict[str, str] = {}
        try:
            with path.open("rb") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.endswith(b"\n"):
                        raise ProtectedOutputIntegrityError(
                            f"{path.name}:{line_number}: incomplete JSONL record"
                        )
                    try:
                        value = json.loads(line)
                    except (UnicodeError, json.JSONDecodeError) as error:
                        raise ProtectedOutputIntegrityError(
                            f"{path.name}:{line_number}: invalid JSON"
                        ) from error
                    if not isinstance(value, dict):
                        raise ProtectedOutputIntegrityError(
                            f"{path.name}:{line_number}: record is not an object"
                        )
                    record = ProtectedOutputRecord.from_dict(value)
                    if canonical_json_bytes(record.to_dict()) != line:
                        raise ProtectedOutputIntegrityError(
                            f"{path.name}:{line_number}: record is not canonical JSON"
                        )
                    records.append(record)
                    hashes[record.example_id] = _record_hash(record.to_dict())
        except OSError as error:
            raise ProtectedOutputIntegrityError(
                f"cannot read shard {path.name}: {error}"
            ) from error
        if not records:
            raise ProtectedOutputIntegrityError(f"empty shard: {path.name}")
        return records, hashes

    @staticmethod
    def _assert_record_identity(
        record: ProtectedOutputRecord, identity: ProtectedOutputIdentity
    ) -> None:
        fields = (
            "output_set_id",
            "protocol_id",
            "model_id",
            "model_checkpoint_sha256",
            "generation_config_id",
            "generation_config_sha256",
        )
        mismatch = [field for field in fields if getattr(record, field) != getattr(identity, field)]
        if mismatch:
            raise ProtectedOutputIntegrityError(
                f"record identity differs from frozen manifest: {mismatch}"
            )


@dataclass(frozen=True, slots=True)
class DeterministicGraderResult:
    """Score contract for a deterministic grader applied to one immutable output."""

    output_set_id: str
    example_id: str
    grader_id: str
    grader_version: str
    grader_kind: str
    grader_config_sha256: str
    source_record_sha256: str
    passed: bool
    score: float
    component_scores: dict[str, float] = field(default_factory=dict)
    failure_codes: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for name in ("output_set_id", "example_id", "grader_id", "grader_version"):
            _validate_text(name, getattr(self, name))
        if self.grader_kind != "DETERMINISTIC":
            raise ProtectedOutputIntegrityError("grader_kind must be DETERMINISTIC")
        _validate_sha256("grader_config_sha256", self.grader_config_sha256)
        _validate_sha256("source_record_sha256", self.source_record_sha256)
        if type(self.passed) is not bool:
            raise ProtectedOutputIntegrityError("passed must be boolean")
        if (
            not isinstance(self.score, (int, float))
            or isinstance(self.score, bool)
            or not math.isfinite(self.score)
        ):
            raise ProtectedOutputIntegrityError("score must be finite")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ProtectedOutputIntegrityError("score must be in [0, 1]")
        if not isinstance(self.component_scores, dict) or any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not 0.0 <= float(value) <= 1.0
            for name, value in self.component_scores.items()
        ):
            raise ProtectedOutputIntegrityError(
                "component scores must be named finite [0, 1] values"
            )
        if (
            not isinstance(self.failure_codes, list)
            or any(not isinstance(value, str) or not value for value in self.failure_codes)
            or len(set(self.failure_codes)) != len(self.failure_codes)
        ):
            raise ProtectedOutputIntegrityError("failure codes must be unique non-empty strings")
        if not isinstance(self.details, dict):
            raise ProtectedOutputIntegrityError("grader details must be an object")
        _ensure_json(self.details, context="grader details")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["score"] = float(self.score)
        value["component_scores"] = {
            name: float(score) for name, score in self.component_scores.items()
        }
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DeterministicGraderResult:
        _require_exact_keys(value, cls.__dataclass_fields__, context="grader result")
        try:
            result = cls(**{name: value[name] for name in cls.__dataclass_fields__})
        except (KeyError, TypeError) as error:
            raise ProtectedOutputIntegrityError(f"invalid grader result: {error}") from error
        result.validate()
        return result


def publish_deterministic_rescore(
    output_directory: Path | str,
    *,
    rescore_id: str,
    grader_id: str,
    grader_version: str,
    grader_config_sha256: str,
    results: Iterable[DeterministicGraderResult],
) -> Path:
    """Publish scores separately while binding them to immutable source records."""

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
            raise ProtectedOutputIntegrityError(f"duplicate grader result: {result.example_id}")
        if verification.record_sha256.get(result.example_id) != result.source_record_sha256:
            raise ProtectedOutputIntegrityError(
                f"grader result is not bound to source record: {result.example_id}"
            )
        seen.add(result.example_id)
    expected = set(verification.manifest.expected_example_ids)
    if seen != expected:
        raise ProtectedOutputIntegrityError(
            f"grader results incomplete; missing={sorted(expected - seen)}, "
            f"extra={sorted(seen - expected)}"
        )

    parent = source / "rescores"
    destination = parent / rescore_id
    if destination.exists():
        raise FileExistsError(f"rescore already exists: {destination}")
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{rescore_id}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        results_path = staging / "results.jsonl"
        with results_path.open("wb") as handle:
            for result in sorted(materialized, key=lambda item: item.example_id):
                handle.write(canonical_json_bytes(result.to_dict()))
            handle.flush()
            os.fsync(handle.fileno())
        source_manifest_path = source / _MANIFEST_NAME
        manifest = {
            "schema_version": 1,
            "status": "COMPLETE",
            "rescore_id": rescore_id,
            "output_set_id": verification.manifest.identity.output_set_id,
            "source_manifest_sha256": sha256_file(source_manifest_path),
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
        os.replace(staging, destination)
        verify_deterministic_rescore(destination, source)
        return destination
    except BaseException:
        broken = staging if staging.exists() else destination
        if broken.exists():
            # Keep broken publication outside the finalized rescore namespace.
            invalid = parent / f"INVALID-{rescore_id}-{uuid.uuid4().hex[:8]}"
            os.replace(broken, invalid)
        raise


def verify_deterministic_rescore(
    rescore_directory: Path | str, output_directory: Path | str
) -> tuple[dict[str, Any], tuple[DeterministicGraderResult, ...]]:
    """Verify that a score set still binds to the original completed generations."""

    rescore = Path(rescore_directory)
    source = Path(output_directory)
    verification = ProtectedOutputWriter(source).verify(require_complete=True)
    required = ("manifest.json", "results.jsonl", "RESCORE_COMPLETE.json")
    missing_files = [name for name in required if not (rescore / name).is_file()]
    if missing_files:
        raise ProtectedOutputIntegrityError(f"incomplete rescore; missing {missing_files}")
    try:
        manifest = json.loads((rescore / "manifest.json").read_text("utf-8"))
        sentinel = json.loads((rescore / "RESCORE_COMPLETE.json").read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProtectedOutputIntegrityError(f"invalid rescore metadata: {error}") from error
    if not isinstance(manifest, dict) or not isinstance(sentinel, dict):
        raise ProtectedOutputIntegrityError("rescore metadata must be JSON objects")
    expected_manifest_fields = {
        "schema_version",
        "status",
        "rescore_id",
        "output_set_id",
        "source_manifest_sha256",
        "source_shard_sha256",
        "grader_id",
        "grader_version",
        "grader_kind",
        "grader_config_sha256",
        "expected_example_ids",
        "result_count",
        "results_sha256",
        "created_at",
    }
    if set(manifest) != expected_manifest_fields:
        raise ProtectedOutputIntegrityError("rescore manifest fields differ from schema")
    if manifest.get("schema_version") != 1 or manifest.get("grader_kind") != "DETERMINISTIC":
        raise ProtectedOutputIntegrityError("invalid rescore schema or grader kind")
    if (
        manifest.get("status") != "COMPLETE"
        or manifest.get("output_set_id") != verification.manifest.identity.output_set_id
        or manifest.get("source_manifest_sha256") != sha256_file(source / _MANIFEST_NAME)
        or manifest.get("source_shard_sha256")
        != {shard.filename: shard.sha256 for shard in verification.manifest.shards}
        or manifest.get("expected_example_ids") != verification.manifest.expected_example_ids
    ):
        raise ProtectedOutputIntegrityError("rescore manifest does not bind source outputs")
    for name in ("rescore_id", "grader_id", "grader_version"):
        _validate_text(name, manifest.get(name))
    _validate_sha256("grader_config_sha256", manifest.get("grader_config_sha256"))
    results_path = rescore / "results.jsonl"
    if manifest.get("results_sha256") != sha256_file(results_path):
        raise ProtectedOutputIntegrityError("rescore result hash mismatch")
    if (
        sentinel.get("status") != "COMPLETE"
        or sentinel.get("rescore_id") != manifest.get("rescore_id")
        or sentinel.get("manifest_sha256") != sha256_file(rescore / "manifest.json")
    ):
        raise ProtectedOutputIntegrityError("rescore completion sentinel mismatch")

    results: list[DeterministicGraderResult] = []
    seen: set[str] = set()
    try:
        with results_path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith(b"\n"):
                    raise ProtectedOutputIntegrityError(
                        f"results.jsonl:{line_number}: incomplete record"
                    )
                try:
                    value = json.loads(line)
                except (UnicodeError, json.JSONDecodeError) as error:
                    raise ProtectedOutputIntegrityError(
                        f"results.jsonl:{line_number}: invalid JSON"
                    ) from error
                if not isinstance(value, dict):
                    raise ProtectedOutputIntegrityError("grader result is not an object")
                result = DeterministicGraderResult.from_dict(value)
                if canonical_json_bytes(result.to_dict()) != line:
                    raise ProtectedOutputIntegrityError("grader result is not canonical JSON")
                if result.example_id in seen:
                    raise ProtectedOutputIntegrityError(
                        f"duplicate grader result: {result.example_id}"
                    )
                if result.output_set_id != verification.manifest.identity.output_set_id:
                    raise ProtectedOutputIntegrityError("grader output-set identity mismatch")
                if result.grader_id != manifest["grader_id"]:
                    raise ProtectedOutputIntegrityError("grader identity mismatch")
                if result.grader_version != manifest["grader_version"]:
                    raise ProtectedOutputIntegrityError("grader version mismatch")
                if result.grader_kind != "DETERMINISTIC":
                    raise ProtectedOutputIntegrityError("grader kind mismatch")
                if result.grader_config_sha256 != manifest["grader_config_sha256"]:
                    raise ProtectedOutputIntegrityError("grader configuration mismatch")
                if verification.record_sha256.get(result.example_id) != result.source_record_sha256:
                    raise ProtectedOutputIntegrityError("grader result source record mismatch")
                seen.add(result.example_id)
                results.append(result)
    except OSError as error:
        raise ProtectedOutputIntegrityError(f"cannot read grader results: {error}") from error
    expected = set(verification.manifest.expected_example_ids)
    if seen != expected or manifest.get("result_count") != len(results):
        raise ProtectedOutputIntegrityError(
            f"rescore result inventory mismatch; missing={sorted(expected - seen)}, "
            f"extra={sorted(seen - expected)}"
        )
    return manifest, tuple(results)
