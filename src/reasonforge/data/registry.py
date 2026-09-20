"""Validated, fail-closed access to the data source registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError

from reasonforge.config import SplitRole

DEFAULT_SOURCE_REGISTRY = Path("data/source_registry.yaml")
DEFAULT_SOURCE_SCHEMA = Path("schemas/source_registry.schema.json")


class SourceRegistryError(ValueError):
    """Base error for malformed or scientifically unsafe source metadata."""


class SourceRegistrySchemaError(SourceRegistryError):
    """The registry or its JSON Schema is invalid."""


class SourceAuthorizationError(SourceRegistryError):
    """A requested source use is forbidden by the registry."""


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """A source entry that has passed schema and governance validation."""

    source_id: str
    name: str
    kind: str
    uri: str
    revision: str
    revision_kind: str
    role: SplitRole
    protection: str
    access_status: str
    governance_status: str
    training_allowed: bool
    raw: dict[str, Any]

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> SourceRecord:
        access = raw["access"]
        governance = raw["governance"]
        return cls(
            source_id=str(raw["source_id"]),
            name=str(raw["name"]),
            kind=str(raw["kind"]),
            uri=str(raw["uri"]),
            revision=str(raw["revision"]),
            revision_kind=str(raw["revision_kind"]),
            role=SplitRole(raw["role"]),
            protection=str(raw["protection"]),
            access_status=str(access["status"]),
            governance_status=str(governance["status"]),
            training_allowed=bool(governance["training_allowed"]),
            raw=raw,
        )

    @property
    def is_blocked(self) -> bool:
        return self.access_status.startswith("BLOCKED") or self.governance_status.startswith(
            "BLOCKED"
        )

    @property
    def is_protected(self) -> bool:
        return self.protection != "NONE" or self.role in {
            SplitRole.LOCKED_FINAL,
            SplitRole.EXTERNAL_PROTECTED,
        }

    def require_acquisition_allowed(self) -> None:
        """Reject acquisition of gated/license-blocked sources before any I/O."""
        if self.is_blocked:
            raise SourceAuthorizationError(
                f"source {self.source_id!r} is blocked: "
                f"access={self.access_status}, governance={self.governance_status}"
            )

    def require_training_allowed(self) -> None:
        """Reject any source that cannot legally and scientifically enter gradients."""
        if self.is_blocked or self.is_protected or self.role is not SplitRole.TRAIN:
            raise SourceAuthorizationError(
                f"source {self.source_id!r} is not eligible for training "
                f"(role={self.role.value}, protection={self.protection}, "
                f"status={self.governance_status})"
            )
        if not self.training_allowed:
            raise SourceAuthorizationError(f"source {self.source_id!r} has training_allowed=false")


@dataclass(frozen=True, slots=True)
class SourceRegistry:
    """Immutable view of the validated registry."""

    registry_version: str
    project: str
    reviewed_at: str
    sources: tuple[SourceRecord, ...]
    raw: dict[str, Any]

    def get(self, source_id: str) -> SourceRecord:
        matches = [source for source in self.sources if source.source_id == source_id]
        if not matches:
            raise KeyError(f"unknown source_id: {source_id}")
        return matches[0]


def _read_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise SourceRegistrySchemaError(f"{label} does not exist: {path}")
    try:
        if path.suffix.lower() == ".json":
            import json

            value = json.loads(path.read_text(encoding="utf-8"))
        else:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise SourceRegistrySchemaError(f"could not parse {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SourceRegistrySchemaError(f"{label} must contain a top-level mapping")
    return value


def _validate_schema(raw: dict[str, Any], schema: dict[str, Any]) -> None:
    try:
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        errors = sorted(validator.iter_errors(raw), key=lambda error: list(error.absolute_path))
    except SchemaError as exc:
        raise SourceRegistrySchemaError(f"invalid source registry schema: {exc.message}") from exc
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path) or "<root>"
        raise SourceRegistrySchemaError(
            f"source registry does not match schema at {location}: {first.message}"
        )


def _validate_governance(entries: list[dict[str, Any]]) -> None:
    ids = [str(entry["source_id"]) for entry in entries]
    if len(ids) != len(set(ids)):
        duplicates = sorted({source_id for source_id in ids if ids.count(source_id) > 1})
        raise SourceRegistryError(f"duplicate source IDs: {duplicates}")

    for entry in entries:
        source = SourceRecord.from_raw(entry)
        if (source.is_protected or source.is_blocked) and source.training_allowed:
            raise SourceRegistryError(
                f"source {source.source_id!r} is protected or blocked but training_allowed=true"
            )
        if source.role is not SplitRole.TRAIN and source.training_allowed:
            raise SourceRegistryError(
                f"source {source.source_id!r} has non-TRAIN role but training_allowed=true"
            )


def validate_source_registry(
    raw: dict[str, Any],
    schema: dict[str, Any],
) -> SourceRegistry:
    """Validate structure plus scientific governance invariants."""
    _validate_schema(raw, schema)
    entries_value = raw["sources"]
    if not isinstance(entries_value, list):  # Defensive; the schema already checks this.
        raise SourceRegistrySchemaError("sources must be a list")
    entries = [dict(entry) for entry in entries_value]
    _validate_governance(entries)
    return SourceRegistry(
        registry_version=str(raw["registry_version"]),
        project=str(raw["project"]),
        reviewed_at=str(raw["reviewed_at"]),
        sources=tuple(SourceRecord.from_raw(entry) for entry in entries),
        raw=raw,
    )


def load_source_registry(
    registry_path: Path = DEFAULT_SOURCE_REGISTRY,
    schema_path: Path = DEFAULT_SOURCE_SCHEMA,
) -> SourceRegistry:
    """Load and validate the registry against its checked-in JSON Schema."""
    raw = _read_mapping(registry_path, label="source registry")
    schema = _read_mapping(schema_path, label="source registry schema")
    return validate_source_registry(raw, schema)
