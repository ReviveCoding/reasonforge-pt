"""Typed checkpoint manifests and scientific identity definitions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class CheckpointError(RuntimeError):
    """Base class for checkpoint integrity and publication failures."""


class CheckpointIntegrityError(CheckpointError):
    """Raised when an existing checkpoint cannot be trusted."""


class ResumeIdentityMismatch(CheckpointError):
    """Raised when resuming would change a scientifically material property."""


class RuntimeCheckpointKind(StrEnum):
    SFT = "SFT"
    DPO = "DPO"
    REWARD_MODEL = "REWARD_MODEL"
    GRPO = "GRPO"
    RCAC = "RCAC"
    STATEFUL_GRPO = "STATEFUL_GRPO"
    STATEFUL_RCAC = "STATEFUL_RCAC"


@dataclass(frozen=True)
class RunIdentity:
    """Properties that must match to continue under an existing run ID."""

    model_family: str
    starting_checkpoint: str
    adapter_architecture: str
    training_objective: str
    optimizer_configuration: str
    dataset_fingerprint: str
    data_role: str
    reward_architecture: str
    reward_weights: str
    sampler_configuration: str
    seed_semantics: str
    protected_access_state: str
    source_configuration: str

    def differences(self, other: RunIdentity) -> dict[str, tuple[str, str]]:
        differences: dict[str, tuple[str, str]] = {}
        for name in self.__dataclass_fields__:
            expected = getattr(self, name)
            actual = getattr(other, name)
            if expected != actual:
                differences[name] = (expected, actual)
        return differences

    def assert_compatible(self, other: RunIdentity) -> None:
        differences = self.differences(other)
        if differences:
            details = ", ".join(
                f"{name}: expected={values[0]!r}, actual={values[1]!r}"
                for name, values in differences.items()
            )
            raise ResumeIdentityMismatch(f"resume identity mismatch ({details})")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunIdentity:
        return cls(**{name: str(value[name]) for name in cls.__dataclass_fields__})


@dataclass
class ProjectCheckpointManifest:
    checkpoint_id: str
    project_status: str
    current_phase: str
    protocol_id: str
    source_tree_hash: str
    environment_identity: str
    completed_experiment_ids: list[str] = field(default_factory=list)
    active_experiment_ids: list[str] = field(default_factory=list)
    pending_experiment_ids: list[str] = field(default_factory=list)
    blocked_experiment_ids: list[str] = field(default_factory=list)
    git_commit: str | None = None
    git_dirty_state: str = "unknown"
    dataset_fingerprints: dict[str, str] = field(default_factory=dict)
    model_checkpoint_identities: dict[str, str] = field(default_factory=dict)
    reward_configuration_identity: str = "UNFROZEN"
    sampler_configuration_identity: str = "UNFROZEN"
    decision_policy_identity: str = "UNFROZEN"
    protected_access_state: str = "NOT_ACCESSED"
    artifact_inventory: list[str] = field(default_factory=list)
    next_authorized_action: str = ""
    created_at: str = field(default_factory=utc_now)
    schema_version: int = 1

    def validate(self) -> None:
        if (
            not isinstance(self.checkpoint_id, str)
            or not self.checkpoint_id
            or "/" in self.checkpoint_id
            or "\\" in self.checkpoint_id
        ):
            raise CheckpointIntegrityError("checkpoint_id must be a non-empty path component")
        required = {
            "project_status": self.project_status,
            "current_phase": self.current_phase,
            "protocol_id": self.protocol_id,
            "source_tree_hash": self.source_tree_hash,
            "environment_identity": self.environment_identity,
            "next_authorized_action": self.next_authorized_action,
        }
        missing = [
            name for name, value in required.items() if not isinstance(value, str) or not value
        ]
        if missing:
            raise CheckpointIntegrityError(
                f"manifest has empty required fields: {', '.join(sorted(missing))}"
            )
        categories = (
            self.completed_experiment_ids,
            self.active_experiment_ids,
            self.pending_experiment_ids,
            self.blocked_experiment_ids,
        )
        if any(
            not isinstance(group, list)
            or any(not isinstance(item, str) or not item for item in group)
            for group in categories
        ):
            raise CheckpointIntegrityError("experiment state categories must be string lists")
        all_ids = [item for group in categories for item in group]
        if len(all_ids) != len(set(all_ids)):
            raise CheckpointIntegrityError("experiment IDs overlap between state categories")
        mappings = (
            self.dataset_fingerprints,
            self.model_checkpoint_identities,
        )
        if any(
            not isinstance(mapping, dict)
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in mapping.items()
            )
            for mapping in mappings
        ):
            raise CheckpointIntegrityError("manifest identities must be string mappings")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProjectCheckpointManifest:
        known = {name for name in cls.__dataclass_fields__}
        manifest = cls(**{name: value[name] for name in known if name in value})
        manifest.validate()
        return manifest


@dataclass
class RuntimeCheckpointManifest:
    experiment_id: str
    run_id: str
    protocol_id: str
    kind: RuntimeCheckpointKind
    checkpoint_step: int
    identity: RunIdentity
    training_config_hash: str
    dataset_fingerprint: str
    source_revision: str
    starting_checkpoint_hash: str
    seed: int
    rng_state_inventory: list[str]
    sampler_state_hash: str
    reward_config_hash: str
    hardware_environment_identity: str
    expected_files: list[str]
    additional_state: dict[str, Any] = field(default_factory=dict)
    checkpoint_byte_size: int = 0
    completed_at: str = field(default_factory=utc_now)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["kind"] = self.kind.value
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RuntimeCheckpointManifest:
        payload = dict(value)
        payload["kind"] = RuntimeCheckpointKind(payload["kind"])
        payload["identity"] = RunIdentity.from_dict(payload["identity"])
        return cls(**payload)
