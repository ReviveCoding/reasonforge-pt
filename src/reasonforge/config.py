"""Shared scientific identities and configuration loading."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SplitRole(StrEnum):
    TRAIN = "TRAIN"
    DEVELOPMENT = "DEVELOPMENT"
    POLICY = "POLICY"
    QUALIFICATION = "QUALIFICATION"
    LOCKED_FINAL = "LOCKED_FINAL"
    EXTERNAL_PROTECTED = "EXTERNAL_PROTECTED"


class ScientificState(StrEnum):
    PILOT = "PILOT"
    DEVELOPMENT = "DEVELOPMENT"
    CANDIDATE_SELECTION = "CANDIDATE_SELECTION"
    FROZEN = "FROZEN"
    QUALIFICATION = "QUALIFICATION"
    LOCKED_FINAL = "LOCKED_FINAL"
    ANALYSIS = "ANALYSIS"
    CLOSED = "CLOSED"


class ExperimentStatus(StrEnum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    COMPLETE = "COMPLETE"
    INTERRUPTED = "INTERRUPTED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(pattern=r"^E\d{2,3}$")
    name: str
    family: str
    priority: str = Field(pattern=r"^P[012]$")
    scientific_state: ScientificState
    backend: str
    heavy_gpu: bool = False
    depends_on: list[str] = Field(default_factory=list)
    question: str
    status: ExperimentStatus = ExperimentStatus.PENDING
    blocked_reason: str | None = None

    @model_validator(mode="after")
    def blocked_has_reason(self) -> ExperimentSpec:
        if self.status == ExperimentStatus.BLOCKED and not self.blocked_reason:
            raise ValueError("blocked experiments require blocked_reason")
        return self


class ExperimentRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_id: str
    frozen: bool = False
    experiments: list[ExperimentSpec]

    @model_validator(mode="after")
    def unique_and_acyclic_refs(self) -> ExperimentRegistry:
        ids = [entry.experiment_id for entry in self.experiments]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate experiment IDs")
        known = set(ids)
        for entry in self.experiments:
            missing = set(entry.depends_on) - known
            if missing:
                raise ValueError(f"{entry.experiment_id} has unknown dependencies: {missing}")
            if entry.experiment_id in entry.depends_on:
                raise ValueError(f"{entry.experiment_id} depends on itself")
        return self


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def identity_hash(value: Any) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_registry(path: Path) -> ExperimentRegistry:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ExperimentRegistry.model_validate(raw)
