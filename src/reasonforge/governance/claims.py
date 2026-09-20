"""Machine-readable claim ledger contracts."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceClass(StrEnum):
    MEASURED = "MEASURED"
    DERIVED = "DERIVED"
    PROJECTED = "PROJECTED"
    RESCORED = "RESCORED"
    PENDING = "PENDING"


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(pattern=r"^C\d{3}$")
    authorized_wording: str
    supporting_experiment_ids: list[str]
    artifact_paths: list[str]
    statistical_qualifier: str
    evidence_class: EvidenceClass
    prohibited_stronger_wording: str
    checkpoint_id: str | None = None

    @model_validator(mode="after")
    def measured_claim_has_evidence(self) -> Claim:
        if self.evidence_class in {
            EvidenceClass.MEASURED,
            EvidenceClass.RESCORED,
        } and (not self.supporting_experiment_ids or not self.artifact_paths):
            raise ValueError("measured claims require experiments and artifacts")
        return self


class ClaimLedger(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = 1
    claims: list[Claim]

    @model_validator(mode="after")
    def unique_ids(self) -> ClaimLedger:
        ids = [claim.claim_id for claim in self.claims]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate claim IDs")
        return self


def load_claim_ledger(path: Path) -> ClaimLedger:
    return ClaimLedger.model_validate_json(path.read_text(encoding="utf-8"))


def write_claim_ledger(ledger: ClaimLedger, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
