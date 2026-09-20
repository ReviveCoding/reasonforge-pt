"""Canonical row schema with exactly one scientific role."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reasonforge.config import SplitRole


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", normalized).split())


def content_hash(prompt: str, reference: str | None = None) -> str:
    material = f"{normalize_text(prompt)}\x1f{normalize_text(reference or '')}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class CanonicalRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    example_id: str
    source: str
    source_revision: str
    source_license: str
    task_family: str
    split_role: SplitRole
    prompt: str = Field(min_length=1)
    reference: str | None = None
    chosen: str | None = None
    rejected: str | None = None
    tool_schema: str | None = None
    difficulty: float | None = Field(default=None, ge=0.0, le=1.0)
    reliability: float | None = Field(default=None, ge=0.0, le=1.0)
    verification_status: str
    token_count: int = Field(ge=0)
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def preference_pair_is_complete(self) -> CanonicalRow:
        if (self.chosen is None) != (self.rejected is None):
            raise ValueError("chosen and rejected must be supplied together")
        expected = content_hash(self.prompt, self.reference)
        if self.content_hash != expected:
            raise ValueError("content_hash does not match normalized content")
        if self.split_role in {
            SplitRole.LOCKED_FINAL,
            SplitRole.EXTERNAL_PROTECTED,
        } and self.metadata.get("gradient_eligible", False):
            raise ValueError("protected rows cannot be gradient eligible")
        return self

    def parquet_record(self) -> dict[str, Any]:
        record = self.model_dump(mode="json")
        record["metadata"] = json.dumps(record["metadata"], sort_keys=True)
        return record
