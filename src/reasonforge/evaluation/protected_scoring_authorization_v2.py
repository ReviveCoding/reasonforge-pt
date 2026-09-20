"""Additive authorization for the qualified IFEval scorer backend."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reasonforge.checkpoints.hashing import sha256_file

from .protected_outputs import ProtectedOutputIntegrityError

AUTHORIZATION_V2_ID = "RFPT-PROTECTED-SCORING-AUTHORIZATION-v2"


def _verify_binding(value: object, label: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ProtectedOutputIntegrityError(f"scoring authorization v2 invalid {label} binding")
    path = Path(str(value.get("path")))
    digest = value.get("sha256")
    if not path.is_file() or not isinstance(digest, str) or sha256_file(path) != digest:
        raise ProtectedOutputIntegrityError(f"scoring authorization v2 {label} identity mismatch")
    return path


def verify_scoring_authorization_v2(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProtectedOutputIntegrityError(f"invalid scoring authorization v2: {error}") from error
    required = {
        "schema_version",
        "status",
        "authorization_id",
        "created_at",
        "parent_authorization_id",
        "recovery_protocol_id",
        "generation_semantics_changed",
        "scoring_routes_changed",
        "scores_computed",
        "semantic_outputs_accessed",
        "failed_preflight",
        "parent_authorization",
        "scoring_config",
        "recovery_config",
        "recovery_amendment",
        "frozen_matrix",
        "benchmark_registry",
        "cp60_manifest",
        "cp60_bindings",
        "backend_config",
        "backend_qualification",
        "requirements_lock",
        "code_inventory",
        "test_inventory",
        "documentation_inventory",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise ProtectedOutputIntegrityError("scoring authorization v2 fields differ from schema")
    if (
        raw.get("schema_version") != 1
        or raw.get("status") != "VERIFIED_SCORING_AUTHORIZATION"
        or raw.get("authorization_id") != AUTHORIZATION_V2_ID
        or raw.get("parent_authorization_id") != "RFPT-PROTECTED-SCORING-ORCHESTRATOR-v1"
        or raw.get("recovery_protocol_id") != "RFPT-PROTECTED-RECOVERY-v2"
        or raw.get("generation_semantics_changed") is not False
        or raw.get("scoring_routes_changed") is not False
        or raw.get("scores_computed") is not False
        or raw.get("semantic_outputs_accessed") is not False
        or not isinstance(raw.get("created_at"), str)
        or not raw.get("created_at")
    ):
        raise ProtectedOutputIntegrityError("scoring authorization v2 policy is invalid")
    failure = raw.get("failed_preflight")
    if (
        not isinstance(failure, Mapping)
        or failure.get("reason") != "PINNED_OFFICIAL_IFEVAL_BACKEND_UNAVAILABLE"
        or failure.get("scores_published") is not False
        or failure.get("rescore_directories") != 0
        or failure.get("scoring_locks") != 0
    ):
        raise ProtectedOutputIntegrityError("scoring authorization v2 failure ancestry is invalid")
    for label in (
        "parent_authorization",
        "scoring_config",
        "recovery_config",
        "recovery_amendment",
        "frozen_matrix",
        "benchmark_registry",
        "cp60_manifest",
        "cp60_bindings",
        "backend_config",
        "backend_qualification",
        "requirements_lock",
    ):
        _verify_binding(raw[label], label)
    for inventory_name in ("code_inventory", "test_inventory", "documentation_inventory"):
        inventory = raw.get(inventory_name)
        if not isinstance(inventory, Mapping) or not inventory:
            raise ProtectedOutputIntegrityError(f"scoring authorization v2 lacks {inventory_name}")
        for candidate, digest in inventory.items():
            file = Path(str(candidate))
            if not file.is_file() or not isinstance(digest, str) or sha256_file(file) != digest:
                raise ProtectedOutputIntegrityError(
                    f"scoring authorization v2 inventory mismatch: {file}"
                )
    return raw
