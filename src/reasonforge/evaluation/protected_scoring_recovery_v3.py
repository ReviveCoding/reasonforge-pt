"""Versioned authorization for Windows path-length scoring recovery."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from reasonforge.checkpoints.hashing import sha256_file

from .protected_outputs import ProtectedOutputIntegrityError

RECOVERY_V3_ID = "RFPT-PROTECTED-SCORING-RECOVERY-v3"


def load_recovery_v3_config(path: Path) -> Mapping[str, Any]:
    raw = yaml.safe_load(path.read_text("utf-8"))
    if (
        not isinstance(raw, Mapping)
        or raw.get("schema_version") != 1
        or raw.get("protocol_id") != RECOVERY_V3_ID
        or raw.get("failure") != "WINDOWS_PATH_LENGTH_BEFORE_FIRST_RESCORE_PUBLICATION"
        or raw.get("published_rescore_count_before_failure") != 0
        or raw.get("short_rescore_id") != "rfpt-sv3"
        or raw.get("suppress_scorer_logging") is not True
        or raw.get("generation_semantics_changed") is not False
        or raw.get("scoring_routes_changed") is not False
        or raw.get("semantic_outputs_inspected") is not False
        or raw.get("scores_computed") is not False
    ):
        raise ProtectedOutputIntegrityError("protected scoring recovery-v3 config is invalid")
    failed_lock = Path(str(raw.get("failed_lock")))
    if not failed_lock.is_file() or sha256_file(failed_lock) != raw.get("failed_lock_sha256"):
        raise ProtectedOutputIntegrityError("failed scoring lock identity changed")
    allowed = Path(str(raw.get("allowed_empty_rescore_directory")))
    if not allowed.is_dir() or any(allowed.iterdir()):
        raise ProtectedOutputIntegrityError("authorized failed rescore directory is not empty")
    return raw


def _verify_binding(value: object, label: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ProtectedOutputIntegrityError(f"recovery-v3 invalid {label} binding")
    path = Path(str(value.get("path")))
    if not path.is_file() or sha256_file(path) != value.get("sha256"):
        raise ProtectedOutputIntegrityError(f"recovery-v3 {label} identity mismatch")
    return path


def verify_recovery_v3_authorization(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProtectedOutputIntegrityError(
            f"invalid recovery-v3 authorization: {error}"
        ) from error
    bindings = {
        "recovery_config",
        "parent_authorization_v2",
        "backend_config",
        "backend_qualification",
        "requirements_lock",
        "scoring_config",
        "recovery_scorer_config",
        "recovery_amendment",
        "frozen_matrix",
        "benchmark_registry",
        "cp60_manifest",
        "cp60_bindings",
        "failed_lock",
    }
    required = {
        "schema_version",
        "status",
        "authorization_id",
        "created_at",
        "failure",
        "published_rescore_count_before_failure",
        "short_rescore_id",
        "suppress_scorer_logging",
        "generation_semantics_changed",
        "scoring_routes_changed",
        "scores_computed",
        "semantic_outputs_inspected",
        "observed_log_disclosure",
        *bindings,
        "code_inventory",
        "test_inventory",
        "documentation_inventory",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise ProtectedOutputIntegrityError("recovery-v3 authorization fields differ from schema")
    if (
        raw.get("schema_version") != 1
        or raw.get("status") != "VERIFIED_SCORING_RECOVERY_AUTHORIZATION"
        or raw.get("authorization_id") != RECOVERY_V3_ID
        or raw.get("failure") != "WINDOWS_PATH_LENGTH_BEFORE_FIRST_RESCORE_PUBLICATION"
        or raw.get("published_rescore_count_before_failure") != 0
        or raw.get("short_rescore_id") != "rfpt-sv3"
        or raw.get("suppress_scorer_logging") is not True
        or raw.get("generation_semantics_changed") is not False
        or raw.get("scoring_routes_changed") is not False
        or raw.get("scores_computed") is not False
        or raw.get("semantic_outputs_inspected") is not False
        or raw.get("observed_log_disclosure")
        != "ASTERISK_ONLY_RESPONSE_FROM_LANGUAGE_DETECTOR_ERROR"
    ):
        raise ProtectedOutputIntegrityError("recovery-v3 authorization policy is invalid")
    for label in bindings:
        _verify_binding(raw[label], label)
    for inventory_name in ("code_inventory", "test_inventory", "documentation_inventory"):
        inventory = raw.get(inventory_name)
        if not isinstance(inventory, Mapping) or not inventory:
            raise ProtectedOutputIntegrityError(f"recovery-v3 lacks {inventory_name}")
        for candidate, digest in inventory.items():
            file = Path(str(candidate))
            if not file.is_file() or sha256_file(file) != digest:
                raise ProtectedOutputIntegrityError(f"recovery-v3 inventory mismatch: {file}")
    return raw
