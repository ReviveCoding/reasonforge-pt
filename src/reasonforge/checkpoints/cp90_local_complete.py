"""Fail-closed qualification inputs for CP90 local completion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from reasonforge.checkpoints.hashing import sha256_file
from reasonforge.checkpoints.models import CheckpointIntegrityError
from reasonforge.checkpoints.project import ProjectCheckpointStore
from reasonforge.config import identity_hash

CP80_ID = "CP80_REPLICATION_COMPLETE"
CP90_ID = "CP90_LOCAL_COMPLETE"
CP80_BINDINGS = (
    "evidence/checkpoints/CP80_REPLICATION_COMPLETE/REPLICATION_COMPLETE_EVIDENCE_BINDINGS.json"
)
E26_ANALYSIS = "evidence/systems/E26/SYSTEMS_ANALYSIS.json"
E29_DECISION = "evidence/decisions/E29/FINAL_DECISION_v2.json"
E29_VALIDATION = "evidence/decisions/E29/VALIDATION_v2.json"
CP60_BINDINGS = "evidence/checkpoints/CP60_PRE_PROTECTED/PRE_PROTECTED_EVIDENCE_BINDINGS.json"
REVIEW_EVIDENCE = "evidence/reviews/FINAL_ADVERSARIAL_REVIEW.json"
HANDOFF = "evidence/RESUME_HANDOFF.md"
FINAL_FILES = (
    "README.md",
    "DATA_CARD.md",
    "MODEL_CARD.md",
    "EVAL_CARD.md",
    "RESULTS.md",
    "REPRODUCIBILITY.md",
    "LIMITATIONS.md",
    "reports/final/TECHNICAL_REPORT.md",
    "reports/final/FINAL_STATUS.md",
    "reports/final/RESUME_BULLETS.md",
    "reports/final/figures/FIGURE_MANIFEST.json",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointIntegrityError(message)


def _safe(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    _require(path != root and root in path.parents, f"CP90 unsafe path: {relative}")
    return path


def _object(root: Path, relative: str) -> dict[str, Any]:
    path = _safe(root, relative)
    _require(path.is_file() and path.stat().st_size > 0, f"CP90 artifact missing: {relative}")
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CheckpointIntegrityError(f"CP90 artifact unreadable: {relative}: {error}") from error
    if not isinstance(value, dict):
        raise CheckpointIntegrityError(f"CP90 artifact is not an object: {relative}")
    return value


def binding(root: Path, relative: str) -> dict[str, Any]:
    path = _safe(root, relative)
    _require(path.is_file() and path.stat().st_size > 0, f"CP90 artifact missing: {relative}")
    return {"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def verify_binding(root: Path, value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CheckpointIntegrityError(f"CP90 {label} binding is invalid")
    relative = value.get("path")
    if not isinstance(relative, str):
        raise CheckpointIntegrityError(f"CP90 {label} binding path is invalid")
    actual = binding(root, relative)
    _require(
        actual["sha256"] == value.get("sha256")
        and ("bytes" not in value or actual["bytes"] == value.get("bytes")),
        f"CP90 {label} binding changed: {relative}",
    )
    return actual


def _verify_registry(root: Path) -> tuple[list[str], list[str]]:
    value = yaml.safe_load(_safe(root, "configs/EXPERIMENT_REGISTRY.yaml").read_text("utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("experiments"), list):
        raise CheckpointIntegrityError("CP90 experiment registry is invalid")
    rows = value["experiments"]
    by_id = {row.get("experiment_id"): row for row in rows if isinstance(row, dict)}
    _require(len(by_id) == len(rows), "CP90 experiment registry IDs are invalid or duplicated")
    for experiment_id in ("E26", "E29"):
        _require(
            by_id.get(experiment_id, {}).get("status") == "COMPLETE",
            f"CP90 requires {experiment_id} COMPLETE",
        )
    required_p0 = [str(row["experiment_id"]) for row in rows if row.get("priority") == "P0"]
    incomplete_p0 = [item for item in required_p0 if by_id[item].get("status") != "COMPLETE"]
    _require(not incomplete_p0, f"CP90 P0 experiments are incomplete: {incomplete_p0}")
    active = [str(row["experiment_id"]) for row in rows if row.get("status") == "ACTIVE"]
    _require(not active, f"CP90 scientific runs remain ACTIVE: {active}")
    blocked = [str(row["experiment_id"]) for row in rows if row.get("status") == "BLOCKED"]
    return required_p0, blocked


def _verify_e26_e29(root: Path) -> dict[str, Any]:
    e26 = _object(root, E26_ANALYSIS)
    _require(e26.get("experiment_id") == "E26", "CP90 E26 identity mismatch")
    _require(e26.get("status") == "ANALYSIS_ARTIFACT_GENERATED", "CP90 E26 is incomplete")
    for index, item in enumerate(e26.get("source_bindings", [])):
        verify_binding(root, item, label=f"E26 source {index}")
    outputs = e26.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise CheckpointIntegrityError("CP90 E26 bound outputs are missing")
    for relative, item in outputs.items():
        if not isinstance(item, dict):
            raise CheckpointIntegrityError(f"CP90 E26 output binding is invalid: {relative}")
        verify_binding(root, {"path": relative, **item}, label=f"E26 output {relative}")

    e29 = _object(root, E29_DECISION)
    _require(
        e29.get("experiment_id") == "E29" and e29.get("status") == "COMPLETE",
        "CP90 E29 is incomplete",
    )
    _require(
        e29.get("decision") in {"PROMOTE", "RETAIN_BASELINE", "ITERATE", "BLOCK"},
        "CP90 E29 final decision is invalid",
    )
    inputs = e29.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise CheckpointIntegrityError("CP90 E29 bound inputs are missing")
    for index, item in enumerate(inputs):
        verify_binding(root, item, label=f"E29 input {index}")
    verify_binding(root, e29.get("report"), label="E29 report")
    validation = _object(root, E29_VALIDATION)
    _require(validation.get("status") == "PASSED", "CP90 E29 validation did not pass")
    _require(
        validation.get("decision_sha256") == sha256_file(_safe(root, E29_DECISION)),
        "CP90 E29 validation does not bind the current decision",
    )
    return {
        "e26": binding(root, E26_ANALYSIS),
        "e29": binding(root, E29_DECISION),
        "e29_validation": binding(root, E29_VALIDATION),
    }


def _verify_finalists(root: Path) -> dict[str, Any]:
    cp60 = _object(root, CP60_BINDINGS)
    finalists = cp60.get("finalists")
    if not isinstance(finalists, dict):
        raise CheckpointIntegrityError("CP90 CP60 finalist register is missing")
    expected = {"e16_terminal_uniform_seed1701", "e18_rcac_seed1701"}
    _require(set(finalists) == expected, "CP90 finalist inventory mismatch")
    result: dict[str, Any] = {}
    for finalist_id in sorted(expected):
        item = finalists[finalist_id]
        if not isinstance(item, dict):
            raise CheckpointIntegrityError(f"CP90 finalist is invalid: {finalist_id}")
        adapter = str(item.get("adapter_path", ""))
        model = f"{adapter}/adapter_model.safetensors"
        completion = (
            f"{adapter.rsplit('/final_adapter', 1)[0]}/checkpoint-32/CHECKPOINT_COMPLETE.json"
        )
        adapter_path = _safe(root, adapter)
        adapter_inventory = {
            path.relative_to(adapter_path).as_posix(): sha256_file(path)
            for path in sorted(adapter_path.rglob("*"))
            if path.is_file()
        }
        _require(bool(adapter_inventory), f"CP90 finalist adapter is empty: {finalist_id}")
        _require(
            identity_hash(adapter_inventory) == item.get("model_checkpoint_sha256"),
            f"CP90 finalist adapter changed: {finalist_id}",
        )
        _require(
            sha256_file(_safe(root, completion)) == item.get("checkpoint_completion_sha256"),
            f"CP90 finalist checkpoint receipt changed: {finalist_id}",
        )
        result[finalist_id] = {
            "adapter": binding(root, model),
            "adapter_inventory_identity": identity_hash(adapter_inventory),
            "checkpoint_completion": binding(root, completion),
        }
    return result


def _verify_review(root: Path) -> dict[str, Any]:
    review = _object(root, REVIEW_EVIDENCE)
    _require(
        review.get("status") == "COMPLETE" and review.get("verdict") == "PASS",
        "CP90 adversarial review is incomplete or did not pass",
    )
    findings = review.get("findings")
    if not isinstance(findings, list):
        raise CheckpointIntegrityError("CP90 adversarial findings must be a list")
    _require(
        not any(
            isinstance(item, dict)
            and item.get("severity") == "BLOCKING"
            and item.get("status") != "RESOLVED"
            for item in findings
        ),
        "CP90 adversarial review has unresolved blocking findings",
    )
    reviewed = review.get("reviewed_artifacts")
    if not isinstance(reviewed, list) or not reviewed:
        raise CheckpointIntegrityError("CP90 adversarial review has no bound artifact inventory")
    for index, item in enumerate(reviewed):
        verify_binding(root, item, label=f"reviewed artifact {index}")
    return binding(root, REVIEW_EVIDENCE)


def _verify_handoff(root: Path) -> dict[str, Any]:
    path = _safe(root, HANDOFF)
    text = path.read_text("utf-8")
    _require(
        "verified publication predecessor is `CP80_REPLICATION_COMPLETE`" in text,
        "CP90 handoff does not identify verified CP80",
    )
    _require(
        "Expected next milestone: `CP90_LOCAL_COMPLETE`" in text,
        "CP90 handoff does not authorize CP90 next",
    )
    _require("E29" in text, "CP90 handoff does not describe E29")
    return binding(root, HANDOFF)


def qualify_cp90(root: Path, *, command_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Return CP90 evidence only after every local-completion gate passes."""
    root = root.resolve()
    latest, _ = ProjectCheckpointStore(root / "evidence/checkpoints").verify_latest()
    _require(
        latest.checkpoint_id == CP80_ID and latest.project_status == "IN_PROGRESS",
        "CP90 requires CP80 to be the latest verified IN_PROGRESS checkpoint",
    )
    _require(not latest.active_experiment_ids, "CP90 CP80 manifest contains active experiments")
    cp80 = _object(root, CP80_BINDINGS)
    _require(cp80.get("checkpoint_id") == CP80_ID, "CP90 CP80 binding identity mismatch")
    e25 = cp80.get("e25_replication")
    if not isinstance(e25, dict) or e25.get("active_run_count") != 0:
        raise CheckpointIntegrityError("CP90 E25 reports an active scientific run")
    _require(
        e25.get("scoring_lock_evidence", {}).get("active") is False,
        "CP90 E25 scoring lock is active",
    )
    if not command_results or not all(item.get("exit_code") == 0 for item in command_results):
        raise CheckpointIntegrityError(
            "CP90 exact reproduction/resume command qualification failed"
        )
    for item in command_results:
        _require(
            isinstance(item.get("command"), list) and item["command"],
            "CP90 command qualification lacks an exact argv",
        )
    required_p0, blocked = _verify_registry(root)
    final_inventory = {relative: binding(root, relative) for relative in FINAL_FILES}
    return {
        "schema_version": 1,
        "checkpoint_id": CP90_ID,
        "status": "QUALIFIED_NOT_PUBLISHED",
        "project_status_on_publish": "LOCAL_COMPLETE",
        "cp80": binding(root, f"evidence/checkpoints/{CP80_ID}/CHECKPOINT_MANIFEST.json"),
        "cp80_evidence": binding(root, CP80_BINDINGS),
        "required_p0_complete": required_p0,
        "allowed_blocked_non_p0": blocked,
        "scientific_active_run_count": 0,
        "protected_shards": "VERIFIED_COMPLETE_BY_PREPUBLISH_INVENTORY",
        "e26_e29": _verify_e26_e29(root),
        "final_inventory": final_inventory,
        "claim_ledger": binding(root, "evidence/claim_ledger.json"),
        "adversarial_review": _verify_review(root),
        "retained_finalists": _verify_finalists(root),
        "handoff": _verify_handoff(root),
        "command_results": command_results,
    }
