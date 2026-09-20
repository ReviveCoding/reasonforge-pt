"""Additive CP80 admission of the exact completed E25 inactive-lock recovery.

The copied validator differs only at lock admission and its diagnostic output.
Original E25 evidence, CP80 code, and E36 disposition/code bindings remain intact.
"""

from __future__ import annotations

from pathlib import Path
from types import FunctionType
from typing import Any, cast

from reasonforge.checkpoints import milestone_evidence as original
from reasonforge.checkpoints.hashing import atomic_write_json
from reasonforge.checkpoints.milestone_evidence import (
    _CP80_E25_BINDING,
    _CP80_E25_BINDING_SHA256,
    _CP80_E25_COMPLETION,
    _CP80_E25_MATRIX,
    _CP80_E25_MATRIX_SHA256,
    _CP80_E25_PROTOCOL,
    _CP80_E25_RESCORE_ID,
    _CP80_E25_ROOT,
    _CP80_E25_SCORING_LOCK,
    _CP80_E25_TRAINING_AUDIT,
    _CP80_POLICY_SHA256,
    MilestoneEvidence,
    _binding,
    _cp80_bound_file,
    _cp80_json_status,
    _json_object,
    _require,
    _safe_path,
)
from reasonforge.checkpoints.models import CheckpointIntegrityError
from reasonforge.config import ExperimentRegistry
from reasonforge.evaluation.e25_analysis_recovery_v1 import verify_completed_scoring_recovery
from reasonforge.evaluation.e25_analysis_recovery_v2 import (
    AUTHORIZATION as E25_ANALYSIS_AUTHORIZATION,
)

ROOT = "evidence/replication/cp80_inactive_lock_admission_v1"
AUTHORIZATION = f"{ROOT}/AUTHORIZATION.json"
VALIDATION = f"{ROOT}/CPU_VALIDATION.json"
POLICY_ID = "RFPT-CP80-INACTIVE-LOCK-ADMISSION-v1"
E25_ANALYSIS_AUTHORIZATION_SHA256 = (
    "bd95df89a52bd47029c63ceb1e7488e91bbcd43fafe9395eb54d1abaa8f0aa2d"
)
CODE_PATHS = (
    "src/reasonforge/checkpoints/cp80_inactive_lock_admission.py",
    "scripts/publish_cp80_inactive_lock_admission.py",
    "tests/test_cp80_inactive_lock_admission.py",
    "docs/CP80_INACTIVE_LOCK_ADMISSION.md",
    "src/reasonforge/checkpoints/milestone_evidence.py",
    "scripts/publish_milestone.py",
)


def _verify_scoring_lock(root: Path, complete: dict[str, Any]) -> dict[str, Any]:
    """Admit absence, or the exact frozen completed recovery; never an active lock."""
    lock = complete.get("scoring_lock")
    if lock == {"path": _CP80_E25_SCORING_LOCK, "present": False}:
        _require(
            not _safe_path(root, _CP80_E25_SCORING_LOCK).exists(),
            "CP80 E25 scoring lock is active or incorrectly reported",
        )
        return {"absent": True, "active": False, "disposition": "ABSENT"}
    _require(
        lock
        == {
            "path": _CP80_E25_SCORING_LOCK,
            "present": True,
            "disposition": "PRESERVED_INACTIVE_COMPLETED_RECOVERY_EVIDENCE",
        }
        and complete.get("active_run_count") == 0,
        "CP80 E25 present scoring lock lacks the exact inactive-recovery disposition",
    )
    try:
        receipt = verify_completed_scoring_recovery(root)
    except (OSError, ValueError) as error:
        raise CheckpointIntegrityError(
            f"CP80 E25 completed scoring recovery is invalid: {error}"
        ) from error
    _require(
        receipt.get("status") == "VERIFIED_COMPLETED_SCORING_RECOVERY"
        and receipt.get("preserved_locks_are_inactive_attempt_evidence") is True
        and isinstance(receipt.get("rescores"), dict)
        and len(receipt["rescores"]) == 28,
        "CP80 E25 scoring recovery receipt is incomplete",
    )
    lock_binding = _binding(root, _CP80_E25_SCORING_LOCK)
    _require(
        receipt.get("bindings", {}).get("original_lock") == lock_binding,
        "CP80 E25 recovery receipt does not bind the exact preserved lock",
    )
    authorization = _cp80_bound_file(
        root,
        complete.get("scoring_recovery_authorization"),
        label="E25 inactive-lock analysis authorization",
        expected_path=E25_ANALYSIS_AUTHORIZATION.as_posix(),
    )
    _, recorded = _json_object(root, E25_ANALYSIS_AUTHORIZATION.as_posix())
    _require(
        authorization["sha256"] == E25_ANALYSIS_AUTHORIZATION_SHA256
        and recorded.get("status") == "VERIFIED_ANALYSIS_RECOVERY_AUTHORIZATION"
        and recorded.get("completed_scoring_recovery") == receipt,
        "CP80 E25 analysis authorization does not bind the verified completed recovery",
    )
    return {
        "absent": False,
        "active": False,
        "disposition": "PRESERVED_INACTIVE_COMPLETED_RECOVERY_EVIDENCE",
        "preserved_lock": lock_binding,
        "analysis_recovery_authorization": authorization,
        "completed_scoring_recovery": receipt,
    }


def _validate_cp80_e25_completion(
    root: Path, cp70: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate the single canonical E25 replication completion contract."""

    _, complete = _json_object(root, _CP80_E25_COMPLETION)
    _require(
        complete.get("schema_version") == 1
        and complete.get("status") == "COMPLETE"
        and complete.get("experiment_id") == "E25"
        and complete.get("protocol_id") == _CP80_E25_PROTOCOL,
        "CP80 E25 replication completion identity is invalid",
    )
    binding = _cp80_bound_file(
        root,
        complete.get("preprotected_v2"),
        label="E25 pre-protected v2",
        expected_path=_CP80_E25_BINDING,
    )
    _require(
        binding["sha256"] == _CP80_E25_BINDING_SHA256,
        "CP80 E25 pre-protected v2 fixed identity mismatch",
    )
    matrix_binding = _cp80_bound_file(
        root,
        complete.get("run_plan_matrix"),
        label="E25 run-plan matrix",
        expected_path=_CP80_E25_MATRIX,
    )
    _require(
        matrix_binding["sha256"] == _CP80_E25_MATRIX_SHA256,
        "CP80 E25 run-plan matrix fixed identity mismatch",
    )
    from reasonforge.evaluation.e25_preprotected import verify_e25_preprotected_binding
    from reasonforge.evaluation.e25_protected_replication import verify_e25_run_plan_matrix

    preprotected = verify_e25_preprotected_binding(
        root,
        root / "configs/evaluation/e25_pre_protected_v2.yaml",
        root / _CP80_E25_BINDING,
    )
    matrix_verification = verify_e25_run_plan_matrix(root, _CP80_E25_MATRIX)
    _require(
        preprotected.get("status") == "VERIFIED_PRE_PROTECTED_BINDING"
        and matrix_verification.get("status") == "VERIFIED"
        and matrix_verification.get("plan_count") == 28,
        "CP80 E25 frozen authorization or plan matrix no longer verifies",
    )
    _, matrix = _json_object(root, _CP80_E25_MATRIX)
    plans = matrix.get("plans")
    if not isinstance(plans, dict) or len(plans) != 28:
        raise CheckpointIntegrityError("CP80 E25 run-plan inventory is not exactly 28")
    plan_ids = set(plans)
    training = _cp80_bound_file(
        root,
        complete.get("training_development_audit"),
        label="E25 training/DEVELOPMENT audit",
        expected_path=_CP80_E25_TRAINING_AUDIT,
    )
    training_value = _cp80_json_status(
        root,
        training,
        label="E25 training/DEVELOPMENT audit",
        statuses={"TRAINING_AND_DEVELOPMENT_COMPLETE_PROTECTED_REPLICATION_PENDING"},
    )
    _require(
        training_value.get("protected_accessed") is False,
        "CP80 E25 training/DEVELOPMENT audit reports protected access",
    )

    new_outputs = complete.get("new_seed_outputs")
    if not isinstance(new_outputs, dict):
        raise CheckpointIntegrityError("CP80 E25 new-seed output inventory is missing")
    output_sets = new_outputs.get("output_sets")
    _require(
        new_outputs.get("count") == 28
        and new_outputs.get("all_complete_and_verified") is True
        and new_outputs.get("incomplete_count") == 0
        and isinstance(output_sets, dict)
        and set(output_sets) == plan_ids,
        "CP80 E25 new-seed output completion inventory mismatch",
    )
    if not isinstance(output_sets, dict):  # pragma: no cover - narrowed by the gate above
        raise CheckpointIntegrityError("CP80 E25 new-seed output inventory is invalid")
    output_bindings: dict[str, Any] = {}
    for output_set_id, raw in sorted(output_sets.items()):
        if not isinstance(raw, dict):
            raise CheckpointIntegrityError(f"CP80 E25 output binding is invalid: {output_set_id}")
        plan = plans[output_set_id]
        if not isinstance(plan, dict) or not isinstance(plan.get("model_id"), str):
            raise CheckpointIntegrityError(f"CP80 E25 plan entry is invalid: {output_set_id}")
        base = f"{_CP80_E25_ROOT}/outputs/{plan['model_id']}/{output_set_id}"
        expected = {
            "output_manifest": f"{base}/manifest.json",
            "output_completion": f"{base}/PROTECTED_OUTPUT_COMPLETE.json",
            "writer_adoption": f"{base}/WRITER_V2_ADOPTION.json",
        }
        bound = {
            name: _cp80_bound_file(
                root, raw.get(name), label=f"E25 {output_set_id} {name}", expected_path=path
            )
            for name, path in expected.items()
        }
        completion = _cp80_json_status(
            root,
            bound["output_completion"],
            label=f"E25 {output_set_id} output completion",
            statuses={"COMPLETE"},
        )
        adoption = _cp80_json_status(
            root,
            bound["writer_adoption"],
            label=f"E25 {output_set_id} writer adoption",
            statuses={"ADOPTED"},
        )
        output_manifest = _cp80_json_status(
            root,
            bound["output_manifest"],
            label=f"E25 {output_set_id} output manifest",
            statuses={"COMPLETE"},
        )
        output_identity = output_manifest.get("identity")
        expected_rows = plan.get("expected_rows")
        _require(
            completion.get("output_set_id") == output_set_id
            and completion.get("manifest_sha256") == bound["output_manifest"]["sha256"]
            and isinstance(output_identity, dict)
            and output_identity.get("output_set_id") == output_set_id
            and isinstance(output_manifest.get("expected_example_ids"), list)
            and len(output_manifest["expected_example_ids"]) == expected_rows
            and adoption.get("run_plan_sha256") == plan.get("sha256"),
            f"CP80 E25 output identity mismatch: {output_set_id}",
        )
        output_bindings[output_set_id] = bound

    rescores = complete.get("rescores")
    if not isinstance(rescores, dict):
        raise CheckpointIntegrityError("CP80 E25 rescore inventory is missing")
    rescored_sets = rescores.get("output_sets")
    _require(
        rescores.get("count") == 28
        and rescores.get("rescore_id") == _CP80_E25_RESCORE_ID
        and rescores.get("all_complete_and_verified") is True
        and isinstance(rescored_sets, dict)
        and set(rescored_sets) == plan_ids,
        "CP80 E25 rescore completion inventory mismatch",
    )
    if not isinstance(rescored_sets, dict):  # pragma: no cover - narrowed by the gate above
        raise CheckpointIntegrityError("CP80 E25 rescore inventory is invalid")
    rescore_bindings: dict[str, Any] = {}
    for output_set_id, raw in sorted(rescored_sets.items()):
        if not isinstance(raw, dict):
            raise CheckpointIntegrityError(f"CP80 E25 rescore binding is invalid: {output_set_id}")
        plan = plans[output_set_id]
        if not isinstance(plan, dict):
            raise CheckpointIntegrityError(f"CP80 E25 plan entry is invalid: {output_set_id}")
        base = (
            f"{_CP80_E25_ROOT}/outputs/{plan['model_id']}/{output_set_id}/rescores/"
            f"{_CP80_E25_RESCORE_ID}"
        )
        expected = {
            "manifest": f"{base}/manifest.json",
            "results": f"{base}/results.jsonl",
            "completion": f"{base}/RESCORE_COMPLETE.json",
        }
        bound = {
            name: _cp80_bound_file(
                root, raw.get(name), label=f"E25 {output_set_id} rescore {name}", expected_path=path
            )
            for name, path in expected.items()
        }
        rescore_complete = _cp80_json_status(
            root,
            bound["completion"],
            label=f"E25 {output_set_id} rescore completion",
            statuses={"COMPLETE"},
        )
        rescore_manifest = _cp80_json_status(
            root,
            bound["manifest"],
            label=f"E25 {output_set_id} rescore manifest",
            statuses={"COMPLETE"},
        )
        result_path = _safe_path(root, str(bound["results"]["path"]))
        try:
            result_count = sum(1 for line in result_path.read_text("utf-8").splitlines() if line)
        except (OSError, UnicodeError) as error:
            raise CheckpointIntegrityError(
                f"CP80 E25 rescore results are unreadable: {output_set_id}: {error}"
            ) from error
        _require(
            rescore_complete.get("rescore_id") == _CP80_E25_RESCORE_ID
            and rescore_complete.get("manifest_sha256") == bound["manifest"]["sha256"]
            and rescore_manifest.get("rescore_id") == _CP80_E25_RESCORE_ID
            and rescore_manifest.get("output_set_id") == output_set_id
            and rescore_manifest.get("result_count") == plan.get("expected_rows")
            and rescore_manifest.get("results_sha256") == bound["results"]["sha256"]
            and result_count == plan.get("expected_rows"),
            f"CP80 E25 rescore identity mismatch: {output_set_id}",
        )
        rescore_bindings[output_set_id] = bound

    reuse = complete.get("seed1701_reuse")
    if not isinstance(reuse, dict):
        raise CheckpointIntegrityError("CP80 E25 seed-1701 reuse evidence is missing")
    cp70_binding = _cp80_bound_file(
        root,
        reuse.get("cp70_binding"),
        label="E25 seed-1701 CP70 binding",
        expected_path="evidence/checkpoints/CP70_PROTECTED_COMPLETE/PROTECTED_COMPLETE_EVIDENCE_BINDINGS.json",
    )
    _require(
        reuse.get("policy") == "REUSE_CP70_IMMUTABLE_OUTPUTS_BYTE_FOR_BYTE"
        and reuse.get("regenerated") is False
        and reuse.get("output_set_count") == 14
        and cp70.get("protected_outputs", {}).get("output_set_count") == 14,
        "CP80 E25 seed-1701 immutable reuse contract mismatch",
    )

    analysis_spec = complete.get("three_seed_analysis")
    if not isinstance(analysis_spec, dict):
        raise CheckpointIntegrityError("CP80 E25 three-seed analysis binding is missing")
    analysis_binding = _cp80_bound_file(root, analysis_spec, label="E25 three-seed analysis")
    analysis = _cp80_json_status(
        root, analysis_binding, label="E25 three-seed analysis", statuses={"COMPLETE"}
    )
    decision = analysis.get("decision")
    _require(
        analysis.get("schema_version") == 1
        and analysis.get("experiment_id") == "E25"
        and analysis.get("protocol_id") == _CP80_E25_PROTOCOL
        and analysis.get("seeds") == [1701, 1702, 1703]
        and analysis.get("seed_count") == 3
        and analysis.get("arms") == ["conventional", "rcac"]
        and analysis.get("output_set_count") == 42
        and analysis.get("every_seed_reported") is True
        and analysis.get("analysis_unit_identity_verified") is True
        and analysis.get("no_tuning") is True
        and analysis.get("frozen_policy_sha256") == _CP80_POLICY_SHA256
        and isinstance(analysis.get("paired_analyses"), (dict, list))
        and bool(analysis.get("paired_analyses"))
        and isinstance(decision, dict)
        and decision.get("status") == "COMPLETE"
        and isinstance(decision.get("outcome"), str)
        and bool(decision.get("outcome")),
        "CP80 E25 three-seed paired analysis or decision is incomplete",
    )
    lock_evidence = _verify_scoring_lock(root, complete)
    _require(
        complete.get("active_run_count") == 0,
        "CP80 E25 completion reports an active scientific run",
    )
    return (
        {
            "completion": _binding(root, _CP80_E25_COMPLETION),
            "preprotected_v2": binding,
            "run_plan_matrix": matrix_binding,
            "training_development_audit": training,
            "new_seed_output_set_count": 28,
            "new_seed_outputs": output_bindings,
            "rescore_count": 28,
            "rescores": rescore_bindings,
            "seed1701_reuse": {**reuse, "cp70_binding": cp70_binding},
            "three_seed_analysis": analysis_binding,
            "scoring_lock_absent": lock_evidence["absent"],
            "scoring_lock_evidence": lock_evidence,
            "active_run_count": 0,
        },
        {
            str(model_id): str(model["model_checkpoint_sha256"])
            for model_id, model in matrix["models"].items()
            if isinstance(model, dict)
        },
    )


def build_admission(root: Path) -> dict[str, Any]:
    _, complete = _json_object(root, _CP80_E25_COMPLETION)
    lock_evidence = _verify_scoring_lock(root, complete)
    _require(
        lock_evidence["absent"] is False, "This admission requires the preserved recovery lock"
    )
    code = {name: _binding(root, name) for name in CODE_PATHS}
    _, validation = _json_object(root, VALIDATION)
    _require(
        validation.get("status") == "PASSED"
        and validation.get("exit_codes") == [0, 0, 0]
        and validation.get("code_inventory") == code
        and validation.get("cuda_used") is False,
        "CP80 inactive-lock CPU validation is invalid",
    )
    _cp80_bound_file(root, validation.get("log"), label="inactive-lock validation log")
    return {
        "schema_version": 1,
        "policy_id": POLICY_ID,
        "status": "AUTHORIZED_READ_ONLY_CP80_ADMISSION",
        "e25_completion": _binding(root, _CP80_E25_COMPLETION),
        "e36_disposition": _binding(root, "evidence/training/E36/FOUR_B_DISPOSITION.json"),
        "scoring_lock_evidence": lock_evidence,
        "code_inventory": code,
        "cpu_validation": _binding(root, VALIDATION),
        "e25_bytes_modified": False,
        "e36_disposition_modified": False,
        "statistical_semantics_changed": False,
        "protected_outputs_regenerated_or_rescored": False,
    }


def publish_admission(root: Path) -> dict[str, Any]:
    _require(not (root / AUTHORIZATION).exists(), "CP80 inactive-lock admission already exists")
    value = build_admission(root)
    atomic_write_json(root / AUTHORIZATION, value)
    return value


def qualify_cp80_evidence(root: Path, registry: ExperimentRegistry) -> MilestoneEvidence:
    _, observed = _json_object(root, AUTHORIZATION)
    _require(observed == build_admission(root), "CP80 inactive-lock admission changed")
    namespace = dict(original.qualify_cp80_evidence.__globals__)
    namespace["_validate_cp80_e25_completion"] = _validate_cp80_e25_completion
    function = original.qualify_cp80_evidence
    bound = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__)
    result = cast(MilestoneEvidence, bound(root, registry))
    result.artifact["inactive_lock_admission"] = _binding(root, AUTHORIZATION)
    return result
