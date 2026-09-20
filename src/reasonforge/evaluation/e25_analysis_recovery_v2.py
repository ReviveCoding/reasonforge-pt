"""Correct the matrix policy label while retaining all frozen E25 analysis semantics."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from types import FunctionType
from typing import Any, cast

from reasonforge.checkpoints.hashing import (
    atomic_write_json,
    canonical_json_bytes,
    sha256_file,
    sha256_json,
)

from . import e25_analysis_recovery_v1 as parent
from . import e25_three_seed_analysis as original
from .e25_analysis_recovery_v1 import ANALYSIS_CONFIG, verify_completed_scoring_recovery
from .e25_three_seed_analysis import (
    E25ThreeSeedConfig,
    _binding,
    _expected_ids,
    _json,
    _load_plan_metadata,
    _read_results_without_generation_text,
    _safe,
    _verify_binding,
    _verify_embedded_binding,
    load_e25_three_seed_config,
)
from .protected_outputs import DeterministicGraderResult, ProtectedOutputIntegrityError

RECOVERY_ID = "RFPT-E25-ANALYSIS-RECOVERY-v2"
AUTHORIZATION = Path(
    "evidence/protected_replication/RFPT-E25-ANALYSIS-RECOVERY-v2/AUTHORIZATION.json"
)
PARENT_SHA = "3cd13e20ad2bde31998be8f23826848459e11ea64a626afdb23bcc8abb42c66a"
QUALIFIED_PAYLOAD_SHA = "a665c9981d21ab42a1b361aee8e0857c43a0cf6919dc1a2f3ea5bae5ba9a3fa3"
_CODE_PATHS = (
    "src/reasonforge/evaluation/e25_analysis_recovery_v2.py",
    "scripts/analyze_e25_recovery_v2.py",
)
_TEST_PATHS = ("tests/test_e25_analysis_recovery_v2.py",)
_DOC_PATHS = ("docs/E25_ANALYSIS_RECOVERY_V2.md",)


def verify_inputs(root: Path) -> dict[str, Any]:
    if sha256_file(root / parent.AUTHORIZATION) != PARENT_SHA:
        raise ProtectedOutputIntegrityError("E25 analysis recovery v1 authorization changed")
    # Verifies all inherited code/config/tests/docs and exact completed score artifacts.
    return parent.prepare_analysis_recovery(root)


def _build_unpublished(root: Path) -> dict[str, Any]:
    verify_inputs(root)
    namespace = dict(original.build_e25_three_seed_analysis.__globals__)
    namespace["_collect_score_sets"] = _collect_score_sets_v2
    function = original.build_e25_three_seed_analysis
    bound = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__)
    return cast(dict[str, Any], bound(root, root / ANALYSIS_CONFIG))


def qualify_real_analysis(root: Path) -> dict[str, Any]:
    """Exercise real score collection and numerical construction; publish no analysis."""
    value = _build_unpublished(root)
    if (
        value.get("status") != "COMPLETE"
        or value.get("output_set_count") != 42
        or value.get("seed_count") != 3
        or value.get("generation_text_read") is not False
        or value.get("analysis_unit_identity_verified") is not True
        or len(value.get("paired_analyses", {})) != 7
    ):
        raise ProtectedOutputIntegrityError("E25 real analysis qualification incomplete")
    return {
        "status": "REAL_COLLECTION_AND_STATISTICS_QUALIFIED",
        "output_set_count": 42,
        "seed_count": 3,
        "family_count": 7,
        "statistics_constructed": True,
        "analysis_published": False,
        "generation_text_read": False,
        "score_values_logged": False,
        "analysis_payload_sha256": sha256_json(value),
    }


def _authorization_payload(root: Path, qualification: dict[str, Any]) -> dict[str, Any]:
    inputs = verify_inputs(root)
    return {
        "schema_version": 1,
        "authorization_id": RECOVERY_ID,
        "status": "VERIFIED_ANALYSIS_RECOVERY_AUTHORIZATION",
        "parent_authorization": _binding(root, parent.AUTHORIZATION),
        "failure": "MATRIX_POLICY_LABEL_CONFUSED_WITH_PREPROTECTED_REUSE_POLICY",
        "matrix_policy_required": "REUSE_IMMUTABLE_DO_NOT_REGENERATE",
        "preprotected_reuse_policy_required": "REUSE_CP70_IMMUTABLE_OUTPUTS_BYTE_FOR_BYTE",
        "statistical_semantics_changed": False,
        "protected_artifacts_changed": False,
        "analysis_configuration": inputs["analysis_configuration"],
        "completed_scoring_recovery": inputs["completed_scoring_recovery"],
        "qualification": qualification,
        "code_inventory": {p: sha256_file(root / p) for p in _CODE_PATHS},
        "test_inventory": {p: sha256_file(root / p) for p in _TEST_PATHS},
        "documentation_inventory": {p: sha256_file(root / p) for p in _DOC_PATHS},
    }


def publish_authorization(root: Path) -> dict[str, Any]:
    path = root / AUTHORIZATION
    if path.exists():
        raise FileExistsError("refusing to overwrite analysis recovery v2 authorization")
    qualification = qualify_real_analysis(root)
    if qualification["analysis_payload_sha256"] != QUALIFIED_PAYLOAD_SHA:
        raise ProtectedOutputIntegrityError("E25 real qualification payload changed")
    value = _authorization_payload(root, qualification)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    return verify_authorization(root)


def verify_authorization(root: Path) -> dict[str, Any]:
    observed = _json(root / AUTHORIZATION)
    qualification = observed.get("qualification")
    if (
        not isinstance(qualification, dict)
        or qualification.get("status") != "REAL_COLLECTION_AND_STATISTICS_QUALIFIED"
        or qualification.get("statistics_constructed") is not True
        or qualification.get("output_set_count") != 42
        or qualification.get("seed_count") != 3
        or qualification.get("family_count") != 7
        or qualification.get("analysis_published") is not False
        or qualification.get("generation_text_read") is not False
        or qualification.get("score_values_logged") is not False
        or qualification.get("analysis_payload_sha256") != QUALIFIED_PAYLOAD_SHA
    ):
        raise ProtectedOutputIntegrityError("E25 analysis recovery v2 lacks real qualification")
    if observed != _authorization_payload(root, qualification):
        raise ProtectedOutputIntegrityError("E25 analysis recovery v2 authorization drift")
    return observed


def publish_analysis(root: Path) -> dict[str, Any]:
    authorization = verify_authorization(root)
    value = _build_unpublished(root)
    if sha256_json(value) != authorization["qualification"]["analysis_payload_sha256"]:
        raise ProtectedOutputIntegrityError("E25 statistics differ from real qualification")
    value["scoring_recovery_authorization"] = _binding(root, AUTHORIZATION)
    config = load_e25_three_seed_config(root / ANALYSIS_CONFIG)
    path = _safe(root, config.analysis_output)
    if path.exists() and _json(path) != value:
        raise ProtectedOutputIntegrityError("refusing to overwrite E25 analysis")
    if not path.exists():
        atomic_write_json(path, value)
    return value


def publish_completion(root: Path) -> dict[str, Any]:
    verify_authorization(root)
    namespace = dict(parent.build_e25_replication_completion.__globals__)
    namespace.update(_collect_score_sets=_collect_score_sets_v2, AUTHORIZATION=AUTHORIZATION)
    function = parent.build_e25_replication_completion
    bound = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__)
    value = cast(dict[str, Any], bound(root, root / ANALYSIS_CONFIG))
    config = load_e25_three_seed_config(root / ANALYSIS_CONFIG)
    path = _safe(root, config.completion_output)
    if path.exists() and _json(path) != value:
        raise ProtectedOutputIntegrityError("refusing to overwrite E25 replication completion")
    if not path.exists():
        atomic_write_json(path, value)
    return value


def _collect_score_sets_v2(
    root: Path, config: E25ThreeSeedConfig
) -> tuple[
    dict[tuple[str, int, str], dict[str, DeterministicGraderResult]],
    dict[str, Any],
]:
    cp70_path = _verify_binding(root, config.cp70_binding, "CP70 protected binding")
    matrix_path = _verify_binding(root, config.run_plan_matrix, "E25 plan matrix")
    _verify_binding(root, config.preprotected_v2, "E25 pre-protected v2")
    _verify_binding(root, config.training_development_audit, "E25 training audit")
    _verify_binding(root, config.frozen_policy, "frozen decision policy")
    verify_completed_scoring_recovery(root)
    cp70, matrix = _json(cp70_path), _json(matrix_path)
    preprotected = _json(_verify_binding(root, config.preprotected_v2, "E25 pre-protected v2"))
    if (
        cp70.get("checkpoint_id") != "CP70_PROTECTED_COMPLETE"
        or cp70.get("protected_access_state") != "LOCKED_FINAL_SCORED_COMPLETE"
        or preprotected.get("status") != "VERIFIED_PRE_PROTECTED_BINDING"
        or preprotected.get("protocol_id") != config.protocol_id
        or preprotected.get("run_plan_count") != 28
        or preprotected.get("new_model_count") != 4
        or preprotected.get("regenerate_seed1701_authorized") is not False
        or preprotected.get("seed1701_reuse", {}).get("policy")
        != "REUSE_CP70_IMMUTABLE_OUTPUTS_BYTE_FOR_BYTE"
        or preprotected.get("seed1701_reuse", {}).get("regenerate") is not False
    ):
        raise ProtectedOutputIntegrityError(
            "protected ancestry or seed-1701 reuse contract changed"
        )
    cp70_sets = cp70.get("protected_outputs", {}).get("output_sets")
    new_plans = matrix.get("plans")
    new_models = matrix.get("models")
    if (
        cp70.get("protected_outputs", {}).get("output_set_count") != 14
        or not isinstance(cp70_sets, Mapping)
        or len(cp70_sets) != 14
    ):
        raise ProtectedOutputIntegrityError("CP70 must bind exactly 14 seed-1701 output sets")
    if (
        matrix.get("status") != "FROZEN"
        or matrix.get("protocol_id") != config.protocol_id
        or matrix.get("plan_count") != 28
        or matrix.get("model_count") != 4
        or matrix.get("outputs_generated") is not False
        or matrix.get("scores_computed") is not False
        or matrix.get("seed1701_outputs") != "REUSE_IMMUTABLE_DO_NOT_REGENERATE"
        or matrix.get("semantic_outputs_inspected") is not False
        or matrix.get("protected_source_rows_loaded") is not False
        or not isinstance(new_plans, Mapping)
        or len(new_plans) != 28
        or not isinstance(new_models, Mapping)
        or len(new_models) != 4
    ):
        raise ProtectedOutputIntegrityError("E25 matrix must bind exactly 28 sets and four models")
    collected: dict[tuple[str, int, str], dict[str, DeterministicGraderResult]] = {}
    inventory: dict[str, Any] = {}
    for output_set_id, raw in sorted(cp70_sets.items()):
        item = cast(Mapping[str, Any], raw)
        plan_binding = cast(Mapping[str, Any], item["run_plan"])
        _verify_embedded_binding(root, plan_binding, label=f"{output_set_id} CP70 run plan")
        plan = _load_plan_metadata(root, plan_binding)
        model_id, family = str(plan["model_id"]), str(plan["benchmark_name"])
        arm = "conventional" if model_id == "e16_terminal_uniform_seed1701" else "rcac"
        if model_id not in {"e16_terminal_uniform_seed1701", "e18_rcac_seed1701"}:
            raise ProtectedOutputIntegrityError("CP70 contains an unexpected seed-1701 model")
        rescore = cast(Mapping[str, Any], item["rescore"])
        results_binding = cast(Mapping[str, Any], rescore["results"])
        manifest_binding = cast(Mapping[str, Any], rescore["manifest"])
        completion_binding = cast(Mapping[str, Any], rescore["completion"])
        manifest_path = _verify_embedded_binding(
            root, manifest_binding, label=f"{output_set_id} CP70 rescore manifest"
        )
        results_path = _verify_embedded_binding(
            root, results_binding, label=f"{output_set_id} CP70 rescore results"
        )
        completion_path = _verify_embedded_binding(
            root, completion_binding, label=f"{output_set_id} CP70 rescore completion"
        )
        rows, bindings = _read_results_without_generation_text(
            root,
            output_set_id=str(output_set_id),
            expected_ids=_expected_ids(plan, output_set_id=str(output_set_id)),
            manifest_path=manifest_path,
            results_path=results_path,
            completion_path=completion_path,
            rescore_id=config.seed1701_rescore_id,
        )
        key = (arm, 1701, family)
        if key in collected or str(output_set_id) in inventory:
            raise ProtectedOutputIntegrityError(f"duplicate protected analysis cell: {key}")
        collected[key] = rows
        inventory[str(output_set_id)] = bindings
    new_output_root = _safe(root, config.new_seed_output_root)
    for output_set_id, raw in sorted(new_plans.items()):
        item = cast(Mapping[str, Any], raw)
        model = cast(Mapping[str, Any], new_models[str(item["model_id"])])
        arm, seed = str(model["arm"]), int(model["seed"])
        family = str(item["benchmark_name"])
        if (
            item.get("arm") != arm
            or item.get("seed") != seed
            or seed not in {1702, 1703}
            or arm not in config.arms
        ):
            raise ProtectedOutputIntegrityError(
                f"new-seed plan/model identity mismatch: {output_set_id}"
            )
        base = new_output_root / str(item["model_id"]) / str(output_set_id)
        rescore_directory = base / "rescores" / config.new_seed_rescore_id
        plan = _load_plan_metadata(root, {"path": item["path"], "sha256": item["sha256"]})
        if plan.get("model_id") != item.get("model_id") or plan.get("benchmark_name") != family:
            raise ProtectedOutputIntegrityError(
                f"new-seed run-plan identity mismatch: {output_set_id}"
            )
        rows, bindings = _read_results_without_generation_text(
            root,
            output_set_id=str(output_set_id),
            expected_ids=_expected_ids(plan, output_set_id=str(output_set_id)),
            manifest_path=rescore_directory / "manifest.json",
            results_path=rescore_directory / "results.jsonl",
            completion_path=rescore_directory / "RESCORE_COMPLETE.json",
            rescore_id=config.new_seed_rescore_id,
        )
        key = (arm, seed, family)
        if key in collected or str(output_set_id) in inventory:
            raise ProtectedOutputIntegrityError(f"duplicate protected analysis cell: {key}")
        collected[key] = rows
        inventory[str(output_set_id)] = bindings
    expected = {
        (arm, seed, family)
        for arm in config.arms
        for seed in config.seeds
        for family in config.endpoints
    }
    if set(collected) != expected or len(inventory) != config.expected_output_sets:
        raise ProtectedOutputIntegrityError("42-set seed/arm/family identity matrix is incomplete")
    return collected, dict(sorted(inventory.items()))
