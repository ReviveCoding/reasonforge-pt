"""Fail-closed additive pre-protected authorization for E25 replication.

The authorization extends the CP60 protocol to four already-trained E25
adapters.  It does not inspect protected predictions or scores and cannot
change any dataset, prompt, generation, scorer, or statistical semantics.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml

from reasonforge.checkpoints.hashing import atomic_write_json
from reasonforge.checkpoints.models import CheckpointIntegrityError
from reasonforge.checkpoints.project import ProjectCheckpointStore
from reasonforge.config import file_sha256, identity_hash


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointIntegrityError(f"E25 pre-protected {label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CheckpointIntegrityError(f"E25 pre-protected {label} must be a sequence")
    return value


def _safe(root: Path, relative: object) -> Path:
    path = (root / Path(str(relative))).resolve()
    if path == root or root not in path.parents:
        raise CheckpointIntegrityError(f"E25 pre-protected path escapes workspace: {relative}")
    return path


def _json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CheckpointIntegrityError(
            f"E25 pre-protected {label} is unreadable: {error}"
        ) from error
    return _mapping(value, label)


def _binding(root: Path, relative: object, expected_sha256: object | None = None) -> dict[str, Any]:
    path = _safe(root, relative)
    if not path.is_file() or path.stat().st_size == 0:
        raise CheckpointIntegrityError(f"E25 pre-protected evidence is missing: {relative}")
    digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise CheckpointIntegrityError(f"E25 pre-protected evidence hash mismatch: {relative}")
    return {"path": Path(str(relative)).as_posix(), "sha256": digest, "bytes": path.stat().st_size}


def _adapter_identity(path: Path) -> str:
    files = {
        item.relative_to(path).as_posix(): file_sha256(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }
    if not files:
        raise CheckpointIntegrityError(f"E25 final adapter is empty: {path}")
    return identity_hash(files)


def _validate_ancestry(
    root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    ancestry = _mapping(config.get("verified_ancestry"), "verified ancestry")
    bound: dict[str, Any] = {}
    for name in ("cp60_manifest", "cp60_bindings", "cp70_manifest", "cp70_bindings"):
        item = _mapping(ancestry.get(name), f"{name} binding")
        bound[name] = _binding(root, item.get("path"), item.get("sha256"))

    latest, checkpoint_path = ProjectCheckpointStore(root / "evidence/checkpoints").verify_latest()
    if latest.checkpoint_id != "CP70_PROTECTED_COMPLETE":
        raise CheckpointIntegrityError("E25 pre-protected binding requires latest VERIFIED CP70")
    if checkpoint_path != _safe(root, "evidence/checkpoints/CP70_PROTECTED_COMPLETE"):
        raise CheckpointIntegrityError("LATEST does not resolve to the bound CP70 directory")
    if (
        file_sha256(checkpoint_path / "CHECKPOINT_MANIFEST.json")
        != bound["cp70_manifest"]["sha256"]
    ):
        raise CheckpointIntegrityError("LATEST does not hash-bind the configured CP70 manifest")

    cp60 = _json(_safe(root, ancestry["cp60_bindings"]["path"]), "CP60 bindings")
    cp70 = _json(_safe(root, ancestry["cp70_bindings"]["path"]), "CP70 bindings")
    cp60_manifest = _json(_safe(root, ancestry["cp60_manifest"]["path"]), "CP60 manifest")
    cp70_manifest = _json(_safe(root, ancestry["cp70_manifest"]["path"]), "CP70 manifest")
    if cp60_manifest.get("checkpoint_id") != "CP60_PRE_PROTECTED":
        raise CheckpointIntegrityError("bound CP60 identity is invalid")
    if cp70_manifest.get("checkpoint_id") != "CP70_PROTECTED_COMPLETE":
        raise CheckpointIntegrityError("bound CP70 identity is invalid")
    if cp70_manifest.get("protected_access_state") != "LOCKED_FINAL_SCORED_COMPLETE":
        raise CheckpointIntegrityError("CP70 protected completion state is invalid")
    parent = _mapping(cp70.get("cp60_verified_binding"), "CP70 CP60 ancestry")
    for name, parent_name in (
        ("cp60_manifest", "manifest"),
        ("cp60_bindings", "pre_protected_bindings"),
    ):
        if _mapping(parent.get(parent_name), parent_name).get("sha256") != bound[name]["sha256"]:
            raise CheckpointIntegrityError("CP70 does not bind the configured CP60 identity")
    return bound, cp60, cp70


def _validate_semantics(
    root: Path,
    config: Mapping[str, Any],
    cp60: Mapping[str, Any],
    cp70: Mapping[str, Any],
) -> dict[str, Any]:
    frozen = _mapping(config.get("frozen_semantics"), "frozen semantics")
    bindings: dict[str, Any] = {}
    for name, item_value in frozen.items():
        item = _mapping(item_value, f"frozen semantic {name}")
        bindings[str(name)] = _binding(root, item.get("path"), item.get("sha256"))
    expected = {
        "benchmark_registry": cp60.get("benchmark_registry"),
        "generation_config": cp60.get("generation_config"),
        "decision_policy": _mapping(cp60.get("analysis_policy"), "analysis policy").get(
            "decision_policy"
        ),
        "statistical_analysis_plan": _mapping(cp60.get("analysis_policy"), "analysis policy").get(
            "statistical_analysis_plan"
        ),
    }
    for name, inherited in expected.items():
        if bindings.get(name) != inherited:
            raise CheckpointIntegrityError(f"E25 {name} differs from the CP60 frozen binding")
    if cp60.get("protected_protocol_id") != cp70.get("protected_protocol_id"):
        raise CheckpointIntegrityError("CP60/CP70 protected protocol identity differs")
    if cp60.get("dataset_identities") != cp70.get("dataset_identities"):
        raise CheckpointIntegrityError("CP60/CP70 protected dataset identities differ")
    if cp60.get("generation_config") != cp70.get("generation_config"):
        raise CheckpointIntegrityError("CP60/CP70 generation semantics differ")
    if cp60.get("analysis_policy") != cp70.get("analysis_policy"):
        raise CheckpointIntegrityError("CP60/CP70 statistical semantics differ")
    return {
        "bindings": bindings,
        "protected_protocol_id": cp60["protected_protocol_id"],
        "dataset_identities": cp60["dataset_identities"],
        "dataset_identity_hash": identity_hash(cp60["dataset_identities"]),
        "generation_identity": identity_hash(cp60["generation_config"]),
        "prompt_and_plan_semantics_identity": identity_hash(cp60["finalist_plans"]),
        "effective_scorer_revisions": cp70["effective_scorer_revisions"],
        "effective_scorer_identity": identity_hash(cp70["effective_scorer_revisions"]),
        "analysis_policy_identity": identity_hash(cp60["analysis_policy"]),
    }


def _validate_models(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    models = _sequence(config.get("new_models"), "new models")
    if len(models) != 4:
        raise CheckpointIntegrityError("E25 pre-protected binding requires exactly four new models")
    expected_pairs = {
        ("conventional", 1702),
        ("conventional", 1703),
        ("rcac", 1702),
        ("rcac", 1703),
    }
    observed_pairs: set[tuple[str, int]] = set()
    result: dict[str, Any] = {}
    for raw in models:
        model = _mapping(raw, "new model")
        arm, seed = str(model.get("arm")), int(model.get("seed", -1))
        observed_pairs.add((arm, seed))
        run_id = str(model.get("run_id"))
        arm_stem = "conventional-terminal-uniform" if arm == "conventional" else "rcac-terminal"
        expected_run = f"E25-{arm_stem}-grpo-seed{seed}-v1"
        if run_id != expected_run:
            raise CheckpointIntegrityError(f"E25 run identity mismatch: {run_id}")
        run_relative = f"artifacts/training/E25/{run_id}/RUN_COMPLETE.json"
        run_binding = _binding(root, run_relative)
        run = _json(_safe(root, run_relative), f"{run_id} completion")
        adapter_identity = str(model.get("adapter_identity"))
        if (
            run.get("status") != "COMPLETE"
            or run.get("experiment_id") != "E25"
            or run.get("run_id") != run_id
            or run.get("protocol_id") != "RFPT-E25-REPLICATION-v1"
            or run.get("execution_mode") != "LOCKED_FINAL_REPLICATION"
            or run.get("seed") != seed
            or run.get("final_adapter_identity") != adapter_identity
        ):
            raise CheckpointIntegrityError(f"E25 completed run identity mismatch: {run_id}")
        adapter_relative = str(run.get("final_adapter_path"))
        adapter_path = _safe(root, adapter_relative)
        if not adapter_path.is_dir() or _adapter_identity(adapter_path) != adapter_identity:
            raise CheckpointIntegrityError(f"E25 adapter hash mismatch: {run_id}")
        checkpoints = {}
        for step in (16, 32):
            relative = f"artifacts/training/E25/{run_id}/checkpoint-{step}/CHECKPOINT_COMPLETE.json"
            complete = _json(_safe(root, relative), f"{run_id} checkpoint {step}")
            if complete.get("status") != "COMPLETE" or complete.get("checkpoint_step") != step:
                raise CheckpointIntegrityError(f"E25 checkpoint {step} is incomplete: {run_id}")
            checkpoints[str(step)] = _binding(root, relative)

        eval_run_id = str(model.get("development_eval_run_id"))
        if eval_run_id != f"{run_id}-dev-eval-v1":
            raise CheckpointIntegrityError(f"E25 DEVELOPMENT run identity mismatch: {run_id}")
        output_relative = f"outputs/{eval_run_id}"
        summary_relative = f"{output_relative}/summary.json"
        predictions_relative = f"{output_relative}/predictions.jsonl"
        diagnostics_relative = f"{output_relative}/model_diagnostics.json"
        summary = _json(_safe(root, summary_relative), f"{eval_run_id} summary")
        diagnostics = _json(_safe(root, diagnostics_relative), f"{eval_run_id} diagnostics")
        predictions_path = _safe(root, predictions_relative)
        try:
            predictions = [
                json.loads(line)
                for line in predictions_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise CheckpointIntegrityError(
                f"E25 DEVELOPMENT predictions are unreadable: {error}"
            ) from error
        if (
            summary.get("status") != "COMPLETE"
            or summary.get("experiment_id") != "E25"
            or summary.get("run_id") != eval_run_id
            or summary.get("scientific_state") != "DEVELOPMENT"
            or summary.get("model_identity") != adapter_identity
            or summary.get("dataset_rows") != 36
            or len(predictions) != 36
        ):
            raise CheckpointIntegrityError(
                f"E25 DEVELOPMENT evaluation is incomplete: {eval_run_id}"
            )
        ids = [row.get("example_id") for row in predictions if isinstance(row, Mapping)]
        if len(ids) != 36 or len(set(ids)) != 36:
            raise CheckpointIntegrityError(
                f"E25 DEVELOPMENT example identity inventory is invalid: {eval_run_id}"
            )
        for row in predictions:
            if not isinstance(row, Mapping) or (
                row.get("split_role") != "DEVELOPMENT"
                or row.get("experiment_id") != "E25"
                or row.get("run_id") != eval_run_id
                or row.get("model_identity") != adapter_identity
            ):
                raise CheckpointIntegrityError(
                    f"E25 DEVELOPMENT prediction identity mismatch: {eval_run_id}"
                )
        if (
            diagnostics.get("model_construction_count") != 1
            or diagnostics.get("adapter_wrapper_count") != 1
            or diagnostics.get("eval_mode") is not True
            or diagnostics.get("optimizer_constructed") is not False
            or diagnostics.get("trainable_parameter_count") != 0
        ):
            raise CheckpointIntegrityError(
                f"E25 DEVELOPMENT model diagnostics are invalid: {eval_run_id}"
            )
        result[run_id] = {
            "arm": arm,
            "seed": seed,
            "adapter_identity": adapter_identity,
            "adapter_path": adapter_relative.replace("\\", "/"),
            "run_complete": run_binding,
            "checkpoint_completions": checkpoints,
            "development_evaluation": {
                "run_id": eval_run_id,
                "model_identity": adapter_identity,
                "row_count": 36,
                "example_id_inventory_sha256": identity_hash(sorted(cast(list[str], ids))),
                "summary": _binding(root, summary_relative),
                "predictions": _binding(root, predictions_relative),
                "model_diagnostics": _binding(root, diagnostics_relative),
            },
        }
    if observed_pairs != expected_pairs:
        raise CheckpointIntegrityError("E25 arm/seed matrix is not the frozen 2x2 design")
    return result


def _validate_plan_matrix(
    root: Path, config: Mapping[str, Any], models: Mapping[str, Any]
) -> dict[str, Any]:
    from reasonforge.evaluation.e25_protected_replication import verify_e25_run_plan_matrix

    specification = _mapping(config.get("run_plan_matrix"), "run-plan matrix specification")
    relative = str(specification.get("path"))
    matrix_binding = _binding(root, relative, specification.get("sha256"))
    verification = verify_e25_run_plan_matrix(root, relative)
    if (
        verification.get("status") != "VERIFIED"
        or verification.get("plan_count") != 28
        or verification.get("model_count") != 4
        or verification.get("benchmark_count") != 7
        or verification.get("protected_source_rows_loaded") is not False
        or verification.get("semantic_outputs_inspected") is not False
        or verification.get("outputs_generated") is not False
        or verification.get("scores_computed") is not False
    ):
        raise CheckpointIntegrityError("E25 deterministic run-plan verification failed")
    matrix = _json(_safe(root, relative), "run-plan matrix")
    plans = _mapping(matrix.get("plans"), "run-plan inventory")
    if len(plans) != int(specification.get("expected_plan_count", -1)):
        raise CheckpointIntegrityError("E25 run-plan matrix must contain exactly 28 plans")
    matrix_models = _mapping(matrix.get("models"), "run-plan model inventory")
    models_by_run = dict(models)
    if len(matrix_models) != len(models_by_run):
        raise CheckpointIntegrityError("E25 matrix model inventory is incomplete")
    matrix_model_to_run: dict[str, str] = {}
    for model_id, raw_model in matrix_models.items():
        model = _mapping(raw_model, f"matrix model {model_id}")
        run_id = str(model.get("run_id"))
        if (
            run_id not in models_by_run
            or model.get("model_checkpoint_sha256") != models_by_run[run_id]["adapter_identity"]
            or model.get("arm") != models_by_run[run_id]["arm"]
            or model.get("seed") != models_by_run[run_id]["seed"]
        ):
            raise CheckpointIntegrityError(f"E25 matrix model identity mismatch: {model_id}")
        matrix_model_to_run[str(model_id)] = run_id
    model_ids: set[str] = set()
    family_ids: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    inventory: dict[str, Any] = {}
    for output_set_id, raw in plans.items():
        item = _mapping(raw, "run-plan inventory item")
        plan_relative = item.get("path", item.get("run_plan"))
        if not isinstance(plan_relative, str):
            raise CheckpointIntegrityError("E25 run-plan inventory omits a plan path")
        plan_binding = _binding(root, plan_relative, item.get("sha256"))
        plan = _json(_safe(root, plan_relative), f"run plan {plan_relative}")
        model_id = str(plan.get("model_id"))
        run_id = matrix_model_to_run.get(model_id, "")
        adapter = str(plan.get("model_checkpoint_sha256", ""))
        family = str(plan.get("benchmark_name", ""))
        plan_id = str(plan.get("output_set_id", ""))
        if run_id not in models_by_run or adapter != models_by_run[run_id]["adapter_identity"]:
            raise CheckpointIntegrityError(
                f"E25 plan references an unauthorized model: {plan_relative}"
            )
        if plan_id != output_set_id or not family or plan_id in inventory:
            raise CheckpointIntegrityError(f"E25 run-plan identity is invalid: {plan_relative}")
        model_ids.add(run_id)
        family_ids.add(family)
        if (run_id, family) in pairs:
            raise CheckpointIntegrityError("E25 run-plan matrix repeats a model/family pair")
        pairs.add((run_id, family))
        inventory[plan_id] = plan_binding
    if len(model_ids) != int(specification.get("expected_model_count", -1)):
        raise CheckpointIntegrityError("E25 run-plan matrix does not cover all four models")
    if len(family_ids) != int(specification.get("expected_benchmark_family_count", -1)):
        raise CheckpointIntegrityError(
            "E25 run-plan matrix does not cover all seven benchmark families"
        )
    if pairs != {(run_id, family) for run_id in model_ids for family in family_ids}:
        raise CheckpointIntegrityError(
            "E25 run-plan matrix is not a complete model-by-family product"
        )
    return {
        "matrix": matrix_binding,
        "verification": verification,
        "plans": dict(sorted(inventory.items())),
    }


def _validate_execution_surface(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    surface = _mapping(config.get("execution_surface"), "execution surface")
    execution_config = _mapping(surface.get("execution_config"), "execution config")
    result: dict[str, Any] = {
        "execution_config": _binding(
            root, execution_config.get("path"), execution_config.get("sha256")
        )
    }
    expected_counts = {"code": 4, "qualification_tests": 1, "documentation": 1}
    for group, expected_count in expected_counts.items():
        values = _sequence(surface.get(group), f"execution {group}")
        if len(values) != expected_count:
            raise CheckpointIntegrityError(f"E25 execution {group} inventory is incomplete")
        inventory: dict[str, Any] = {}
        for raw in values:
            item = _mapping(raw, f"execution {group} item")
            path = str(item.get("path"))
            if path in inventory:
                raise CheckpointIntegrityError(f"E25 execution {group} repeats {path}")
            inventory[path] = _binding(root, path, item.get("sha256"))
        result[group] = dict(sorted(inventory.items()))
    return result


def build_e25_preprotected_binding(workspace: Path, config_path: Path) -> dict[str, Any]:
    """Build and fully verify the deterministic E25 authorization payload."""

    root = workspace.resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = _mapping(raw, "configuration")
    authorization_id = config.get("authorization_id")
    if (
        config.get("schema_version") != 1
        or authorization_id not in {"RFPT-E25-PRE-PROTECTED-v1", "RFPT-E25-PRE-PROTECTED-v2"}
        or config.get("protocol_id") != "RFPT-E25-PROTECTED-REPLICATION-v1"
        or config.get("scientific_state") != "LOCKED_FINAL"
    ):
        raise CheckpointIntegrityError("E25 pre-protected configuration identity is invalid")
    if authorization_id == "RFPT-E25-PRE-PROTECTED-v2" and (
        config.get("recovery_of") != "RFPT-E25-PRE-PROTECTED-v1"
        or config.get("recovery_reason")
        != "POST_PUBLICATION_EXECUTION_CODE_RACE_CAUGHT_BEFORE_INFERENCE"
    ):
        raise CheckpointIntegrityError("E25 pre-protected v2 recovery ancestry is invalid")
    ancestry, cp60, cp70 = _validate_ancestry(root, config)
    semantics = _validate_semantics(root, config, cp60, cp70)
    models = _validate_models(root, config)
    plans = _validate_plan_matrix(root, config, models)
    execution_surface = _validate_execution_surface(root, config)
    reuse = dict(_mapping(config.get("seed1701_reuse"), "seed-1701 reuse policy"))
    finalists = _mapping(cp70.get("finalists"), "CP70 finalists")
    if (
        reuse.get("policy") != "REUSE_CP70_IMMUTABLE_OUTPUTS_BYTE_FOR_BYTE"
        or reuse.get("regenerate") is not False
        or reuse.get("conventional_adapter_identity")
        != _mapping(finalists.get("e16_terminal_uniform_seed1701"), "conventional finalist").get(
            "model_checkpoint_sha256"
        )
        or reuse.get("rcac_adapter_identity")
        != _mapping(finalists.get("e18_rcac_seed1701"), "RCAC finalist").get(
            "model_checkpoint_sha256"
        )
    ):
        raise CheckpointIntegrityError("E25 seed-1701 immutable-output reuse identity mismatch")
    prohibitions = list(_sequence(config.get("scientific_prohibitions"), "prohibitions"))
    required = {
        "TUNE_FROM_SEED1701_PROTECTED_OUTCOMES",
        "TUNE_FROM_NEW_SEED_PROTECTED_OUTCOMES",
        "MUTATE_CP60_OR_CP70",
        "REGENERATE_OR_OVERWRITE_SEED1701_OUTPUTS",
        "CHANGE_DATASET_PROMPT_GENERATION_SCORER_OR_STATISTICAL_SEMANTICS",
        "ACCESS_PROTECTED_OUTCOMES_DURING_AUTHORIZATION",
    }
    if set(prohibitions) != required:
        raise CheckpointIntegrityError("E25 scientific prohibitions are incomplete or changed")
    return {
        "schema_version": 1,
        "status": "VERIFIED_PRE_PROTECTED_BINDING",
        "authorization_id": config["authorization_id"],
        "protocol_id": config["protocol_id"],
        "parent_replication_protocol_id": config["parent_replication_protocol_id"],
        "scientific_state": "LOCKED_FINAL",
        "configuration": _binding(root, config_path.relative_to(root)),
        "verified_ancestry": ancestry,
        "frozen_semantics": semantics,
        "new_models": models,
        "new_model_count": 4,
        "development_evaluation_count": 4,
        "run_plan_inventory": plans,
        "run_plan_count": 28,
        "execution_config": execution_surface["execution_config"],
        "execution_code_inventory": execution_surface["code"],
        "execution_test_inventory": execution_surface["qualification_tests"],
        "execution_documentation_inventory": execution_surface["documentation"],
        "seed1701_reuse": reuse,
        "scientific_prohibitions": sorted(prohibitions),
        "protected_outputs_read": False,
        "protected_scores_read": False,
        "protected_generation_authorized": True,
        "regenerate_seed1701_authorized": False,
        "cuda_used": False,
        "recovery_of": config.get("recovery_of"),
        "recovery_reason": config.get("recovery_reason"),
    }


def verify_e25_preprotected_binding(
    workspace: Path, config_path: Path, binding_path: Path
) -> dict[str, Any]:
    expected = build_e25_preprotected_binding(workspace, config_path)
    observed = _json(binding_path.resolve(), "published binding")
    if observed != expected:
        raise CheckpointIntegrityError(
            "published E25 pre-protected binding differs from live evidence"
        )
    return expected


def publish_e25_preprotected_binding(
    workspace: Path, config_path: Path, binding_path: Path
) -> dict[str, Any]:
    value = build_e25_preprotected_binding(workspace, config_path)
    if binding_path.exists():
        observed = _json(binding_path.resolve(), "published binding")
        if observed != value:
            raise FileExistsError(f"refusing to overwrite mismatched E25 binding: {binding_path}")
        return value
    atomic_write_json(binding_path, value)
    return verify_e25_preprotected_binding(workspace, config_path, binding_path)
