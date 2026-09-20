"""Fail-closed evidence qualification for project milestone publication."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from reasonforge.config import ExperimentRegistry, ExperimentStatus, file_sha256, identity_hash
from reasonforge.training.runtime import validate_native_trainer_checkpoint

from .hashing import sha256_json
from .models import CheckpointIntegrityError
from .project import ProjectCheckpointStore


@dataclass(frozen=True, slots=True)
class MilestoneEvidence:
    """Verified material to embed in a project checkpoint."""

    artifact: dict[str, Any]
    model_checkpoint_identities: dict[str, str]
    last_runtime_checkpoint: str


def _safe_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if candidate == resolved_root or resolved_root not in candidate.parents:
        raise CheckpointIntegrityError(f"evidence path escapes workspace: {relative!r}")
    return candidate


def _json_object(root: Path, relative: str) -> tuple[Path, dict[str, Any]]:
    path = _safe_path(root, relative)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CheckpointIntegrityError(
            f"required evidence is unreadable: {relative}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise CheckpointIntegrityError(f"required evidence is not a JSON object: {relative}")
    return path, value


def _binding(root: Path, relative: str) -> dict[str, str | int]:
    path = _safe_path(root, relative)
    if not path.is_file() or path.stat().st_size == 0:
        raise CheckpointIntegrityError(f"required evidence is missing or empty: {relative}")
    return {"path": relative, "sha256": file_sha256(path), "bytes": path.stat().st_size}


def _require(value: bool, message: str) -> None:
    if not value:
        raise CheckpointIntegrityError(message)


def _validate_registry(registry: ExperimentRegistry) -> None:
    by_id = {item.experiment_id: item for item in registry.experiments}
    absent = sorted({"E08", "E09", "E10", "E11"} - set(by_id))
    if absent:
        raise CheckpointIntegrityError(f"CP30 registry omits prerequisites: {absent}")
    incomplete = sorted(
        experiment_id
        for experiment_id in ("E08", "E09", "E10", "E11")
        if by_id[experiment_id].status is not ExperimentStatus.COMPLETE
    )
    if incomplete:
        raise CheckpointIntegrityError(f"CP30 prerequisites are not COMPLETE: {incomplete}")


def _validate_complete_registry(
    registry: ExperimentRegistry, checkpoint_id: str, experiment_ids: tuple[str, ...]
) -> None:
    by_id = {item.experiment_id: item for item in registry.experiments}
    absent = sorted(set(experiment_ids) - set(by_id))
    if absent:
        raise CheckpointIntegrityError(f"{checkpoint_id} registry omits prerequisites: {absent}")
    incomplete = sorted(
        experiment_id
        for experiment_id in experiment_ids
        if by_id[experiment_id].status is not ExperimentStatus.COMPLETE
    )
    if incomplete:
        raise CheckpointIntegrityError(
            f"{checkpoint_id} prerequisites are not COMPLETE: {incomplete}"
        )


def _validate_run(
    root: Path, relative: str, experiment_id: str, protocol_id: str
) -> tuple[dict[str, Any], dict[str, str | int]]:
    _, run = _json_object(root, relative)
    _require(run.get("status") == "COMPLETE", f"{experiment_id} run is not COMPLETE")
    _require(run.get("experiment_id") == experiment_id, f"{experiment_id} run identity mismatch")
    _require(run.get("protocol_id") == protocol_id, f"{experiment_id} protocol mismatch")
    adapter_id = run.get("final_adapter_identity")
    adapter_path = run.get("final_adapter_path")
    _require(
        isinstance(adapter_id, str) and len(adapter_id) == 64,
        f"{experiment_id} adapter identity is invalid",
    )
    if not isinstance(adapter_path, str) or not adapter_path:
        raise CheckpointIntegrityError(f"{experiment_id} adapter path is invalid")
    _require(_safe_path(root, adapter_path).is_dir(), f"{experiment_id} final adapter is missing")
    return run, _binding(root, relative)


def _prediction_rows(root: Path, relative: str) -> list[dict[str, Any]]:
    path = _safe_path(root, relative)
    try:
        rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CheckpointIntegrityError(
            f"prediction evidence is unreadable: {relative}: {error}"
        ) from error
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise CheckpointIntegrityError(f"prediction evidence is empty or malformed: {relative}")
    return rows


def _validate_development_evaluation(
    root: Path,
    experiment_id: str,
    run: dict[str, Any],
    summary_relative: str,
    predictions_relative: str,
) -> dict[str, Any]:
    _, summary = _json_object(root, summary_relative)
    rows = _prediction_rows(root, predictions_relative)
    expected_adapter = str(run["final_adapter_path"]).replace("\\", "/")
    observed_adapter = str(summary.get("adapter", "")).replace("\\", "/")
    adapter_id = str(run["final_adapter_identity"])
    _require(
        summary.get("status") == "COMPLETE", f"{experiment_id} DEVELOPMENT summary is not COMPLETE"
    )
    _require(
        summary.get("experiment_id") == experiment_id,
        f"{experiment_id} DEVELOPMENT summary identity mismatch",
    )
    _require(
        summary.get("scientific_state") == "DEVELOPMENT",
        f"{experiment_id} evaluation is not DEVELOPMENT",
    )
    _require(
        observed_adapter == expected_adapter, f"{experiment_id} evaluation adapter path mismatch"
    )
    model_identity = summary.get("model_identity")
    _require(
        isinstance(model_identity, str) and adapter_id[:8] in model_identity,
        f"{experiment_id} evaluation adapter identity mismatch",
    )
    _require(
        summary.get("dataset_rows") == len(rows), f"{experiment_id} evaluation row count mismatch"
    )
    example_ids = [row.get("example_id") for row in rows]
    _require(
        all(isinstance(item, str) and item for item in example_ids),
        f"{experiment_id} predictions have invalid example IDs",
    )
    _require(
        len(example_ids) == len(set(example_ids)), f"{experiment_id} predictions repeat example IDs"
    )
    for row in rows:
        _require(
            row.get("split_role") == "DEVELOPMENT",
            f"{experiment_id} predictions cross the DEVELOPMENT boundary",
        )
        _require(
            row.get("experiment_id") == experiment_id,
            f"{experiment_id} prediction identity mismatch",
        )
        _require(
            row.get("run_id") == summary.get("run_id"),
            f"{experiment_id} prediction run identity mismatch",
        )
        _require(
            row.get("model_identity") == model_identity,
            f"{experiment_id} prediction model identity mismatch",
        )
    return {
        "run_id": summary.get("run_id"),
        "model_identity": model_identity,
        "adapter_identity": adapter_id,
        "dataset_rows": len(rows),
        "summary": _binding(root, summary_relative),
        "predictions": _binding(root, predictions_relative),
    }


def _validate_preference_side(
    root: Path,
    side: dict[str, Any],
    experiment_id: str,
    run: dict[str, Any],
    dataset: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require(
        side.get("experiment_id") == experiment_id,
        f"{experiment_id} preference side identity mismatch",
    )
    _require(
        side.get("adapter_identity") == run.get("final_adapter_identity"),
        f"{experiment_id} preference adapter identity mismatch",
    )
    run_id = side.get("run_id")
    _require(
        isinstance(run_id, str) and bool(run_id), f"{experiment_id} preference run ID is invalid"
    )
    output_root = f"outputs/{run_id}"
    summary_relative = f"{output_root}/summary.json"
    rows_relative = f"{output_root}/preference_scores.jsonl"
    _, summary = _json_object(root, summary_relative)
    rows = _prediction_rows(root, rows_relative)
    artifact_hashes = side.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict):
        raise CheckpointIntegrityError(f"{experiment_id} preference artifact hashes are missing")
    _require(
        artifact_hashes.get("summary.json") == file_sha256(_safe_path(root, summary_relative)),
        f"{experiment_id} preference summary hash mismatch",
    )
    _require(
        artifact_hashes.get("preference_scores.jsonl")
        == file_sha256(_safe_path(root, rows_relative)),
        f"{experiment_id} preference row hash mismatch",
    )
    adapter_id = str(run["final_adapter_identity"])
    model_identity = side.get("model_identity")
    _require(
        summary.get("status") == "COMPLETE", f"{experiment_id} preference summary is not COMPLETE"
    )
    _require(
        summary.get("experiment_id") == experiment_id,
        f"{experiment_id} preference summary experiment mismatch",
    )
    _require(summary.get("run_id") == run_id, f"{experiment_id} preference summary run mismatch")
    _require(
        summary.get("scientific_state") == "DEVELOPMENT",
        f"{experiment_id} preference summary is not DEVELOPMENT",
    )
    _require(
        summary.get("adapter_identity") == adapter_id,
        f"{experiment_id} preference summary adapter mismatch",
    )
    _require(
        summary.get("model_identity") == model_identity and adapter_id[:8] in str(model_identity),
        f"{experiment_id} preference model identity mismatch",
    )
    _require(
        summary.get("dataset_rows") == 10 and len(rows) == 10,
        f"{experiment_id} preference evidence must contain 10 rows",
    )
    _require(
        summary.get("dataset") == dataset.get("path"),
        f"{experiment_id} preference dataset path mismatch",
    )
    _require(
        summary.get("dataset_sha256") == dataset.get("sha256"),
        f"{experiment_id} preference dataset hash mismatch",
    )
    scoring = summary.get("scoring")
    _require(
        isinstance(scoring, dict) and scoring.get("pair_count") == 10,
        f"{experiment_id} preference pair count mismatch",
    )
    for row in rows:
        _require(
            row.get("experiment_id") == experiment_id,
            f"{experiment_id} preference row experiment mismatch",
        )
        _require(row.get("run_id") == run_id, f"{experiment_id} preference row run mismatch")
        _require(
            row.get("scientific_state") == "DEVELOPMENT" and row.get("split_role") == "DEVELOPMENT",
            f"{experiment_id} preference rows cross the DEVELOPMENT boundary",
        )
        _require(
            row.get("adapter_identity") == adapter_id,
            f"{experiment_id} preference row adapter mismatch",
        )
        _require(
            row.get("model_identity") == model_identity,
            f"{experiment_id} preference row model mismatch",
        )
        _require(
            row.get("dataset_sha256") == dataset.get("sha256"),
            f"{experiment_id} preference row dataset mismatch",
        )
    return (
        {
            "run_id": run_id,
            "model_identity": model_identity,
            "adapter_identity": adapter_id,
            "summary": _binding(root, summary_relative),
            "rows": _binding(root, rows_relative),
        },
        rows,
    )


def _validate_preference_comparison(
    root: Path, protocol_id: str, e10: dict[str, Any], e11: dict[str, Any]
) -> dict[str, Any]:
    comparison_relative = "evidence/evaluation/E11_DPO_PREFERENCE_COMPARISON.json"
    _, comparison = _json_object(root, comparison_relative)
    _require(comparison.get("status") == "COMPLETE", "E11 preference comparison is not COMPLETE")
    _require(
        comparison.get("experiment_id") == "E11", "E11 preference comparison identity mismatch"
    )
    _require(
        comparison.get("protocol_id") == protocol_id, "E11 preference comparison protocol mismatch"
    )
    _require(
        comparison.get("scientific_state") == "DEVELOPMENT",
        "E11 preference comparison is not DEVELOPMENT",
    )
    _require(
        comparison.get("protected_data_accessed") is False,
        "E11 preference comparison reports protected-data access",
    )
    dataset = comparison.get("dataset")
    if not isinstance(dataset, dict):
        raise CheckpointIntegrityError("E11 preference comparison dataset identity is missing")
    _require(
        dataset.get("split_role") == "DEVELOPMENT",
        "E11 preference comparison dataset is not DEVELOPMENT",
    )
    _require(dataset.get("pair_count") == 10, "E11 preference comparison must contain 10 pairs")
    baseline_side = comparison.get("baseline")
    candidate_side = comparison.get("candidate")
    if not isinstance(baseline_side, dict) or not isinstance(candidate_side, dict):
        raise CheckpointIntegrityError("E11 preference comparison sides are missing")
    baseline, baseline_rows = _validate_preference_side(root, baseline_side, "E10", e10, dataset)
    candidate, candidate_rows = _validate_preference_side(root, candidate_side, "E11", e11, dataset)
    baseline_ids = [row.get("example_id") for row in baseline_rows]
    candidate_ids = [row.get("example_id") for row in candidate_rows]
    _require(
        len(set(baseline_ids)) == 10 and baseline_ids == candidate_ids,
        "E10/E11 preference rows do not contain 10 aligned DEVELOPMENT IDs",
    )
    _require(
        [row.get("content_hash") for row in baseline_rows]
        == [row.get("content_hash") for row in candidate_rows],
        "E10/E11 preference rows are not content-aligned",
    )
    paired_rows = comparison.get("paired_rows")
    _require(
        isinstance(paired_rows, list)
        and [row.get("example_id") for row in paired_rows if isinstance(row, dict)] == baseline_ids,
        "preference comparison paired rows do not align with source rows",
    )
    return {
        "aggregate": _binding(root, comparison_relative),
        "evidence_id": comparison.get("evidence_id"),
        "dataset": dataset,
        "baseline": baseline,
        "candidate": candidate,
        "aligned_development_ids": len(baseline_ids),
        "protected_data_accessed": False,
    }


def qualify_cp30_evidence(root: Path, registry: ExperimentRegistry) -> MilestoneEvidence:
    """Validate and bind every material CP30 baseline artifact."""

    root = root.resolve()
    _validate_registry(registry)
    protocol_id = registry.protocol_id

    e08_path = "evidence/training/E08-qwen3-1p7b-dev-v2.json"
    _, e08 = _json_object(root, e08_path)
    _require(
        e08.get("status") == "COMPLETE" and e08.get("experiment_id") == "E08",
        "E08 evidence is not a complete E08 result",
    )
    _require(e08.get("scientific_state") == "DEVELOPMENT", "E08 evidence is not DEVELOPMENT")
    _require(
        e08.get("protected_data_accessed") is False, "E08 evidence reports protected-data access"
    )
    e08_summary = "outputs/E08-qwen3-1p7b-dev-v2/summary.json"
    e08_predictions = "outputs/E08-qwen3-1p7b-dev-v2/predictions.jsonl"
    _require(
        e08.get("summary_sha256") == file_sha256(_safe_path(root, e08_summary)),
        "E08 summary hash does not match its evidence record",
    )
    _require(
        e08.get("predictions_sha256") == file_sha256(_safe_path(root, e08_predictions)),
        "E08 prediction hash does not match its evidence record",
    )

    e09_path = "evidence/training/E09/ADAPTER_SELECTION.json"
    _, e09 = _json_object(root, e09_path)
    _require(
        e09.get("status") == "SELECTED" and e09.get("experiment_id") == "E09",
        "E09 adapter selection is not final",
    )
    _require(e09.get("protocol_id") == protocol_id, "E09 protocol mismatch")
    _require(
        e09.get("protected_data_accessed") is False, "E09 selection reports protected-data access"
    )
    ranking = e09.get("candidate_ranking")
    if not isinstance(ranking, list) or not ranking:
        raise CheckpointIntegrityError("E09 candidate ranking is missing")
    selected = next(
        (
            item
            for item in ranking
            if isinstance(item, dict) and item.get("candidate") == e09.get("selected_candidate")
        ),
        None,
    )
    if not isinstance(selected, dict):
        raise CheckpointIntegrityError("E09 selected candidate is absent from its ranking")
    selected_run_relative = str(selected["run_manifest"])
    selected_run, selected_run_binding = _validate_run(
        root, selected_run_relative, "E09", protocol_id
    )
    _require(
        file_sha256(_safe_path(root, selected_run_relative)) == selected.get("run_manifest_sha256"),
        "E09 selected run hash mismatch",
    )

    e10_relative = "artifacts/training/E10/E10-main-sft-seed1710-v1/RUN_COMPLETE.json"
    e11_relative = "artifacts/training/E11/E11-offline-dpo-seed1711-v1/RUN_COMPLETE.json"
    e10, e10_binding = _validate_run(root, e10_relative, "E10", protocol_id)
    e11, e11_binding = _validate_run(root, e11_relative, "E11", protocol_id)
    _require(
        e11.get("starting_identity") == e10.get("final_adapter_identity"),
        "E11 does not start from the completed E10 adapter",
    )
    _require(
        e11.get("dpo_reference_identity") == e10.get("final_adapter_identity"),
        "E11 frozen DPO reference is not the E10 adapter",
    )

    e10_eval = _validate_development_evaluation(
        root,
        "E10",
        e10,
        "outputs/E10-main-sft-dev-eval-v1/summary.json",
        "outputs/E10-main-sft-dev-eval-v1/predictions.jsonl",
    )
    e11_eval = _validate_development_evaluation(
        root,
        "E11",
        e11,
        "outputs/E11-offline-dpo-dev-eval-v1/summary.json",
        "outputs/E11-offline-dpo-dev-eval-v1/predictions.jsonl",
    )
    preference_comparison = _validate_preference_comparison(root, protocol_id, e10, e11)

    identity_path, identity = _json_object(
        root, "artifacts/training/E11/E11-offline-dpo-seed1711-v1/RUN_IDENTITY.json"
    )
    checkpoints: dict[str, Any] = {}
    for step in (24, 48):
        relative = f"artifacts/training/E11/E11-offline-dpo-seed1711-v1/checkpoint-{step}"
        observed_step = validate_native_trainer_checkpoint(_safe_path(root, relative), identity)
        _require(observed_step == step, f"E11 checkpoint-{step} step mismatch")
        completion = f"{relative}/CHECKPOINT_COMPLETE.json"
        checkpoints[str(step)] = {
            "path": relative,
            "completion_manifest": _binding(root, completion),
        }

    last_runtime = "artifacts/training/E11/E11-offline-dpo-seed1711-v1/checkpoint-48"
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint_id": "CP30_POSTTRAIN_BASELINES",
        "protocol_id": protocol_id,
        "protected_access_state": "NOT_ACCESSED_BY_BASELINE_EXPERIMENTS",
        "experiments": {
            "E08": {
                "model_identity": e08.get("model_identity"),
                "evidence": _binding(root, e08_path),
                "summary": _binding(root, e08_summary),
                "predictions": _binding(root, e08_predictions),
            },
            "E09": {
                "selected_candidate": e09.get("selected_candidate"),
                "selected_adapter_identity": selected_run.get("final_adapter_identity"),
                "selection": _binding(root, e09_path),
                "selected_run": selected_run_binding,
            },
            "E10": {"run": e10_binding, "development_evaluation": e10_eval},
            "E11": {
                "run": e11_binding,
                "development_evaluation": e11_eval,
                "run_identity": _binding(root, identity_path.relative_to(root).as_posix()),
                "checkpoints": checkpoints,
            },
        },
        "preference_comparison": preference_comparison,
        "last_complete_runtime_checkpoint": last_runtime,
    }
    model_identities = {
        "e08_frozen_reference": str(e08.get("model_identity")),
        "e09_selected_adapter": str(selected_run["final_adapter_identity"]),
        "e10_sft_adapter": str(e10["final_adapter_identity"]),
        "e11_dpo_adapter": str(e11["final_adapter_identity"]),
        "e11_checkpoint_24_completion": str(checkpoints["24"]["completion_manifest"]["sha256"]),
        "e11_checkpoint_48_completion": str(checkpoints["48"]["completion_manifest"]["sha256"]),
    }
    return MilestoneEvidence(artifact, model_identities, last_runtime)


_CP40_RUNS = {
    "E12": ("E12-uniform-grpo-seed1701-v1", "E12-uniform-grpo-dev-eval-v1"),
    "E13": ("E13-difficulty-grpo-seed1701-v1", "E13-difficulty-grpo-dev-eval-v1"),
    "E15": (
        "E15-reward-variance-grpo-seed1701-v1",
        "E15-reward-variance-grpo-dev-eval-v1",
    ),
}
_CP40_AGGREGATES = {
    "E12": "evidence/training/E12-uniform-grpo-seed1701-v1.json",
    "E13": "evidence/training/E13-difficulty-grpo-seed1701-v1.json",
    "E15": "evidence/training/E15-reward-variance-grpo-seed1701-v1.json",
}
_CP40_COMPARISONS = {
    "E12": "evidence/evaluation/E12_UNIFORM_GRPO_DEVELOPMENT_COMPARISON.json",
    "E13": "evidence/evaluation/E13_DIFFICULTY_GRPO_DEVELOPMENT_COMPARISON.json",
    "E15": "evidence/evaluation/E15_REWARD_VARIANCE_GRPO_DEVELOPMENT_COMPARISON.json",
}


def _adapter_inventory_identity(adapter: Path) -> str:
    inventory = {
        path.relative_to(adapter).as_posix(): file_sha256(path)
        for path in sorted(adapter.rglob("*"))
        if path.is_file()
    }
    _require(bool(inventory), f"final adapter has no files: {adapter}")
    return identity_hash(inventory)


def _validate_cp40_run(root: Path, experiment_id: str, run_id: str) -> dict[str, Any]:
    run_root = f"artifacts/training/{experiment_id}/{run_id}"
    run_relative = f"{run_root}/RUN_COMPLETE.json"
    identity_relative = f"{run_root}/RUN_IDENTITY.json"
    _, run = _json_object(root, run_relative)
    _, run_identity = _json_object(root, identity_relative)
    for value, message in (
        (run.get("status") == "COMPLETE", "run is not COMPLETE"),
        (run.get("experiment_id") == experiment_id, "run experiment identity mismatch"),
        (run.get("run_id") == run_id, "run ID mismatch"),
        (run.get("protocol_id") == "RFPT-RL-DEVELOPMENT-v2", "run protocol mismatch"),
        (run.get("execution_mode") == "FULL", "run is not the FULL arm"),
        (run.get("optimizer_steps") == 32, "optimizer-step count is not 32"),
        (run.get("completed_rollout_groups") == 64, "rollout-group count is not 64"),
        (run.get("completed_rollouts") == 512, "rollout count is not 512"),
        (run_identity.get("experiment_id") == experiment_id, "identity experiment mismatch"),
        (run_identity.get("run_id") == run_id, "identity run mismatch"),
        (
            run_identity.get("protected_access_state") == "NOT_ACCESSED",
            "run identity reports protected access",
        ),
    ):
        _require(value, f"{experiment_id} {message}")
    identity_fields = (
        "protocol_id",
        "execution_mode",
        "starting_adapter_identity",
        "suite_config_hash",
        "schedule_hash",
    )
    for field in identity_fields:
        _require(
            run.get(field) == run_identity.get(field),
            f"{experiment_id} run/identity {field} mismatch",
        )
    _require(
        run.get("execution_identity_hash") == identity_hash(run_identity),
        f"{experiment_id} execution identity hash mismatch",
    )
    adapter_id = run.get("final_adapter_identity")
    adapter_relative = run.get("final_adapter_path")
    _require(
        isinstance(adapter_id, str) and len(adapter_id) == 64,
        f"{experiment_id} final adapter identity is invalid",
    )
    if not isinstance(adapter_relative, str) or not adapter_relative:
        raise CheckpointIntegrityError(f"{experiment_id} final adapter path is invalid")
    adapter_path = _safe_path(root, adapter_relative)
    _require(adapter_path.is_dir(), f"{experiment_id} final adapter is missing")
    _require(
        _adapter_inventory_identity(adapter_path) == adapter_id,
        f"{experiment_id} final adapter inventory identity mismatch",
    )
    reward_relative = run.get("rollout_reward_evidence")
    _require(
        isinstance(reward_relative, str)
        and run.get("rollout_reward_evidence_sha256")
        == file_sha256(_safe_path(root, reward_relative)),
        f"{experiment_id} rollout reward evidence hash mismatch",
    )

    checkpoint_relative = f"{run_root}/checkpoint-32"
    try:
        observed_step = validate_native_trainer_checkpoint(
            _safe_path(root, checkpoint_relative), run_identity
        )
    except (OSError, ValueError) as error:
        raise CheckpointIntegrityError(
            f"{experiment_id} checkpoint-32 validation failed: {error}"
        ) from error
    _require(observed_step == 32, f"{experiment_id} checkpoint-32 step mismatch")
    completion_relative = f"{checkpoint_relative}/CHECKPOINT_COMPLETE.json"
    _, completion = _json_object(root, completion_relative)
    inventory = completion.get("file_inventory")
    if (
        completion.get("status") != "COMPLETE"
        or completion.get("checkpoint_step") != 32
        or not isinstance(inventory, dict)
        or not inventory
    ):
        raise CheckpointIntegrityError(
            f"{experiment_id} checkpoint-32 completion inventory is invalid"
        )
    return {
        "run": run,
        "identity": run_identity,
        "run_binding": _binding(root, run_relative),
        "identity_binding": _binding(root, identity_relative),
        "checkpoint": {
            "path": checkpoint_relative,
            "step": 32,
            "file_count": len(inventory),
            "completion_manifest": _binding(root, completion_relative),
        },
    }


def _validate_cp40_evaluation(
    root: Path,
    experiment_id: str,
    evaluation_id: str,
    run: dict[str, Any],
    aggregate: dict[str, Any],
    diagnostics_name: str = "MODEL_DIAGNOSTICS.json",
) -> dict[str, Any]:
    output_root = f"outputs/{evaluation_id}"
    summary_relative = f"{output_root}/summary.json"
    predictions_relative = f"{output_root}/predictions.jsonl"
    diagnostics_relative = f"{output_root}/{diagnostics_name}"
    memory_relative = f"{output_root}/memory_telemetry.jsonl"
    _, summary = _json_object(root, summary_relative)
    _, diagnostics = _json_object(root, diagnostics_relative)
    rows = _prediction_rows(root, predictions_relative)
    memory_rows = _prediction_rows(root, memory_relative)
    expected_adapter = str(run["final_adapter_path"]).replace("\\", "/")
    observed_adapter = str(summary.get("adapter", "")).replace("\\", "/")
    model_identity = summary.get("model_identity")
    adapter_id = str(run["final_adapter_identity"])
    _require(summary.get("status") == "COMPLETE", f"{experiment_id} eval is not COMPLETE")
    _require(
        summary.get("experiment_id") == experiment_id and summary.get("run_id") == evaluation_id,
        f"{experiment_id} eval identity mismatch",
    )
    _require(
        summary.get("scientific_state") == "DEVELOPMENT",
        f"{experiment_id} eval is not DEVELOPMENT",
    )
    _require(summary.get("dataset_rows") == 36, f"{experiment_id} eval must contain 36 rows")
    _require(observed_adapter == expected_adapter, f"{experiment_id} eval adapter mismatch")
    _require(
        isinstance(model_identity, str) and adapter_id[:8] in model_identity,
        f"{experiment_id} eval model/adapter identity mismatch",
    )
    example_ids = [row.get("example_id") for row in rows]
    _require(
        len(rows) == 36 and len(set(example_ids)) == 36 and all(example_ids),
        f"{experiment_id} eval predictions are not 36 unique examples",
    )
    for row in rows:
        _require(
            row.get("split_role") == "DEVELOPMENT"
            and row.get("experiment_id") == experiment_id
            and row.get("run_id") == evaluation_id
            and row.get("model_identity") == model_identity,
            f"{experiment_id} prediction identity or role mismatch",
        )
    _require(
        diagnostics.get("eval_mode") is True
        and diagnostics.get("loaded_in_4bit") is True
        and diagnostics.get("model_construction_count") == 1
        and diagnostics.get("adapter_wrapper_count") == 1
        and diagnostics.get("trainable_parameter_count") == 0
        and diagnostics.get("optimizer_constructed") is False,
        f"{experiment_id} bounded evaluation diagnostics are invalid",
    )
    _require(
        len(memory_rows) == 36
        and memory_rows[-1].get("completed_rows") == 36
        and all(row.get("bounded_memory") is True for row in memory_rows),
        f"{experiment_id} memory telemetry is incomplete or unbounded",
    )
    development = aggregate.get("development_evaluation")
    if not isinstance(development, dict):
        raise CheckpointIntegrityError(f"{experiment_id} aggregate omits DEVELOPMENT evidence")
    expected_hashes = {
        "summary_sha256": summary_relative,
        "predictions_sha256": predictions_relative,
        "model_diagnostics_sha256": diagnostics_relative,
        "memory_telemetry_sha256": memory_relative,
    }
    for field, relative in expected_hashes.items():
        _require(
            development.get(field) == file_sha256(_safe_path(root, relative)),
            f"{experiment_id} aggregate {field} mismatch",
        )
    _require(
        development.get("status") == "COMPLETE"
        and development.get("run_id") == evaluation_id
        and development.get("dataset_rows") == 36
        and development.get("split_role") == "DEVELOPMENT"
        and development.get("bounded_memory") is True,
        f"{experiment_id} aggregate DEVELOPMENT identity mismatch",
    )
    return {
        "run_id": evaluation_id,
        "model_identity": model_identity,
        "rows": 36,
        "generation_config_id": summary.get("generation_config_id"),
        "summary": _binding(root, summary_relative),
        "predictions": _binding(root, predictions_relative),
        "diagnostics": _binding(root, diagnostics_relative),
        "memory_telemetry": _binding(root, memory_relative),
    }


def _validate_cp40_comparison(
    root: Path, experiment_id: str, evaluation: dict[str, Any], aggregate: dict[str, Any]
) -> dict[str, Any]:
    relative = _CP40_COMPARISONS[experiment_id]
    _, comparison = _json_object(root, relative)
    _require(comparison.get("status") == "COMPLETE", f"{experiment_id} comparison incomplete")
    _require(
        comparison.get("scientific_state") == "DEVELOPMENT"
        and comparison.get("protected_data_accessed") is False,
        f"{experiment_id} comparison crosses the protected boundary",
    )
    _require(
        comparison.get("candidate_run_id") == evaluation["run_id"]
        and comparison.get("analysis_units") == "36 unique matched example_ids"
        and comparison.get("seed_count") == 1,
        f"{experiment_id} comparison analysis identity mismatch",
    )
    inputs = comparison.get("inputs")
    if not isinstance(inputs, dict) or "candidate" not in inputs:
        raise CheckpointIntegrityError(f"{experiment_id} comparison inputs are missing")
    aligned_ids: list[str] | None = None
    bound_inputs: dict[str, Any] = {}
    for label, raw_input in inputs.items():
        if not isinstance(raw_input, dict):
            raise CheckpointIntegrityError(f"{experiment_id} comparison input {label} is invalid")
        summary_relative = raw_input.get("summary")
        predictions_relative = raw_input.get("predictions")
        if not isinstance(summary_relative, str) or not isinstance(predictions_relative, str):
            raise CheckpointIntegrityError(f"{experiment_id} comparison input paths are invalid")
        _, summary = _json_object(root, summary_relative)
        rows = _prediction_rows(root, predictions_relative)
        ids = [str(row.get("example_id")) for row in rows]
        _require(
            len(rows) == 36
            and len(set(ids)) == 36
            and summary.get("scientific_state") == "DEVELOPMENT"
            and summary.get("status") == "COMPLETE"
            and summary.get("run_id") == raw_input.get("run_id")
            and all(row.get("split_role") == "DEVELOPMENT" for row in rows),
            f"{experiment_id} comparison input {label} is not aligned DEVELOPMENT evidence",
        )
        _require(
            raw_input.get("summary_sha256") == file_sha256(_safe_path(root, summary_relative))
            and raw_input.get("predictions_sha256")
            == file_sha256(_safe_path(root, predictions_relative)),
            f"{experiment_id} comparison input {label} hash mismatch",
        )
        if aligned_ids is None:
            aligned_ids = ids
        _require(ids == aligned_ids, f"{experiment_id} comparison input IDs are not aligned")
        bound_inputs[str(label)] = {
            "run_id": raw_input.get("run_id"),
            "summary": _binding(root, summary_relative),
            "predictions": _binding(root, predictions_relative),
        }
    comparisons = comparison.get("comparisons")
    if not isinstance(comparisons, dict) or not comparisons:
        raise CheckpointIntegrityError(f"{experiment_id} paired comparisons are missing")
    for label, value in comparisons.items():
        if not isinstance(value, dict):
            raise CheckpointIntegrityError(f"{experiment_id}/{label} comparison is invalid")
        binary = value.get("binary_pass")
        continuous = value.get("continuous_score")
        _require(
            isinstance(binary, dict)
            and binary.get("n") == 36
            and binary.get("seed_count") == 1
            and isinstance(continuous, dict)
            and continuous.get("analysis_unit_count") == 36
            and continuous.get("observation_count") == 36
            and continuous.get("seed_count") == 1,
            f"{experiment_id}/{label} paired statistics have invalid units",
        )
    aggregate_comparison = next(
        (
            value
            for key, value in aggregate.items()
            if "comparison" in key and isinstance(value, dict) and value.get("path") == relative
        ),
        None,
    )
    _require(
        isinstance(aggregate_comparison, dict)
        and aggregate_comparison.get("sha256") == file_sha256(_safe_path(root, relative)),
        f"{experiment_id} aggregate comparison hash mismatch",
    )
    return {
        "aggregate": _binding(root, relative),
        "aligned_development_ids": len(aligned_ids or []),
        "inputs": bound_inputs,
    }


def _validate_e14_equivalence(
    root: Path, e12: dict[str, Any], e12_identity: dict[str, Any]
) -> dict[str, Any]:
    relative = "evidence/training/E14/POLICY_EQUIVALENCE.json"
    _, value = _json_object(root, relative)
    required = {
        "status": "MEASURED_POLICY_EQUIVALENCE",
        "evidence_class": "MEASURED",
        "experiment_id": "E14",
        "source_experiment_id": "E12",
        "method": "EXACT_POLICY_AND_SEEDED_TASK_SEQUENCE_EQUIVALENCE",
        "disposition": "NO_REDUNDANT_CUDA_RUN_REQUIRED",
        "cuda_run_executed_for_e14": False,
        "protected_access_state": "NOT_ACCESSED",
        "protocol_id": "RFPT-RL-DEVELOPMENT-v2",
        "all_tasks_reliability_eligible": True,
        "uniform_distribution_equals_reliability_distribution": True,
        "computed_seeded_sequences_equal": True,
        "computed_sequence_equals_frozen_e12_schedule": True,
        "candidate_task_count": 96,
        "q_value_count": 96,
        "q_unique_values": [1.0],
        "rollout_group_count": 64,
    }
    for field, expected in required.items():
        _require(value.get(field) == expected, f"E14 policy-equivalence {field} mismatch")
    _require(
        value.get("starting_adapter_identity") == e12.get("starting_adapter_identity")
        and value.get("suite_config_hash") == e12.get("suite_config_hash")
        and value.get("e12_schedule_hash") == e12.get("schedule_hash")
        and value.get("fairness_budget_hash") == e12_identity.get("fairness_budget_hash")
        and value.get("reward_config_hash") == e12_identity.get("reward_config_hash")
        and value.get("sampler_config_hash") == e12_identity.get("sampler_config_hash"),
        "E14/E12 frozen identities differ",
    )
    estimates_relative = value.get("adaptive_estimates_path")
    schedule_relative = value.get("e12_schedule_path")
    run_relative = value.get("e12_run_complete_path")
    if not all(
        isinstance(item, str) for item in (estimates_relative, schedule_relative, run_relative)
    ):
        raise CheckpointIntegrityError("E14 source paths are invalid")
    _, estimates = _json_object(root, str(estimates_relative))
    _, schedule = _json_object(root, str(schedule_relative))
    task_estimates = estimates.get("task_estimates")
    _require(
        estimates.get("status") == "QUALIFIED"
        and isinstance(task_estimates, list)
        and len(task_estimates) == 96
        and all(isinstance(item, dict) and item.get("q") == 1.0 for item in task_estimates),
        "E14 adaptive reliability estimates are not 96 qualified q=1 values",
    )
    task_sequence = value.get("task_sequence")
    _require(
        isinstance(value.get("candidate_pool_identity"), str)
        and len(value["candidate_pool_identity"]) == 64,
        "E14 candidate-pool identity is invalid",
    )
    _require(
        isinstance(task_sequence, list)
        and task_sequence == schedule.get("task_ids")
        and len(task_sequence) == 64
        and value.get("task_sequence_identity") == identity_hash(task_sequence),
        "E14 task sequence does not exactly match E12",
    )
    hashes = (
        ("adaptive_estimates_sha256", str(estimates_relative)),
        ("e12_schedule_file_sha256", str(schedule_relative)),
        ("e12_run_complete_sha256", str(run_relative)),
    )
    for field, source in hashes:
        _require(
            value.get(field) == file_sha256(_safe_path(root, source)),
            f"E14 {field} mismatch",
        )
    return {
        "evidence": _binding(root, relative),
        "source_experiment_id": "E12",
        "cuda_run_executed": False,
        "policy_and_seeded_sequence_equivalent": True,
        "adaptive_estimates": _binding(root, str(estimates_relative)),
        "e12_schedule": _binding(root, str(schedule_relative)),
    }


def qualify_cp40_evidence(root: Path, registry: ExperimentRegistry) -> MilestoneEvidence:
    """Fail closed unless every frozen RL-baseline artifact is complete and hash-bound."""

    root = root.resolve()
    _validate_complete_registry(registry, "CP40_RL_BASELINES", ("E12", "E13", "E14", "E15"))
    experiments: dict[str, Any] = {}
    model_identities: dict[str, str] = {}
    runs: dict[str, dict[str, Any]] = {}
    run_identities: dict[str, dict[str, Any]] = {}
    shared_start: str | None = None
    shared_suite: str | None = None
    for experiment_id, (run_id, evaluation_id) in _CP40_RUNS.items():
        qualified = _validate_cp40_run(root, experiment_id, run_id)
        run = qualified["run"]
        runs[experiment_id] = run
        run_identities[experiment_id] = qualified["identity"]
        if shared_start is None:
            shared_start = str(run["starting_adapter_identity"])
            shared_suite = str(run["suite_config_hash"])
        _require(
            run.get("starting_adapter_identity") == shared_start
            and run.get("suite_config_hash") == shared_suite,
            f"{experiment_id} violates the frozen RL starting/suite identity",
        )
        aggregate_relative = _CP40_AGGREGATES[experiment_id]
        _, aggregate = _json_object(root, aggregate_relative)
        _require(
            aggregate.get("artifact_status") == "COMPLETE"
            and aggregate.get("experiment_status") == "COMPLETE"
            and aggregate.get("experiment_id") == experiment_id
            and aggregate.get("run_id") == run_id
            and aggregate.get("protocol_id") == "RFPT-RL-DEVELOPMENT-v2"
            and aggregate.get("protected_data_accessed") is False,
            f"{experiment_id} aggregate identity/status is invalid",
        )
        _require(
            aggregate.get("run_complete_sha256") == qualified["run_binding"]["sha256"]
            and aggregate.get("run_identity_sha256") == qualified["identity_binding"]["sha256"]
            and aggregate.get("final_adapter_identity") == run.get("final_adapter_identity"),
            f"{experiment_id} aggregate training hashes mismatch",
        )
        checkpoint_record = aggregate.get("checkpoint")
        _require(
            isinstance(checkpoint_record, dict)
            and checkpoint_record.get("step") == 32
            and checkpoint_record.get("status") == "VERIFIED"
            and checkpoint_record.get("complete_manifest_sha256")
            == qualified["checkpoint"]["completion_manifest"]["sha256"],
            f"{experiment_id} aggregate checkpoint binding mismatch",
        )
        evaluation = _validate_cp40_evaluation(root, experiment_id, evaluation_id, run, aggregate)
        comparison = _validate_cp40_comparison(root, experiment_id, evaluation, aggregate)
        experiments[experiment_id] = {
            "run": qualified["run_binding"],
            "run_identity": qualified["identity_binding"],
            "aggregate": _binding(root, aggregate_relative),
            "checkpoint": qualified["checkpoint"],
            "development_evaluation": evaluation,
            "paired_comparison": comparison,
        }
        model_identities[f"{experiment_id.lower()}_adapter"] = str(run["final_adapter_identity"])
        model_identities[f"{experiment_id.lower()}_checkpoint_32_completion"] = str(
            qualified["checkpoint"]["completion_manifest"]["sha256"]
        )
    e14 = _validate_e14_equivalence(root, runs["E12"], run_identities["E12"])
    experiments["E14"] = e14
    model_identities["e14_reliability_policy_source_adapter"] = str(
        runs["E12"]["final_adapter_identity"]
    )
    last_runtime = "artifacts/training/E15/E15-reward-variance-grpo-seed1701-v1/checkpoint-32"
    artifact = {
        "schema_version": 1,
        "checkpoint_id": "CP40_RL_BASELINES",
        "protocol_id": registry.protocol_id,
        "rl_protocol_id": "RFPT-RL-DEVELOPMENT-v2",
        "protected_access_state": "ACQUIRED_REGISTERED_NOT_SCORED",
        "frozen_starting_adapter_identity": shared_start,
        "frozen_suite_config_hash": shared_suite,
        "experiments": experiments,
        "last_complete_runtime_checkpoint": last_runtime,
    }
    return MilestoneEvidence(artifact, model_identities, last_runtime)


_CP50_RUNS = {
    "E16": ("E16-terminal-uniform-grpo-seed1701-v1", "E16-terminal-uniform-grpo-dev-eval-v1"),
    "E18": ("E18-rcac-grpo-seed1701-v1", "E18-rcac-grpo-dev-eval-v1"),
    "E20": (
        "E20-rcac-no-difficulty-grpo-seed1701-v1",
        "E20-rcac-no-difficulty-grpo-dev-eval-v1",
    ),
}
_CP50_AGGREGATES = {
    "E16": "evidence/training/E16-terminal-uniform-grpo-seed1701-v1.json",
    "E18": "evidence/training/E18-rcac-grpo-seed1701-v1.json",
    "E20": "evidence/training/E20-rcac-no-difficulty-grpo-seed1701-v1.json",
}
_CP50_COMPARISONS = {
    "E16": ("evidence/evaluation/E16_TERMINAL_VS_STRUCTURED_DEVELOPMENT_COMPARISON.json",),
    "E18": ("evidence/evaluation/E18_RCAC_DEVELOPMENT_COMPARISON.json",),
    "E20": (
        "evidence/evaluation/E20_VS_E18_DEVELOPMENT_COMPARISON.json",
        "evidence/evaluation/E20_VS_E16_DEVELOPMENT_COMPARISON.json",
    ),
}


def _validate_frozen_decision_policy(root: Path) -> dict[str, str | int]:
    relative = "docs/FROZEN_DECISION_POLICY.md"
    path = _safe_path(root, relative)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise CheckpointIntegrityError(f"frozen decision policy is unreadable: {error}") from error
    header = "\n".join(text.splitlines()[:20]).upper()
    _require("STATUS: FROZEN" in header, "decision policy is not explicitly FROZEN")
    _require("STATUS: UNFROZEN" not in header, "decision policy remains UNFROZEN")
    _require(
        "CP50_CANDIDATE_FREEZE" in header,
        "decision policy is not bound to CP50_CANDIDATE_FREEZE",
    )
    lowered = text.lower()
    for phrase in (
        "robust reasoning",
        "stateful tool",
        "final-state",
        "instruction following",
        "safety",
        "truthfulness",
        "broad capability",
        "holm",
    ):
        _require(phrase in lowered, f"frozen decision policy omits {phrase!r}")
    _require(
        "| pending" not in lowered and "pending |" not in lowered,
        "decision policy retains pending endpoint values",
    )
    return _binding(root, relative)


def _validate_cp50_precision_register(root: Path) -> dict[str, str | int]:
    relative = "evidence/policy/CP50_PILOT_VARIANCE_SUMMARY.json"
    _, value = _json_object(root, relative)
    _require(
        value.get("status") == "FROZEN_INPUT"
        and value.get("artifact_type") == "CP50_PILOT_VARIANCE_AND_PRECISION_REGISTER"
        and value.get("source_role") == "DEVELOPMENT"
        and value.get("protected_data_accessed") is False,
        "CP50 precision register is not frozen DEVELOPMENT-only evidence",
    )
    expected_runs = {
        "E12-uniform-grpo-dev-eval-v1",
        "E13-difficulty-grpo-dev-eval-v1",
        "E15-reward-variance-grpo-dev-eval-v1",
        "E16-terminal-uniform-grpo-dev-eval-v1",
        "E18-rcac-grpo-dev-eval-v1",
        "E20-rcac-no-difficulty-grpo-dev-eval-v1",
    }
    _require(
        set(value.get("eligible_run_ids", [])) == expected_runs,
        "CP50 precision register eligible-run set mismatch",
    )
    input_hashes = value.get("input_sha256")
    if not isinstance(input_hashes, dict) or set(input_hashes) != expected_runs:
        raise CheckpointIntegrityError("CP50 precision input inventory mismatch")
    for run_id, recorded_hash in input_hashes.items():
        predictions = f"outputs/{run_id}/predictions.jsonl"
        rows = _prediction_rows(root, predictions)
        _require(
            recorded_hash == file_sha256(_safe_path(root, predictions))
            and len(rows) == 36
            and all(row.get("split_role") == "DEVELOPMENT" for row in rows),
            f"CP50 precision input is not hash-bound DEVELOPMENT evidence: {run_id}",
        )
    endpoints = value.get("endpoints")
    required_endpoints = {
        "robust_reasoning_success",
        "stateful_tool_final_state_success",
        "instruction_following",
        "safety_authorization",
        "truthfulness_mc",
        "broad_capability_mc",
    }
    if not isinstance(endpoints, dict) or set(endpoints) != required_endpoints:
        raise CheckpointIntegrityError("CP50 precision endpoint register mismatch")
    for endpoint, record in endpoints.items():
        _require(
            isinstance(record, dict)
            and record.get("rule") in {"MATERIAL_IMPROVEMENT", "NONINFERIORITY"}
            and isinstance(record.get("planned_pairs"), int)
            and record["planned_pairs"] > 0
            and isinstance(record.get("decision_margin"), (float, int))
            and record["decision_margin"] > 0
            and record.get("precision_status") in {"ADEQUATE", "UNDERPOWERED_WITHIN_BUDGET"},
            f"CP50 precision endpoint is incomplete: {endpoint}",
        )
    return _binding(root, relative)


def _inventory_records(value: object, label: str) -> dict[str, str]:
    records: dict[str, str] = {}
    if isinstance(value, dict):
        raw_items = list(value.items())
    elif isinstance(value, list):
        raw_items = [
            (item.get("path"), item.get("sha256")) for item in value if isinstance(item, dict)
        ]
    else:
        raise CheckpointIntegrityError(f"{label} is not a path/hash inventory")
    for raw_path, raw_record in raw_items:
        digest = raw_record.get("sha256") if isinstance(raw_record, dict) else raw_record
        if not isinstance(raw_path, str) or not isinstance(digest, str):
            raise CheckpointIntegrityError(f"{label} has an invalid path/hash record")
        if raw_path in records:
            raise CheckpointIntegrityError(f"{label} repeats path: {raw_path}")
        records[raw_path] = digest
    return records


def _validate_cp50_protected_protocol_freeze(root: Path) -> dict[str, Any]:
    relative = "evidence/protected_protocol/CP50_PROTECTED_PROTOCOL_FREEZE.json"
    _, value = _json_object(root, relative)
    required = {
        "status": "FROZEN",
        "protocol_id": "RFPT-PROTECTED-v1",
        "protected_access_state": "ACQUIRED_REGISTERED_NOT_SCORED",
    }
    for field, expected in required.items():
        _require(value.get(field) == expected, f"protected protocol freeze {field} mismatch")
    invariants = value.get("invariants")
    if not isinstance(invariants, dict):
        raise CheckpointIntegrityError("protected protocol freeze invariant register is missing")
    for field in (
        "cp60_required_for_generation",
        "batch_size_one",
        "bounded_memory",
        "separate_scoring",
        "resume_missing_ids_only",
        "duplicate_example_ids_rejected",
        "immutable_completed_outputs",
        "references_excluded_from_output_records",
    ):
        _require(invariants.get(field) is True, f"protected protocol invariant {field} is not true")
    actions = value.get("qualification_actions")
    _require(
        isinstance(actions, dict)
        and actions.get("model_loaded") is False
        and actions.get("cuda_used") is False
        and actions.get("outputs_generated") is False
        and actions.get("scores_computed") is False
        and actions.get("observed_existing_output_manifests") == 0,
        "protected protocol freeze reports model execution, scoring, or existing outputs",
    )

    registry = value.get("benchmark_registry")
    generation = value.get("generation_config")
    if not isinstance(registry, dict) or not isinstance(generation, dict):
        raise CheckpointIntegrityError("protected protocol config bindings are missing")
    registry_path = registry.get("path")
    generation_path = generation.get("path")
    if not isinstance(registry_path, str) or not isinstance(generation_path, str):
        raise CheckpointIntegrityError("protected protocol config paths are invalid")
    _require(
        registry_path == "configs/evaluation/protected_benchmarks.yaml"
        and generation_path == "configs/evaluation/protected_generation.yaml"
        and registry.get("sha256") == file_sha256(_safe_path(root, registry_path))
        and generation.get("sha256") == file_sha256(_safe_path(root, generation_path)),
        "protected protocol config binding mismatch",
    )
    _, registry_config = _yaml_mapping(root, registry_path)
    _, generation_config = _yaml_mapping(root, generation_path)
    _require(
        generation_config.get("batch_size") == 1
        and generation_config.get("do_sample") is False
        and generation_config.get("bounded_memory") is True,
        "protected generation configuration is not deterministic batch-size one",
    )
    _require(
        generation.get("identity_sha256") == sha256_json(generation_config),
        "protected generation semantic identity mismatch",
    )

    expected_code = {
        "src/reasonforge/evaluation/protected_protocol.py",
        "src/reasonforge/evaluation/protected_inference.py",
        "src/reasonforge/evaluation/protected_outputs.py",
        "scripts/prepare_protected_run.py",
        "scripts/run_protected_generation.py",
        "scripts/verify_protected_run.py",
        "scripts/score_protected_outputs.py",
        "scripts/build_protected_safety_suite.py",
        "scripts/freeze_protected_finalist_matrix.py",
    }
    code_inventory = _inventory_records(value.get("code_inventory"), "protected code inventory")
    _require(set(code_inventory) == expected_code, "protected code inventory path set mismatch")
    for path, digest in code_inventory.items():
        _require(
            digest == file_sha256(_safe_path(root, path)),
            f"protected code inventory hash mismatch: {path}",
        )

    benchmark_specs = registry_config.get("benchmarks")
    datasets = value.get("dataset_inventory")
    if not isinstance(benchmark_specs, dict) or not isinstance(datasets, dict) or not datasets:
        raise CheckpointIntegrityError("protected dataset/spec inventory is missing")
    dataset_by_id: dict[str, dict[str, Any]] = {}
    for record in datasets.values():
        if not isinstance(record, dict) or not isinstance(record.get("benchmark_id"), str):
            raise CheckpointIntegrityError("protected dataset inventory record is invalid")
        benchmark_id = record["benchmark_id"]
        _require(
            benchmark_id not in dataset_by_id, f"duplicate protected benchmark: {benchmark_id}"
        )
        dataset_by_id[benchmark_id] = record
    spec_by_id = {
        str(spec.get("benchmark_id")): spec
        for spec in benchmark_specs.values()
        if isinstance(spec, dict)
    }
    _require(
        set(dataset_by_id) == set(spec_by_id),
        "protected dataset inventory does not match benchmark registry",
    )
    for benchmark_id, record in dataset_by_id.items():
        spec = spec_by_id[benchmark_id]
        dataset_path = record.get("path")
        if not isinstance(dataset_path, str):
            raise CheckpointIntegrityError(f"protected dataset path is invalid: {benchmark_id}")
        for field in (
            "source_sha256",
            "id_inventory_sha256",
            "example_identity_inventory_sha256",
        ):
            record_digest = record.get(field)
            _require(
                isinstance(record_digest, str) and len(record_digest) == 64,
                f"protected dataset {benchmark_id} has invalid {field}",
            )
        _require(
            record.get("source_sha256") == file_sha256(_safe_path(root, dataset_path))
            and record.get("source_sha256") == spec.get("source_sha256")
            and record.get("path") == str(spec.get("path")).replace("\\", "/")
            and record.get("role") == spec.get("role")
            and record.get("revision") == spec.get("revision")
            and record.get("expected_rows") == spec.get("expected_rows")
            and record.get("observed_rows") == spec.get("expected_rows")
            and record.get("generation_mode", "generate") == spec.get("generation_mode", "generate")
            and record.get("grader_id") == spec.get("grader_id")
            and record.get("grader_revision") == spec.get("grader_revision"),
            f"protected dataset identity does not match registry: {benchmark_id}",
        )
    safety = value.get("local_safety_suite")
    if not isinstance(safety, dict):
        raise CheckpointIntegrityError("protected local safety-suite binding is missing")
    expected_safety = {
        "rows_path": "data/protected/reasonforge_authorization_v1/rows.jsonl",
        "manifest_path": ("data/protected/reasonforge_authorization_v1/ACQUISITION_MANIFEST.json"),
    }
    safety_sources: dict[str, dict[str, str | int]] = {}
    for path_field, path in expected_safety.items():
        hash_field = path_field.replace("_path", "_sha256")
        _require(
            safety.get(path_field) == path
            and safety.get(hash_field) == file_sha256(_safe_path(root, path)),
            f"protected local safety-suite {path_field} binding mismatch",
        )
        safety_sources[path] = _binding(root, path)

    matrix_record = value.get("finalist_plan_matrix")
    plan_inventory = value.get("finalist_plan_inventory")
    if (
        value.get("finalist_plans_status") != "FROZEN"
        or not isinstance(matrix_record, dict)
        or not isinstance(plan_inventory, dict)
    ):
        raise CheckpointIntegrityError("protected finalist plan freeze is missing or not FROZEN")
    matrix_relative = matrix_record.get("path")
    _require(
        matrix_relative == "evidence/protected_protocol/FINALIST_RUN_PLAN_MATRIX.json",
        "protected finalist plan matrix path mismatch",
    )
    matrix_path = _safe_path(root, str(matrix_relative))
    _require(
        matrix_record.get("sha256") == file_sha256(matrix_path),
        "protected finalist plan matrix hash mismatch",
    )
    _, matrix = _json_object(root, str(matrix_relative))
    matrix_config_relative = matrix.get("matrix_path")
    _require(
        matrix.get("status") == "FROZEN"
        and matrix.get("protocol_id") == "RFPT-PROTECTED-v1"
        and matrix.get("model_loaded") is False
        and matrix.get("outputs_generated") is False
        and matrix.get("scores_computed") is False
        and matrix_config_relative == "configs/evaluation/protected_finalist_matrix.yaml",
        "protected finalist plan matrix status or execution boundary mismatch",
    )
    matrix_config_path, matrix_config = _yaml_mapping(root, str(matrix_config_relative))
    _require(
        matrix.get("matrix_sha256") == file_sha256(matrix_config_path)
        and matrix_config.get("protocol_id") == "RFPT-PROTECTED-v1",
        "protected finalist matrix configuration binding mismatch",
    )
    finalist_models = matrix_config.get("models")
    finalist_experiments = matrix_config.get("experiments")
    matrix_plans = matrix.get("plans")
    if (
        not isinstance(finalist_models, dict)
        or not isinstance(finalist_experiments, list)
        or not isinstance(matrix_plans, dict)
    ):
        raise CheckpointIntegrityError("protected finalist matrix structure is invalid")
    _require(
        set(finalist_models)
        == {
            "e16_terminal_uniform_seed1701",
            "e18_rcac_seed1701",
        },
        "protected finalist matrix model set mismatch",
    )
    expected_plan_tuples: set[tuple[str, str, str]] = set()
    for experiment in finalist_experiments:
        if not isinstance(experiment, dict) or not isinstance(experiment.get("benchmarks"), list):
            raise CheckpointIntegrityError("protected finalist experiment matrix is invalid")
        experiment_id = experiment.get("experiment_id")
        _require(
            experiment_id in {"E21", "E22", "E23", "E24"},
            "protected finalist experiment identity mismatch",
        )
        for benchmark_name in experiment["benchmarks"]:
            _require(
                isinstance(benchmark_name, str) and benchmark_name in datasets,
                "protected finalist matrix references an unregistered benchmark",
            )
            for model_id in finalist_models:
                expected_plan_tuples.add((str(experiment_id), model_id, benchmark_name))
    _require(
        matrix.get("plan_count") == 14
        and len(plan_inventory) == 14
        and set(plan_inventory) == set(matrix_plans),
        "protected finalist plan inventory/count mismatch",
    )
    observed_plan_tuples: set[tuple[str, str, str]] = set()
    plan_bindings: dict[str, dict[str, str | int]] = {}
    for output_set_id, inventory_record in plan_inventory.items():
        matrix_plan = matrix_plans[output_set_id]
        if not isinstance(inventory_record, dict) or not isinstance(matrix_plan, dict):
            raise CheckpointIntegrityError("protected finalist plan inventory record is invalid")
        for field in (
            "path",
            "sha256",
            "experiment_id",
            "model_id",
            "model_checkpoint_sha256",
            "benchmark_name",
        ):
            _require(
                inventory_record.get(field) == matrix_plan.get(field),
                f"protected finalist plan matrix/inventory {field} mismatch: {output_set_id}",
            )
        plan_relative = inventory_record.get("path")
        if not isinstance(plan_relative, str):
            raise CheckpointIntegrityError("protected finalist plan path is invalid")
        _require(
            inventory_record.get("sha256") == file_sha256(_safe_path(root, plan_relative)),
            f"protected finalist plan hash mismatch: {output_set_id}",
        )
        _, plan = _json_object(root, plan_relative)
        model_id = inventory_record.get("model_id")
        model_spec = finalist_models.get(model_id)
        benchmark_name = inventory_record.get("benchmark_name")
        dataset_record = datasets.get(benchmark_name)
        if not isinstance(model_spec, dict) or not isinstance(dataset_record, dict):
            raise CheckpointIntegrityError("protected finalist plan source identity is invalid")
        expected_ids = plan.get("expected_example_ids")
        example_identities = plan.get("example_identity_sha256")
        if not isinstance(expected_ids, list) or not isinstance(example_identities, dict):
            raise CheckpointIntegrityError(
                f"protected finalist plan example inventory is invalid: {output_set_id}"
            )
        _require(
            plan.get("schema_version") == 1
            and plan.get("protocol_id") == "RFPT-PROTECTED-v1"
            and plan.get("output_set_id") == output_set_id
            and plan.get("experiment_id") == inventory_record.get("experiment_id")
            and plan.get("model_id") == model_id
            and plan.get("benchmark_name") == benchmark_name
            and plan.get("model_checkpoint_sha256") == model_spec.get("model_checkpoint_sha256")
            and plan.get("adapter_path") == model_spec.get("adapter_path")
            and plan.get("base_model_id") == model_spec.get("base_model_id")
            and plan.get("base_model_revision") == model_spec.get("base_model_revision")
            and plan.get("tokenizer_id") == model_spec.get("tokenizer_id")
            and plan.get("generation_config") == generation_config
            and plan.get("generation_config_sha256") == generation.get("identity_sha256")
            and plan.get("protected_dataset_fingerprint")
            == matrix_plan.get("protected_dataset_fingerprint")
            and matrix_plan.get("expected_rows") == dataset_record.get("expected_rows")
            and len(expected_ids) == dataset_record.get("expected_rows")
            and len(example_identities) == dataset_record.get("expected_rows")
            and set(expected_ids) == set(example_identities),
            f"protected finalist plan identity/content mismatch: {output_set_id}",
        )
        observed_plan_tuples.add(
            (
                str(inventory_record.get("experiment_id")),
                str(model_id),
                str(benchmark_name),
            )
        )
        plan_bindings[str(output_set_id)] = _binding(root, plan_relative)
    _require(
        observed_plan_tuples == expected_plan_tuples,
        "protected finalist plan coverage does not match frozen matrix",
    )
    bound_finalist_models: dict[str, dict[str, Any]] = {}
    for model_id, model_spec in finalist_models.items():
        if not isinstance(model_spec, dict):
            raise CheckpointIntegrityError("protected finalist model specification is invalid")
        adapter_path = model_spec.get("adapter_path")
        if not isinstance(adapter_path, str):
            raise CheckpointIntegrityError("protected finalist adapter path is invalid")
        completion_path = str(
            Path(adapter_path).parent / "checkpoint-32" / "CHECKPOINT_COMPLETE.json"
        )
        completion_path = completion_path.replace("\\", "/")
        model_plan_records = [
            record
            for record in matrix_plans.values()
            if isinstance(record, dict) and record.get("model_id") == model_id
        ]
        _require(
            len(model_plan_records) == 7
            and all(
                record.get("checkpoint_completion_sha256")
                == model_spec.get("checkpoint_completion_sha256")
                for record in model_plan_records
            )
            and model_spec.get("checkpoint_completion_sha256")
            == file_sha256(_safe_path(root, completion_path)),
            f"protected finalist checkpoint-completion binding mismatch: {model_id}",
        )
        bound_finalist_models[str(model_id)] = {
            key: model_spec.get(key)
            for key in (
                "model_id",
                "base_model_id",
                "base_model_revision",
                "adapter_path",
                "model_checkpoint_sha256",
                "checkpoint_completion_sha256",
                "tokenizer_id",
            )
        }
    return {
        "freeze_manifest": _binding(root, relative),
        "benchmark_registry": _binding(root, registry_path),
        "generation_config": _binding(root, generation_path),
        "code_inventory": {path: _binding(root, path) for path in sorted(code_inventory)},
        "dataset_count": len(dataset_by_id),
        "dataset_identities": {
            benchmark_id: {
                key: record.get(key)
                for key in (
                    "path",
                    "source_sha256",
                    "role",
                    "revision",
                    "expected_rows",
                    "id_inventory_sha256",
                    "example_identity_inventory_sha256",
                    "generation_mode",
                    "grader_id",
                    "grader_revision",
                )
            }
            for benchmark_id, record in sorted(dataset_by_id.items())
        },
        "supporting_safety_sources": safety_sources,
        "finalist_plan_matrix": _binding(root, str(matrix_relative)),
        "finalist_matrix_config": _binding(root, str(matrix_config_relative)),
        "finalist_plan_count": len(plan_bindings),
        "finalist_plans": plan_bindings,
        "finalist_models": bound_finalist_models,
        "protected_access_state": "ACQUIRED_REGISTERED_NOT_SCORED",
    }


def _yaml_mapping(root: Path, relative: str) -> tuple[Path, dict[str, Any]]:
    path = _safe_path(root, relative)
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise CheckpointIntegrityError(
            f"required configuration is unreadable: {relative}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise CheckpointIntegrityError(f"required configuration is not a mapping: {relative}")
    return path, value


def _validate_cp50_configs(root: Path) -> dict[str, Any]:
    reward_relative = "configs/reward.yaml"
    sampler_relative = "configs/rcac.yaml"
    suite_relative = "configs/rl/e12_e20.yaml"
    reward_path, reward = _yaml_mapping(root, reward_relative)
    sampler_path, sampler = _yaml_mapping(root, sampler_relative)
    _, suite = _yaml_mapping(root, suite_relative)
    common = suite.get("common")
    experiments = suite.get("experiments")
    if not isinstance(common, dict) or not isinstance(experiments, dict):
        raise CheckpointIntegrityError("RL suite omits common or experiment configuration")
    _require(suite.get("protocol_id") == "RFPT-RL-DEVELOPMENT-v2", "RL suite protocol mismatch")
    _require(
        common.get("reward_config_sha256") == file_sha256(reward_path)
        and common.get("sampler_config_sha256") == file_sha256(sampler_path),
        "RL suite config-source hashes mismatch",
    )
    _require(reward.get("state") == "FROZEN", "reward configuration is not FROZEN")
    expected_arms = {
        "E16": ("uniform", "terminal"),
        "E18": ("rcac", "selected_from_E16"),
        "E19": ("rcac_no_reliability", "selected_from_E16"),
        "E20": ("rcac_no_difficulty", "selected_from_E16"),
    }
    for experiment_id, (sampler_mode, reward_architecture) in expected_arms.items():
        experiment = experiments.get(experiment_id)
        arms = experiment.get("arms") if isinstance(experiment, dict) else None
        _require(
            isinstance(arms, list)
            and len(arms) >= 1
            and any(
                isinstance(arm, dict)
                and arm.get("sampler") == sampler_mode
                and arm.get("reward_architecture") == reward_architecture
                for arm in arms
            ),
            f"RL suite {experiment_id} finalist/ablation arm mismatch",
        )
    required_sampler = {
        "tau_q": 0.75,
        "p_target": 0.55,
        "sigma": 0.2,
        "lambda_u": 0.5,
        "family_balance": "inverse_sqrt_frequency",
        "selection_rng_seed": 1701,
    }
    for field, expected in required_sampler.items():
        _require(sampler.get(field) == expected, f"RCAC source parameter {field} mismatch")
    return {
        "reward": _binding(root, reward_relative),
        "rcac_source": _binding(root, sampler_relative),
        "rl_suite_file": _binding(root, suite_relative),
        "rl_suite_semantic_hash": common.get("suite_config_hash", "BOUND_BY_RUN_IDENTITIES"),
        "rcac_frozen_for_finalists": required_sampler,
        "freeze_disposition": "FROZEN_FOR_FINALISTS_AT_CP50_WITHOUT_MUTATING_SOURCE",
    }


def _validate_selected_reward(root: Path) -> tuple[dict[str, Any], dict[str, str | int]]:
    relative = "evidence/training/E16/SELECTED_REWARD.json"
    _, selected = _json_object(root, relative)
    recorded_identity = selected.get("selection_identity")
    identity_payload = dict(selected)
    identity_payload.pop("selection_identity", None)
    _require(
        recorded_identity == identity_hash(identity_payload),
        "selected reward identity does not reproduce",
    )
    required = {
        "status": "SELECTED",
        "experiment_id": "E16",
        "protocol_id": "RFPT-RL-DEVELOPMENT-v2",
        "scientific_state": "DEVELOPMENT",
        "selected_reward_architecture": "terminal",
        "selected_arm_id": "E16_terminal",
        "protected_access_state": "NOT_ACCESSED",
        "training_seed_count": 1,
        "sample_size": 36,
        "terminal_eligible": True,
    }
    for field, expected in required.items():
        _require(selected.get(field) == expected, f"selected reward {field} mismatch")
    _require(
        selected.get("selected_reward_config_sha256")
        == file_sha256(_safe_path(root, "configs/reward.yaml")),
        "selected reward configuration hash mismatch",
    )
    sources = selected.get("source_artifacts")
    if not isinstance(sources, dict) or not sources:
        raise CheckpointIntegrityError("selected reward source bindings are missing")
    for name, record in sources.items():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise CheckpointIntegrityError(f"selected reward source {name} is invalid")
        _require(
            record.get("sha256") == file_sha256(_safe_path(root, record["path"])),
            f"selected reward source {name} hash mismatch",
        )
    return selected, _binding(root, relative)


def _validate_candidate_comparison(
    root: Path, relative: str, candidate_run_id: str
) -> dict[str, Any]:
    _, comparison = _json_object(root, relative)
    _require(
        comparison.get("status") == "COMPLETE"
        and comparison.get("scientific_state") == "DEVELOPMENT"
        and comparison.get("protected_data_accessed") is False,
        f"candidate comparison is incomplete or crosses protected boundary: {relative}",
    )
    _require(
        comparison.get("candidate_run_id") == candidate_run_id
        and comparison.get("analysis_units") == "36 unique matched example_ids"
        and comparison.get("seed_count") == 1,
        f"candidate comparison analysis identity mismatch: {relative}",
    )
    inputs = comparison.get("inputs")
    if not isinstance(inputs, dict) or "candidate" not in inputs:
        raise CheckpointIntegrityError(f"candidate comparison inputs are missing: {relative}")
    aligned_ids: list[str] | None = None
    bindings: dict[str, Any] = {}
    for label, record in inputs.items():
        if not isinstance(record, dict):
            raise CheckpointIntegrityError(f"comparison input {label} is invalid: {relative}")
        summary_relative = record.get("summary")
        predictions_relative = record.get("predictions")
        if not isinstance(summary_relative, str) or not isinstance(predictions_relative, str):
            raise CheckpointIntegrityError(f"comparison input {label} paths are invalid")
        _, summary = _json_object(root, summary_relative)
        predictions = _prediction_rows(root, predictions_relative)
        example_ids = [str(row.get("example_id")) for row in predictions]
        _require(
            len(predictions) == 36
            and len(set(example_ids)) == 36
            and summary.get("status") == "COMPLETE"
            and summary.get("scientific_state") == "DEVELOPMENT"
            and summary.get("run_id") == record.get("run_id")
            and all(row.get("split_role") == "DEVELOPMENT" for row in predictions),
            f"comparison input {label} is not 36-row DEVELOPMENT evidence",
        )
        _require(
            record.get("summary_sha256") == file_sha256(_safe_path(root, summary_relative))
            and record.get("predictions_sha256")
            == file_sha256(_safe_path(root, predictions_relative)),
            f"comparison input {label} hash mismatch: {relative}",
        )
        if aligned_ids is None:
            aligned_ids = example_ids
        _require(example_ids == aligned_ids, f"comparison IDs are not aligned: {relative}")
        bindings[str(label)] = {
            "run_id": record.get("run_id"),
            "summary": _binding(root, summary_relative),
            "predictions": _binding(root, predictions_relative),
        }
    comparisons = comparison.get("comparisons")
    if not isinstance(comparisons, dict) or not comparisons:
        raise CheckpointIntegrityError(f"paired statistics are missing: {relative}")
    for label, result in comparisons.items():
        binary = result.get("binary_pass") if isinstance(result, dict) else None
        continuous = result.get("continuous_score") if isinstance(result, dict) else None
        _require(
            isinstance(binary, dict)
            and binary.get("n") == 36
            and binary.get("seed_count") == 1
            and isinstance(continuous, dict)
            and continuous.get("analysis_unit_count") == 36
            and continuous.get("observation_count") == 36
            and continuous.get("seed_count") == 1,
            f"comparison {label} has invalid statistical units: {relative}",
        )
    return {
        "aggregate": _binding(root, relative),
        "aligned_development_ids": len(aligned_ids or []),
        "inputs": bindings,
    }


def _validate_cp50_run(
    root: Path,
    experiment_id: str,
    run_id: str,
    evaluation_id: str,
    selected_reward: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    qualified = _validate_cp40_run(root, experiment_id, run_id)
    run = qualified["run"]
    identity = qualified["identity"]
    aggregate_relative = _CP50_AGGREGATES[experiment_id]
    _, aggregate = _json_object(root, aggregate_relative)
    expected_sampler = {"E16": "uniform", "E18": "rcac", "E20": "rcac_no_difficulty"}[experiment_id]
    reward_config_hash = file_sha256(_safe_path(root, "configs/reward.yaml"))
    sampler_config_hash = file_sha256(_safe_path(root, "configs/rcac.yaml"))
    _require(
        run.get("reward_architecture") == "terminal"
        and identity.get("reward_architecture") == "terminal"
        and aggregate.get("reward_architecture") == "terminal"
        and aggregate.get("sampler_mode") == expected_sampler,
        f"{experiment_id} frozen method identity mismatch",
    )
    _require(
        identity.get("reward_config_hash") == reward_config_hash
        and identity.get("sampler_config_hash") == sampler_config_hash,
        f"{experiment_id} frozen reward/sampler config hash mismatch",
    )
    if experiment_id in {"E18", "E20"}:
        _require(
            run.get("selected_reward_identity") == selected_reward.get("selection_identity")
            and identity.get("selected_reward_identity")
            == selected_reward.get("selection_identity")
            and run.get("selected_reward_sha256")
            == file_sha256(_safe_path(root, "evidence/training/E16/SELECTED_REWARD.json")),
            f"{experiment_id} selected reward binding mismatch",
        )
        _require(
            aggregate.get("selected_reward_identity") == selected_reward.get("selection_identity"),
            f"{experiment_id} aggregate selected reward identity mismatch",
        )
    else:
        reward_selection = aggregate.get("reward_selection")
        _require(
            isinstance(reward_selection, dict)
            and reward_selection.get("selected_reward_architecture") == "terminal"
            and reward_selection.get("selection_identity")
            == selected_reward.get("selection_identity")
            and reward_selection.get("sha256")
            == file_sha256(_safe_path(root, "evidence/training/E16/SELECTED_REWARD.json")),
            "E16 aggregate reward selection binding mismatch",
        )
    _require(
        aggregate.get("artifact_status") == "COMPLETE"
        and aggregate.get("experiment_status") == "COMPLETE"
        and aggregate.get("experiment_id") == experiment_id
        and aggregate.get("run_id") == run_id
        and aggregate.get("scientific_state") == "CANDIDATE_SELECTION"
        and aggregate.get("protocol_id") == "RFPT-RL-DEVELOPMENT-v2"
        and aggregate.get("protected_data_accessed") is False,
        f"{experiment_id} candidate aggregate identity/status mismatch",
    )
    _require(
        aggregate.get("run_complete_sha256") == qualified["run_binding"]["sha256"]
        and aggregate.get("run_identity_sha256") == qualified["identity_binding"]["sha256"]
        and aggregate.get("final_adapter_identity") == run.get("final_adapter_identity"),
        f"{experiment_id} candidate aggregate hashes mismatch",
    )
    checkpoint = aggregate.get("checkpoint")
    _require(
        isinstance(checkpoint, dict)
        and checkpoint.get("step") == 32
        and checkpoint.get("status") == "VERIFIED"
        and checkpoint.get("complete_manifest_sha256")
        == qualified["checkpoint"]["completion_manifest"]["sha256"],
        f"{experiment_id} candidate checkpoint binding mismatch",
    )
    evaluation = _validate_cp40_evaluation(
        root,
        experiment_id,
        evaluation_id,
        run,
        aggregate,
        diagnostics_name="model_diagnostics.json",
    )
    comparisons = [
        _validate_candidate_comparison(root, relative, evaluation_id)
        for relative in _CP50_COMPARISONS[experiment_id]
    ]
    aggregate_comparisons: list[dict[str, Any]] = []
    if experiment_id in {"E16", "E18"}:
        record = aggregate.get("paired_comparison")
        if isinstance(record, dict):
            aggregate_comparisons.append(record)
    else:
        records = aggregate.get("paired_comparisons")
        if isinstance(records, dict):
            aggregate_comparisons.extend(
                value for value in records.values() if isinstance(value, dict)
            )
    for relative in _CP50_COMPARISONS[experiment_id]:
        record = next(
            (item for item in aggregate_comparisons if item.get("path") == relative), None
        )
        _require(
            isinstance(record, dict)
            and record.get("sha256") == file_sha256(_safe_path(root, relative)),
            f"{experiment_id} aggregate comparison hash mismatch: {relative}",
        )
    return run, {
        "run": qualified["run_binding"],
        "run_identity": qualified["identity_binding"],
        "aggregate": _binding(root, aggregate_relative),
        "checkpoint": qualified["checkpoint"],
        "development_evaluation": evaluation,
        "development_comparisons": comparisons,
    }


def _validate_policy_finalist_bindings(
    root: Path,
    runs: dict[str, dict[str, Any]],
    experiments: dict[str, Any],
    selected_reward: dict[str, Any],
) -> None:
    policy = _safe_path(root, "docs/FROZEN_DECISION_POLICY.md").read_text(encoding="utf-8")
    required_identities = {
        "E16 run": str(runs["E16"]["run_id"]),
        "E16 adapter": str(runs["E16"]["final_adapter_identity"]),
        "E16 checkpoint": str(experiments["E16"]["checkpoint"]["completion_manifest"]["sha256"]),
        "E18 run": str(runs["E18"]["run_id"]),
        "E18 adapter": str(runs["E18"]["final_adapter_identity"]),
        "E18 checkpoint": str(experiments["E18"]["checkpoint"]["completion_manifest"]["sha256"]),
        "starting adapter": str(runs["E16"]["starting_adapter_identity"]),
        "reward selection": str(selected_reward["selection_identity"]),
        "RL suite": str(runs["E16"]["suite_config_hash"]),
        "precision register": file_sha256(
            _safe_path(root, "evidence/policy/CP50_PILOT_VARIANCE_SUMMARY.json")
        ),
    }
    missing = [label for label, identity in required_identities.items() if identity not in policy]
    _require(not missing, f"frozen decision policy omits finalist bindings: {missing}")


def _validate_e19_equivalence(
    root: Path, e18: dict[str, Any], selected_reward: dict[str, Any]
) -> dict[str, Any]:
    relative = "evidence/training/E19/POLICY_EQUIVALENCE.json"
    _, value = _json_object(root, relative)
    identity_payload = dict(value)
    recorded_identity = identity_payload.pop("evidence_identity", None)
    _require(recorded_identity == identity_hash(identity_payload), "E19 evidence identity mismatch")
    required = {
        "status": "MEASURED_POLICY_EQUIVALENCE",
        "experiment_id": "E19",
        "source_experiment_id": "E18",
        "method": "EXACT_SCORE_DISTRIBUTION_AND_SEEDED_SEQUENCE_EQUIVALENCE",
        "disposition": "NO_REDUNDANT_CUDA_RUN_REQUIRED",
        "cuda_run_executed_for_e19": False,
        "protected_access_state": "NOT_ACCESSED",
        "protocol_id": "RFPT-RL-DEVELOPMENT-v2",
        "selected_reward_architecture": "terminal",
        "all_q_exactly_one": True,
        "score_distributions_exactly_equal_at_every_selection": True,
        "computed_seeded_sequences_equal": True,
        "computed_sequence_equals_frozen_e18_schedule": True,
        "rollout_group_count": 64,
    }
    for field, expected in required.items():
        _require(value.get(field) == expected, f"E19 equivalence {field} mismatch")
    e18_root = "artifacts/training/E18/E18-rcac-grpo-seed1701-v1"
    _require(
        value.get("source_run_complete_sha256")
        == file_sha256(_safe_path(root, f"{e18_root}/RUN_COMPLETE.json"))
        and value.get("source_schedule_sha256")
        == file_sha256(_safe_path(root, f"{e18_root}/ROLLOUT_SCHEDULE.json"))
        and value.get("source_schedule_hash") == e18.get("schedule_hash")
        and value.get("starting_adapter_identity") == e18.get("starting_adapter_identity")
        and value.get("suite_config_hash") == e18.get("suite_config_hash")
        and value.get("selected_reward_identity") == selected_reward.get("selection_identity"),
        "E19 equivalence source/frozen identity mismatch",
    )
    return {
        "evidence": _binding(root, relative),
        "policy_and_seeded_sequence_equivalent": True,
        "cuda_run_executed": False,
        "source_adapter_identity": e18.get("final_adapter_identity"),
    }


def _validate_e20_policy_qualification(
    root: Path, e16: dict[str, Any], selected_reward: dict[str, Any]
) -> dict[str, Any]:
    relative = "evidence/training/E20/POLICY_QUALIFICATION.json"
    _, value = _json_object(root, relative)
    identity_payload = dict(value)
    recorded_identity = identity_payload.pop("evidence_identity", None)
    _require(recorded_identity == identity_hash(identity_payload), "E20 policy identity mismatch")
    required = {
        "status": "QUALIFIED_DISTINCT_POLICY",
        "experiment_id": "E20",
        "method": "EXACT_NON_EQUIVALENCE_AUDIT",
        "candidate_sampler": "rcac_no_difficulty",
        "distinct_cuda_run_required": True,
        "cuda_reuse_authorized": False,
        "protected_access_state": "NOT_ACCESSED",
        "protocol_id": "RFPT-RL-DEVELOPMENT-v2",
        "selected_reward_architecture": "terminal",
        "score_distributions_exactly_equal_at_every_selection": False,
        "computed_seeded_sequences_equal": False,
        "computed_sequence_equals_frozen_e16_schedule": False,
        "rollout_group_count": 64,
    }
    for field, expected in required.items():
        _require(value.get(field) == expected, f"E20 policy qualification {field} mismatch")
    e16_root = "artifacts/training/E16/E16-terminal-uniform-grpo-seed1701-v1"
    _require(
        value.get("proposed_reuse_run_complete_sha256")
        == file_sha256(_safe_path(root, f"{e16_root}/RUN_COMPLETE.json"))
        and value.get("proposed_reuse_schedule_sha256")
        == file_sha256(_safe_path(root, f"{e16_root}/ROLLOUT_SCHEDULE.json"))
        and value.get("proposed_reuse_schedule_hash") == e16.get("schedule_hash")
        and value.get("starting_adapter_identity") == e16.get("starting_adapter_identity")
        and value.get("suite_config_hash") == e16.get("suite_config_hash")
        and value.get("selected_reward_identity") == selected_reward.get("selection_identity"),
        "E20 policy qualification source/frozen identity mismatch",
    )
    return {"evidence": _binding(root, relative), "distinct_cuda_run_required": True}


def qualify_cp50_evidence(root: Path, registry: ExperimentRegistry) -> MilestoneEvidence:
    """Qualify and bind the immutable candidate/finalist freeze without protected outcomes."""

    root = root.resolve()
    _validate_complete_registry(registry, "CP50_CANDIDATE_FREEZE", ("E16", "E18", "E19", "E20"))
    decision_policy = {
        "document": _validate_frozen_decision_policy(root),
        "precision_register": _validate_cp50_precision_register(root),
    }
    protected_protocol = _validate_cp50_protected_protocol_freeze(root)
    configurations = _validate_cp50_configs(root)
    selected_reward, selected_reward_binding = _validate_selected_reward(root)
    runs: dict[str, dict[str, Any]] = {}
    experiments: dict[str, Any] = {}
    for experiment_id, (run_id, evaluation_id) in _CP50_RUNS.items():
        run, evidence = _validate_cp50_run(
            root, experiment_id, run_id, evaluation_id, selected_reward
        )
        runs[experiment_id] = run
        experiments[experiment_id] = evidence
    common_start = runs["E16"].get("starting_adapter_identity")
    common_suite = runs["E16"].get("suite_config_hash")
    for experiment_id in ("E18", "E20"):
        _require(
            runs[experiment_id].get("starting_adapter_identity") == common_start
            and runs[experiment_id].get("suite_config_hash") == common_suite,
            f"{experiment_id} does not share the frozen finalist starting/suite identity",
        )
    protected_finalists = protected_protocol.get("finalist_models")
    if not isinstance(protected_finalists, dict):
        raise CheckpointIntegrityError("protected finalist model bindings are missing")
    for experiment_id, model_id in (
        ("E16", "e16_terminal_uniform_seed1701"),
        ("E18", "e18_rcac_seed1701"),
    ):
        protected_model = protected_finalists.get(model_id)
        if not isinstance(protected_model, dict):
            raise CheckpointIntegrityError(f"protected finalist model is missing: {model_id}")
        _require(
            protected_model.get("model_checkpoint_sha256")
            == runs[experiment_id].get("final_adapter_identity")
            and protected_model.get("adapter_path") == runs[experiment_id].get("final_adapter_path")
            and protected_model.get("checkpoint_completion_sha256")
            == experiments[experiment_id]["checkpoint"]["completion_manifest"]["sha256"],
            f"protected finalist {model_id} does not bind frozen {experiment_id}",
        )
    _validate_policy_finalist_bindings(root, runs, experiments, selected_reward)
    configurations["rl_suite_semantic_hash"] = common_suite
    experiments["E19"] = _validate_e19_equivalence(root, runs["E18"], selected_reward)
    experiments["E20"]["distinct_policy_qualification"] = _validate_e20_policy_qualification(
        root, runs["E16"], selected_reward
    )
    e20_identity = experiments["E20"]["run_identity"]
    _, e20_run_identity = _json_object(root, str(e20_identity["path"]))
    dependencies = e20_run_identity.get("dependency_artifact_sha256")
    _require(
        isinstance(dependencies, dict)
        and dependencies.get("E18")
        == file_sha256(
            _safe_path(
                root,
                "artifacts/training/E18/E18-rcac-grpo-seed1701-v1/RUN_COMPLETE.json",
            )
        )
        and dependencies.get("E19_equivalence") == experiments["E19"]["evidence"]["sha256"]
        and dependencies.get("E20_qualification")
        == experiments["E20"]["distinct_policy_qualification"]["evidence"]["sha256"],
        "E20 dependency artifact bindings mismatch",
    )
    last_runtime = "artifacts/training/E20/E20-rcac-no-difficulty-grpo-seed1701-v1/checkpoint-32"
    artifact = {
        "schema_version": 1,
        "checkpoint_id": "CP50_CANDIDATE_FREEZE",
        "protocol_id": registry.protocol_id,
        "rl_protocol_id": "RFPT-RL-DEVELOPMENT-v2",
        "protected_access_state": "ACQUIRED_REGISTERED_NOT_SCORED",
        "decision_policy": decision_policy,
        "protected_protocol": protected_protocol,
        "configurations": configurations,
        "selected_reward": {
            "architecture": "terminal",
            "selection_identity": selected_reward.get("selection_identity"),
            "evidence": selected_reward_binding,
        },
        "strongest_conventional_finalist": {
            "experiment_id": "E16",
            "run_id": _CP50_RUNS["E16"][0],
            "adapter_identity": runs["E16"].get("final_adapter_identity"),
            "checkpoint_step": 32,
        },
        "rcac_finalist": {
            "experiment_id": "E18",
            "run_id": _CP50_RUNS["E18"][0],
            "adapter_identity": runs["E18"].get("final_adapter_identity"),
            "checkpoint_step": 32,
        },
        "experiments": experiments,
        "last_complete_runtime_checkpoint": last_runtime,
        "scientific_prohibition": "NO_LOCKED_FINAL_TUNING_AFTER_VERIFIED_CP50",
    }
    model_identities = {
        "strongest_conventional_e16_adapter": str(runs["E16"]["final_adapter_identity"]),
        "strongest_conventional_e16_checkpoint_32_completion": str(
            experiments["E16"]["checkpoint"]["completion_manifest"]["sha256"]
        ),
        "rcac_e18_adapter": str(runs["E18"]["final_adapter_identity"]),
        "rcac_e18_checkpoint_32_completion": str(
            experiments["E18"]["checkpoint"]["completion_manifest"]["sha256"]
        ),
        "e19_equivalence_source_adapter": str(runs["E18"]["final_adapter_identity"]),
        "e20_no_difficulty_adapter": str(runs["E20"]["final_adapter_identity"]),
        "e20_checkpoint_32_completion": str(
            experiments["E20"]["checkpoint"]["completion_manifest"]["sha256"]
        ),
    }
    return MilestoneEvidence(artifact, model_identities, last_runtime)


def _validate_cp60_registry(registry: ExperimentRegistry) -> None:
    by_id = {item.experiment_id: item for item in registry.experiments}
    required = {"E16", "E18", "E21", "E22", "E23", "E24"}
    absent = sorted(required - set(by_id))
    if absent:
        raise CheckpointIntegrityError(f"CP60 registry omits prerequisites: {absent}")
    _require(
        by_id["E16"].status is ExperimentStatus.COMPLETE
        and by_id["E18"].status is ExperimentStatus.COMPLETE,
        "CP60 finalist experiments E16/E18 are not COMPLETE",
    )
    invalid_protected = sorted(
        experiment_id
        for experiment_id in ("E21", "E22", "E23", "E24")
        if by_id[experiment_id].status is not ExperimentStatus.PENDING
    )
    _require(
        not invalid_protected,
        f"CP60 protected experiments must remain PENDING before generation: {invalid_protected}",
    )


def _validate_cp60_cp50_checkpoint(
    root: Path, registry: ExperimentRegistry, protected: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    store = ProjectCheckpointStore(root / "evidence/checkpoints")
    manifest, checkpoint = store.verify_latest()
    _require(
        manifest.checkpoint_id == "CP50_CANDIDATE_FREEZE",
        "CP60 requires CP50_CANDIDATE_FREEZE to be the latest VERIFIED checkpoint",
    )
    _require(
        manifest.protocol_id == registry.protocol_id
        and manifest.protected_access_state == "ACQUIRED_REGISTERED_NOT_SCORED",
        "CP50 protocol or protected-access state mismatch",
    )
    relative_checkpoint = checkpoint.relative_to(root).as_posix()
    cp50_relative = f"{relative_checkpoint}/CANDIDATE_FREEZE_EVIDENCE_BINDINGS.json"
    _, cp50 = _json_object(root, cp50_relative)
    _require(
        cp50.get("checkpoint_id") == "CP50_CANDIDATE_FREEZE"
        and cp50.get("protocol_id") == registry.protocol_id
        and cp50.get("protected_access_state") == "ACQUIRED_REGISTERED_NOT_SCORED",
        "CP50 candidate-freeze evidence identity mismatch",
    )
    cp50_protected = cp50.get("protected_protocol")
    cp50_decision = cp50.get("decision_policy")
    if not isinstance(cp50_protected, dict) or not isinstance(cp50_decision, dict):
        raise CheckpointIntegrityError("CP50 omits protected-protocol or decision-policy evidence")
    for field in (
        "freeze_manifest",
        "benchmark_registry",
        "generation_config",
        "code_inventory",
        "dataset_identities",
        "finalist_plan_matrix",
        "finalist_matrix_config",
        "finalist_plans",
        "finalist_models",
    ):
        _require(
            cp50_protected.get(field) == protected.get(field),
            f"current protected protocol differs from verified CP50 binding: {field}",
        )
    decision_document = cp50_decision.get("document")
    precision_register = cp50_decision.get("precision_register")
    _require(
        isinstance(decision_document, dict)
        and isinstance(precision_register, dict)
        and decision_document == _binding(root, "docs/FROZEN_DECISION_POLICY.md")
        and precision_register
        == _binding(root, "evidence/policy/CP50_PILOT_VARIANCE_SUMMARY.json"),
        "current decision policy differs from verified CP50 binding",
    )
    expected_manifest_models = {
        "strongest_conventional_e16_adapter": (
            "e16_terminal_uniform_seed1701",
            "model_checkpoint_sha256",
        ),
        "strongest_conventional_e16_checkpoint_32_completion": (
            "e16_terminal_uniform_seed1701",
            "checkpoint_completion_sha256",
        ),
        "rcac_e18_adapter": ("e18_rcac_seed1701", "model_checkpoint_sha256"),
        "rcac_e18_checkpoint_32_completion": (
            "e18_rcac_seed1701",
            "checkpoint_completion_sha256",
        ),
    }
    protected_models = protected.get("finalist_models")
    if not isinstance(protected_models, dict):
        raise CheckpointIntegrityError("protected finalist models are missing")
    for manifest_key, (model_id, model_field) in expected_manifest_models.items():
        model = protected_models.get(model_id)
        _require(
            isinstance(model, dict)
            and manifest.model_checkpoint_identities.get(manifest_key) == model.get(model_field),
            f"verified CP50 manifest finalist identity mismatch: {manifest_key}",
        )
    return (
        {
            "latest_pointer": _binding(root, "evidence/checkpoints/LATEST.json"),
            "manifest": _binding(root, f"{relative_checkpoint}/CHECKPOINT_MANIFEST.json"),
            "artifact_hashes": _binding(root, f"{relative_checkpoint}/ARTIFACT_HASHES.json"),
            "candidate_freeze_bindings": _binding(root, cp50_relative),
            "verified_checkpoint_id": manifest.checkpoint_id,
        },
        cp50,
    )


def _validate_cp60_analysis_policy(root: Path) -> dict[str, Any]:
    decision_relative = "docs/FROZEN_DECISION_POLICY.md"
    statistics_relative = "docs/STATISTICAL_ANALYSIS_PLAN.md"
    claim_relative = "docs/CLAIM_BOUNDARY.md"
    decision_text = _safe_path(root, decision_relative).read_text("utf-8")
    statistics_text = _safe_path(root, statistics_relative).read_text("utf-8")
    claim_text = _safe_path(root, claim_relative).read_text("utf-8")
    _require(
        "STATUS: FROZEN FOR CP50_CANDIDATE_FREEZE" in decision_text
        and "RFPT-PROTECTED-v1" in decision_text
        and "Holm correction at alpha 0.05" in decision_text
        and "Parser-invalid or missing completed outputs score as failures" in decision_text
        and "all expected IDs" in decision_text
        and "exist exactly once" in decision_text
        and "before CP70" in decision_text,
        "frozen decision policy omits a CP60 analysis/output rule",
    )
    _require(
        "FROZEN FOR RFPT-PROTECTED-v1 at CP50" in statistics_text
        and "Binary paired outcomes" in statistics_text
        and "Continuous paired outcomes" in statistics_text
        and "Holm correction" in statistics_text
        and "Missing and parser-invalid completed" in statistics_text
        and "outputs are scored as failures" in statistics_text,
        "statistical analysis plan is not frozen or complete for CP60",
    )
    _require(
        "Protected evaluation passed" in claim_text
        and "before CP60 and CP70 are verified" in claim_text
        and "MEASURED" in claim_text
        and "PROJECTED" in claim_text
        and "PENDING" in claim_text,
        "claim-boundary policy omits protected-evidence restrictions",
    )
    precision_relative = "evidence/policy/CP50_PILOT_VARIANCE_SUMMARY.json"
    _, precision = _json_object(root, precision_relative)
    endpoints = precision.get("endpoints")
    if not isinstance(endpoints, dict):
        raise CheckpointIntegrityError("frozen endpoint/margin register is missing")
    expected_margins = {
        "robust_reasoning_success": ("MATERIAL_IMPROVEMENT", 0.1414508159514583, 16),
        "stateful_tool_final_state_success": (
            "MATERIAL_IMPROVEMENT",
            0.1414508159514583,
            16,
        ),
        "instruction_following": ("NONINFERIORITY", 0.05, 541),
        "safety_authorization": ("NONINFERIORITY", 0.02, 24),
        "truthfulness_mc": ("NONINFERIORITY", 0.03, 817),
        "broad_capability_mc": ("NONINFERIORITY", 0.03, 256),
    }
    margin_register: dict[str, dict[str, Any]] = {}
    for endpoint, (rule, margin, planned_pairs) in expected_margins.items():
        record = endpoints.get(endpoint)
        _require(
            isinstance(record, dict)
            and record.get("rule") == rule
            and record.get("decision_margin") == margin
            and record.get("planned_pairs") == planned_pairs,
            f"CP60 frozen endpoint policy mismatch: {endpoint}",
        )
        margin_register[endpoint] = {
            "rule": rule,
            "decision_margin": margin,
            "planned_pairs": planned_pairs,
        }
    claim_ledger_relative = "evidence/claim_ledger.json"
    _, claim_ledger = _json_object(root, claim_ledger_relative)
    claims = claim_ledger.get("claims")
    allowed_classes = {"MEASURED", "DERIVED", "PROJECTED", "RESCORED", "PENDING"}
    _require(
        claim_ledger.get("version") == 1
        and isinstance(claims, list)
        and all(
            isinstance(claim, dict)
            and claim.get("evidence_class") in allowed_classes
            and isinstance(claim.get("authorized_wording"), str)
            and isinstance(claim.get("prohibited_stronger_wording"), str)
            for claim in claims
        ),
        "machine-readable claim ledger is invalid",
    )
    return {
        "decision_policy": _binding(root, decision_relative),
        "statistical_analysis_plan": _binding(root, statistics_relative),
        "precision_register": _binding(root, precision_relative),
        "claim_boundary": _binding(root, claim_relative),
        "claim_ledger": _binding(root, claim_ledger_relative),
        "margins": margin_register,
        "multiplicity": {
            "primary_family": [
                "robust_reasoning_success",
                "stateful_tool_final_state_success",
            ],
            "guardrail_family": [
                "instruction_following",
                "safety_authorization",
                "truthfulness_mc",
                "broad_capability_mc",
            ],
            "method": "HOLM",
            "familywise_alpha": 0.05,
            "exploratory_method": "BH_FDR_WHERE_APPROPRIATE",
        },
        "missing_output_rules": {
            "parser_invalid_completed_output": "SCORE_AS_FAILURE",
            "missing_completed_output": "SCORE_AS_FAILURE",
            "interrupted_shard": "INCOMPLETE_UNTIL_RESUMED",
            "completion_requirement": "ALL_EXPECTED_IDS_EXACTLY_ONCE_BEFORE_CP70",
            "resume_rule": "GENERATE_MISSING_IDS_ONLY",
            "scorer_fix_rule": "RESCORE_IMMUTABLE_OUTPUTS_WITH_NEW_SCORER_VERSION",
        },
    }


def _validate_cp60_tool_schemas(root: Path, protected: dict[str, Any]) -> dict[str, Any]:
    from reasonforge.evaluation.protected_protocol import (
        load_benchmark_specs,
        load_generation_examples,
        load_run_plan,
    )

    registry_path = root / "configs/evaluation/protected_benchmarks.yaml"
    specs = load_benchmark_specs(registry_path)
    plans = protected.get("finalist_plans")
    if not isinstance(plans, dict):
        raise CheckpointIntegrityError("protected finalist plans are missing")
    result: dict[str, Any] = {}
    for benchmark_name, revision in (
        ("reasonforge_tool", "rf-stateful-tools-v1"),
        ("reasonforge_authorization", "rf-authorization-safety-v1"),
    ):
        spec = specs[benchmark_name]
        resolved_spec = spec.model_copy(update={"path": root / spec.path})
        examples = load_generation_examples(resolved_spec)
        schema_by_id = {
            example.example_id: sha256_json(json.loads(example.tool_schema or "null"))
            for example in examples
        }
        _require(
            all(example.tool_schema is not None for example in examples),
            f"protected tool benchmark has a missing schema: {benchmark_name}",
        )
        matching_plans = [
            load_run_plan(_safe_path(root, str(binding["path"])))
            for output_set_id, binding in plans.items()
            if f"-{benchmark_name}-" in output_set_id and isinstance(binding, dict)
        ]
        _require(
            len(matching_plans) == 2,
            f"protected tool benchmark finalist-plan coverage mismatch: {benchmark_name}",
        )
        example_identities = {
            example.example_id: sha256_json(example.identity_payload()) for example in examples
        }
        _require(
            all(plan.example_identity_sha256 == example_identities for plan in matching_plans),
            f"protected tool-schema identities differ from frozen plans: {benchmark_name}",
        )
        if benchmark_name == "reasonforge_tool":
            import pyarrow.parquet as pq

            rows = pq.read_table(resolved_spec.path, columns=["example_id", "metadata"])
            metadata_by_id = {
                str(row["example_id"]): json.loads(str(row["metadata"]))
                for row in rows.to_pylist()
                if str(row["example_id"]).startswith("locked-v2-tool-")
            }
            _require(
                len(metadata_by_id) == len(examples)
                and {value.get("tool_registry_revision") for value in metadata_by_id.values()}
                == {revision},
                "ReasonForge tool-registry revision mismatch",
            )
        result[benchmark_name] = {
            "tool_schema_revision": revision,
            "example_count": len(examples),
            "schema_inventory_sha256": sha256_json(schema_by_id),
            "unique_schema_count": len(set(schema_by_id.values())),
            "source": _binding(root, spec.path.as_posix()),
        }
    result["environment_implementation"] = _binding(
        root, "src/reasonforge/environments/stateful.py"
    )
    return result


def _validate_cp60_no_outputs(root: Path) -> dict[str, Any]:
    output_root = root / "evidence/protected_outputs"
    files = (
        sorted(
            path.relative_to(root).as_posix() for path in output_root.rglob("*") if path.is_file()
        )
        if output_root.exists()
        else []
    )
    _require(not files, f"protected outputs exist before CP60: {files[:5]}")
    return {
        "output_root": "evidence/protected_outputs",
        "files_observed": 0,
        "outputs_generated": False,
        "scores_computed": False,
    }


def qualify_cp60_evidence(root: Path, registry: ExperimentRegistry) -> MilestoneEvidence:
    """Qualify the immutable pre-protected protocol boundary without model inference."""

    root = root.resolve()
    _validate_cp60_registry(registry)
    protected = _validate_cp50_protected_protocol_freeze(root)
    cp50_binding, cp50 = _validate_cp60_cp50_checkpoint(root, registry, protected)
    analysis_policy = _validate_cp60_analysis_policy(root)
    tool_schemas = _validate_cp60_tool_schemas(root, protected)
    output_state = _validate_cp60_no_outputs(root)
    protected_models = protected["finalist_models"]
    _require(isinstance(protected_models, dict), "protected finalist model bindings are invalid")
    expected_model_common = {
        "base_model_id": "Qwen/Qwen3-1.7B",
        "base_model_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "tokenizer_id": "Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    }
    finalists: dict[str, dict[str, Any]] = {}
    for model_id in ("e16_terminal_uniform_seed1701", "e18_rcac_seed1701"):
        model = protected_models.get(model_id)
        if not isinstance(model, dict):
            raise CheckpointIntegrityError(f"protected finalist is missing: {model_id}")
        for field, expected in expected_model_common.items():
            _require(
                model.get(field) == expected, f"protected finalist {model_id} {field} mismatch"
            )
        finalists[model_id] = dict(model)
    dataset_identities = protected.get("dataset_identities")
    if not isinstance(dataset_identities, dict) or len(dataset_identities) != 7:
        raise CheckpointIntegrityError("CP60 requires exactly seven protected dataset identities")
    scorer_revisions = {
        benchmark_id: {
            "grader_id": record.get("grader_id"),
            "grader_revision": record.get("grader_revision"),
        }
        for benchmark_id, record in dataset_identities.items()
        if isinstance(record, dict)
    }
    _require(len(scorer_revisions) == 7, "CP60 scorer-revision inventory is incomplete")
    last_runtime = "artifacts/training/E20/E20-rcac-no-difficulty-grpo-seed1701-v1/checkpoint-32"
    artifact = {
        "schema_version": 1,
        "checkpoint_id": "CP60_PRE_PROTECTED",
        "protocol_id": registry.protocol_id,
        "protected_protocol_id": "RFPT-PROTECTED-v1",
        "protected_access_state": "ACQUIRED_REGISTERED_NOT_SCORED",
        "cp50_verified_binding": cp50_binding,
        "cp50_candidate_freeze_identity": {
            "strongest_conventional_finalist": cp50["strongest_conventional_finalist"],
            "rcac_finalist": cp50["rcac_finalist"],
        },
        "finalists": finalists,
        "generation_config": protected["generation_config"],
        "benchmark_registry": protected["benchmark_registry"],
        "dataset_identities": dataset_identities,
        "scorer_revisions": scorer_revisions,
        "tool_schemas": tool_schemas,
        "finalist_matrix": protected["finalist_plan_matrix"],
        "finalist_matrix_config": protected["finalist_matrix_config"],
        "finalist_plan_count": protected["finalist_plan_count"],
        "finalist_plans": protected["finalist_plans"],
        "protected_code_inventory": protected["code_inventory"],
        "analysis_policy": analysis_policy,
        "protected_output_state": output_state,
        "last_complete_runtime_checkpoint": last_runtime,
        "next_authorized_action": "GENERATE_FROZEN_E21_E24_PLANS_ONLY_AFTER_VERIFIED_CP60",
        "scientific_prohibitions": [
            "NO_LOCKED_FINAL_TUNING",
            "NO_PLAN_OR_IDENTITY_MUTATION_UNDER_RFPT-PROTECTED-v1",
            "NO_SCORING_IN_GENERATION_PROCESS",
            "NO_REGENERATION_FOR_ANALYSIS_CONVENIENCE",
        ],
    }
    model_identities = {
        "strongest_conventional_e16_adapter": str(
            finalists["e16_terminal_uniform_seed1701"]["model_checkpoint_sha256"]
        ),
        "strongest_conventional_e16_checkpoint_32_completion": str(
            finalists["e16_terminal_uniform_seed1701"]["checkpoint_completion_sha256"]
        ),
        "rcac_e18_adapter": str(finalists["e18_rcac_seed1701"]["model_checkpoint_sha256"]),
        "rcac_e18_checkpoint_32_completion": str(
            finalists["e18_rcac_seed1701"]["checkpoint_completion_sha256"]
        ),
        "protected_run_plan_matrix": str(protected["finalist_plan_matrix"]["sha256"]),
        "protected_generation_config": str(protected["generation_config"]["sha256"]),
    }
    return MilestoneEvidence(artifact, model_identities, last_runtime)


def _validate_cp70_registry(registry: ExperimentRegistry) -> None:
    by_id = {item.experiment_id: item for item in registry.experiments}
    required = {"E21", "E22", "E23", "E24"}
    absent = sorted(required - set(by_id))
    if absent:
        raise CheckpointIntegrityError(f"CP70 registry omits protected experiments: {absent}")
    incomplete = sorted(
        experiment_id
        for experiment_id in required
        if by_id[experiment_id].status is not ExperimentStatus.COMPLETE
    )
    _require(
        not incomplete,
        f"CP70 protected experiments are not COMPLETE: {incomplete}",
    )


def _validate_cp70_cp60_checkpoint(
    root: Path, registry: ExperimentRegistry, protected: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    store = ProjectCheckpointStore(root / "evidence/checkpoints")
    manifest, checkpoint = store.verify_latest()
    _require(
        manifest.checkpoint_id == "CP60_PRE_PROTECTED",
        "CP70 requires CP60_PRE_PROTECTED to be the latest VERIFIED checkpoint",
    )
    _require(
        manifest.protocol_id == registry.protocol_id
        and manifest.protected_access_state == "ACQUIRED_REGISTERED_NOT_SCORED",
        "CP60 protocol or protected-access state mismatch",
    )
    relative_checkpoint = checkpoint.relative_to(root).as_posix()
    cp60_relative = f"{relative_checkpoint}/PRE_PROTECTED_EVIDENCE_BINDINGS.json"
    _, cp60 = _json_object(root, cp60_relative)
    _require(
        cp60.get("checkpoint_id") == "CP60_PRE_PROTECTED"
        and cp60.get("protocol_id") == registry.protocol_id
        and cp60.get("protected_protocol_id") == "RFPT-PROTECTED-v1"
        and cp60.get("protected_access_state") == "ACQUIRED_REGISTERED_NOT_SCORED",
        "CP60 pre-protected evidence identity mismatch",
    )
    _require(
        cp60.get("finalist_plan_count") == 14
        and cp60.get("finalist_plans") == protected.get("finalist_plans")
        and cp60.get("finalist_matrix") == protected.get("finalist_plan_matrix")
        and cp60.get("finalist_matrix_config") == protected.get("finalist_matrix_config")
        and cp60.get("generation_config") == protected.get("generation_config")
        and cp60.get("dataset_identities") == protected.get("dataset_identities")
        and cp60.get("protected_code_inventory") == protected.get("code_inventory"),
        "current protected protocol differs from verified CP60 binding",
    )
    cp60_finalists = cp60.get("finalists")
    if not isinstance(cp60_finalists, dict):
        raise CheckpointIntegrityError("CP60 finalist identity register is missing")
    e16 = cp60_finalists.get("e16_terminal_uniform_seed1701")
    e18 = cp60_finalists.get("e18_rcac_seed1701")
    if not isinstance(e16, dict) or not isinstance(e18, dict):
        raise CheckpointIntegrityError("CP60 finalist model bindings are incomplete")
    expected_manifest_identities = {
        "strongest_conventional_e16_adapter": e16.get("model_checkpoint_sha256"),
        "rcac_e18_adapter": e18.get("model_checkpoint_sha256"),
        "protected_run_plan_matrix": protected["finalist_plan_matrix"]["sha256"],
        "protected_generation_config": protected["generation_config"]["sha256"],
    }
    _require(
        all(
            manifest.model_checkpoint_identities.get(key) == expected
            for key, expected in expected_manifest_identities.items()
        ),
        "verified CP60 manifest model/protocol identity mismatch",
    )
    output_state = cp60.get("protected_output_state")
    _require(
        isinstance(output_state, dict)
        and output_state.get("files_observed") == 0
        and output_state.get("outputs_generated") is False
        and output_state.get("scores_computed") is False,
        "CP60 does not attest a clean pre-protected boundary",
    )
    return (
        {
            "latest_pointer": _binding(root, "evidence/checkpoints/LATEST.json"),
            "manifest": _binding(root, f"{relative_checkpoint}/CHECKPOINT_MANIFEST.json"),
            "artifact_hashes": _binding(root, f"{relative_checkpoint}/ARTIFACT_HASHES.json"),
            "pre_protected_bindings": _binding(root, cp60_relative),
            "verified_checkpoint_id": manifest.checkpoint_id,
        },
        cp60,
    )


def _validate_cp70_recovery_amendment(
    root: Path, cp60_binding: dict[str, Any]
) -> tuple[dict[str, Any], Any]:
    from reasonforge.evaluation.protected_orchestration_v2 import (
        verify_recovery_amendment,
    )
    from reasonforge.evaluation.protected_outputs import ProtectedOutputIntegrityError
    from reasonforge.evaluation.protected_recovery_v2 import load_recovery_config

    amendment_relative = "evidence/protected_recovery/RFPT-PROTECTED-RECOVERY-v2/AMENDMENT.json"
    amendment_path = _safe_path(root, amendment_relative)
    try:
        amendment = verify_recovery_amendment(amendment_path)
    except (OSError, ValueError, ProtectedOutputIntegrityError) as error:
        raise CheckpointIntegrityError(
            f"protected recovery amendment is invalid: {error}"
        ) from error
    required = {
        "schema_version": 1,
        "status": "VERIFIED_RECOVERY_AMENDMENT",
        "recovery_protocol_id": "RFPT-PROTECTED-RECOVERY-v2",
        "parent_protocol_id": "RFPT-PROTECTED-v1",
        "cp60_checkpoint_id": "CP60_PRE_PROTECTED",
        "generation_semantics_changed": False,
        "regeneration_authorized": False,
        "immutable_rescore_only": True,
        "scores_computed": False,
        "semantic_outputs_inspected": False,
        "cuda_used": False,
    }
    for field, expected in required.items():
        _require(amendment.get(field) == expected, f"recovery amendment {field} mismatch")
    expected_defects = {
        "MMLU_FIRST_ISOLATED_LETTER_AMBIGUITY",
        "SAFETY_MALFORMED_TOOL_MARKUP_FALSE_PASS",
        "IFEVAL_LOOSE_TRANSFORM_DIVERGENCE",
        "SHARD_MANIFEST_PUBLICATION_CRASH_WINDOW",
        "COMPLETION_SENTINEL_PUBLICATION_CRASH_WINDOW",
        "FINISH_REASON_LEGACY_DEFAULT_NOT_EVIDENTIARY",
    }
    _require(
        set(amendment.get("defects", [])) == expected_defects,
        "recovery amendment defect inventory mismatch",
    )
    cp60_manifest = amendment.get("cp60_manifest")
    cp60_bindings = amendment.get("cp60_bindings")
    _require(
        isinstance(cp60_manifest, dict)
        and isinstance(cp60_bindings, dict)
        and cp60_manifest.get("sha256") == cp60_binding["manifest"]["sha256"]
        and cp60_bindings.get("sha256") == cp60_binding["pre_protected_bindings"]["sha256"],
        "recovery amendment does not bind the verified CP60 artifacts",
    )
    recovery_config = amendment.get("recovery_config")
    code_inventory = amendment.get("recovery_code_inventory")
    if not isinstance(recovery_config, dict) or not isinstance(code_inventory, dict):
        raise CheckpointIntegrityError("recovery configuration/code inventory is missing")
    config_relative = recovery_config.get("path")
    _require(
        config_relative == "configs/evaluation/protected_recovery_v2.yaml"
        and recovery_config.get("sha256") == file_sha256(_safe_path(root, str(config_relative))),
        "recovery configuration identity mismatch",
    )
    expected_code = {
        "scripts/adopt_protected_outputs_v2.py",
        "scripts/publish_protected_recovery_amendment.py",
        "scripts/run_protected_generation_v2.py",
        "scripts/score_protected_outputs_v2.py",
        "src/reasonforge/evaluation/protected_orchestration_v2.py",
        "src/reasonforge/evaluation/protected_outputs_v2.py",
        "src/reasonforge/evaluation/protected_recovery_v2.py",
    }
    _require(set(code_inventory) == expected_code, "recovery code path inventory mismatch")
    for path, digest in code_inventory.items():
        _require(
            digest == file_sha256(_safe_path(root, path)),
            f"recovery code hash mismatch: {path}",
        )
    config = load_recovery_config(_safe_path(root, str(config_relative)))
    expected_corrections = {
        "google-ifeval": (
            "google.ifeval.strict-loose",
            "5594ac0ee7a13c77eb1e0b98f0f305a2ca65b6c4",
            "5594ac0ee7a13c77eb1e0b98f0f305a2ca65b6c4+rf-wrapper-v2",
        ),
        "tiger-lab-mmlu-pro-test-256": (
            "reasonforge.mmlu-pro.exact-choice",
            "1.0.0",
            "1.1.0",
        ),
        "reasonforge-authorization-safety": (
            "reasonforge.authorization",
            "1.0.0",
            "1.1.0",
        ),
    }
    _require(set(config.scorers) == set(expected_corrections), "recovery scorer set mismatch")
    for benchmark_id, expected in expected_corrections.items():
        correction = config.scorers[benchmark_id]
        _require(
            (correction.grader_id, correction.old_revision, correction.new_revision) == expected,
            f"recovery scorer identity mismatch: {benchmark_id}",
        )
    _require(
        config.writer.old_protocol_id == "RFPT-PROTECTED-v1"
        and config.writer.new_protocol_id == "RFPT-PROTECTED-WRITER-v2"
        and config.writer.write_ahead_append_journal
        and config.writer.write_ahead_completion_journal
        and config.writer.adopts_valid_v1_outputs_without_record_mutation
        and config.finish_reason.existing_record_semantics
        == "LEGACY_FIELD_UNVERIFIED_DO_NOT_USE_FOR_CLAIMS",
        "recovery writer/adoption contract mismatch",
    )
    observed_outputs = amendment.get("observed_outputs")
    if not isinstance(observed_outputs, dict) or len(observed_outputs) != 9:
        raise CheckpointIntegrityError("recovery amendment observed-output inventory mismatch")
    for output_set_id, record in observed_outputs.items():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise CheckpointIntegrityError("recovery amendment output record is invalid")
        output_directory = _safe_path(root, record["path"])
        _require(
            record.get("status") == "COMPLETE"
            and record.get("missing_id_count") == 0
            and record.get("finish_reason_semantics")
            == "LEGACY_FIELD_UNVERIFIED_DO_NOT_USE_FOR_CLAIMS"
            and record.get("manifest_sha256") == file_sha256(output_directory / "manifest.json")
            and record.get("run_plan_sha256") == file_sha256(output_directory / "RUN_PLAN.json"),
            f"recovery amendment observed output changed: {output_set_id}",
        )
        shard_inventory = record.get("shard_sha256")
        _require(
            isinstance(shard_inventory, dict)
            and all(
                digest == file_sha256(output_directory / shard_name)
                for shard_name, digest in shard_inventory.items()
            ),
            f"recovery amendment observed shard changed: {output_set_id}",
        )
    recovery_document_relative = "docs/PROTECTED_RECOVERY_V2.md"
    recovery_document = _safe_path(root, recovery_document_relative).read_text("utf-8")
    _require(
        "additive amendment" in recovery_document
        and "RFPT-PROTECTED-v1" in recovery_document
        and "Regeneration is not" in recovery_document
        and "authorized." in recovery_document
        and "LEGACY_FIELD_UNVERIFIED_DO_NOT_USE_FOR_CLAIMS" in recovery_document
        and "Continue using v1 scorers for unaffected" in recovery_document,
        "protected recovery-v2 documentation omits a required scientific boundary",
    )
    return (
        {
            "protocol_id": config.protocol_id,
            "parent_protocol_id": config.parent_protocol_id,
            "amendment": _binding(root, amendment_relative),
            "documentation": _binding(root, recovery_document_relative),
            "configuration": _binding(root, str(config_relative)),
            "code_inventory": {path: _binding(root, path) for path in sorted(code_inventory)},
            "corrected_scorers": {
                benchmark_id: {
                    "grader_id": correction.grader_id,
                    "old_revision": correction.old_revision,
                    "new_revision": correction.new_revision,
                    "correction": correction.correction,
                }
                for benchmark_id, correction in sorted(config.scorers.items())
            },
            "writer_protocol_id": config.writer.new_protocol_id,
            "finish_reason_semantics": config.finish_reason.existing_record_semantics,
            "generation_semantics_changed": False,
            "regeneration_authorized": False,
            "amendment_observed_output_count": len(observed_outputs),
        },
        config,
    )


def _validate_cp70_scoring_authorization(
    root: Path,
    cp60_binding: dict[str, Any],
    protected: dict[str, Any],
    recovery: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    """Bind the sole authorized whole-matrix protected scoring pass."""

    from reasonforge.evaluation.protected_outputs import ProtectedOutputIntegrityError
    from reasonforge.evaluation.protected_scoring_orchestrator import (
        load_scoring_config,
        verify_scoring_authorization,
    )

    authorization_relative = (
        "evidence/protected_scoring/RFPT-PROTECTED-SCORING-ORCHESTRATOR-v1/AUTHORIZATION.json"
    )
    authorization_path = _safe_path(root, authorization_relative)
    expected_authorization_sha256 = (
        "b0fbd365de949015347d4fef8dde4e60b275b4b32133aa9615b82f9515f3c0df"
    )
    _require(
        file_sha256(authorization_path) == expected_authorization_sha256,
        "protected scoring authorization identity mismatch",
    )
    try:
        authorization = verify_scoring_authorization(authorization_path)
    except (OSError, ValueError, ProtectedOutputIntegrityError) as error:
        raise CheckpointIntegrityError(
            f"protected scoring authorization is invalid: {error}"
        ) from error

    expected_routes = {
        "ifeval": "recovery_v2",
        "mmlu_pro_test_256": "recovery_v2",
        "reasonforge_authorization": "recovery_v2",
        "reasonforge_instruction": "frozen_v1",
        "reasonforge_reasoning": "frozen_v1",
        "reasonforge_tool": "frozen_v1",
        "truthfulqa_mc": "frozen_v1",
    }
    required = {
        "schema_version": 1,
        "status": "VERIFIED_SCORING_AUTHORIZATION",
        "authorization_id": "RFPT-PROTECTED-SCORING-ORCHESTRATOR-v1",
        "parent_protocol_id": "RFPT-PROTECTED-v1",
        "recovery_protocol_id": "RFPT-PROTECTED-RECOVERY-v2",
        "cp60_checkpoint_id": "CP60_PRE_PROTECTED",
        "expected_output_sets": 14,
        "routes": expected_routes,
        "all_complete_before_any_score": True,
        "exactly_one_rescore_per_set": True,
        "refuse_existing_rescore_state": True,
        "semantic_output_logging": False,
        "score_values_logged": False,
        "generation_semantics_changed": False,
        "registry_mutated": False,
        "cuda_used": False,
        "scores_computed": False,
    }
    for field, expected in required.items():
        _require(
            authorization.get(field) == expected,
            f"protected scoring authorization {field} mismatch",
        )

    expected_bound_paths = {
        "scoring_config": "configs/evaluation/protected_scoring_orchestrator_v1.yaml",
        "recovery_config": str(recovery["configuration"]["path"]),
        "recovery_amendment": str(recovery["amendment"]["path"]),
        "frozen_matrix": str(protected["finalist_plan_matrix"]["path"]),
        "benchmark_registry": str(protected["benchmark_registry"]["path"]),
        "cp60_manifest": str(cp60_binding["manifest"]["path"]),
        "cp60_bindings": str(cp60_binding["pre_protected_bindings"]["path"]),
    }
    expected_bound_hashes = {
        "recovery_config": recovery["configuration"]["sha256"],
        "recovery_amendment": recovery["amendment"]["sha256"],
        "frozen_matrix": protected["finalist_plan_matrix"]["sha256"],
        "benchmark_registry": protected["benchmark_registry"]["sha256"],
        "cp60_manifest": cp60_binding["manifest"]["sha256"],
        "cp60_bindings": cp60_binding["pre_protected_bindings"]["sha256"],
    }
    for label, expected_relative in expected_bound_paths.items():
        value = authorization.get(label)
        if not isinstance(value, dict):
            raise CheckpointIntegrityError(
                f"protected scoring authorization {label} binding is missing"
            )
        raw_path = value.get("path")
        _require(
            isinstance(raw_path, str)
            and Path(raw_path).resolve() == _safe_path(root, expected_relative).resolve()
            and value.get("sha256") == file_sha256(_safe_path(root, expected_relative))
            and (
                label not in expected_bound_hashes
                or value.get("sha256") == expected_bound_hashes[label]
            ),
            f"protected scoring authorization {label} binding mismatch",
        )

    expected_inventories = {
        "code_inventory": {
            "scripts/publish_protected_scoring_authorization.py",
            "scripts/score_protected_matrix.py",
            "src/reasonforge/evaluation/protected_scoring_orchestrator.py",
        },
        "test_inventory": {"tests/test_protected_scoring_orchestrator.py"},
        "documentation_inventory": {"docs/PROTECTED_SCORING_ORCHESTRATOR.md"},
    }
    inventory_bindings: dict[str, dict[str, dict[str, str | int]]] = {}
    for inventory_name, expected_paths in expected_inventories.items():
        inventory = authorization.get(inventory_name)
        if not isinstance(inventory, dict) or set(inventory) != expected_paths:
            raise CheckpointIntegrityError(
                f"protected scoring authorization {inventory_name} path mismatch"
            )
        inventory_bindings[inventory_name] = {}
        for path in sorted(expected_paths):
            _require(
                inventory[path] == file_sha256(_safe_path(root, path)),
                f"protected scoring authorization {inventory_name} hash mismatch: {path}",
            )
            inventory_bindings[inventory_name][path] = _binding(root, path)

    config_relative = expected_bound_paths["scoring_config"]
    config = load_scoring_config(_safe_path(root, config_relative))
    _require(
        config.protocol_id == authorization["authorization_id"]
        and config.parent_protocol_id == authorization["parent_protocol_id"]
        and config.recovery_protocol_id == authorization["recovery_protocol_id"]
        and config.expected_output_sets == authorization["expected_output_sets"]
        and config.routes == expected_routes
        and config.require_complete
        and config.require_writer_v2_adoption
        and config.refuse_any_existing_rescore_state
        and not config.semantic_output_logging,
        "protected scoring orchestrator configuration mismatch",
    )
    return (
        {
            "authorization_id": authorization["authorization_id"],
            "authorization": _binding(root, authorization_relative),
            "configuration": _binding(root, config_relative),
            "routes": dict(sorted(expected_routes.items())),
            "authorized_family_route_counts": {"frozen_v1": 4, "recovery_v2": 3},
            "expected_output_sets": 14,
            "rescore_id_suffix": config.rescore_id_suffix,
            "cp60_manifest_sha256": cp60_binding["manifest"]["sha256"],
            "cp60_bindings_sha256": cp60_binding["pre_protected_bindings"]["sha256"],
            "recovery_amendment_sha256": recovery["amendment"]["sha256"],
            "recovery_config_sha256": recovery["configuration"]["sha256"],
            "frozen_matrix_sha256": protected["finalist_plan_matrix"]["sha256"],
            "benchmark_registry_sha256": protected["benchmark_registry"]["sha256"],
            **inventory_bindings,
            "all_complete_before_any_score": True,
            "exactly_one_rescore_per_set": True,
            "refuse_existing_rescore_state": True,
            "scores_computed_at_authorization": False,
            "semantic_outputs_logged": False,
            "score_values_logged": False,
        },
        config,
    )


def _validate_cp70_scoring_recovery_v3(
    root: Path,
    cp60_binding: dict[str, Any],
    protected: dict[str, Any],
    recovery: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    """Bind the final operational scoring recovery without changing science."""

    from reasonforge.evaluation.protected_outputs import ProtectedOutputIntegrityError
    from reasonforge.evaluation.protected_scoring_orchestrator import load_scoring_config
    from reasonforge.evaluation.protected_scoring_recovery_v3 import (
        verify_recovery_v3_authorization,
    )

    authorization_relative = (
        "evidence/protected_scoring/RFPT-PROTECTED-SCORING-RECOVERY-v3/AUTHORIZATION.json"
    )
    authorization_path = _safe_path(root, authorization_relative)
    authorization_sha256 = "acb28312cb3b5aced2e64ecb0df4bb7359664c00c84d6fef2a9d57b5e83c6444"
    _require(
        file_sha256(authorization_path) == authorization_sha256,
        "protected scoring recovery-v3 authorization identity mismatch",
    )
    try:
        authorization = verify_recovery_v3_authorization(authorization_path)
    except (OSError, ValueError, ProtectedOutputIntegrityError) as error:
        raise CheckpointIntegrityError(
            f"protected scoring recovery-v3 authorization is invalid: {error}"
        ) from error

    expected_policy = {
        "schema_version": 1,
        "status": "VERIFIED_SCORING_RECOVERY_AUTHORIZATION",
        "authorization_id": "RFPT-PROTECTED-SCORING-RECOVERY-v3",
        "failure": "WINDOWS_PATH_LENGTH_BEFORE_FIRST_RESCORE_PUBLICATION",
        "published_rescore_count_before_failure": 0,
        "short_rescore_id": "rfpt-sv3",
        "suppress_scorer_logging": True,
        "observed_log_disclosure": "ASTERISK_ONLY_RESPONSE_FROM_LANGUAGE_DETECTOR_ERROR",
        "generation_semantics_changed": False,
        "scoring_routes_changed": False,
        "semantic_outputs_inspected": False,
        "scores_computed": False,
    }
    for field, expected in expected_policy.items():
        _require(
            authorization.get(field) == expected,
            f"protected scoring recovery-v3 {field} mismatch",
        )

    expected_paths = {
        "recovery_config": "configs/evaluation/protected_scoring_recovery_v3.yaml",
        "parent_authorization_v2": (
            "evidence/protected_scoring/RFPT-PROTECTED-SCORING-AUTHORIZATION-v2/AUTHORIZATION.json"
        ),
        "backend_config": "configs/evaluation/ifeval_backend_v1.yaml",
        "backend_qualification": "evidence/protected_scoring/IFEVAL_BACKEND_QUALIFICATION.json",
        "requirements_lock": "requirements-eval-lock.txt",
        "scoring_config": "configs/evaluation/protected_scoring_orchestrator_v1.yaml",
        "recovery_scorer_config": str(recovery["configuration"]["path"]),
        "recovery_amendment": str(recovery["amendment"]["path"]),
        "frozen_matrix": str(protected["finalist_plan_matrix"]["path"]),
        "benchmark_registry": str(protected["benchmark_registry"]["path"]),
        "cp60_manifest": str(cp60_binding["manifest"]["path"]),
        "cp60_bindings": str(cp60_binding["pre_protected_bindings"]["path"]),
        "failed_lock": "evidence/protected_scoring/SCORING_IN_PROGRESS.json",
    }
    expected_hashes = {
        "backend_qualification": (
            "1c72d2a56af1ae0d372e7857602d8b906fdcc808c19766ab13f446139f197827"
        ),
        "requirements_lock": ("434e763fad92935a846c66c64d3d8ddb3a9200196a81c94dcb847dd0256108f7"),
        "recovery_scorer_config": recovery["configuration"]["sha256"],
        "recovery_amendment": recovery["amendment"]["sha256"],
        "frozen_matrix": protected["finalist_plan_matrix"]["sha256"],
        "benchmark_registry": protected["benchmark_registry"]["sha256"],
        "cp60_manifest": cp60_binding["manifest"]["sha256"],
        "cp60_bindings": cp60_binding["pre_protected_bindings"]["sha256"],
        "failed_lock": ("8ad41f79858f3088a7ba25b1de266e627441bfdaff477a003355ea2133fe71f7"),
    }
    path_bindings: dict[str, dict[str, str | int]] = {}
    for label, expected_relative in expected_paths.items():
        value = authorization.get(label)
        if not isinstance(value, dict):
            raise CheckpointIntegrityError(f"recovery-v3 {label} binding is missing")
        raw_path = value.get("path")
        expected_path = _safe_path(root, expected_relative)
        _require(
            isinstance(raw_path, str)
            and Path(raw_path).resolve() == expected_path.resolve()
            and value.get("sha256") == file_sha256(expected_path)
            and (label not in expected_hashes or value.get("sha256") == expected_hashes[label]),
            f"protected scoring recovery-v3 {label} binding mismatch",
        )
        path_bindings[label] = _binding(root, expected_relative)

    expected_inventories = {
        "code_inventory": {
            "scripts/publish_protected_scoring_recovery_v3.py",
            "scripts/score_protected_matrix_recovery_v3.py",
            "src/reasonforge/evaluation/ifeval_backend.py",
            "src/reasonforge/evaluation/protected_scoring_orchestrator.py",
            "src/reasonforge/evaluation/protected_scoring_recovery_v3.py",
        },
        "test_inventory": {"tests/test_protected_scoring_recovery_v3.py"},
        "documentation_inventory": {"docs/PROTECTED_SCORING_RECOVERY_V3.md"},
    }
    inventory_bindings: dict[str, dict[str, dict[str, str | int]]] = {}
    for inventory_name, expected_inventory in expected_inventories.items():
        inventory = authorization.get(inventory_name)
        if not isinstance(inventory, dict) or set(inventory) != expected_inventory:
            raise CheckpointIntegrityError(f"recovery-v3 {inventory_name} path mismatch")
        inventory_bindings[inventory_name] = {}
        for relative in sorted(expected_inventory):
            _require(
                inventory[relative] == file_sha256(_safe_path(root, relative)),
                f"recovery-v3 {inventory_name} hash mismatch: {relative}",
            )
            inventory_bindings[inventory_name][relative] = _binding(root, relative)

    _, recovery_config = _yaml_mapping(root, expected_paths["recovery_config"])
    allowed_failed_directory = (
        "evidence/protected_outputs/e16_terminal_uniform_seed1701/"
        "e21-e16_terminal_uniform_seed1701-ifeval-v1/rescores"
    )
    recovery_lock = "evidence/protected_scoring/SCORING_RECOVERY_V3_IN_PROGRESS.json"
    _require(
        recovery_config.get("protocol_id") == authorization["authorization_id"]
        and recovery_config.get("parent_authorization_id")
        == "RFPT-PROTECTED-SCORING-AUTHORIZATION-v2"
        and recovery_config.get("failure") == authorization["failure"]
        and recovery_config.get("failed_lock") == expected_paths["failed_lock"]
        and recovery_config.get("failed_lock_sha256") == expected_hashes["failed_lock"]
        and recovery_config.get("allowed_empty_rescore_directory") == allowed_failed_directory
        and recovery_config.get("published_rescore_count_before_failure") == 0
        and recovery_config.get("short_rescore_id") == "rfpt-sv3"
        and recovery_config.get("recovery_lock") == recovery_lock
        and recovery_config.get("suppress_scorer_logging") is True
        and recovery_config.get("observed_log_disclosure")
        == "ASTERISK_ONLY_RESPONSE_FROM_LANGUAGE_DETECTOR_ERROR"
        and recovery_config.get("generation_semantics_changed") is False
        and recovery_config.get("scoring_routes_changed") is False
        and recovery_config.get("semantic_outputs_inspected") is False
        and recovery_config.get("scores_computed") is False,
        "protected scoring recovery-v3 configuration mismatch",
    )

    _, parent_v2 = _json_object(root, expected_paths["parent_authorization_v2"])
    _require(
        parent_v2.get("authorization_id") == "RFPT-PROTECTED-SCORING-AUTHORIZATION-v2"
        and parent_v2.get("status") == "VERIFIED_SCORING_AUTHORIZATION"
        and parent_v2.get("scores_computed") is False
        and parent_v2.get("semantic_outputs_accessed") is False
        and parent_v2.get("generation_semantics_changed") is False
        and parent_v2.get("scoring_routes_changed") is False,
        "protected scoring recovery-v3 parent authorization mismatch",
    )
    _, backend = _json_object(root, expected_paths["backend_qualification"])
    _require(
        backend.get("status") == "QUALIFIED"
        and backend.get("backend_id") == "RFPT-IFEVAL-BACKEND-v1"
        and backend.get("source_commit") == "5594ac0ee7a13c77eb1e0b98f0f305a2ca65b6c4"
        and backend.get("instruction_count") == 25
        and backend.get("langdetect_seed") == 0
        and backend.get("requirements_lock_sha256") == expected_hashes["requirements_lock"]
        and backend.get("semantic_outputs_accessed") is False
        and backend.get("scores_computed") is False,
        "protected scoring recovery-v3 backend qualification mismatch",
    )

    scoring_config = load_scoring_config(_safe_path(root, expected_paths["scoring_config"]))
    expected_routes = {
        "ifeval": "recovery_v2",
        "mmlu_pro_test_256": "recovery_v2",
        "reasonforge_authorization": "recovery_v2",
        "reasonforge_instruction": "frozen_v1",
        "reasonforge_reasoning": "frozen_v1",
        "reasonforge_tool": "frozen_v1",
        "truthfulqa_mc": "frozen_v1",
    }
    _require(
        scoring_config.routes == expected_routes
        and scoring_config.expected_output_sets == 14
        and scoring_config.require_complete
        and scoring_config.require_writer_v2_adoption
        and not scoring_config.semantic_output_logging,
        "protected scoring recovery-v3 changed scorer routes or boundaries",
    )
    recovery_document = _safe_path(root, "docs/PROTECTED_SCORING_RECOVERY_V3.md").read_text("utf-8")
    _require(
        "failed before publishing" in recovery_document
        and "asterisk-only protected response" in recovery_document
        and "No response was inspected for tuning" in recovery_document
        and "rescore ID `rfpt-sv3`" in recovery_document
        and "scorer routes, scorer versions, and model" in recovery_document
        and "identities are unchanged" in recovery_document,
        "protected scoring recovery-v3 documentation omits a required boundary",
    )
    return (
        {
            "authorization_id": authorization["authorization_id"],
            "authorization": _binding(root, authorization_relative),
            "bound_inputs": path_bindings,
            **inventory_bindings,
            "routes": dict(sorted(expected_routes.items())),
            "authorized_family_route_counts": {"frozen_v1": 4, "recovery_v2": 3},
            "expected_output_sets": 14,
            "short_rescore_id": "rfpt-sv3",
            "allowed_failed_rescore_directory": allowed_failed_directory,
            "recovery_lock": recovery_lock,
            "failure": authorization["failure"],
            "published_rescore_count_before_failure": 0,
            "observed_log_disclosure": authorization["observed_log_disclosure"],
            "suppress_scorer_logging": True,
            "generation_semantics_changed": False,
            "scoring_routes_changed": False,
            "semantic_outputs_inspected": False,
            "tuning_from_disclosure": False,
            "scores_computed_at_authorization": False,
        },
        scoring_config,
    )


def _validate_cp70_protected_outputs(
    root: Path,
    protected: dict[str, Any],
    recovery_config: Any,
    scoring_config: Any,
    scoring_recovery: dict[str, Any],
) -> dict[str, Any]:
    from reasonforge.evaluation.protected_outputs import (
        ProtectedOutputIntegrityError,
        ProtectedOutputWriter,
        verify_deterministic_rescore,
    )
    from reasonforge.evaluation.protected_protocol import (
        load_benchmark_specs,
        load_run_plan,
    )

    plans = protected.get("finalist_plans")
    if not isinstance(plans, dict) or len(plans) != 14:
        raise CheckpointIntegrityError("CP70 requires exactly fourteen frozen plans")
    failed_lock = _safe_path(root, "evidence/protected_scoring/SCORING_IN_PROGRESS.json")
    _require(
        failed_lock.is_file()
        and file_sha256(failed_lock)
        == "8ad41f79858f3088a7ba25b1de266e627441bfdaff477a003355ea2133fe71f7",
        "original protected scoring interruption lock was not preserved exactly",
    )
    _require(
        not _safe_path(root, str(scoring_recovery["recovery_lock"])).exists(),
        "protected scoring recovery-v3 interruption lock remains present",
    )
    specs = load_benchmark_specs(root / "configs/evaluation/protected_benchmarks.yaml")
    output_root = root / "evidence/protected_outputs"
    _require(output_root.is_dir(), "protected output root is missing")
    expected_directories: dict[str, Path] = {}
    for output_set_id, binding in plans.items():
        if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
            raise CheckpointIntegrityError("CP70 frozen plan binding is invalid")
        plan = load_run_plan(_safe_path(root, binding["path"]))
        expected_directories[str(output_set_id)] = output_root / plan.model_id / str(output_set_id)
    discovered_directories = {
        output_directory.resolve()
        for model_directory in output_root.iterdir()
        if model_directory.is_dir()
        for output_directory in model_directory.iterdir()
        if output_directory.is_dir()
    }
    _require(
        all(path.is_dir() for path in output_root.iterdir()),
        "protected output root contains an unexpected top-level file",
    )
    _require(
        discovered_directories == {path.resolve() for path in expected_directories.values()},
        "protected output-set directory inventory differs from fourteen frozen plans",
    )

    output_bindings: dict[str, Any] = {}
    route_counts = {"frozen_v1": 0, "recovery_v2": 0}
    total_records = 0
    total_shards = 0
    for output_set_id, output_directory in sorted(expected_directories.items()):
        binding = plans[output_set_id]
        plan_relative = str(binding["path"])
        plan = load_run_plan(_safe_path(root, plan_relative))
        spec = specs.get(plan.benchmark_name)
        if spec is None:
            raise CheckpointIntegrityError(
                f"CP70 plan references an unknown benchmark: {plan.benchmark_name}"
            )
        run_plan_path = output_directory / "RUN_PLAN.json"
        _require(
            run_plan_path.is_file()
            and file_sha256(run_plan_path) == binding.get("sha256")
            and load_run_plan(run_plan_path) == plan,
            f"protected output RUN_PLAN identity mismatch: {output_set_id}",
        )
        adoption_relative = (
            output_directory.relative_to(root).as_posix() + "/WRITER_V2_ADOPTION.json"
        )
        _, adoption = _json_object(root, adoption_relative)
        _require(
            set(adoption)
            == {
                "schema_version",
                "status",
                "writer_protocol_id",
                "run_plan_sha256",
                "adopted_manifest_sha256",
                "finish_reason_semantics",
            }
            and adoption.get("schema_version") == 1
            and adoption.get("status") == "ADOPTED"
            and adoption.get("writer_protocol_id") == "RFPT-PROTECTED-WRITER-v2"
            and adoption.get("run_plan_sha256") == binding.get("sha256")
            and isinstance(adoption.get("adopted_manifest_sha256"), str)
            and len(adoption["adopted_manifest_sha256"]) == 64
            and all(
                character in "0123456789abcdef" for character in adoption["adopted_manifest_sha256"]
            )
            and adoption.get("finish_reason_semantics")
            == "LEGACY_FIELD_UNVERIFIED_DO_NOT_USE_FOR_CLAIMS"
            and not (output_directory / "WRITER_V2_PENDING_APPEND.json").exists()
            and not (output_directory / "WRITER_V2_PENDING_COMPLETE.json").exists(),
            f"protected writer-v2 adoption/journal state mismatch: {output_set_id}",
        )
        try:
            verification = ProtectedOutputWriter(output_directory).verify(require_complete=True)
        except ProtectedOutputIntegrityError as error:
            raise CheckpointIntegrityError(
                f"protected output set is not complete/valid: {output_set_id}: {error}"
            ) from error
        _require(
            verification.manifest.identity == plan.identity
            and verification.manifest.expected_example_ids == plan.expected_example_ids
            and not verification.missing_example_ids
            and len(verification.records) == len(plan.expected_example_ids)
            and all(shard.status == "COMPLETE" for shard in verification.manifest.shards),
            f"protected output set does not match frozen plan: {output_set_id}",
        )
        for record in verification.records:
            _require(
                record.example_id in plan.example_identity_sha256
                and record.benchmark_id == spec.benchmark_id
                and record.benchmark_revision == spec.revision
                and record.prompt_template_id == spec.prompt_template_id,
                f"protected output record benchmark identity mismatch: {output_set_id}",
            )
        rescore_root = output_directory / "rescores"
        invalid_rescores = (
            [path for path in rescore_root.iterdir() if path.name.startswith("INVALID-")]
            if rescore_root.is_dir()
            else []
        )
        complete_rescores = (
            [
                path
                for path in rescore_root.iterdir()
                if path.is_dir()
                and not path.name.startswith(".")
                and not path.name.startswith("INVALID-")
            ]
            if rescore_root.is_dir()
            else []
        )
        _require(
            not invalid_rescores and len(complete_rescores) == 1,
            f"protected output set requires exactly one valid rescore: {output_set_id}",
        )
        expected_rescore_id = str(scoring_recovery["short_rescore_id"])
        rescore_directory = complete_rescores[0]
        _require(
            rescore_directory.name == expected_rescore_id,
            f"protected rescore was not published under the authorized identity: {output_set_id}",
        )
        try:
            rescore_manifest, results = verify_deterministic_rescore(
                rescore_directory, output_directory
            )
        except ProtectedOutputIntegrityError as error:
            raise CheckpointIntegrityError(
                f"protected rescore is not complete/valid: {output_set_id}: {error}"
            ) from error
        authorized_route = scoring_config.routes.get(plan.benchmark_name)
        _require(
            authorized_route in {"frozen_v1", "recovery_v2"},
            f"protected benchmark lacks an authorized scoring route: {output_set_id}",
        )
        correction = recovery_config.scorers.get(spec.benchmark_id)
        if correction is None:
            _require(
                authorized_route == "frozen_v1",
                f"protected scorer route/recovery mismatch: {output_set_id}",
            )
            effective_grader_revision = spec.grader_revision
            scorer_protocol = "RFPT-PROTECTED-v1"
            grader_config_sha256 = sha256_json(
                {
                    "grader_id": spec.grader_id,
                    "revision": spec.grader_revision,
                    "benchmark_spec": spec.model_dump(mode="json"),
                }
            )
        else:
            _require(
                authorized_route == "recovery_v2",
                f"protected scorer route/recovery mismatch: {output_set_id}",
            )
            _require(
                correction.grader_id == spec.grader_id
                and correction.old_revision == spec.grader_revision,
                f"protected recovery scorer ancestry mismatch: {output_set_id}",
            )
            effective_grader_revision = correction.new_revision
            scorer_protocol = recovery_config.protocol_id
            grader_config_sha256 = sha256_json(
                {
                    "recovery_protocol_id": recovery_config.protocol_id,
                    "benchmark_spec": spec.model_dump(mode="json"),
                    "correction": correction.model_dump(mode="json"),
                }
            )
        route_counts[authorized_route] += 1
        _require(
            rescore_manifest.get("rescore_id") == expected_rescore_id
            and rescore_manifest.get("grader_id") == spec.grader_id
            and rescore_manifest.get("grader_version") == effective_grader_revision
            and rescore_manifest.get("grader_config_sha256") == grader_config_sha256
            and len(results) == len(plan.expected_example_ids),
            f"protected rescore grader identity mismatch: {output_set_id}",
        )
        allowed_top_level = {
            "RUN_PLAN.json",
            "manifest.json",
            "PROTECTED_OUTPUT_COMPLETE.json",
            "WRITER_V2_ADOPTION.json",
            "rescores",
            *(shard.filename for shard in verification.manifest.shards),
        }
        _require(
            {path.name for path in output_directory.iterdir()} == allowed_top_level,
            f"protected output set contains an unexpected or incomplete artifact: {output_set_id}",
        )
        _require(
            {path.name for path in rescore_directory.iterdir()}
            == {"manifest.json", "results.jsonl", "RESCORE_COMPLETE.json"},
            f"protected rescore contains an unexpected or incomplete artifact: {output_set_id}",
        )
        _require(
            {path.name for path in rescore_root.iterdir()} == {expected_rescore_id},
            f"protected rescore root contains non-v3 failure or partial state: {output_set_id}",
        )
        total_records += len(verification.records)
        total_shards += len(verification.manifest.shards)
        relative_output = output_directory.relative_to(root).as_posix()
        relative_rescore = rescore_directory.relative_to(root).as_posix()
        output_bindings[output_set_id] = {
            "output_directory": relative_output,
            "run_plan": _binding(root, f"{relative_output}/RUN_PLAN.json"),
            "manifest": _binding(root, f"{relative_output}/manifest.json"),
            "completion": _binding(root, f"{relative_output}/PROTECTED_OUTPUT_COMPLETE.json"),
            "writer_v2_adoption": _binding(root, adoption_relative),
            "shards": {
                shard.filename: _binding(root, f"{relative_output}/{shard.filename}")
                for shard in verification.manifest.shards
            },
            "record_count": len(verification.records),
            "record_inventory_sha256": sha256_json(verification.record_sha256),
            "rescore": {
                "directory": relative_rescore,
                "manifest": _binding(root, f"{relative_rescore}/manifest.json"),
                "results": _binding(root, f"{relative_rescore}/results.jsonl"),
                "completion": _binding(root, f"{relative_rescore}/RESCORE_COMPLETE.json"),
                "grader_id": spec.grader_id,
                "grader_revision_at_cp60": spec.grader_revision,
                "effective_grader_revision": effective_grader_revision,
                "scorer_protocol": scorer_protocol,
                "authorized_route": authorized_route,
                "scoring_authorization_id": scoring_recovery["authorization_id"],
                "result_count": len(results),
            },
        }
    dataset_identities = protected.get("dataset_identities")
    if not isinstance(dataset_identities, dict):
        raise CheckpointIntegrityError("protected dataset identities are missing")
    expected_total_records = 2 * sum(
        int(record.get("expected_rows", 0))
        for record in dataset_identities.values()
        if isinstance(record, dict)
    )
    _require(
        total_records == expected_total_records == 3372,
        "protected output total does not match the frozen two-finalist dataset inventory",
    )
    _require(
        route_counts == {"frozen_v1": 8, "recovery_v2": 6},
        "protected output-set routing differs from the authorized fourteen-set matrix",
    )
    failed_rescore_root = _safe_path(
        root, str(scoring_recovery["allowed_failed_rescore_directory"])
    )
    _require(
        {path.name for path in failed_rescore_root.iterdir()}
        == {str(scoring_recovery["short_rescore_id"])},
        "recorded v1 empty-directory failure was not recovered by only the valid v3 rescore",
    )
    return {
        "output_set_count": len(output_bindings),
        "total_records": total_records,
        "total_shards": total_shards,
        "all_expected_ids_present_exactly_once": True,
        "all_shards_complete_and_hash_verified": True,
        "immutable_outputs_bound_to_separate_scores": True,
        "authorized_output_set_route_counts": route_counts,
        "regeneration_check": {
            "observed": False,
            "evidence_basis": (
                "EXACT_FROZEN_OUTPUT_DIRECTORY_INVENTORY_AND_UNIQUE_EXPECTED_ID_COMPLETION"
            ),
        },
        "output_sets": output_bindings,
    }


def qualify_cp70_evidence(root: Path, registry: ExperimentRegistry) -> MilestoneEvidence:
    """Qualify complete immutable protected outputs and their deterministic scores."""

    root = root.resolve()
    _validate_cp70_registry(registry)
    protected = _validate_cp50_protected_protocol_freeze(root)
    cp60_binding, cp60 = _validate_cp70_cp60_checkpoint(root, registry, protected)
    recovery, recovery_config = _validate_cp70_recovery_amendment(root, cp60_binding)
    scoring_authorization, scoring_config = _validate_cp70_scoring_recovery_v3(
        root, cp60_binding, protected, recovery
    )
    protected_outputs = _validate_cp70_protected_outputs(
        root, protected, recovery_config, scoring_config, scoring_authorization
    )
    _require(
        protected_outputs["output_set_count"] == cp60.get("finalist_plan_count") == 14,
        "CP70 output count differs from verified CP60 plan count",
    )
    last_runtime = "artifacts/training/E20/E20-rcac-no-difficulty-grpo-seed1701-v1/checkpoint-32"
    artifact = {
        "schema_version": 1,
        "checkpoint_id": "CP70_PROTECTED_COMPLETE",
        "protocol_id": registry.protocol_id,
        "protected_protocol_id": "RFPT-PROTECTED-v1",
        "protected_access_state": "LOCKED_FINAL_SCORED_COMPLETE",
        "cp60_verified_binding": cp60_binding,
        "finalists": cp60["finalists"],
        "generation_config": cp60["generation_config"],
        "finalist_matrix": cp60["finalist_matrix"],
        "finalist_plans": cp60["finalist_plans"],
        "dataset_identities": cp60["dataset_identities"],
        "scorer_revisions_at_cp60": cp60["scorer_revisions"],
        "recovery_v2": recovery,
        "scoring_recovery_v3": scoring_authorization,
        "effective_scorer_revisions": {
            benchmark_id: (
                {
                    "grader_id": recovery["corrected_scorers"][benchmark_id]["grader_id"],
                    "grader_revision": recovery["corrected_scorers"][benchmark_id]["new_revision"],
                    "scorer_protocol": "RFPT-PROTECTED-RECOVERY-v2",
                }
                if benchmark_id in recovery["corrected_scorers"]
                else {
                    **record,
                    "scorer_protocol": "RFPT-PROTECTED-v1",
                }
            )
            for benchmark_id, record in cp60["scorer_revisions"].items()
        },
        "analysis_policy": cp60["analysis_policy"],
        "protected_outputs": protected_outputs,
        "last_complete_runtime_checkpoint": last_runtime,
        "scoring_disposition": (
            "DETERMINISTIC_SEPARATE_RESCORES_COMPLETE_UNDER_SCORING_RECOVERY_V3_"
            "WITH_UNCHANGED_RECOVERY_V2_SCORERS"
        ),
        "regeneration_authorized": False,
        "next_authorized_action": "ANALYZE_IMMUTABLE_PROTECTED_OUTPUTS_WITH_FROZEN_POLICY",
    }
    model_identities = {
        "strongest_conventional_e16_adapter": str(
            cp60["finalists"]["e16_terminal_uniform_seed1701"]["model_checkpoint_sha256"]
        ),
        "rcac_e18_adapter": str(cp60["finalists"]["e18_rcac_seed1701"]["model_checkpoint_sha256"]),
        "protected_output_inventory": sha256_json(protected_outputs["output_sets"]),
        "protected_score_inventory": sha256_json(
            {
                output_set_id: binding["rescore"]
                for output_set_id, binding in protected_outputs["output_sets"].items()
            }
        ),
        "protected_scoring_recovery_v3_authorization": str(
            scoring_authorization["authorization"]["sha256"]
        ),
    }
    return MilestoneEvidence(artifact, model_identities, last_runtime)


_CP80_E25_PROTOCOL = "RFPT-E25-PROTECTED-REPLICATION-v1"
_CP80_E25_ROOT = "evidence/protected_replication/RFPT-E25-PROTECTED-REPLICATION-v1"
_CP80_E25_COMPLETION = f"{_CP80_E25_ROOT}/REPLICATION_COMPLETE.json"
_CP80_E25_BINDING = "evidence/protected_replication/RFPT-E25-PRE-PROTECTED-v2/BINDING.json"
_CP80_E25_BINDING_SHA256 = "9090c3118602068be8093d53945367ab0c5596281d006ff24636d17d6b56d405"
_CP80_E25_MATRIX = f"{_CP80_E25_ROOT}/RUN_PLAN_MATRIX.json"
_CP80_E25_MATRIX_SHA256 = "68e800b31f585e789d5c25300b0ca649bd8f91b5c71b346cf40758af9e467746"
_CP80_E25_TRAINING_AUDIT = "evidence/training/E25/TRAINING_DEVELOPMENT_COMPLETE.json"
_CP80_E25_SCORING_LOCK = f"{_CP80_E25_ROOT}/SCORING_IN_PROGRESS.json"
_CP80_E25_RESCORE_ID = "rfpt-e25-s1"
_CP80_POLICY_SHA256 = "30680634e3db628169d0ebfa80c06e146431b2f606fec9f3ce58f2a4cf852b8c"
_CP80_E36_DISPOSITION = "evidence/training/E36/FOUR_B_DISPOSITION.json"
_CP80_SCALE_MODEL = "Qwen/Qwen3-4B"
_CP80_SCALE_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


def _cp80_registry_entry(registry: ExperimentRegistry, experiment_id: str) -> Any:
    matches = [item for item in registry.experiments if item.experiment_id == experiment_id]
    if len(matches) != 1:
        raise CheckpointIntegrityError(f"CP80 registry must contain exactly one {experiment_id}")
    return matches[0]


def _validate_cp80_registry(registry: ExperimentRegistry) -> str:
    """Require completed E25 and an explicit, protocol-valid E36 disposition."""

    e25 = _cp80_registry_entry(registry, "E25")
    e36 = _cp80_registry_entry(registry, "E36")
    _require(e25.status is ExperimentStatus.COMPLETE, "CP80 E25 replication is not COMPLETE")
    _require(
        e36.depends_on == ["E25"] and e36.priority == "P1",
        "CP80 E36 registry dependency or priority changed",
    )
    if e36.status is ExperimentStatus.COMPLETE:
        return "EXECUTED"
    if e36.status is ExperimentStatus.BLOCKED:
        _require(
            e36.blocked_reason in {"BLOCKED_LOCAL_COMPUTE", "BLOCKED_PROTOCOL_UNDERSPECIFIED"},
            "CP80 E36 blocker must be BLOCKED_LOCAL_COMPUTE or BLOCKED_PROTOCOL_UNDERSPECIFIED",
        )
        return str(e36.blocked_reason)
    raise CheckpointIntegrityError(
        "CP80 requires E36 COMPLETE or explicitly BLOCKED_LOCAL_COMPUTE / "
        "BLOCKED_PROTOCOL_UNDERSPECIFIED"
    )


def _validate_cp80_cp70_checkpoint(
    root: Path, registry: ExperimentRegistry
) -> tuple[dict[str, str | int], dict[str, Any]]:
    store = ProjectCheckpointStore(root / "evidence/checkpoints")
    manifest, checkpoint = store.verify_latest()
    _require(
        manifest.checkpoint_id == "CP70_PROTECTED_COMPLETE",
        "CP80 requires CP70_PROTECTED_COMPLETE to be the latest VERIFIED checkpoint",
    )
    _require(
        manifest.protocol_id == registry.protocol_id
        and manifest.protected_access_state == "LOCKED_FINAL_SCORED_COMPLETE",
        "CP70 protocol or protected-access state mismatch",
    )
    relative = checkpoint.relative_to(root).as_posix()
    binding_relative = f"{relative}/PROTECTED_COMPLETE_EVIDENCE_BINDINGS.json"
    _, binding = _json_object(root, binding_relative)
    _require(
        binding.get("checkpoint_id") == "CP70_PROTECTED_COMPLETE"
        and binding.get("protocol_id") == registry.protocol_id
        and binding.get("protected_access_state") == "LOCKED_FINAL_SCORED_COMPLETE",
        "CP70 protected-complete evidence identity mismatch",
    )
    return _binding(root, f"{relative}/CHECKPOINT_MANIFEST.json"), binding


def _cp80_bound_file(
    root: Path,
    raw: object,
    *,
    label: str,
    expected_path: str | None = None,
) -> dict[str, str | int]:
    if not isinstance(raw, dict):
        raise CheckpointIntegrityError(f"CP80 {label} binding is missing")
    path = raw.get("path")
    if not isinstance(path, str) or not path:
        raise CheckpointIntegrityError(f"CP80 {label} path is invalid")
    if expected_path is not None:
        _require(path == expected_path, f"CP80 {label} path mismatch")
    observed = _binding(root, path)
    _require(
        raw.get("sha256") == observed["sha256"] and raw.get("bytes") == observed["bytes"],
        f"CP80 {label} hash or byte size mismatch",
    )
    return observed


def _cp80_json_status(
    root: Path,
    binding: dict[str, str | int],
    *,
    label: str,
    statuses: set[str],
) -> dict[str, Any]:
    _, value = _json_object(root, str(binding["path"]))
    _require(value.get("status") in statuses, f"CP80 {label} is not complete")
    return value


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
    scoring_lock = complete.get("scoring_lock")
    _require(
        scoring_lock == {"path": _CP80_E25_SCORING_LOCK, "present": False}
        and not _safe_path(root, _CP80_E25_SCORING_LOCK).exists(),
        "CP80 E25 scoring lock is active or incorrectly reported",
    )
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
            "scoring_lock_absent": True,
            "active_run_count": 0,
        },
        {
            str(model_id): str(model["model_checkpoint_sha256"])
            for model_id, model in matrix["models"].items()
            if isinstance(model, dict)
        },
    )


def _cp80_validate_frozen_4b_recipe(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CheckpointIntegrityError("CP80 E36 frozen 4B recipe is missing")
    _require(
        value.get("model_id") == _CP80_SCALE_MODEL
        and value.get("revision") == _CP80_SCALE_REVISION
        and value.get("recipes") == ["C0", "C1", "C2"]
        and value.get("scientific_recipe_preserved") is True
        and value.get("substitute_experiment_run") is False,
        "CP80 E36 frozen 4B recipe identity changed",
    )
    return value


def _validate_cp80_e36_disposition(root: Path, expected: str) -> dict[str, Any]:
    from reasonforge.orchestration.four_b_transfer import (
        FourBIntegrityError,
        verify_e36_disposition,
    )

    try:
        if expected == "BLOCKED_PROTOCOL_UNDERSPECIFIED":
            from reasonforge.orchestration.four_b_protocol_disposition import (
                verify_protocol_block_disposition,
            )

            value = verify_protocol_block_disposition(root)
        else:
            value = verify_e36_disposition(root, expected_status=expected)
    except (OSError, FourBIntegrityError) as error:
        raise CheckpointIntegrityError(f"CP80 E36 disposition is invalid: {error}") from error
    recipe = _cp80_validate_frozen_4b_recipe(value.get("frozen_recipe"))
    result = {
        "disposition": _binding(root, _CP80_E36_DISPOSITION),
        "status": expected,
        "frozen_recipe": recipe,
    }
    if expected == "EXECUTED":
        result["runs"] = value["runs"]
    elif expected == "BLOCKED_PROTOCOL_UNDERSPECIFIED":
        for key in (
            "block_reason", "protocol_blocker_evidence", "bounded_c0_profiles",
            "c0_bounded_real_cuda_qualified", "four_b_transfer_completed",
            "four_b_confirmation_available", "local_compute_limitation_verified",
            "administrative_p1_disposition_complete", "disposition_policy_id",
        ):
            result[key] = value[key]
    else:
        result["block_reason"] = value["block_reason"]
        result["preflight_evidence"] = value["preflight_evidence"]
    return result


def qualify_cp80_evidence(root: Path, registry: ExperimentRegistry) -> MilestoneEvidence:
    """Qualify final replication evidence and the 4B execution/block disposition."""

    root = root.resolve()
    e36_status = _validate_cp80_registry(registry)
    cp70_manifest, cp70 = _validate_cp80_cp70_checkpoint(root, registry)
    e25, replication_models = _validate_cp80_e25_completion(root, cp70)
    e36 = _validate_cp80_e36_disposition(root, e36_status)
    last_runtime = "artifacts/training/E25/E25-rcac-terminal-grpo-seed1703-v1/checkpoint-32"
    artifact = {
        "schema_version": 1,
        "checkpoint_id": "CP80_REPLICATION_COMPLETE",
        "protocol_id": registry.protocol_id,
        "protected_access_state": "LOCKED_FINAL_MULTI_SEED_SCORED_COMPLETE",
        "cp70_verified_manifest": cp70_manifest,
        "e25_replication": e25,
        "e36_four_b_disposition": e36,
        "e26_required_at_cp80": False,
        "e26_requirement": "COMPLETE_BEFORE_E29_FINAL_RELEASE_DECISION",
        "next_authorized_action": "FINALIZE_E26_THEN_EXECUTE_E29",
    }
    model_identities = {
        f"e25_replication_{model_id}": identity
        for model_id, identity in sorted(replication_models.items())
    }
    model_identities["e36_four_b_disposition"] = str(e36["status"])
    return MilestoneEvidence(artifact, model_identities, last_runtime)


def infer_latest_native_checkpoint(root: Path) -> str | None:
    """Return the newest verified native Trainer checkpoint, if one exists."""

    candidates: list[tuple[str, str]] = []
    for completion_path in (root / "artifacts" / "training").glob(
        "*/*/checkpoint-*/CHECKPOINT_COMPLETE.json"
    ):
        checkpoint = completion_path.parent
        identity_path = checkpoint.parent / "RUN_IDENTITY.json"
        if not identity_path.is_file():
            continue
        try:
            identity = json.loads(identity_path.read_text("utf-8"))
            completion = json.loads(completion_path.read_text("utf-8"))
            validate_native_trainer_checkpoint(checkpoint, identity)
            timestamp = str(completion.get("completion_timestamp", ""))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            continue
        candidates.append((timestamp, checkpoint.relative_to(root).as_posix()))
    return max(candidates, default=("", ""))[1] or None
