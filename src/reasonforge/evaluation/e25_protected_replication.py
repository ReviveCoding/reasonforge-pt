"""Additive protected-evaluation protocol for E25 replication seeds.

The planner derives protected prompt/dataset identities from the already-frozen
CP60 run plans.  It deliberately does not reload protected source rows.  The
execution helpers are gated by a separately published E25 pre-protected binding.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from reasonforge.checkpoints.hashing import atomic_write_json, sha256_file

from .protected_outputs import ProtectedOutputIntegrityError
from .protected_protocol import (
    FrozenProtectedRunPlan,
    directory_tree_sha256,
    load_run_plan,
)

E25_PROTECTED_PROTOCOL_ID = "RFPT-E25-PROTECTED-REPLICATION-v1"
E25_PLAN_COUNT = 28
E25_MODEL_COUNT = 4
E25_BENCHMARK_COUNT = 7
_BASE_MODEL_ID = "Qwen/Qwen3-1.7B"
_BASE_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
_TOKENIZER_ID = f"{_BASE_MODEL_ID}@{_BASE_REVISION}"
_RUNS = {
    ("conventional", 1702): "E25-conventional-terminal-uniform-grpo-seed1702-v1",
    ("conventional", 1703): "E25-conventional-terminal-uniform-grpo-seed1703-v1",
    ("rcac", 1702): "E25-rcac-terminal-grpo-seed1702-v1",
    ("rcac", 1703): "E25-rcac-terminal-grpo-seed1703-v1",
}
_SEMANTIC_FIELDS = (
    "base_model_id",
    "base_model_revision",
    "tokenizer_id",
    "benchmark_name",
    "benchmark_spec_sha256",
    "protected_dataset_fingerprint",
    "generation_config",
    "generation_config_sha256",
    "expected_example_ids",
    "example_identity_sha256",
)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProtectedOutputIntegrityError(f"invalid JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ProtectedOutputIntegrityError(f"JSON artifact must be an object: {path}")
    return value


def _under(root: Path, raw: str, label: str) -> Path:
    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ProtectedOutputIntegrityError(f"{label} escapes workspace: {raw}") from error
    return candidate


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _source_plan_pairs(root: Path) -> dict[str, tuple[Path, Path]]:
    matrix_path = root / "evidence/protected_protocol/FINALIST_RUN_PLAN_MATRIX.json"
    matrix = _json(matrix_path)
    plans = matrix.get("plans")
    if (
        matrix.get("schema_version") != 1
        or matrix.get("status") != "FROZEN"
        or matrix.get("protocol_id") != "RFPT-PROTECTED-v1"
        or matrix.get("plan_count") != 14
        or not isinstance(plans, Mapping)
        or len(plans) != 14
    ):
        raise ProtectedOutputIntegrityError("CP60 source matrix identity is invalid")
    grouped: dict[str, dict[str, Path]] = {}
    for output_set_id, raw_entry in plans.items():
        if not isinstance(raw_entry, Mapping):
            raise ProtectedOutputIntegrityError("CP60 source matrix entry is invalid")
        raw_path = raw_entry.get("path")
        expected_sha = raw_entry.get("sha256")
        model_id = raw_entry.get("model_id")
        benchmark = raw_entry.get("benchmark_name")
        if not all(isinstance(item, str) for item in (raw_path, expected_sha, model_id, benchmark)):
            raise ProtectedOutputIntegrityError("CP60 source matrix identity is incomplete")
        path = _under(root, str(raw_path), "CP60 run plan")
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ProtectedOutputIntegrityError(f"CP60 run-plan hash mismatch: {output_set_id}")
        grouped.setdefault(str(benchmark), {})[str(model_id)] = path
    if len(grouped) != E25_BENCHMARK_COUNT:
        raise ProtectedOutputIntegrityError("CP60 benchmark-family count changed")
    result: dict[str, tuple[Path, Path]] = {}
    for benchmark, by_model in grouped.items():
        expected = {"e16_terminal_uniform_seed1701", "e18_rcac_seed1701"}
        if set(by_model) != expected:
            raise ProtectedOutputIntegrityError(
                f"CP60 benchmark lacks both frozen seed-1701 models: {benchmark}"
            )
        conventional = by_model["e16_terminal_uniform_seed1701"]
        rcac = by_model["e18_rcac_seed1701"]
        left, right = load_run_plan(conventional), load_run_plan(rcac)
        left_values, right_values = left.to_dict(), right.to_dict()
        if any(left_values[field] != right_values[field] for field in _SEMANTIC_FIELDS):
            raise ProtectedOutputIntegrityError(
                f"CP60 seed-1701 plans disagree on protected semantics: {benchmark}"
            )
        result[benchmark] = (conventional, rcac)
    return result


def _new_models(root: Path) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for (arm, seed), run_id in sorted(_RUNS.items()):
        run_dir = root / "artifacts/training/E25" / run_id
        run_complete_path = run_dir / "RUN_COMPLETE.json"
        checkpoint_complete_path = run_dir / "checkpoint-32/CHECKPOINT_COMPLETE.json"
        run = _json(run_complete_path)
        checkpoint = _json(checkpoint_complete_path)
        adapter = run_dir / "final_adapter"
        expected_source = "E16" if arm == "conventional" else "E18"
        if (
            run.get("status") != "COMPLETE"
            or run.get("experiment_id") != "E25"
            or run.get("run_id") != run_id
            or run.get("seed") != seed
            or run.get("source_experiment_id") != expected_source
            or run.get("optimizer_steps") != 32
            or run.get("completed_rollouts") != 512
            or not isinstance(run.get("final_adapter_identity"), str)
            or checkpoint.get("status") != "COMPLETE"
            or checkpoint.get("checkpoint_step") != 32
            or not adapter.is_dir()
        ):
            raise ProtectedOutputIntegrityError(
                f"E25 run is not a complete frozen finalist: {run_id}"
            )
        model_id = f"e25_{arm}_seed{seed}"
        models[model_id] = {
            "model_id": model_id,
            "arm": arm,
            "seed": seed,
            "run_id": run_id,
            "run_complete": _relative(root, run_complete_path),
            "run_complete_sha256": sha256_file(run_complete_path),
            "checkpoint_complete": _relative(root, checkpoint_complete_path),
            "checkpoint_complete_sha256": sha256_file(checkpoint_complete_path),
            "adapter_path": _relative(root, adapter),
            "model_checkpoint_sha256": str(run["final_adapter_identity"]),
            "adapter_tree_sha256": directory_tree_sha256(adapter),
            "base_model_id": _BASE_MODEL_ID,
            "base_model_revision": _BASE_REVISION,
            "tokenizer_id": _TOKENIZER_ID,
        }
    if len(models) != E25_MODEL_COUNT:
        raise ProtectedOutputIntegrityError("E25 protected model count changed")
    return models


def build_e25_run_plan_matrix(workspace: Path | str) -> dict[str, Any]:
    """Build a metadata-only 28-plan matrix from frozen CP60 identities."""

    root = Path(workspace).resolve()
    source_pairs = _source_plan_pairs(root)
    models = _new_models(root)
    plans: dict[str, dict[str, Any]] = {}
    plan_values: dict[str, dict[str, Any]] = {}
    for model_id, model in sorted(models.items()):
        for benchmark, (source_path, corroborating_path) in sorted(source_pairs.items()):
            source = load_run_plan(source_path)
            output_set_id = f"e25-{model_id}-{benchmark}-v1"
            plan = replace(
                source,
                protocol_id=E25_PROTECTED_PROTOCOL_ID,
                experiment_id="E25",
                output_set_id=output_set_id,
                model_id=model_id,
                model_checkpoint_sha256=str(model["model_checkpoint_sha256"]),
                adapter_path=str(model["adapter_path"]),
                adapter_tree_sha256=str(model["adapter_tree_sha256"]),
            )
            plan.validate()
            plan_values[output_set_id] = plan.to_dict()
            plans[output_set_id] = {
                "experiment_id": "E25",
                "model_id": model_id,
                "arm": model["arm"],
                "seed": model["seed"],
                "run_id": model["run_id"],
                "model_checkpoint_sha256": model["model_checkpoint_sha256"],
                "checkpoint_completion_sha256": model["checkpoint_complete_sha256"],
                "benchmark_name": benchmark,
                "protected_dataset_fingerprint": plan.protected_dataset_fingerprint,
                "generation_config_sha256": plan.generation_config_sha256,
                "expected_rows": len(plan.expected_example_ids),
                "source_plan": {
                    "path": _relative(root, source_path),
                    "sha256": sha256_file(source_path),
                },
                "corroborating_source_plan": {
                    "path": _relative(root, corroborating_path),
                    "sha256": sha256_file(corroborating_path),
                },
            }
    if len(plans) != E25_PLAN_COUNT:
        raise ProtectedOutputIntegrityError("E25 protected plan count changed")
    return {
        "schema_version": 1,
        "status": "FROZEN",
        "protocol_id": E25_PROTECTED_PROTOCOL_ID,
        "parent_protocol_id": "RFPT-PROTECTED-v1",
        "experiment_id": "E25",
        "model_count": len(models),
        "benchmark_count": len(source_pairs),
        "plan_count": len(plans),
        "parent_matrix": {
            "path": "evidence/protected_protocol/FINALIST_RUN_PLAN_MATRIX.json",
            "sha256": sha256_file(
                root / "evidence/protected_protocol/FINALIST_RUN_PLAN_MATRIX.json"
            ),
        },
        "models": models,
        "plans": plans,
        "plan_values": plan_values,
        "seed1701_outputs": "REUSE_IMMUTABLE_DO_NOT_REGENERATE",
        "protected_source_rows_loaded": False,
        "semantic_outputs_inspected": False,
        "outputs_generated": False,
        "scores_computed": False,
    }


def publish_e25_run_plan_matrix(
    workspace: Path | str,
    output_directory: Path | str = Path(
        "evidence/protected_replication/RFPT-E25-PROTECTED-REPLICATION-v1"
    ),
) -> dict[str, Any]:
    """Atomically publish plans and matrix; existing non-identical files fail closed."""

    root = Path(workspace).resolve()
    output = _under(root, Path(output_directory).as_posix(), "E25 protected plan directory")
    built = build_e25_run_plan_matrix(root)
    plan_values = built.pop("plan_values")
    plans_dir = output / "plans"
    for output_set_id, value in sorted(plan_values.items()):
        path = plans_dir / f"{output_set_id}.json"
        if path.exists() and _json(path) != value:
            raise ProtectedOutputIntegrityError(f"refusing to overwrite E25 plan: {path}")
        if not path.exists():
            atomic_write_json(path, value)
        built["plans"][output_set_id]["path"] = _relative(root, path)
        built["plans"][output_set_id]["sha256"] = sha256_file(path)
    matrix_path = output / "RUN_PLAN_MATRIX.json"
    if matrix_path.exists() and _json(matrix_path) != built:
        raise ProtectedOutputIntegrityError("refusing to overwrite E25 protected matrix")
    if not matrix_path.exists():
        atomic_write_json(matrix_path, built)
    return verify_e25_run_plan_matrix(root, matrix_path)


def verify_e25_run_plan_matrix(
    workspace: Path | str,
    matrix_path: Path | str = Path(
        "evidence/protected_replication/RFPT-E25-PROTECTED-REPLICATION-v1/RUN_PLAN_MATRIX.json"
    ),
) -> dict[str, Any]:
    """Rebuild and verify the exact additive matrix without loading protected rows."""

    root = Path(workspace).resolve()
    path = _under(root, Path(matrix_path).as_posix(), "E25 protected matrix")
    observed = _json(path)
    rebuilt = build_e25_run_plan_matrix(root)
    plan_values = rebuilt.pop("plan_values")
    plans = observed.get("plans")
    if not isinstance(plans, Mapping) or len(plans) != E25_PLAN_COUNT:
        raise ProtectedOutputIntegrityError("E25 protected matrix plan inventory is invalid")
    for output_set_id, value in plan_values.items():
        entry = plans.get(output_set_id)
        if not isinstance(entry, Mapping):
            raise ProtectedOutputIntegrityError(f"missing E25 plan: {output_set_id}")
        raw_plan_path, expected_sha = entry.get("path"), entry.get("sha256")
        if not isinstance(raw_plan_path, str) or not isinstance(expected_sha, str):
            raise ProtectedOutputIntegrityError(f"invalid E25 plan binding: {output_set_id}")
        plan_path = _under(root, raw_plan_path, "E25 run plan")
        if (
            not plan_path.is_file()
            or sha256_file(plan_path) != expected_sha
            or _json(plan_path) != value
        ):
            raise ProtectedOutputIntegrityError(f"E25 run plan identity mismatch: {output_set_id}")
        rebuilt["plans"][output_set_id]["path"] = raw_plan_path
        rebuilt["plans"][output_set_id]["sha256"] = expected_sha
    if observed != rebuilt:
        raise ProtectedOutputIntegrityError("E25 protected matrix differs from deterministic build")
    return {
        "status": "VERIFIED",
        "protocol_id": E25_PROTECTED_PROTOCOL_ID,
        "matrix_path": _relative(root, path),
        "matrix_sha256": sha256_file(path),
        "model_count": E25_MODEL_COUNT,
        "benchmark_count": E25_BENCHMARK_COUNT,
        "plan_count": E25_PLAN_COUNT,
        "protected_source_rows_loaded": False,
        "semantic_outputs_inspected": False,
        "outputs_generated": False,
        "scores_computed": False,
    }


def verify_e25_plan_is_bound(
    workspace: Path | str,
    matrix_path: Path | str,
    plan_path: Path | str,
) -> tuple[FrozenProtectedRunPlan, dict[str, Any]]:
    """Verify one plan is an exact member of the immutable E25 matrix."""

    root = Path(workspace).resolve()
    verify_e25_run_plan_matrix(root, matrix_path)
    matrix = _json(_under(root, Path(matrix_path).as_posix(), "E25 protected matrix"))
    resolved = _under(root, Path(plan_path).as_posix(), "E25 run plan")
    plan = load_run_plan(resolved)
    entry = matrix["plans"].get(plan.output_set_id)
    if (
        not isinstance(entry, Mapping)
        or entry.get("path") != _relative(root, resolved)
        or entry.get("sha256") != sha256_file(resolved)
        or plan.model_id not in matrix["models"]
        or plan.protocol_id != E25_PROTECTED_PROTOCOL_ID
        or plan.experiment_id != "E25"
    ):
        raise ProtectedOutputIntegrityError("plan is not bound by the E25 protected matrix")
    return plan, dict(entry)
