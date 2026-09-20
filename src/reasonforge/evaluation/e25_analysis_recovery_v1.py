"""Admission of the exact completed E25 scoring recovery to unchanged statistics."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from types import FunctionType
from typing import Any, cast

from reasonforge.checkpoints.hashing import atomic_write_json, canonical_json_bytes, sha256_file

from . import e25_three_seed_analysis as original
from .e25_protected_execution import load_e25_execution_config
from .e25_scoring_recovery_v1 import DEFAULT_CONFIG as SCORING_CONFIG
from .e25_scoring_recovery_v1 import load_config as load_scoring_recovery_config
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

ANALYSIS_RECOVERY_ID = "RFPT-E25-ANALYSIS-RECOVERY-v1"
ANALYSIS_CONFIG = Path("configs/evaluation/e25_three_seed_analysis_v1.yaml")
ANALYSIS_CONFIG_SHA = "9ee5af1bca7ecb8088192d141daa1d4333ea4bc374d6337c02019e92724e26d1"
AUTHORIZATION = Path(
    "evidence/protected_replication/RFPT-E25-ANALYSIS-RECOVERY-v1/AUTHORIZATION.json"
)
_EXACT_RECOVERY_HASHES = {
    "authorization": "04cc4e385f93eec0cf02404ba0d9eed4ae21095864c15358069464f4f98f5c7e",
    "completion": "f428733b6c8495827a18ce624c346650420e7cb856aefc33d2b26bd0c3b9bb22",
    "recovery_lock": "06f2809b44cdf64ed13428a272c89060577663f3fd35b18b1616d7d16a00c771",
}
_CODE_PATHS = (
    "src/reasonforge/evaluation/e25_analysis_recovery_v1.py",
    "src/reasonforge/evaluation/e25_three_seed_analysis.py",
    "src/reasonforge/statistics/__init__.py",
    "src/reasonforge/statistics/paired.py",
    "src/reasonforge/statistics/policy.py",
    "scripts/analyze_e25_recovery_v1.py",
)
_TEST_PATHS = ("tests/test_e25_analysis_recovery_v1.py", "tests/test_e25_three_seed_analysis.py")
_DOC_PATHS = ("docs/E25_ANALYSIS_RECOVERY_V1.md",)


def verify_completed_scoring_recovery(root: Path) -> dict[str, Any]:
    """Verify opaque bytes and metadata; do not decode generations or result values."""
    root = root.resolve()
    config = load_scoring_recovery_config(root / SCORING_CONFIG)
    paths = {
        "authorization": root / config.authorization,
        "completion": (root / config.recovery_lock).parent / "COMPLETE.json",
        "recovery_lock": root / config.recovery_lock,
        "original_lock": root / config.failed_lock,
        "failed_results": root / config.failed_directory / "results.jsonl",
    }
    expected_hashes = {
        **_EXACT_RECOVERY_HASHES,
        "original_lock": config.failed_lock_sha256,
        "failed_results": config.failed_results_sha256,
    }
    for name, path in paths.items():
        if not path.is_file() or sha256_file(path) != expected_hashes[name]:
            raise ProtectedOutputIntegrityError(f"E25 completed recovery {name} differs")
    authorization, completion = _json(paths["authorization"]), _json(paths["completion"])
    if (
        authorization.get("authorization_id") != config.authorization_id
        or authorization.get("status") != "VERIFIED_SCORING_RECOVERY_AUTHORIZATION"
        or completion.get("status") != "COMPLETE"
        or completion.get("authorization_id") != config.authorization_id
        or completion.get("target_count") != 28
        or completion.get("semantic_outputs_logged") is not False
        or completion.get("score_values_logged") is not False
        or completion.get("original_failure_evidence_preserved") is not True
    ):
        raise ProtectedOutputIntegrityError("E25 recovery completion contract differs")
    for label, binding in authorization["bindings"].items():
        path = _safe(root, Path(binding["path"]))
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise ProtectedOutputIntegrityError(f"E25 scoring authorization binding drift: {label}")
    for inventory_name in ("code_inventory", "test_inventory", "documentation_inventory"):
        for name, digest in authorization[inventory_name].items():
            path = _safe(root, Path(name))
            if not path.is_file() or sha256_file(path) != digest:
                raise ProtectedOutputIntegrityError("E25 scoring recovery inventory drift")
    execution = load_e25_execution_config(root / config.execution_config)
    if authorization["routes"] != execution.routes:
        raise ProtectedOutputIntegrityError("E25 scoring routes differ")
    matrix = _json(root / execution.matrix)
    plans = matrix["plans"]
    receipts = completion.get("rescores")
    if (
        not isinstance(receipts, list)
        or len(receipts) != 28
        or len(plans) != 28
        or [r.get("output_set_id") for r in receipts] != sorted(plans)
        or set(authorization["output_manifest_inventory"]) != set(plans)
    ):
        raise ProtectedOutputIntegrityError("E25 completion receipt matrix differs")
    output_root = _safe(root, execution.output_root)
    expected_directories = {
        output_root / entry["model_id"] / output_id for output_id, entry in plans.items()
    }
    discovered: set[Path] = set()
    for model in output_root.iterdir():
        if not model.is_dir():
            raise ProtectedOutputIntegrityError("E25 output namespace has extra file")
        discovered.update(model.iterdir())
    if discovered != expected_directories:
        raise ProtectedOutputIntegrityError("E25 completed output namespace differs")
    inventory: dict[str, Any] = {}
    for receipt in receipts:
        output_id = receipt["output_set_id"]
        entry = plans[output_id]
        base = output_root / entry["model_id"] / output_id
        plan = _load_plan_metadata(root, entry)
        if sha256_file(base / "RUN_PLAN.json") != entry["sha256"]:
            raise ProtectedOutputIntegrityError("E25 completed output plan differs")
        output_manifest = _json(base / "manifest.json")
        source_hash = sha256_file(base / "manifest.json")
        if source_hash != authorization["output_manifest_inventory"][output_id]:
            raise ProtectedOutputIntegrityError("E25 source manifest differs from authorization")
        source_complete = _json(base / "PROTECTED_OUTPUT_COMPLETE.json")
        adoption = _json(base / "WRITER_V2_ADOPTION.json")
        if (
            output_manifest.get("status") != "COMPLETE"
            or output_manifest.get("identity", {}).get("output_set_id") != output_id
            or source_complete.get("status") != "COMPLETE"
            or source_complete.get("output_set_id") != output_id
            or source_complete.get("manifest_sha256") != source_hash
            or adoption.get("status") != "ADOPTED"
            or adoption.get("run_plan_sha256") != entry["sha256"]
        ):
            raise ProtectedOutputIntegrityError("E25 source completion differs")
        rescores = base / "rescores"
        expected_children: set[str] = {config.rescore_id}
        if output_id == config.failed_output_set_id:
            expected_children.add(config.failed_directory.name)
            failed = root / config.failed_directory
            if failed.parent != rescores or {p.name for p in failed.iterdir()} != {"results.jsonl"}:
                raise ProtectedOutputIntegrityError("E25 failed namespace differs")
        if {p.name for p in rescores.iterdir()} != expected_children:
            raise ProtectedOutputIntegrityError("E25 prior, partial, or extra rescore namespace")
        destination = rescores / config.rescore_id
        if {p.name for p in destination.iterdir()} != {
            "manifest.json",
            "results.jsonl",
            "RESCORE_COMPLETE.json",
        }:
            raise ProtectedOutputIntegrityError("E25 completed rescore contents differ")
        manifest = _json(destination / "manifest.json")
        sentinel = _json(destination / "RESCORE_COMPLETE.json")
        manifest_hash = sha256_file(destination / "manifest.json")
        expected_ids = _expected_ids(plan, output_set_id=output_id)
        if (
            receipt.get("rescore_id") != config.rescore_id
            or receipt.get("manifest_sha256") != manifest_hash
            or manifest.get("status") != "COMPLETE"
            or manifest.get("rescore_id") != config.rescore_id
            or manifest.get("output_set_id") != output_id
            or manifest.get("source_manifest_sha256") != source_hash
            or manifest.get("expected_example_ids") != expected_ids
            or manifest.get("result_count") != len(expected_ids)
            or manifest.get("results_sha256") != sha256_file(destination / "results.jsonl")
            or sentinel
            != {
                "status": "COMPLETE",
                "rescore_id": config.rescore_id,
                "manifest_sha256": manifest_hash,
            }
        ):
            raise ProtectedOutputIntegrityError("E25 completed rescore identity differs")
        source_shards = {s["filename"]: s["sha256"] for s in output_manifest["shards"]}
        if manifest.get("source_shard_sha256") != source_shards or {
            p.name for p in base.glob("shard_*.jsonl")
        } != set(source_shards):
            raise ProtectedOutputIntegrityError("E25 source shard inventory differs")
        for filename, digest in source_shards.items():
            shard = _safe(root, (base / filename).relative_to(root))
            if sha256_file(shard) != digest:
                raise ProtectedOutputIntegrityError("E25 source shard hash differs")
        for pending in ("WRITER_V2_PENDING_APPEND.json", "WRITER_V2_PENDING_COMPLETE.json"):
            if (base / pending).exists():
                raise ProtectedOutputIntegrityError("E25 writer pending state exists")
        inventory[output_id] = {
            name: _binding(root, (destination / name).relative_to(root))
            for name in ("manifest.json", "results.jsonl", "RESCORE_COMPLETE.json")
        }
    staging = root / config.staging_root
    if not staging.is_dir() or any(staging.iterdir()):
        raise ProtectedOutputIntegrityError("E25 recovery staging is not empty")
    return {
        "status": "VERIFIED_COMPLETED_SCORING_RECOVERY",
        "bindings": {name: _binding(root, path.relative_to(root)) for name, path in paths.items()},
        "rescores": inventory,
        "preserved_locks_are_inactive_attempt_evidence": True,
        "generation_text_read": False,
        "score_values_read": False,
    }


def prepare_analysis_recovery(root: Path, *, publish_authorization: bool = False) -> dict[str, Any]:
    root = root.resolve()
    if sha256_file(root / ANALYSIS_CONFIG) != ANALYSIS_CONFIG_SHA:
        raise ProtectedOutputIntegrityError("E25 frozen analysis configuration differs")
    config = load_e25_three_seed_config(root / ANALYSIS_CONFIG)
    for name in original._FIXED_BINDING_SHA256:
        _verify_binding(root, getattr(config, name), name)
    receipt = verify_completed_scoring_recovery(root)
    value = {
        "schema_version": 1,
        "status": "VERIFIED_ANALYSIS_RECOVERY_AUTHORIZATION",
        "authorization_id": ANALYSIS_RECOVERY_ID,
        "analysis_configuration": _binding(root, ANALYSIS_CONFIG),
        "completed_scoring_recovery": receipt,
        "statistical_semantics_changed": False,
        "scores_changed": False,
        "generation_text_read": False,
        "score_values_read": False,
        "code_inventory": {p: sha256_file(root / p) for p in _CODE_PATHS},
        "test_inventory": {p: sha256_file(root / p) for p in _TEST_PATHS},
        "documentation_inventory": {p: sha256_file(root / p) for p in _DOC_PATHS},
    }
    path = root / AUTHORIZATION
    if publish_authorization:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    if not path.is_file() or _json(path) != value:
        raise ProtectedOutputIntegrityError("E25 analysis recovery authorization drift")
    return value


def build_recovery_analysis(root: Path) -> dict[str, Any]:
    prepare_analysis_recovery(root)
    namespace = dict(original.build_e25_three_seed_analysis.__globals__)
    namespace["_collect_score_sets"] = _collect_score_sets
    function = original.build_e25_three_seed_analysis
    bound = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__)
    result = cast(dict[str, Any], bound(root, root / ANALYSIS_CONFIG))
    result["scoring_recovery_authorization"] = _binding(root, AUTHORIZATION)
    return result


def publish_recovery_analysis(root: Path) -> dict[str, Any]:
    value = build_recovery_analysis(root)
    config = load_e25_three_seed_config(root / ANALYSIS_CONFIG)
    path = _safe(root, config.analysis_output)
    if path.exists() and _json(path) != value:
        raise ProtectedOutputIntegrityError("refusing to overwrite E25 analysis")
    if not path.exists():
        atomic_write_json(path, value)
    return value


def publish_recovery_completion(root: Path) -> dict[str, Any]:
    prepare_analysis_recovery(root)
    value = build_e25_replication_completion(root, root / ANALYSIS_CONFIG)
    config = load_e25_three_seed_config(root / ANALYSIS_CONFIG)
    path = _safe(root, config.completion_output)
    if path.exists() and _json(path) != value:
        raise ProtectedOutputIntegrityError("refusing to overwrite E25 completion")
    if not path.exists():
        atomic_write_json(path, value)
    return value


# Copied identity/collection logic; only the explicit lock admission differs.
def _collect_score_sets(
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
        or matrix.get("seed1701_outputs") != "REUSE_CP70_IMMUTABLE_OUTPUTS_BYTE_FOR_BYTE"
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


def build_e25_replication_completion(workspace: Path, config_path: Path) -> dict[str, Any]:
    """Bind all completed E25 outputs, scores, and analysis without reading output text."""

    root = workspace.resolve()
    config = load_e25_three_seed_config(_safe(root, config_path))
    matrix = _json(_verify_binding(root, config.run_plan_matrix, "E25 plan matrix"))
    _, analysis_inputs = _collect_score_sets(root, config)
    analysis_path = _safe(root, config.analysis_output)
    analysis = _json(analysis_path)
    if (
        analysis.get("status") != "COMPLETE"
        or analysis.get("output_set_count") != 42
        or analysis.get("generation_text_read") is not False
        or analysis.get("analysis_id") != config.analysis_id
        or analysis.get("protocol_id") != config.protocol_id
        or analysis.get("configuration")
        != _binding(root, _safe(root, config_path).relative_to(root))
        or analysis.get("input_inventory") != analysis_inputs
        or analysis.get("scoring_recovery_authorization") != _binding(root, AUTHORIZATION)
        or analysis.get("decision", {}).get("outcome")
        not in {"PROMOTE", "RETAIN_BASELINE", "BLOCK"}
    ):
        raise ProtectedOutputIntegrityError("E25 three-seed analysis is incomplete")
    raw_plans = matrix.get("plans")
    if not isinstance(raw_plans, Mapping) or len(raw_plans) != 28:
        raise ProtectedOutputIntegrityError("E25 completion requires exactly 28 new plans")
    plans = cast(Mapping[str, Mapping[str, Any]], raw_plans)
    output_sets: dict[str, Any] = {}
    rescores: dict[str, Any] = {}
    output_root = _safe(root, config.new_seed_output_root)
    for output_set_id, plan in sorted(plans.items()):
        if not isinstance(plan, Mapping) or not isinstance(plan.get("model_id"), str):
            raise ProtectedOutputIntegrityError(f"invalid completion plan: {output_set_id}")
        base = output_root / str(plan["model_id"]) / output_set_id
        plan_metadata = _load_plan_metadata(
            root, {"path": plan.get("path"), "sha256": plan.get("sha256")}
        )
        expected_ids = _expected_ids(plan_metadata, output_set_id=output_set_id)
        manifest_path = base / "manifest.json"
        completion_path = base / "PROTECTED_OUTPUT_COMPLETE.json"
        adoption_path = base / "WRITER_V2_ADOPTION.json"
        output_manifest = _json(manifest_path)
        output_completion = _json(completion_path)
        adoption = _json(adoption_path)
        identity = output_manifest.get("identity")
        if (
            output_manifest.get("status") != "COMPLETE"
            or not isinstance(identity, Mapping)
            or identity.get("output_set_id") != output_set_id
            or output_manifest.get("expected_example_ids") != expected_ids
            or plan.get("expected_rows") != len(expected_ids)
            or output_completion.get("status") != "COMPLETE"
            or output_completion.get("output_set_id") != output_set_id
            or output_completion.get("manifest_sha256") != sha256_file(manifest_path)
            or adoption.get("status") != "ADOPTED"
            or adoption.get("run_plan_sha256") != plan.get("sha256")
        ):
            raise ProtectedOutputIntegrityError(
                f"new-seed output identity mismatch: {output_set_id}"
            )
        output_sets[output_set_id] = {
            "output_manifest": _binding(root, manifest_path.relative_to(root)),
            "output_completion": _binding(root, completion_path.relative_to(root)),
            "writer_adoption": _binding(root, adoption_path.relative_to(root)),
        }
        rescore_directory = base / "rescores" / config.new_seed_rescore_id
        _, score_bindings = _read_results_without_generation_text(
            root,
            output_set_id=output_set_id,
            expected_ids=expected_ids,
            manifest_path=rescore_directory / "manifest.json",
            results_path=rescore_directory / "results.jsonl",
            completion_path=rescore_directory / "RESCORE_COMPLETE.json",
            rescore_id=config.new_seed_rescore_id,
        )
        if score_bindings != analysis_inputs[output_set_id]:
            raise ProtectedOutputIntegrityError(f"analysis rescore binding drift: {output_set_id}")
        rescores[output_set_id] = score_bindings
        for pending in (
            "WRITER_V2_PENDING_APPEND.json",
            "WRITER_V2_PENDING_COMPLETE.json",
        ):
            if (base / pending).exists():
                raise ProtectedOutputIntegrityError(f"pending writer state: {output_set_id}")
    verify_completed_scoring_recovery(root)
    return {
        "schema_version": 1,
        "status": "COMPLETE",
        "experiment_id": "E25",
        "protocol_id": config.protocol_id,
        "preprotected_v2": _binding(root, config.preprotected_v2.path),
        "run_plan_matrix": _binding(root, config.run_plan_matrix.path),
        "training_development_audit": _binding(root, config.training_development_audit.path),
        "new_seed_outputs": {
            "count": 28,
            "all_complete_and_verified": True,
            "incomplete_count": 0,
            "output_sets": output_sets,
        },
        "rescores": {
            "count": 28,
            "rescore_id": config.new_seed_rescore_id,
            "all_complete_and_verified": True,
            "output_sets": rescores,
        },
        "seed1701_reuse": {
            "policy": "REUSE_CP70_IMMUTABLE_OUTPUTS_BYTE_FOR_BYTE",
            "regenerated": False,
            "output_set_count": 14,
            "cp70_binding": _binding(root, config.cp70_binding.path),
        },
        "three_seed_analysis": _binding(root, config.analysis_output),
        "scoring_lock": {
            "path": config.scoring_lock.as_posix(),
            "present": True,
            "disposition": "PRESERVED_INACTIVE_COMPLETED_RECOVERY_EVIDENCE",
        },
        "scoring_recovery_authorization": _binding(root, AUTHORIZATION),
        "active_run_count": 0,
    }
