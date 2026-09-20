"""Read-only E36 feasibility preflight and fail-closed disposition evidence.

This module deliberately does not import torch or load a model.  Static host evidence
can authorize a bounded CUDA qualification, but cannot by itself prove that the frozen
4B recipe is locally infeasible.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple, cast

import yaml

from reasonforge.checkpoints.hashing import (
    atomic_write_json,
    canonical_json_bytes,
    sha256_file,
    sha256_json,
)

E36_CONFIG = "configs/replication/e36_four_b_transfer.yaml"
E36_PREFLIGHT = "evidence/training/E36/FOUR_B_PREFLIGHT.json"
E36_DISPOSITION = "evidence/training/E36/FOUR_B_DISPOSITION.json"
E36_PROTOCOL = "RFPT-E36-4B-TRANSFER-v1"
E36_MODEL = "Qwen/Qwen3-4B"
E36_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
E36_RECIPES = ("C0", "C1", "C2")


class FourBIntegrityError(ValueError):
    """Raised when E36 evidence cannot support the requested disposition."""


class DiskUsage(NamedTuple):
    total: int
    used: int
    free: int


DiskUsageReader = Callable[[str | bytes | Path], tuple[int, int, int]]


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FourBIntegrityError(f"E36 {label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise FourBIntegrityError(f"E36 {label} must be a sequence")
    return value


def _safe_path(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise FourBIntegrityError(f"E36 path must be workspace-relative: {relative!r}")
    resolved_root = root.resolve()
    candidate = (resolved_root / relative).resolve()
    if candidate == resolved_root or resolved_root not in candidate.parents:
        raise FourBIntegrityError(f"E36 path escapes workspace: {relative!r}")
    return candidate


def _load_yaml(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise FourBIntegrityError(f"E36 {label} is unreadable: {path}: {error}") from error
    return _mapping(value, label)


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FourBIntegrityError(f"E36 {label} is unreadable: {path}: {error}") from error
    return _mapping(value, label)


def _binding(root: Path, relative: str) -> dict[str, str | int]:
    path = _safe_path(root, relative)
    if not path.is_file():
        raise FourBIntegrityError(f"E36 required artifact is missing: {relative}")
    return {"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _verify_binding(root: Path, value: object, label: str) -> dict[str, str | int]:
    raw = _mapping(value, label)
    relative = raw.get("path")
    if not isinstance(relative, str):
        raise FourBIntegrityError(f"E36 {label} path is invalid")
    observed = _binding(root, relative)
    if raw.get("sha256") != observed["sha256"] or raw.get("bytes") != observed["bytes"]:
        raise FourBIntegrityError(f"E36 {label} hash or size mismatch")
    return observed


def _config(root: Path, relative: str = E36_CONFIG) -> tuple[Mapping[str, Any], str]:
    path = _safe_path(root, relative)
    config = _load_yaml(path, "configuration")
    if (
        config.get("schema_version") != 1
        or config.get("experiment_id") != "E36"
        or config.get("protocol_id") != E36_PROTOCOL
        or config.get("scientific_state") != "LOCKED_FINAL"
    ):
        raise FourBIntegrityError("E36 configuration identity is invalid")
    model = _mapping(config.get("model"), "model configuration")
    if model.get("model_id") != E36_MODEL or model.get("revision") != E36_REVISION:
        raise FourBIntegrityError("E36 scale-model identity changed")
    model_registry_relative = model.get("registry_path")
    model_registry_key = model.get("registry_key")
    if not isinstance(model_registry_relative, str) or not isinstance(model_registry_key, str):
        raise FourBIntegrityError("E36 model-registry binding is invalid")
    registry = _load_yaml(_safe_path(root, model_registry_relative), "model registry")
    registered = _mapping(registry.get(model_registry_key), "registered scale model")
    for field in ("model_id", "revision", "weight_bytes"):
        if model.get(field) != registered.get(field):
            raise FourBIntegrityError(f"E36 model-registry {field} mismatch")
    frozen = _mapping(config.get("frozen_recipe"), "frozen recipe")
    recipes = tuple(str(item) for item in _sequence(frozen.get("recipes"), "recipe list"))
    definitions = _mapping(frozen.get("definitions"), "recipe definitions")
    if (
        recipes != E36_RECIPES
        or set(definitions) != set(E36_RECIPES)
        or frozen.get("scientific_recipe_preserved") is not True
        or frozen.get("substitute_experiment_run") is not False
    ):
        raise FourBIntegrityError("E36 frozen C0/C1/C2 recipe changed or permits a substitute")
    for recipe_id in E36_RECIPES:
        definition = _mapping(definitions[recipe_id], f"{recipe_id} recipe")
        if definition.get("run_id") != _expected_run_id(recipe_id):
            raise FourBIntegrityError(f"E36 {recipe_id} run identity changed")
    return config, sha256_file(path)


def _expected_run_id(recipe_id: str) -> str:
    return {
        "C0": "E36-C0-frozen-reference-v1",
        "C1": "E36-C1-conventional-v1",
        "C2": "E36-C2-rcac-v1",
    }[recipe_id]


def _frozen_recipe() -> dict[str, object]:
    return {
        "model_id": E36_MODEL,
        "revision": E36_REVISION,
        "recipes": list(E36_RECIPES),
        "scientific_recipe_preserved": True,
        "substitute_experiment_run": False,
    }


def _directory_bytes(path: Path) -> int:
    total = 0
    if not path.is_dir():
        return total
    for candidate in path.rglob("*"):
        try:
            if candidate.is_file():
                total += candidate.stat().st_size
        except OSError:
            continue
    return total


def build_e36_preflight(
    root: Path,
    *,
    config_relative: str = E36_CONFIG,
    disk_usage_reader: DiskUsageReader = shutil.disk_usage,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a model-free, read-only static feasibility record for the frozen recipe."""

    root = root.resolve()
    config, config_sha256 = _config(root, config_relative)
    policy = _mapping(config.get("resource_policy"), "resource policy")
    environment_relative = "evidence/environment/CP00_environment.json"
    environment = _load_json(_safe_path(root, environment_relative), "CP00 environment")
    cuda = _mapping(environment.get("cuda"), "CP00 CUDA evidence")
    nvidia_smi = _mapping(environment.get("nvidia_smi"), "CP00 NVIDIA evidence")
    rows = _sequence(nvidia_smi.get("rows"), "CP00 NVIDIA rows")

    total, used, free = disk_usage_reader(root)
    reserve = int(policy["disk_safety_reserve_bytes"])
    materialization = int(policy["estimated_model_and_runtime_bytes"])
    checkpoints = int(policy["estimated_checkpoint_retention_bytes"])
    required_free = reserve + materialization + checkpoints
    disk_sufficient = free >= required_free

    cache_relative = str(policy["canonical_cache_path"])
    cache = _safe_path(root, cache_relative)
    snapshot = cache / "snapshots" / E36_REVISION
    cache_bytes = _directory_bytes(cache)
    cache_sufficient = snapshot.is_dir() and cache_bytes >= int(policy["minimum_cache_bytes"])

    expected_gpu_count = int(policy["expected_physical_gpu_count"])
    expected_gpu_name = str(policy["expected_gpu_name"])
    vram_bytes = 0
    if rows and isinstance(rows[0], str):
        fields = [field.strip() for field in rows[0].split(",")]
        if len(fields) >= 5:
            try:
                vram_bytes = int(fields[4]) * 1024 * 1024
            except ValueError:
                vram_bytes = 0
    hardware_matches = (
        cuda.get("available") is True
        and cuda.get("device_count") == expected_gpu_count
        and cuda.get("name") == expected_gpu_name
        and cuda.get("bf16_supported") is True
        and nvidia_smi.get("qualified") is True
        and vram_bytes >= int(policy["minimum_vram_bytes_for_bounded_pilot"])
    )

    if not hardware_matches:
        status = "STATIC_HARDWARE_MISMATCH"
    elif not disk_sufficient:
        status = "INSUFFICIENT_DISK_FOR_FROZEN_RECIPE"
    elif not cache_sufficient:
        status = "CACHE_ACQUISITION_REQUIRED"
    else:
        status = "READY_FOR_BOUNDED_CUDA_QUALIFICATION"

    command_template = str(
        _mapping(config.get("execution_contract"), "execution contract")["command_template"]
    )
    commands = [command_template.replace("{recipe}", recipe_id) for recipe_id in E36_RECIPES]
    return {
        "schema_version": 1,
        "status": status,
        "experiment_id": "E36",
        "protocol_id": E36_PROTOCOL,
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "model_loaded": False,
        "cuda_process_launched": False,
        "protected_outputs_accessed": False,
        "final_block_authorized": False,
        "frozen_recipe": _frozen_recipe(),
        "configuration": {
            "path": config_relative,
            "sha256": config_sha256,
            "bytes": _safe_path(root, config_relative).stat().st_size,
        },
        "environment": _binding(root, environment_relative),
        "hardware": {
            "static_identity_matches": hardware_matches,
            "physical_gpu_count": cuda.get("device_count"),
            "gpu_name": cuda.get("name"),
            "vram_bytes": vram_bytes,
            "bf16_supported": cuda.get("bf16_supported"),
            "bounded_cuda_qualification_required": True,
        },
        "disk": {
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": free,
            "safety_reserve_bytes": reserve,
            "estimated_model_and_runtime_bytes": materialization,
            "estimated_checkpoint_retention_bytes": checkpoints,
            "required_free_bytes": required_free,
            "capacity_sufficient": disk_sufficient,
        },
        "cache": {
            "path": cache_relative,
            "snapshot_revision": E36_REVISION,
            "bytes": cache_bytes,
            "minimum_bytes": int(policy["minimum_cache_bytes"]),
            "complete": cache_sufficient,
            "missing_cache_is_not_compute_blocker": True,
        },
        "authorized_fallback_ladder": list(
            _sequence(config.get("authorized_fallback_ladder"), "fallback ladder")
        ),
        "future_execution_commands": commands,
        "interpretation": (
            "Static evidence is not sufficient for BLOCKED_LOCAL_COMPUTE; a bounded CUDA "
            "qualification and durable fallback-attempt evidence are required."
        ),
    }


def publish_e36_preflight(root: Path, *, created_at: str | None = None) -> dict[str, Any]:
    """Atomically publish the current read-only preflight without touching disposition."""

    root = root.resolve()
    if _safe_path(root, E36_DISPOSITION).exists():
        raise FourBIntegrityError("E36 preflight is immutable after disposition publication")
    value = build_e36_preflight(root, created_at=created_at)
    destination = _safe_path(root, E36_PREFLIGHT)
    atomic_write_json(destination, value)
    return value


def _verify_preflight(root: Path, binding: object) -> Mapping[str, Any]:
    verified = _verify_binding(root, binding, "preflight binding")
    if verified["path"] != E36_PREFLIGHT:
        raise FourBIntegrityError("E36 disposition must bind the canonical preflight")
    value = _load_json(_safe_path(root, E36_PREFLIGHT), "preflight")
    if (
        value.get("schema_version") != 1
        or value.get("experiment_id") != "E36"
        or value.get("protocol_id") != E36_PROTOCOL
        or value.get("model_loaded") is not False
        or value.get("cuda_process_launched") is not False
        or value.get("final_block_authorized") is not False
        or value.get("frozen_recipe") != _frozen_recipe()
    ):
        raise FourBIntegrityError("E36 preflight identity or model-free boundary is invalid")
    if "post_acquisition_recovery" in value:
        from .four_b_disk_recovery import verify_disk_recovery

        verify_disk_recovery(root, value, live=False)
    return value


def verify_e36_preflight(root: Path) -> dict[str, Any]:
    """Verify and return the canonical static preflight."""

    root = root.resolve()
    value = _verify_preflight(root, _binding(root, E36_PREFLIGHT))
    if "post_acquisition_recovery" in value:
        from .four_b_disk_recovery import verify_disk_recovery

        verify_disk_recovery(root, value, live=True)
    return dict(value)


def verify_e36_run_completion(root: Path, recipe_id: str) -> dict[str, Any]:
    """Verify one canonical frozen-recipe completion and return its binding."""

    if recipe_id not in E36_RECIPES:
        raise FourBIntegrityError(f"E36 recipe is invalid: {recipe_id}")
    root = root.resolve()
    config, _ = _config(root)
    definitions = _mapping(
        _mapping(config.get("frozen_recipe"), "frozen recipe").get("definitions"),
        "recipe definitions",
    )
    definition = _mapping(definitions[recipe_id], f"{recipe_id} recipe")
    relative = str(definition["completion_path"])
    run = _load_json(_safe_path(root, relative), f"{recipe_id} completion")
    run_directory = _safe_path(root, relative).parent
    identity = _load_json(run_directory / "RUN_IDENTITY.json", f"{recipe_id} run identity")
    identity_hash = sha256_json(identity)
    if (
        run.get("status") != "COMPLETE"
        or run.get("experiment_id") != "E36"
        or run.get("protocol_id") != E36_PROTOCOL
        or run.get("recipe_id") != recipe_id
        or run.get("run_id") != definition.get("run_id")
        or run.get("model_id") != E36_MODEL
        or run.get("model_revision") != E36_REVISION
        or run.get("run_identity_hash") != identity_hash
        or run.get("scientific_recipe_preserved") is not True
        or run.get("substitute_experiment_run") is not False
        or identity.get("schema_version") != 1
        or identity.get("experiment_id") != "E36"
        or identity.get("protocol_id") != E36_PROTOCOL
        or identity.get("recipe_id") != recipe_id
        or identity.get("run_id") != definition.get("run_id")
        or identity.get("model_id") != E36_MODEL
        or identity.get("model_revision") != E36_REVISION
        or identity.get("scientific_recipe_preserved") is not True
        or identity.get("substitute_experiment_run") is not False
    ):
        raise FourBIntegrityError(f"E36 {recipe_id} completion identity is invalid")
    _verify_binding(root, identity.get("configuration"), f"{recipe_id} configuration")
    worker_binding = _verify_binding(root, run.get("worker_result"), f"{recipe_id} worker result")
    worker = _load_json(_safe_path(root, str(worker_binding["path"])), f"{recipe_id} worker result")
    if (
        worker.get("status") != "COMPLETE"
        or worker.get("experiment_id") != "E36"
        or worker.get("protocol_id") != E36_PROTOCOL
        or worker.get("recipe_id") != recipe_id
        or worker.get("run_id") != definition.get("run_id")
        or worker.get("run_identity_hash") != identity_hash
        or worker.get("model_id") != E36_MODEL
        or worker.get("model_revision") != E36_REVISION
    ):
        raise FourBIntegrityError(f"E36 {recipe_id} worker result identity is invalid")
    return _binding(root, relative)


def _verify_completed_runs(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    del config  # the public verifier reloads and binds the canonical configuration
    runs: dict[str, Any] = {}
    for recipe_id in E36_RECIPES:
        runs[recipe_id] = verify_e36_run_completion(root, recipe_id)
    return runs


def _verify_failure_evidence(
    root: Path, raw_binding: object, preflight_binding: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, str | int]:
    binding = _verify_binding(root, raw_binding, "local-compute failure evidence")
    value = _load_json(_safe_path(root, str(binding["path"])), "local-compute failure evidence")
    ladder = list(_sequence(config.get("authorized_fallback_ladder"), "fallback ladder"))
    failure_mode = value.get("failure_mode")
    common_valid = (
        value.get("schema_version") == 1
        and value.get("status") == "BLOCKED_LOCAL_COMPUTE"
        and value.get("experiment_id") == "E36"
        and value.get("protocol_id") == E36_PROTOCOL
        and value.get("model_id") == E36_MODEL
        and value.get("model_revision") == E36_REVISION
        and value.get("affected_recipes") == list(E36_RECIPES)
        and value.get("scientific_recipe_preserved") is True
        and value.get("substitute_experiment_run") is False
        and value.get("authorized_fallback_ladder_exhausted") is True
        and value.get("preflight") == dict(preflight_binding)
    )
    if not common_valid:
        raise FourBIntegrityError("E36 local-compute failure identity is invalid")
    attempts = _sequence(value.get("attempts"), "failure attempts")
    if failure_mode in {"CUDA_OOM", "CUDA_RUNTIME_INFEASIBLE"}:
        preflight = _load_json(_safe_path(root, E36_PREFLIGHT), "preflight")
        if preflight.get("status") != "READY_FOR_BOUNDED_CUDA_QUALIFICATION":
            raise FourBIntegrityError(
                "E36 CUDA block requires a preflight ready for bounded qualification"
            )
        observed_steps: list[object] = []
        for index, raw in enumerate(attempts):
            attempt = _mapping(raw, f"failure attempt {index}")
            observed_steps.append(attempt.get("fallback_step"))
            evidence = _verify_binding(root, attempt.get("evidence"), f"failure attempt {index}")
            record = _load_json(_safe_path(root, str(evidence["path"])), f"failure attempt {index}")
            if (
                record.get("status") != "FAILED"
                or record.get("experiment_id") != "E36"
                or record.get("protocol_id") != E36_PROTOCOL
                or record.get("model_id") != E36_MODEL
                or record.get("model_revision") != E36_REVISION
                or record.get("recipe_id") not in E36_RECIPES
                or record.get("fallback_step") != attempt.get("fallback_step")
                or record.get("attempt_index") != index
                or record.get("failure_type") != failure_mode
                or record.get("failure_observed") is not True
                or record.get("cuda_process_launched") is not True
                or record.get("cuda_available") is not True
                or record.get("cuda_device_name")
                != _mapping(config.get("resource_policy"), "resource policy").get(
                    "expected_gpu_name"
                )
                or record.get("model_instances_created") != 1
                or record.get("child_model_processes_created") != 0
                or not isinstance(record.get("peak_allocated_bytes"), int)
                or int(record["peak_allocated_bytes"]) <= 0
                or record.get("scientific_recipe_preserved") is not True
                or record.get("substitute_experiment_run") is not False
            ):
                raise FourBIntegrityError(f"E36 fallback attempt {index} identity is invalid")
            _verify_binding(root, record.get("diagnostic_log"), f"failure attempt {index} log")
        if observed_steps != ladder:
            raise FourBIntegrityError(
                "E36 CUDA failure does not exhaust the ordered fallback ladder"
            )
    elif failure_mode == "DISK_CAPACITY":
        preflight = _load_json(_safe_path(root, E36_PREFLIGHT), "preflight")
        disk = _mapping(preflight.get("disk"), "preflight disk evidence")
        if (
            disk.get("capacity_sufficient") is not False
            or value.get("retention_minimized") is not True
            or value.get("claim_preserving_artifacts_retained") is not True
            or attempts
        ):
            raise FourBIntegrityError("E36 disk-capacity block is not rigorously established")
    else:
        raise FourBIntegrityError("E36 local-compute failure mode is not authorized")
    return binding


def build_e36_disposition(
    root: Path,
    *,
    status: str,
    failure_evidence_relative: str | None = None,
    block_reason: str | None = None,
) -> dict[str, Any]:
    """Build an EXECUTED or rigorously evidenced BLOCKED_LOCAL_COMPUTE record."""

    root = root.resolve()
    config, config_sha256 = _config(root)
    preflight_binding = _binding(root, E36_PREFLIGHT)
    preflight = _verify_preflight(root, preflight_binding)
    base: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "E36",
        "protocol_id": E36_PROTOCOL,
        "frozen_recipe": _frozen_recipe(),
        "configuration": {
            "path": E36_CONFIG,
            "sha256": config_sha256,
            "bytes": _safe_path(root, E36_CONFIG).stat().st_size,
        },
        "preflight": preflight_binding,
    }
    if status == "EXECUTED":
        if failure_evidence_relative is not None or block_reason is not None:
            raise FourBIntegrityError("E36 EXECUTED disposition cannot include blocker evidence")
        if preflight.get("status") != "READY_FOR_BOUNDED_CUDA_QUALIFICATION":
            raise FourBIntegrityError("E36 EXECUTED disposition requires a ready preflight")
        return {**base, "status": status, "runs": _verify_completed_runs(root, config)}
    if status != "BLOCKED_LOCAL_COMPUTE":
        raise FourBIntegrityError(f"E36 disposition status is not authorized: {status}")
    if not failure_evidence_relative or not block_reason or not block_reason.strip():
        raise FourBIntegrityError("E36 blocked disposition requires failure evidence and a reason")
    failure_binding = _verify_failure_evidence(
        root,
        _binding(root, failure_evidence_relative),
        preflight_binding,
        config,
    )
    return {
        **base,
        "status": status,
        "local_compute_limitation_verified": True,
        "authorized_fallback_ladder_exhausted": True,
        "block_reason": block_reason.strip(),
        "preflight_evidence": [preflight_binding, failure_binding],
    }


def publish_e36_disposition(
    root: Path,
    *,
    status: str,
    failure_evidence_relative: str | None = None,
    block_reason: str | None = None,
) -> dict[str, Any]:
    """Atomically publish a verified disposition, refusing identity-changing overwrite."""

    root = root.resolve()
    value = build_e36_disposition(
        root,
        status=status,
        failure_evidence_relative=failure_evidence_relative,
        block_reason=block_reason,
    )
    destination = _safe_path(root, E36_DISPOSITION)
    if destination.exists() and destination.read_bytes() != canonical_json_bytes(value):
        raise FourBIntegrityError("E36 disposition already exists with different content")
    atomic_write_json(destination, value)
    return value


def verify_e36_disposition(root: Path, *, expected_status: str) -> dict[str, Any]:
    """Rebuild the canonical disposition and require exact content identity."""

    root = root.resolve()
    observed = _load_json(_safe_path(root, E36_DISPOSITION), "disposition")
    failure_relative: str | None = None
    block_reason: str | None = None
    if expected_status == "BLOCKED_LOCAL_COMPUTE":
        evidence = _sequence(observed.get("preflight_evidence"), "preflight evidence")
        if len(evidence) != 2:
            raise FourBIntegrityError("E36 blocked disposition must bind preflight and failure")
        failure = _mapping(evidence[1], "failure evidence")
        failure_path = failure.get("path")
        if not isinstance(failure_path, str):
            raise FourBIntegrityError("E36 failure evidence path is invalid")
        failure_relative = failure_path
        raw_reason = observed.get("block_reason")
        block_reason = raw_reason if isinstance(raw_reason, str) else None
    rebuilt = build_e36_disposition(
        root,
        status=expected_status,
        failure_evidence_relative=failure_relative,
        block_reason=block_reason,
    )
    if observed != rebuilt:
        raise FourBIntegrityError("E36 disposition is not canonical or its evidence changed")
    return rebuilt
