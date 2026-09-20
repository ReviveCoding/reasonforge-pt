"""Fail-closed E36 CUDA-worker boundary with injectable runtime execution.

This module validates scientific/run identities and publishes worker results.  Model
loading is delegated to a separately qualified runtime and is never performed during
CPU/synthetic qualification.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from reasonforge.checkpoints.hashing import atomic_write_json, sha256_json

from .four_b_transfer import (
    E36_CONFIG,
    E36_MODEL,
    E36_PROTOCOL,
    E36_RECIPES,
    E36_REVISION,
    FourBIntegrityError,
    _binding,
    _config,
    _load_json,
    _mapping,
    _safe_path,
    _sequence,
    verify_e36_preflight,
)

E36_CUDA_RUNTIME = "src/reasonforge/orchestration/four_b_cuda_runtime.py"
E36_CUDA_RUNTIME_QUALIFICATION = "evidence/training/E36/CUDA_RUNTIME_QUALIFICATION.json"
E36_WORKER_CPU_QUALIFICATION = "evidence/training/E36/CUDA_WORKER_CPU_QUALIFICATION.json"


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    workspace: str
    config_path: str
    recipe_id: str
    recipe_role: str
    source_experiment: str | None
    sampler: str | None
    reward_architecture: str | None
    run_id: str
    run_identity_hash: str
    model_id: str
    model_revision: str
    fallback_step: str
    fallback_index: int
    microbatch_size: int
    preserve_frozen_global_work: bool
    gradient_checkpointing: bool
    quantization_mode: str
    resume_from: str | None
    completed_optimizer_step_only: bool
    partial_rollout_groups_excluded: bool
    protected_access_prohibited: bool


@dataclass(frozen=True, slots=True)
class RuntimeOutcome:
    cuda_available: bool
    cuda_device_name: str
    peak_allocated_bytes: int
    model_instances_created: int
    child_model_processes_created: int
    optimizer_created: bool
    completed_optimizer_steps: int
    final_checkpoint_completion: Mapping[str, str | int] | None
    resumed_from: str | None
    protected_outputs_accessed: bool


class E36CudaOutOfMemory(RuntimeError):
    """Typed OOM signal carrying auditable device telemetry."""

    def __init__(
        self,
        *,
        device_name: str,
        peak_allocated_bytes: int,
        model_instances_created: int = 1,
        child_model_processes_created: int = 0,
    ) -> None:
        super().__init__("CUDA out of memory during frozen E36 recipe")
        self.device_name = device_name
        self.peak_allocated_bytes = peak_allocated_bytes
        self.model_instances_created = model_instances_created
        self.child_model_processes_created = child_model_processes_created


RuntimeExecutor = Callable[[WorkerRequest], RuntimeOutcome]


def _recipe_definition(config: Mapping[str, Any], recipe_id: str) -> Mapping[str, Any]:
    definitions = _mapping(
        _mapping(config.get("frozen_recipe"), "frozen recipe").get("definitions"),
        "recipe definitions",
    )
    if recipe_id not in E36_RECIPES:
        raise FourBIntegrityError(f"E36 worker recipe is invalid: {recipe_id}")
    return _mapping(definitions[recipe_id], f"{recipe_id} definition")


def _fallback_profile(ladder: Sequence[Any], fallback_step: str) -> dict[str, Any]:
    string_ladder = [str(item) for item in ladder]
    if fallback_step not in string_ladder:
        raise FourBIntegrityError(f"E36 worker fallback step is unauthorized: {fallback_step}")
    index = string_ladder.index(fallback_step)
    return {
        "fallback_index": index,
        "microbatch_size": 1,
        "preserve_frozen_global_work": True,
        "gradient_checkpointing": index >= 2,
        "quantization_mode": "QUALIFIED_QLORA" if index >= 3 else "FROZEN_PRIMARY",
    }


def _run_directory(root: Path, run_id: str) -> Path:
    return root / "artifacts" / "training" / "E36" / run_id


def _verify_run_identity(
    root: Path, *, recipe_id: str, run_id: str, run_identity_hash: str
) -> Mapping[str, Any]:
    identity = _load_json(_run_directory(root, run_id) / "RUN_IDENTITY.json", "run identity")
    if (
        sha256_json(identity) != run_identity_hash
        or identity.get("experiment_id") != "E36"
        or identity.get("protocol_id") != E36_PROTOCOL
        or identity.get("scientific_state") != "LOCKED_FINAL"
        or identity.get("recipe_id") != recipe_id
        or identity.get("run_id") != run_id
        or identity.get("model_id") != E36_MODEL
        or identity.get("model_revision") != E36_REVISION
        or identity.get("scientific_recipe_preserved") is not True
        or identity.get("substitute_experiment_run") is not False
        or identity.get("completed_optimizer_step_only") is not True
        or identity.get("partial_rollout_policy") != "PRESERVE_EXCLUDE_AND_DISCARD"
        or identity.get("configuration") != _binding(root, E36_CONFIG)
        or identity.get("preflight")
        != _binding(root, "evidence/training/E36/FOUR_B_PREFLIGHT.json")
    ):
        raise FourBIntegrityError("E36 worker run identity is incompatible")
    return identity


def _verify_resume_checkpoint(
    root: Path, *, run_id: str, run_identity_hash: str, resume_from: Path | None
) -> str | None:
    if resume_from is None:
        return None
    run_directory = _run_directory(root, run_id).resolve()
    checkpoint = resume_from.resolve()
    if run_directory not in checkpoint.parents or not checkpoint.name.startswith("checkpoint-"):
        raise FourBIntegrityError("E36 worker resume checkpoint escapes its run")
    marker = checkpoint / "CHECKPOINT_COMPLETE.json"
    value = _load_json(marker, "resume checkpoint completion")
    step = value.get("checkpoint_step")
    if (
        value.get("status") != "COMPLETE"
        or value.get("run_identity_hash") != run_identity_hash
        or not isinstance(step, int)
        or step <= 0
        or checkpoint.name != f"checkpoint-{step}"
    ):
        raise FourBIntegrityError("E36 worker resume checkpoint is not a compatible completed step")
    return checkpoint.relative_to(root).as_posix()


def build_worker_request(
    root: Path,
    *,
    config_path: Path,
    recipe_id: str,
    run_id: str,
    run_identity_hash: str,
    fallback_step: str,
    result_path: Path,
    resume_from: Path | None,
) -> WorkerRequest:
    """Validate all worker inputs without importing a neural runtime."""

    root = root.resolve()
    if config_path.resolve() != _safe_path(root, E36_CONFIG):
        raise FourBIntegrityError("E36 worker accepts only the canonical frozen config")
    config, _ = _config(root)
    definition = _recipe_definition(config, recipe_id)
    if run_id != definition.get("run_id"):
        raise FourBIntegrityError("E36 worker run ID does not match the frozen recipe")
    run_directory = _run_directory(root, run_id).resolve()
    resolved_result = result_path.resolve()
    if run_directory not in resolved_result.parents or "attempts" not in resolved_result.parts:
        raise FourBIntegrityError("E36 worker result path is outside the run attempt directory")
    if resolved_result.exists():
        raise FourBIntegrityError("E36 worker refuses to overwrite an existing result")
    preflight = verify_e36_preflight(root)
    if preflight.get("status") != "READY_FOR_BOUNDED_CUDA_QUALIFICATION":
        raise FourBIntegrityError("E36 worker requires a ready static preflight")
    _verify_run_identity(
        root,
        recipe_id=recipe_id,
        run_id=run_id,
        run_identity_hash=run_identity_hash,
    )
    resume = _verify_resume_checkpoint(
        root,
        run_id=run_id,
        run_identity_hash=run_identity_hash,
        resume_from=resume_from,
    )
    if recipe_id == "C0" and resume is not None:
        raise FourBIntegrityError("E36 C0 frozen reference has no training checkpoint to resume")
    profile = _fallback_profile(
        _sequence(config.get("authorized_fallback_ladder"), "fallback ladder"), fallback_step
    )
    role = str(definition.get("role"))
    expected = {
        "C0": ("frozen_reference", None, None, None),
        "C1": ("strongest_conventional_finalist", "E16", "uniform", "terminal"),
        "C2": ("rcac_finalist", "E18", "rcac", "selected_from_E16"),
    }[recipe_id]
    observed = (
        role,
        definition.get("source_experiment"),
        definition.get("source_sampler"),
        definition.get("source_reward_architecture"),
    )
    if observed != expected:
        raise FourBIntegrityError(f"E36 {recipe_id} recipe boundary changed")
    return WorkerRequest(
        workspace=str(root),
        config_path=str(config_path.resolve()),
        recipe_id=recipe_id,
        recipe_role=role,
        source_experiment=expected[1],
        sampler=expected[2],
        reward_architecture=expected[3],
        run_id=run_id,
        run_identity_hash=run_identity_hash,
        model_id=E36_MODEL,
        model_revision=E36_REVISION,
        fallback_step=fallback_step,
        resume_from=resume,
        completed_optimizer_step_only=True,
        partial_rollout_groups_excluded=True,
        protected_access_prohibited=True,
        **profile,
    )


def _validate_outcome(root: Path, request: WorkerRequest, outcome: RuntimeOutcome) -> None:
    if (
        outcome.cuda_available is not True
        or outcome.cuda_device_name != "NVIDIA GeForce RTX 4090 Laptop GPU"
        or outcome.peak_allocated_bytes <= 0
        or outcome.model_instances_created != 1
        or outcome.child_model_processes_created != 0
        or outcome.protected_outputs_accessed is not False
        or outcome.resumed_from != request.resume_from
    ):
        raise FourBIntegrityError("E36 runtime violated single-process/device or access semantics")
    if request.recipe_id == "C0":
        if (
            outcome.optimizer_created
            or outcome.completed_optimizer_steps != 0
            or outcome.final_checkpoint_completion is not None
        ):
            raise FourBIntegrityError("E36 C0 crossed the frozen-reference recipe boundary")
        return
    if not outcome.optimizer_created or outcome.completed_optimizer_steps <= 0:
        raise FourBIntegrityError(f"E36 {request.recipe_id} did not execute finalist training")
    binding = _mapping(outcome.final_checkpoint_completion, "final checkpoint binding")
    relative = binding.get("path")
    if not isinstance(relative, str) or dict(binding) != _binding(root, relative):
        raise FourBIntegrityError("E36 training result final-checkpoint binding is invalid")
    marker = _load_json(_safe_path(root, relative), "final checkpoint completion")
    if (
        marker.get("status") != "COMPLETE"
        or marker.get("run_identity_hash") != request.run_identity_hash
        or marker.get("checkpoint_step") != outcome.completed_optimizer_steps
    ):
        raise FourBIntegrityError("E36 training result checkpoint is not the completed final step")


def _publish_result(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists():
        raise FourBIntegrityError("E36 worker result already exists")
    payload = dict(value)
    atomic_write_json(path, payload)
    return payload


def execute_worker(
    root: Path,
    *,
    config_path: Path,
    recipe_id: str,
    run_id: str,
    run_identity_hash: str,
    fallback_step: str,
    result_path: Path,
    resume_from: Path | None,
    runtime: RuntimeExecutor,
) -> tuple[int, dict[str, Any]]:
    """Execute one qualified runtime attempt and atomically publish its result."""

    root = root.resolve()
    request = build_worker_request(
        root,
        config_path=config_path,
        recipe_id=recipe_id,
        run_id=run_id,
        run_identity_hash=run_identity_hash,
        fallback_step=fallback_step,
        result_path=result_path,
        resume_from=resume_from,
    )
    common = {
        "schema_version": 1,
        "experiment_id": "E36",
        "protocol_id": E36_PROTOCOL,
        "recipe_id": recipe_id,
        "run_id": run_id,
        "run_identity_hash": run_identity_hash,
        "model_id": E36_MODEL,
        "model_revision": E36_REVISION,
        "fallback_step": fallback_step,
        "scientific_recipe_preserved": True,
        "substitute_experiment_run": False,
        "completed_optimizer_step_only": True,
        "partial_rollout_groups_excluded": True,
        "worker_pid": os.getpid(),
    }
    try:
        outcome = runtime(request)
    except E36CudaOutOfMemory as error:
        if error.model_instances_created != 1 or error.child_model_processes_created != 0:
            raise FourBIntegrityError(
                "E36 OOM violated the one-model/one-process boundary"
            ) from error
        return 42, _publish_result(
            result_path,
            {
                **common,
                "status": "FAILED",
                "failure_type": "CUDA_OOM",
                "cuda_process_launched": True,
                "cuda_available": True,
                "cuda_device_name": error.device_name,
                "peak_allocated_bytes": error.peak_allocated_bytes,
                "model_instances_created": error.model_instances_created,
                "child_model_processes_created": error.child_model_processes_created,
                "protected_outputs_accessed": False,
                "request_identity": sha256_json(asdict(request)),
            },
        )
    _validate_outcome(root, request, outcome)
    return 0, _publish_result(
        result_path,
        {
            **common,
            "status": "COMPLETE",
            "failure_type": None,
            "cuda_process_launched": True,
            "cuda_available": outcome.cuda_available,
            "cuda_device_name": outcome.cuda_device_name,
            "peak_allocated_bytes": outcome.peak_allocated_bytes,
            "model_instances_created": outcome.model_instances_created,
            "child_model_processes_created": outcome.child_model_processes_created,
            "optimizer_created": outcome.optimizer_created,
            "completed_optimizer_steps": outcome.completed_optimizer_steps,
            "final_checkpoint_completion": outcome.final_checkpoint_completion,
            "resumed_from": outcome.resumed_from,
            "protected_outputs_accessed": outcome.protected_outputs_accessed,
            "request_identity": sha256_json(asdict(request)),
        },
    )


def load_qualified_cuda_runtime(root: Path) -> RuntimeExecutor:
    """Load the real runtime only after a hash-bound real-CUDA qualification."""

    root = root.resolve()
    qualification = _load_json(
        _safe_path(root, E36_CUDA_RUNTIME_QUALIFICATION), "CUDA runtime qualification"
    )
    code = _mapping(qualification.get("runtime_code"), "CUDA runtime code binding")
    relative = code.get("path")
    if (
        qualification.get("schema_version") != 1
        or qualification.get("status") != "QUALIFIED"
        or qualification.get("qualification_scope") != "REAL_CUDA_QWEN3_4B"
        or qualification.get("experiment_id") != "E36"
        or qualification.get("protocol_id") != E36_PROTOCOL
        or qualification.get("model_id") != E36_MODEL
        or qualification.get("model_revision") != E36_REVISION
        or qualification.get("real_cuda_qualified") is not True
        or qualification.get("single_model_instance") is not True
        or qualification.get("single_gpu_process") is not True
        or qualification.get("protected_outputs_accessed") is not False
        or relative != E36_CUDA_RUNTIME
        or dict(code) != _binding(root, E36_CUDA_RUNTIME)
    ):
        raise FourBIntegrityError("E36 real CUDA runtime is not qualified")
    runtime_path = _safe_path(root, E36_CUDA_RUNTIME)
    spec = importlib.util.spec_from_file_location("reasonforge_e36_cuda_runtime", runtime_path)
    if spec is None or spec.loader is None:
        raise FourBIntegrityError("E36 CUDA runtime module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime = getattr(module, "execute_e36_request", None)
    if not callable(runtime):
        raise FourBIntegrityError("E36 CUDA runtime does not expose execute_e36_request")
    return cast(RuntimeExecutor, runtime)


def publish_worker_cpu_qualification(
    root: Path, *, test_exit_code: int, test_log_relative: str
) -> dict[str, Any]:
    """Publish model-free worker qualification without authorizing real CUDA."""

    root = root.resolve()
    if test_exit_code != 0:
        raise FourBIntegrityError("E36 CUDA-worker synthetic tests did not pass")
    paths = (
        "scripts/e36_four_b_cuda_worker.py",
        "scripts/qualify_e36_cuda_worker.py",
        "src/reasonforge/orchestration/four_b_cuda_worker.py",
        "src/reasonforge/orchestration/four_b_transfer.py",
        "tests/test_four_b_cuda_worker.py",
    )
    artifact = {
        "schema_version": 1,
        "status": "QUALIFIED_CPU_SYNTHETIC",
        "experiment_id": "E36",
        "protocol_id": E36_PROTOCOL,
        "qualification_scope": "CPU_SYNTHETIC_WORKER_BOUNDARY_ONLY",
        "real_cuda_qualified": False,
        "cuda_used": False,
        "model_loaded": False,
        "protected_outputs_accessed": False,
        "test_exit_code": test_exit_code,
        "test_log": _binding(root, test_log_relative),
        "code_inventory": {relative: _binding(root, relative) for relative in paths},
        "qualified_properties": [
            "CANONICAL_INPUT_IDENTITIES",
            "C0_REFERENCE_BOUNDARY",
            "C1_CONVENTIONAL_BOUNDARY",
            "C2_RCAC_BOUNDARY",
            "AUTHORIZED_FALLBACK_MAPPING",
            "SINGLE_MODEL_INSTANCE_RESULT_GATE",
            "COMPLETED_STEP_RESUME_GATE",
            "ATOMIC_RESULT_PUBLICATION",
            "REAL_RUNTIME_FAIL_CLOSED",
        ],
        "remaining_gate": E36_CUDA_RUNTIME_QUALIFICATION,
    }
    destination = _safe_path(root, E36_WORKER_CPU_QUALIFICATION)
    atomic_write_json(destination, artifact)
    return artifact
