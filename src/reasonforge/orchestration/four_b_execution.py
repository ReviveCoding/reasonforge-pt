"""Fail-closed CPU orchestration for the frozen E36 single-GPU worker.

The orchestrator owns identities, ordering, durable attempt accounting, and restart
selection.  The separately qualified CUDA worker owns model loading and computation.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from reasonforge.checkpoints.hashing import atomic_write_json, sha256_file, sha256_json
from reasonforge.config import ExperimentStatus, file_sha256, load_registry
from reasonforge.systems import GpuOwnerLease

from .four_b_transfer import (
    E36_CONFIG,
    E36_MODEL,
    E36_PREFLIGHT,
    E36_PROTOCOL,
    E36_RECIPES,
    E36_REVISION,
    FourBIntegrityError,
    _binding,
    _config,
    _expected_run_id,
    _load_json,
    _mapping,
    _safe_path,
    _sequence,
    verify_e36_preflight,
    verify_e36_run_completion,
)

E36_CPU_EXECUTOR_QUALIFICATION = "evidence/training/E36/EXECUTOR_QUALIFICATION.json"
E36_CUDA_WORKER_QUALIFICATION = "evidence/training/E36/CUDA_WORKER_QUALIFICATION.json"
E36_CUDA_WORKER = "scripts/e36_four_b_cuda_worker.py"
E36_CUDA_WORKER_CPU_QUALIFICATION = "evidence/training/E36/CUDA_WORKER_CPU_QUALIFICATION.json"
E36_CUDA_RUNTIME_QUALIFICATION = "evidence/training/E36/CUDA_RUNTIME_QUALIFICATION.json"
E36_CUDA_RUNTIME = "src/reasonforge/orchestration/four_b_cuda_runtime.py"
E25_COMPLETION = (
    "evidence/protected_replication/RFPT-E25-PROTECTED-REPLICATION-v1/REPLICATION_COMPLETE.json"
)


class ProcessResult(Protocol):
    returncode: int
    stdout: str
    stderr: str


ProcessRunner = Callable[[Sequence[str], Path], ProcessResult]
LeaseFactory = Callable[[Path, str, str], Any]


def _run_process(command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), cwd=cwd, text=True, capture_output=True, check=False)


def _recipe_definition(config: Mapping[str, Any], recipe_id: str) -> Mapping[str, Any]:
    if recipe_id not in E36_RECIPES:
        raise FourBIntegrityError(f"E36 recipe is invalid: {recipe_id}")
    frozen = _mapping(config.get("frozen_recipe"), "frozen recipe")
    definitions = _mapping(frozen.get("definitions"), "recipe definitions")
    return _mapping(definitions[recipe_id], f"{recipe_id} recipe")


def _verify_cpu_qualification(root: Path) -> dict[str, str | int]:
    binding = _binding(root, E36_CPU_EXECUTOR_QUALIFICATION)
    value = _load_json(_safe_path(root, E36_CPU_EXECUTOR_QUALIFICATION), "executor qualification")
    expected_code = {
        "scripts/qualify_e36_executor.py",
        "scripts/run_e36_four_b_transfer.py",
        "src/reasonforge/orchestration/four_b_execution.py",
        "src/reasonforge/orchestration/four_b_transfer.py",
        "tests/test_four_b_execution.py",
    }
    code = _mapping(value.get("code_inventory"), "executor code inventory")
    if (
        value.get("schema_version") != 1
        or value.get("status") != "QUALIFIED"
        or value.get("experiment_id") != "E36"
        or value.get("protocol_id") != E36_PROTOCOL
        or value.get("cuda_used") is not False
        or value.get("model_loaded") is not False
        or value.get("test_exit_code") != 0
        or set(code) != expected_code
    ):
        raise FourBIntegrityError("E36 CPU executor qualification identity is invalid")
    for relative, digest in code.items():
        if digest != file_sha256(_safe_path(root, str(relative))):
            raise FourBIntegrityError(f"E36 executor code changed after qualification: {relative}")
    _verify_bound_artifact(root, value.get("test_log"), "executor qualification log")
    return binding


def _verify_bound_artifact(root: Path, raw: object, label: str) -> dict[str, str | int]:
    binding = _mapping(raw, label)
    relative = binding.get("path")
    if not isinstance(relative, str):
        raise FourBIntegrityError(f"E36 {label} path is invalid")
    observed = _binding(root, relative)
    if dict(binding) != observed:
        raise FourBIntegrityError(f"E36 {label} binding mismatch")
    return observed


def _verify_worker_qualification(root: Path, config_sha256: str) -> dict[str, str | int]:
    binding = _binding(root, E36_CUDA_WORKER_QUALIFICATION)
    value = _load_json(_safe_path(root, E36_CUDA_WORKER_QUALIFICATION), "CUDA worker qualification")
    worker = _verify_bound_artifact(root, value.get("worker_code"), "CUDA worker code")
    cpu_qualification = _verify_bound_artifact(
        root, value.get("worker_cpu_qualification"), "CUDA worker CPU qualification"
    )
    runtime_qualification = _verify_bound_artifact(
        root, value.get("runtime_qualification"), "CUDA runtime qualification"
    )
    runtime = _load_json(
        _safe_path(root, E36_CUDA_RUNTIME_QUALIFICATION), "CUDA runtime qualification"
    )
    runtime_code = _verify_bound_artifact(root, runtime.get("runtime_code"), "CUDA runtime code")
    if (
        value.get("schema_version") != 1
        or value.get("status") != "QUALIFIED"
        or value.get("qualification_scope") != "REAL_CUDA_QWEN3_4B_WORKER"
        or value.get("experiment_id") != "E36"
        or value.get("protocol_id") != E36_PROTOCOL
        or value.get("model_id") != E36_MODEL
        or value.get("model_revision") != E36_REVISION
        or value.get("config_sha256") != config_sha256
        or value.get("real_cuda_qualified") is not True
        or value.get("recipes") != list(E36_RECIPES)
        or value.get("completed_optimizer_step_only") is not True
        or value.get("partial_rollout_groups_excluded") is not True
        or value.get("single_model_instance") is not True
        or value.get("single_gpu_process") is not True
        or value.get("protected_outputs_accessed") is not False
        or worker.get("path") != E36_CUDA_WORKER
        or cpu_qualification.get("path") != E36_CUDA_WORKER_CPU_QUALIFICATION
        or runtime_qualification.get("path") != E36_CUDA_RUNTIME_QUALIFICATION
        or runtime.get("status") != "QUALIFIED"
        or runtime.get("qualification_scope") != "REAL_CUDA_QWEN3_4B"
        or runtime.get("real_cuda_qualified") is not True
        or runtime.get("model_id") != E36_MODEL
        or runtime.get("model_revision") != E36_REVISION
        or runtime.get("single_model_instance") is not True
        or runtime.get("single_gpu_process") is not True
        or runtime.get("protected_outputs_accessed") is not False
        or runtime_code.get("path") != E36_CUDA_RUNTIME
    ):
        raise FourBIntegrityError("E36 CUDA worker qualification identity is invalid")
    return binding


def _verify_e25_dependency(root: Path) -> dict[str, str | int]:
    registry = load_registry(root / "configs/EXPERIMENT_REGISTRY.yaml")
    matches = [item for item in registry.experiments if item.experiment_id == "E25"]
    if len(matches) != 1 or matches[0].status is not ExperimentStatus.COMPLETE:
        raise FourBIntegrityError("E36 dependency E25 is not registry-COMPLETE")
    value = _load_json(_safe_path(root, E25_COMPLETION), "E25 completion")
    if (
        value.get("schema_version") != 1
        or value.get("status") != "COMPLETE"
        or value.get("experiment_id") != "E25"
        or value.get("protocol_id") != "RFPT-E25-PROTECTED-REPLICATION-v1"
    ):
        raise FourBIntegrityError("E36 dependency E25 completion identity is invalid")
    return _binding(root, E25_COMPLETION)


def _run_directory(root: Path, recipe_id: str) -> Path:
    return root / "artifacts" / "training" / "E36" / _expected_run_id(recipe_id)


def _latest_complete_checkpoint(run_dir: Path, run_identity_hash: str) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for marker in run_dir.glob("checkpoint-*/CHECKPOINT_COMPLETE.json"):
        value = _load_json(marker, "runtime checkpoint completion")
        step = value.get("checkpoint_step")
        if (
            value.get("status") != "COMPLETE"
            or not isinstance(step, int)
            or step < 0
            or value.get("run_identity_hash") != run_identity_hash
            or marker.parent.name != f"checkpoint-{step}"
        ):
            continue
        candidates.append((step, marker.parent))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _completed_fallbacks(root: Path, run_dir: Path, ladder: Sequence[Any]) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    for index, step in enumerate(ladder):
        path = run_dir / "attempts" / f"FALLBACK_{index:02d}.json"
        if not path.is_file():
            break
        value = dict(_load_json(path, f"fallback {index}"))
        if (
            value.get("status") != "FAILED"
            or value.get("failure_type") != "CUDA_OOM"
            or value.get("attempt_index") != index
            or value.get("fallback_step") != step
        ):
            raise FourBIntegrityError(f"E36 fallback {index} is not a valid CUDA OOM record")
        value["binding"] = _binding(root, path.relative_to(root).as_posix())
        completed.append(value)
    extra = (
        list((run_dir / "attempts").glob("FALLBACK_*.json"))
        if (run_dir / "attempts").is_dir()
        else []
    )
    if len(extra) != len(completed):
        raise FourBIntegrityError("E36 fallback evidence is non-contiguous")
    return completed


def plan_e36_recipe(
    root: Path,
    *,
    config_path: Path,
    recipe_id: str,
    resume: bool = False,
) -> dict[str, Any]:
    """Build a non-mutating E36 launch/resume plan."""

    root = root.resolve()
    canonical_config = _safe_path(root, E36_CONFIG)
    if config_path.resolve() != canonical_config:
        raise FourBIntegrityError("E36 executor accepts only the canonical frozen config path")
    config, config_sha256 = _config(root)
    definition = _recipe_definition(config, recipe_id)
    reasons: list[str] = []
    bindings: dict[str, Any] = {
        "configuration": _binding(root, E36_CONFIG),
        "preflight": _binding(root, E36_PREFLIGHT),
    }
    try:
        preflight = verify_e36_preflight(root)
        if preflight.get("status") != "READY_FOR_BOUNDED_CUDA_QUALIFICATION":
            reasons.append(f"preflight status is {preflight.get('status')}")
        if (
            _mapping(preflight.get("configuration"), "preflight configuration").get("sha256")
            != config_sha256
        ):
            reasons.append("preflight does not bind the current frozen configuration")
    except (OSError, FourBIntegrityError) as error:
        reasons.append(str(error))
    try:
        bindings["cpu_executor_qualification"] = _verify_cpu_qualification(root)
    except (OSError, FourBIntegrityError) as error:
        reasons.append(str(error))
    try:
        bindings["cuda_worker_qualification"] = _verify_worker_qualification(root, config_sha256)
    except (OSError, FourBIntegrityError) as error:
        reasons.append(str(error))
    try:
        bindings["e25_completion"] = _verify_e25_dependency(root)
    except (OSError, FourBIntegrityError, ValueError) as error:
        reasons.append(str(error))

    recipe_index = E36_RECIPES.index(recipe_id)
    prerequisite_runs: dict[str, Any] = {}
    for predecessor in E36_RECIPES[:recipe_index]:
        try:
            prerequisite_runs[predecessor] = verify_e36_run_completion(root, predecessor)
        except (OSError, FourBIntegrityError) as error:
            reasons.append(f"{recipe_id} predecessor {predecessor} is incomplete: {error}")
    bindings["prerequisite_runs"] = prerequisite_runs

    run_id = str(definition["run_id"])
    output_dir = _run_directory(root, recipe_id)
    already_complete = False
    try:
        bindings["existing_completion"] = verify_e36_run_completion(root, recipe_id)
        already_complete = True
    except (OSError, FourBIntegrityError):
        pass
    identity = {
        "schema_version": 1,
        "experiment_id": "E36",
        "protocol_id": E36_PROTOCOL,
        "scientific_state": "LOCKED_FINAL",
        "recipe_id": recipe_id,
        "run_id": run_id,
        "model_id": E36_MODEL,
        "model_revision": E36_REVISION,
        "configuration": bindings["configuration"],
        "preflight": bindings["preflight"],
        "e25_completion": bindings.get("e25_completion"),
        "cpu_executor_qualification": bindings.get("cpu_executor_qualification"),
        "cuda_worker_qualification": bindings.get("cuda_worker_qualification"),
        "authorized_fallback_ladder": list(
            _sequence(config.get("authorized_fallback_ladder"), "fallback ladder")
        ),
        "completed_optimizer_step_only": True,
        "partial_rollout_policy": "PRESERVE_EXCLUDE_AND_DISCARD",
        "scientific_recipe_preserved": True,
        "substitute_experiment_run": False,
    }
    identity_hash = sha256_json(identity)
    latest_checkpoint = _latest_complete_checkpoint(output_dir, identity_hash)
    populated = output_dir.is_dir() and any(output_dir.iterdir())
    if already_complete:
        action = "ALREADY_COMPLETE"
    else:
        action = "RESUME" if resume else "START"
        if populated and not resume:
            reasons.append("incomplete E36 output exists; explicit --resume is required")
        if resume and not populated:
            reasons.append("--resume requested but no incomplete E36 output exists")
        if resume and (output_dir / "RUN_IDENTITY.json").is_file():
            prior = _load_json(output_dir / "RUN_IDENTITY.json", "prior run identity")
            if prior != identity:
                reasons.append("existing E36 run identity is incompatible with resume")
    return {
        **identity,
        "run_identity_hash": identity_hash,
        "action": action,
        "resume_requested": resume,
        "latest_complete_checkpoint": (
            latest_checkpoint.relative_to(root).as_posix() if latest_checkpoint else None
        ),
        "single_gpu_lease_required": True,
        "one_worker_process_at_a_time": True,
        "bindings": bindings,
        "launch_authorized": already_complete or not reasons,
        "blocking_reasons": reasons,
        "gpu_touched": False,
        "model_loaded": False,
        "protected_outputs_accessed": False,
    }


def _worker_command(
    root: Path,
    plan: Mapping[str, Any],
    *,
    fallback_step: str,
    result_path: Path,
    resume_checkpoint: str | None,
) -> list[str]:
    command = [
        sys.executable,
        str(_safe_path(root, E36_CUDA_WORKER)),
        "--workspace",
        str(root),
        "--config",
        str(_safe_path(root, E36_CONFIG)),
        "--recipe",
        str(plan["recipe_id"]),
        "--run-id",
        str(plan["run_id"]),
        "--run-identity-hash",
        str(plan["run_identity_hash"]),
        "--fallback-step",
        fallback_step,
        "--result-path",
        str(result_path),
    ]
    if resume_checkpoint is not None:
        command.extend(("--resume-from", str(_safe_path(root, resume_checkpoint))))
    return command


def _validate_worker_result(
    root: Path, value: Mapping[str, Any], plan: Mapping[str, Any], fallback_step: str
) -> None:
    if (
        value.get("schema_version") != 1
        or value.get("experiment_id") != "E36"
        or value.get("protocol_id") != E36_PROTOCOL
        or value.get("recipe_id") != plan.get("recipe_id")
        or value.get("run_id") != plan.get("run_id")
        or value.get("run_identity_hash") != plan.get("run_identity_hash")
        or value.get("model_id") != E36_MODEL
        or value.get("model_revision") != E36_REVISION
        or value.get("fallback_step") != fallback_step
        or value.get("cuda_process_launched") is not True
        or value.get("cuda_available") is not True
        or value.get("cuda_device_name") != "NVIDIA GeForce RTX 4090 Laptop GPU"
        or value.get("model_instances_created") != 1
        or value.get("child_model_processes_created") != 0
        or not isinstance(value.get("peak_allocated_bytes"), int)
        or int(value["peak_allocated_bytes"]) <= 0
        or value.get("scientific_recipe_preserved") is not True
        or value.get("substitute_experiment_run") is not False
        or value.get("completed_optimizer_step_only") is not True
        or value.get("partial_rollout_groups_excluded") is not True
    ):
        raise FourBIntegrityError("E36 CUDA worker result identity is invalid")
    if value.get("status") == "COMPLETE" and plan.get("recipe_id") != "C0":
        checkpoint = _verify_bound_artifact(
            root, value.get("final_checkpoint_completion"), "final checkpoint completion"
        )
        marker = _load_json(
            _safe_path(root, str(checkpoint["path"])), "final checkpoint completion"
        )
        step = marker.get("checkpoint_step")
        if (
            marker.get("status") != "COMPLETE"
            or marker.get("run_identity_hash") != plan.get("run_identity_hash")
            or not isinstance(step, int)
            or step <= 0
            or Path(str(checkpoint["path"])).parent.name != f"checkpoint-{step}"
        ):
            raise FourBIntegrityError("E36 final checkpoint is not a compatible completed step")
    elif value.get("status") == "COMPLETE" and value.get("final_checkpoint_completion") is not None:
        raise FourBIntegrityError("E36 C0 reference must not claim a training checkpoint")


def _publish_cuda_failure(
    root: Path,
    plan: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    preflight = _binding(root, E36_PREFLIGHT)
    artifact = {
        "schema_version": 1,
        "status": "BLOCKED_LOCAL_COMPUTE",
        "experiment_id": "E36",
        "protocol_id": E36_PROTOCOL,
        "model_id": E36_MODEL,
        "model_revision": E36_REVISION,
        "affected_recipes": list(E36_RECIPES),
        "failure_mode": "CUDA_OOM",
        "scientific_recipe_preserved": True,
        "substitute_experiment_run": False,
        "authorized_fallback_ladder_exhausted": True,
        "preflight": preflight,
        "source_recipe_id": plan["recipe_id"],
        "source_run_id": plan["run_id"],
        "attempts": [
            {"fallback_step": item["fallback_step"], "evidence": item["binding"]}
            for item in attempts
        ],
    }
    destination = root / "evidence/training/E36/CUDA_QUALIFICATION_FAILURE.json"
    if destination.exists() and _load_json(destination, "prior CUDA failure") != artifact:
        raise FourBIntegrityError("refusing to overwrite different E36 CUDA failure evidence")
    atomic_write_json(destination, artifact)
    return artifact


def execute_e36_recipe(
    root: Path,
    *,
    config_path: Path,
    recipe_id: str,
    resume: bool = False,
    process_runner: ProcessRunner = _run_process,
    lease_factory: LeaseFactory = GpuOwnerLease,
) -> dict[str, Any]:
    """Execute one recipe through the qualified worker, retrying only ordered OOMs."""

    root = root.resolve()
    plan = plan_e36_recipe(root, config_path=config_path, recipe_id=recipe_id, resume=resume)
    if plan["action"] == "ALREADY_COMPLETE":
        return {**plan, "status": "COMPLETE", "reused_existing_completion": True}
    if not plan["launch_authorized"]:
        raise FourBIntegrityError(f"E36 launch is blocked: {plan['blocking_reasons']}")
    run_dir = _run_directory(root, recipe_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    identity_path = run_dir / "RUN_IDENTITY.json"
    if identity_path.exists() and _load_json(identity_path, "run identity") != {
        key: plan[key]
        for key in (
            "schema_version",
            "experiment_id",
            "protocol_id",
            "scientific_state",
            "recipe_id",
            "run_id",
            "model_id",
            "model_revision",
            "configuration",
            "preflight",
            "e25_completion",
            "cpu_executor_qualification",
            "cuda_worker_qualification",
            "authorized_fallback_ladder",
            "completed_optimizer_step_only",
            "partial_rollout_policy",
            "scientific_recipe_preserved",
            "substitute_experiment_run",
        )
    }:
        raise FourBIntegrityError("E36 existing RUN_IDENTITY is incompatible")
    identity = {
        key: plan[key]
        for key in (
            "schema_version",
            "experiment_id",
            "protocol_id",
            "scientific_state",
            "recipe_id",
            "run_id",
            "model_id",
            "model_revision",
            "configuration",
            "preflight",
            "e25_completion",
            "cpu_executor_qualification",
            "cuda_worker_qualification",
            "authorized_fallback_ladder",
            "completed_optimizer_step_only",
            "partial_rollout_policy",
            "scientific_recipe_preserved",
            "substitute_experiment_run",
        )
    }
    if not identity_path.exists():
        atomic_write_json(identity_path, identity)
    atomic_write_json(run_dir / "PRELAUNCH.json", plan)

    ladder = _sequence(plan["authorized_fallback_ladder"], "fallback ladder")
    completed = _completed_fallbacks(root, run_dir, ladder)
    attempts_dir = run_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    with lease_factory(root, "E36", str(plan["run_id"])):
        for index in range(len(completed), len(ladder)):
            fallback_step = str(ladder[index])
            prior_instances = list(attempts_dir.glob(f"{index:02d}_*/IN_PROGRESS.json"))
            for prior in prior_instances:
                if not (prior.parent / "TERMINAL.json").is_file():
                    recovery = prior.parent / "INTERRUPTION_RECOVERY.json"
                    if not recovery.exists():
                        atomic_write_json(
                            recovery,
                            {
                                "status": "PARTIAL_ATTEMPT_PRESERVED_FOR_RESUME",
                                "experiment_id": "E36",
                                "protocol_id": E36_PROTOCOL,
                                "recipe_id": recipe_id,
                                "fallback_step": fallback_step,
                                "prior_in_progress": _binding(
                                    root, prior.relative_to(root).as_posix()
                                ),
                                "partial_gradient_accounting": "EXCLUDED",
                            },
                        )
            instance_index = len(prior_instances)
            instance = attempts_dir / f"{index:02d}_{instance_index:02d}"
            instance.mkdir(parents=True)
            in_progress = {
                "status": "IN_PROGRESS",
                "experiment_id": "E36",
                "protocol_id": E36_PROTOCOL,
                "recipe_id": recipe_id,
                "run_id": plan["run_id"],
                "attempt_index": index,
                "fallback_step": fallback_step,
                "run_identity_hash": plan["run_identity_hash"],
            }
            atomic_write_json(instance / "IN_PROGRESS.json", in_progress)
            result_path = instance / "WORKER_RESULT.json"
            latest_checkpoint = _latest_complete_checkpoint(run_dir, str(plan["run_identity_hash"]))
            resume_checkpoint = (
                latest_checkpoint.relative_to(root).as_posix() if latest_checkpoint else None
            )
            command = _worker_command(
                root,
                plan,
                fallback_step=fallback_step,
                result_path=result_path,
                resume_checkpoint=str(resume_checkpoint) if resume_checkpoint else None,
            )
            process = process_runner(command, root)
            log_path = instance / "worker.log"
            log_path.write_text(
                f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}\n",
                encoding="utf-8",
            )
            if not result_path.is_file():
                atomic_write_json(
                    instance / "PROCESS_INTERRUPTED.json",
                    {**in_progress, "status": "INTERRUPTED", "returncode": process.returncode},
                )
                raise FourBIntegrityError(
                    "E36 worker ended without durable result; preserve and use --resume"
                )
            result = _load_json(result_path, "CUDA worker result")
            _validate_worker_result(root, result, plan, fallback_step)
            if process.returncode == 0 and result.get("status") == "COMPLETE":
                atomic_write_json(
                    instance / "TERMINAL.json",
                    {**in_progress, "status": "COMPLETE", "returncode": process.returncode},
                )
                completion = {
                    "schema_version": 1,
                    "status": "COMPLETE",
                    "experiment_id": "E36",
                    "protocol_id": E36_PROTOCOL,
                    "recipe_id": recipe_id,
                    "run_id": plan["run_id"],
                    "model_id": E36_MODEL,
                    "model_revision": E36_REVISION,
                    "run_identity_hash": plan["run_identity_hash"],
                    "worker_result": _binding(root, result_path.relative_to(root).as_posix()),
                    "final_checkpoint_completion": result.get("final_checkpoint_completion"),
                    "fallback_step": fallback_step,
                    "resume_from_checkpoint": resume_checkpoint,
                    "scientific_recipe_preserved": True,
                    "substitute_experiment_run": False,
                }
                atomic_write_json(run_dir / "RUN_COMPLETE.json", completion)
                return completion
            if process.returncode == 0 or result.get("status") != "FAILED":
                raise FourBIntegrityError("E36 worker exit status and result disagree")
            if result.get("failure_type") != "CUDA_OOM":
                raise FourBIntegrityError(
                    "E36 non-OOM worker failure is not fallback-authorized: "
                    f"{result.get('failure_type')}"
                )
            fallback = {
                **dict(result),
                "status": "FAILED",
                "failure_observed": True,
                "attempt_index": index,
                "diagnostic_log": _binding(root, log_path.relative_to(root).as_posix()),
            }
            fallback_path = attempts_dir / f"FALLBACK_{index:02d}.json"
            atomic_write_json(fallback_path, fallback)
            atomic_write_json(
                instance / "TERMINAL.json",
                {**in_progress, "status": "FAILED_CUDA_OOM", "returncode": process.returncode},
            )
            completed.append(
                {
                    "fallback_step": fallback_step,
                    "binding": _binding(root, fallback_path.relative_to(root).as_posix()),
                }
            )
    failure = _publish_cuda_failure(root, plan, completed)
    return {"status": "BLOCKED_LOCAL_COMPUTE", "failure_evidence": failure}


def publish_executor_qualification(
    root: Path, *, test_exit_code: int, test_log_relative: str
) -> dict[str, Any]:
    """Publish CPU qualification only after the synthetic executor tests actually pass."""

    root = root.resolve()
    if test_exit_code != 0:
        raise FourBIntegrityError("E36 executor tests did not pass")
    paths = (
        "scripts/qualify_e36_executor.py",
        "scripts/run_e36_four_b_transfer.py",
        "src/reasonforge/orchestration/four_b_execution.py",
        "src/reasonforge/orchestration/four_b_transfer.py",
        "tests/test_four_b_execution.py",
    )
    artifact = {
        "schema_version": 1,
        "status": "QUALIFIED",
        "experiment_id": "E36",
        "protocol_id": E36_PROTOCOL,
        "qualification_scope": "CPU_ORCHESTRATION_ONLY",
        "cuda_worker_qualified": False,
        "cuda_used": False,
        "model_loaded": False,
        "test_exit_code": test_exit_code,
        "test_log": _binding(root, test_log_relative),
        "code_inventory": {relative: sha256_file(root / relative) for relative in paths},
        "qualified_properties": [
            "FAIL_CLOSED_IDENTITY",
            "SEQUENTIAL_RECIPE_ORDER",
            "SINGLE_GPU_LEASE",
            "ORDERED_OOM_FALLBACK",
            "COMPLETED_STEP_RESUME_SELECTION",
            "INTERRUPTED_ATTEMPT_PRESERVATION",
            "DRY_RUN_NON_MUTATION",
        ],
    }
    destination = _safe_path(root, E36_CPU_EXECUTOR_QUALIFICATION)
    atomic_write_json(destination, artifact)
    return artifact
