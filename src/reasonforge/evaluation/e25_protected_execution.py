"""Fail-closed generation and scoring for the E25 additive protected matrix."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from reasonforge.checkpoints.hashing import atomic_write_json, canonical_json_bytes, sha256_file
from reasonforge.checkpoints.models import utc_now
from reasonforge.checkpoints.project import verify_latest

from .e25_preprotected import verify_e25_preprotected_binding
from .e25_protected_replication import (
    E25_PLAN_COUNT,
    verify_e25_plan_is_bound,
    verify_e25_run_plan_matrix,
)
from .protected_orchestration_v2 import verify_recovery_amendment
from .protected_outputs import (
    ProtectedOutputIntegrityError,
    ProtectedOutputRecord,
    ProtectedOutputVerification,
    ProtectedOutputWriter,
    verify_deterministic_rescore,
)
from .protected_outputs_v2 import JournaledProtectedOutputWriter
from .protected_protocol import (
    FrozenProtectedRunPlan,
    PredictionFunction,
    ProtectedBenchmarkSpec,
    load_benchmark_specs,
    load_run_plan,
    score_protected_outputs,
    verify_run_plan,
)
from .protected_recovery_v2 import (
    ProtectedRecoveryConfig,
    score_protected_outputs_v2,
)

ScorerRoute = Literal["frozen_v1", "recovery_v2"]
_ROUTES: dict[str, ScorerRoute] = {
    "ifeval": "recovery_v2",
    "mmlu_pro_test_256": "recovery_v2",
    "reasonforge_authorization": "recovery_v2",
    "reasonforge_instruction": "frozen_v1",
    "reasonforge_reasoning": "frozen_v1",
    "reasonforge_tool": "frozen_v1",
    "truthfulqa_mc": "frozen_v1",
}
_PENDING_WRITER_FILES = {
    "WRITER_V2_PENDING_APPEND.json",
    "WRITER_V2_PENDING_COMPLETE.json",
}


class E25ProtectedExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    protocol_id: Literal["RFPT-E25-PROTECTED-REPLICATION-v1"]
    scoring_protocol_id: Literal["RFPT-E25-PROTECTED-SCORING-v1"]
    binding: Path
    binding_config: Path
    matrix: Path
    benchmark_registry: Path
    generation_config: Path
    recovery_amendment: Path
    recovery_scorer_config: Path
    ifeval_backend_config: Path
    ifeval_backend_qualification: Path
    output_root: Path
    scoring_lock: Path
    expected_models: Literal[4]
    expected_benchmarks: Literal[7]
    expected_output_sets: Literal[28]
    shard_max_records: int = Field(gt=0, le=128)
    rescore_id: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    routes: dict[str, ScorerRoute]
    latest_checkpoint_required_before_inference: Literal["CP70_PROTECTED_COMPLETE"]
    seed1701_outputs: Literal["REUSE_IMMUTABLE_DO_NOT_REGENERATE"]
    require_all_outputs_complete_before_scoring: Literal[True]
    refuse_existing_rescore_state: Literal[True]
    semantic_output_logging: Literal[False]
    score_values_logging: Literal[False]

    @model_validator(mode="after")
    def validate_routes(self) -> E25ProtectedExecutionConfig:
        if self.routes != _ROUTES:
            raise ValueError("E25 scorer routes differ from the CP60/recovery-v2 map")
        return self


@dataclass(frozen=True, slots=True)
class E25ScoringTarget:
    output_set_id: str
    plan: FrozenProtectedRunPlan
    spec: ProtectedBenchmarkSpec
    route: ScorerRoute
    output_directory: Path
    output_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class E25ScoringPreflight:
    protocol_id: str
    matrix_sha256: str
    binding_sha256: str
    routes: dict[str, int]
    targets: tuple[E25ScoringTarget, ...]

    def public_summary(self) -> dict[str, Any]:
        return {
            "status": "DRY_RUN_READY",
            "protocol_id": self.protocol_id,
            "matrix_sha256": self.matrix_sha256,
            "binding_sha256": self.binding_sha256,
            "target_count": len(self.targets),
            "route_counts": dict(sorted(self.routes.items())),
            "output_set_ids": [item.output_set_id for item in self.targets],
            "all_complete_and_verified": True,
            "existing_rescore_count": 0,
            "semantic_outputs_logged": False,
            "score_values_logged": False,
            "scores_computed": False,
        }


def load_e25_execution_config(path: Path) -> E25ProtectedExecutionConfig:
    return E25ProtectedExecutionConfig.model_validate(yaml.safe_load(path.read_text("utf-8")))


def _root_path(root: Path, path: Path, label: str) -> Path:
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ProtectedOutputIntegrityError(f"E25 {label} escapes workspace") from error
    return candidate


def _require_cp70(root: Path, config: E25ProtectedExecutionConfig) -> None:
    latest = verify_latest(root / "evidence/checkpoints")
    manifest = latest.get("manifest")
    checkpoint_id = manifest.get("checkpoint_id") if isinstance(manifest, Mapping) else None
    if checkpoint_id != config.latest_checkpoint_required_before_inference:
        raise ProtectedOutputIntegrityError(
            f"E25 protected inference requires latest VERIFIED CP70; observed {checkpoint_id}"
        )


def _verify_authorization(
    root: Path, config_path: Path, config: E25ProtectedExecutionConfig
) -> dict[str, Any]:
    _require_cp70(root, config)
    binding = verify_e25_preprotected_binding(
        root,
        _root_path(root, config.binding_config, "binding config"),
        _root_path(root, config.binding, "binding"),
    )
    matrix_path = _root_path(root, config.matrix, "matrix")
    matrix_verification = verify_e25_run_plan_matrix(root, matrix_path)
    inventory = binding.get("run_plan_inventory")
    matrix_binding = inventory.get("matrix") if isinstance(inventory, Mapping) else None
    if (
        not isinstance(matrix_binding, Mapping)
        or matrix_binding.get("path") != config.matrix.as_posix()
        or matrix_binding.get("sha256") != matrix_verification["matrix_sha256"]
    ):
        raise ProtectedOutputIntegrityError("E25 binding does not authorize the exact run matrix")
    execution_binding = binding.get("execution_config")
    if (
        not isinstance(execution_binding, Mapping)
        or execution_binding.get("path") != config_path.relative_to(root).as_posix()
        or execution_binding.get("sha256") != sha256_file(config_path)
    ):
        raise ProtectedOutputIntegrityError("E25 binding does not authorize execution code/config")
    return binding


def preflight_e25_generation(
    workspace: Path | str,
    config_path: Path | str,
    plan_path: Path | str,
) -> tuple[E25ProtectedExecutionConfig, FrozenProtectedRunPlan, ProtectedBenchmarkSpec]:
    """Identity-only preflight; does not load protected rows or a model."""

    root = Path(workspace).resolve()
    resolved_config = _root_path(root, Path(config_path), "execution config")
    config = load_e25_execution_config(resolved_config)
    _verify_authorization(root, resolved_config, config)
    plan, _ = verify_e25_plan_is_bound(root, config.matrix, plan_path)
    specs = load_benchmark_specs(_root_path(root, config.benchmark_registry, "registry"))
    spec = specs.get(plan.benchmark_name)
    if spec is None or plan.model_id.startswith(("e16_", "e18_")):
        raise ProtectedOutputIntegrityError("E25 plan is not an authorized new-seed target")
    return config, plan, spec


def run_e25_protected_predictions(
    *,
    workspace: Path | str,
    config_path: Path | str,
    plan_path: Path | str,
    predict: PredictionFunction,
) -> ProtectedOutputVerification:
    """Generate or resume only the exact authorized new-seed E25 output set."""

    root = Path(workspace).resolve()
    config, plan, spec = preflight_e25_generation(root, config_path, plan_path)
    verify_recovery_amendment(_root_path(root, config.recovery_amendment, "recovery amendment"))
    examples = verify_run_plan(plan, spec)
    output_root = _root_path(root, config.output_root, "output root")
    legacy_root = (root / "evidence/protected_outputs").resolve()
    if output_root == legacy_root or legacy_root in output_root.parents:
        raise ProtectedOutputIntegrityError("E25 outputs must not share the seed-1701 namespace")
    destination = output_root / plan.model_id / plan.output_set_id
    if destination.exists():
        output_plan = destination / "RUN_PLAN.json"
        if (
            not output_plan.is_file()
            or json.loads(output_plan.read_text("utf-8")) != plan.to_dict()
        ):
            raise ProtectedOutputIntegrityError(
                "existing E25 output plan differs from authorization"
            )
        writer = JournaledProtectedOutputWriter.open(destination)
        verification = writer.verify()
        if verification.manifest.identity != plan.identity:
            raise ProtectedOutputIntegrityError("existing E25 output identity differs from plan")
    else:
        base = ProtectedOutputWriter.create(
            output_root,
            plan.identity,
            plan.expected_example_ids,
            shard_max_records=config.shard_max_records,
        )
        atomic_write_json(base.output_directory / "RUN_PLAN.json", plan.to_dict())
        writer = JournaledProtectedOutputWriter.adopt(
            base.output_directory,
            run_plan_sha256=sha256_file(base.output_directory / "RUN_PLAN.json"),
        )
    missing = set(writer.verify().missing_example_ids)
    by_id = {item.example_id: item for item in examples}
    for example_id in plan.expected_example_ids:
        if example_id not in missing:
            continue
        output_text, metadata = predict(by_id[example_id], spec.generation_mode)
        writer.append(
            ProtectedOutputRecord(
                output_set_id=plan.output_set_id,
                protocol_id=plan.protocol_id,
                model_id=plan.model_id,
                model_checkpoint_sha256=plan.model_checkpoint_sha256,
                generation_config_id=str(plan.generation_config["generation_config_id"]),
                generation_config_sha256=plan.generation_config_sha256,
                example_id=example_id,
                benchmark_id=spec.benchmark_id,
                benchmark_revision=spec.revision,
                prompt_template_id=spec.prompt_template_id,
                output_text=output_text,
                finish_reason=str(metadata.pop("finish_reason", "stop")),
                metadata={
                    "example_identity_sha256": plan.example_identity_sha256[example_id],
                    "generation_mode": spec.generation_mode,
                    **metadata,
                },
            )
        )
    return writer.complete()


def _verify_writer(output_directory: Path, plan_path: Path) -> None:
    pending = [name for name in _PENDING_WRITER_FILES if (output_directory / name).exists()]
    if pending:
        raise ProtectedOutputIntegrityError(f"E25 writer has pending state: {pending}")
    adoption = json.loads((output_directory / "WRITER_V2_ADOPTION.json").read_text("utf-8"))
    if (
        not isinstance(adoption, Mapping)
        or adoption.get("status") != "ADOPTED"
        or adoption.get("writer_protocol_id") != "RFPT-PROTECTED-WRITER-v2"
        or adoption.get("run_plan_sha256") != sha256_file(plan_path)
    ):
        raise ProtectedOutputIntegrityError("E25 writer-v2 adoption identity mismatch")


def preflight_e25_scoring(
    workspace: Path | str,
    config_path: Path | str,
) -> E25ScoringPreflight:
    """Require all 28 immutable outputs before any E25 score is published."""

    root = Path(workspace).resolve()
    resolved_config = _root_path(root, Path(config_path), "execution config")
    config = load_e25_execution_config(resolved_config)
    _verify_authorization(root, resolved_config, config)
    verify_recovery_amendment(_root_path(root, config.recovery_amendment, "recovery amendment"))
    matrix_path = _root_path(root, config.matrix, "matrix")
    matrix = json.loads(matrix_path.read_text("utf-8"))
    specs = load_benchmark_specs(_root_path(root, config.benchmark_registry, "registry"))
    output_root = _root_path(root, config.output_root, "output root")
    expected_directories = {
        (str(entry["model_id"]), str(output_set_id))
        for output_set_id, entry in matrix["plans"].items()
    }
    discovered_directories = {
        (model_dir.name, output_dir.name)
        for model_dir in output_root.iterdir()
        if model_dir.is_dir()
        for output_dir in model_dir.iterdir()
        if output_dir.is_dir()
    }
    unexpected_files = [
        path
        for path in output_root.rglob("*")
        if path.is_file()
        and path.parent == output_root
        or (path.is_file() and path.parent.parent == output_root)
    ]
    if discovered_directories != expected_directories or unexpected_files:
        raise ProtectedOutputIntegrityError("E25 protected output namespace is not exact")
    targets: list[E25ScoringTarget] = []
    route_counts = {"frozen_v1": 0, "recovery_v2": 0}
    for output_set_id, entry in sorted(matrix["plans"].items()):
        plan_path = _root_path(root, Path(str(entry["path"])), "plan")
        if sha256_file(plan_path) != entry["sha256"]:
            raise ProtectedOutputIntegrityError(f"E25 plan hash mismatch: {output_set_id}")
        plan = load_run_plan(plan_path)
        if (
            plan.output_set_id != output_set_id
            or plan.model_id != entry["model_id"]
            or plan.protocol_id != config.protocol_id
        ):
            raise ProtectedOutputIntegrityError(f"E25 plan identity mismatch: {output_set_id}")
        spec = specs[plan.benchmark_name]
        verify_run_plan(plan, spec)
        output_directory = output_root / plan.model_id / output_set_id
        verification = ProtectedOutputWriter(output_directory).verify(require_complete=True)
        if verification.manifest.identity != plan.identity:
            raise ProtectedOutputIntegrityError(f"E25 output identity mismatch: {output_set_id}")
        if not (output_directory / "RUN_PLAN.json").is_file() or sha256_file(
            output_directory / "RUN_PLAN.json"
        ) != sha256_file(plan_path):
            raise ProtectedOutputIntegrityError(f"E25 output plan mismatch: {output_set_id}")
        _verify_writer(output_directory, plan_path)
        rescore_root = output_directory / "rescores"
        if rescore_root.exists():
            raise ProtectedOutputIntegrityError(
                f"prior or partial E25 rescore state exists: {output_set_id}"
            )
        route = config.routes[plan.benchmark_name]
        route_counts[route] += 1
        targets.append(
            E25ScoringTarget(
                output_set_id=output_set_id,
                plan=plan,
                spec=spec,
                route=route,
                output_directory=output_directory,
                output_manifest_sha256=sha256_file(output_directory / "manifest.json"),
            )
        )
    if len(targets) != E25_PLAN_COUNT:
        raise ProtectedOutputIntegrityError("E25 scoring matrix does not contain 28 targets")
    return E25ScoringPreflight(
        protocol_id=config.scoring_protocol_id,
        matrix_sha256=sha256_file(matrix_path),
        binding_sha256=sha256_file(_root_path(root, config.binding, "binding")),
        routes=route_counts,
        targets=tuple(targets),
    )


V1Scorer = Callable[..., Path]
V2Scorer = Callable[..., Path]


def execute_e25_scoring(
    *,
    preflight: E25ScoringPreflight,
    config: E25ProtectedExecutionConfig,
    recovery: ProtectedRecoveryConfig,
    lock_path: Path,
    v1_scorer: V1Scorer = score_protected_outputs,
    v2_scorer: V2Scorer = score_protected_outputs_v2,
) -> dict[str, Any]:
    """Publish one deterministic score namespace per E25 output, or fail closed."""

    if len(preflight.targets) != config.expected_output_sets:
        raise ProtectedOutputIntegrityError("E25 scoring preflight target count changed")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise ProtectedOutputIntegrityError("E25 protected scoring lock already exists") from error
    completed: list[dict[str, str]] = []
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                canonical_json_bytes(
                    {
                        "schema_version": 1,
                        "status": "IN_PROGRESS",
                        "protocol_id": preflight.protocol_id,
                        "matrix_sha256": preflight.matrix_sha256,
                        "binding_sha256": preflight.binding_sha256,
                        "target_count": len(preflight.targets),
                        "started_at": utc_now(),
                    }
                )
            )
            handle.flush()
            os.fsync(handle.fileno())
        for target in preflight.targets:
            if (target.output_directory / "rescores").exists():
                raise ProtectedOutputIntegrityError(
                    f"E25 rescore state appeared after preflight: {target.output_set_id}"
                )
            verification = ProtectedOutputWriter(target.output_directory).verify(
                require_complete=True
            )
            if (
                verification.manifest.identity != target.plan.identity
                or sha256_file(target.output_directory / "manifest.json")
                != target.output_manifest_sha256
            ):
                raise ProtectedOutputIntegrityError(
                    f"E25 output changed after preflight: {target.output_set_id}"
                )
        for target in preflight.targets:
            scorer = v2_scorer if target.route == "recovery_v2" else v1_scorer
            kwargs: dict[str, Any] = {
                "plan": target.plan,
                "spec": target.spec,
                "output_directory": target.output_directory,
                "rescore_id": config.rescore_id,
            }
            if target.route == "recovery_v2":
                kwargs["recovery"] = recovery
            destination = scorer(**kwargs)
            manifest, _ = verify_deterministic_rescore(destination, target.output_directory)
            if manifest.get("rescore_id") != config.rescore_id:
                raise ProtectedOutputIntegrityError("E25 rescore identity mismatch")
            completed.append(
                {
                    "output_set_id": target.output_set_id,
                    "rescore_id": config.rescore_id,
                    "manifest_sha256": sha256_file(destination / "manifest.json"),
                }
            )
        lock_path.unlink()
    except BaseException:
        # Preserve the lock and any partial rescore namespaces as interruption evidence.
        raise
    return {
        "status": "COMPLETE",
        "protocol_id": preflight.protocol_id,
        "target_count": len(completed),
        "rescores": completed,
        "semantic_outputs_logged": False,
        "score_values_logged": False,
    }
