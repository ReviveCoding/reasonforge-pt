"""Fail-closed whole-matrix orchestration for CP60 protected scoring.

The preflight validates all fourteen frozen output sets before a single score
artifact can be published.  Reports deliberately contain identities, hashes,
and counts only; generated text and score values are never returned or logged.
"""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from reasonforge.checkpoints.hashing import canonical_json_bytes, sha256_file
from reasonforge.checkpoints.models import utc_now

from .protected_orchestration_v2 import verify_recovery_amendment
from .protected_outputs import (
    ProtectedOutputIntegrityError,
    ProtectedOutputWriter,
    verify_deterministic_rescore,
)
from .protected_protocol import (
    FrozenProtectedRunPlan,
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

_EXPECTED_ROUTES: dict[str, ScorerRoute] = {
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
SCORING_AUTHORIZATION_ID = "RFPT-PROTECTED-SCORING-ORCHESTRATOR-v1"


class ProtectedScoringConfig(BaseModel):
    """Frozen policy for the one permitted whole-matrix scoring pass."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    protocol_id: str = Field(min_length=1)
    parent_protocol_id: str = Field(min_length=1)
    recovery_protocol_id: str = Field(min_length=1)
    expected_output_sets: Literal[14] = 14
    rescore_id_suffix: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    routes: dict[str, ScorerRoute]
    require_complete: Literal[True] = True
    require_writer_v2_adoption: Literal[True] = True
    refuse_any_existing_rescore_state: Literal[True] = True
    semantic_output_logging: Literal[False] = False

    @model_validator(mode="after")
    def validate_routes(self) -> ProtectedScoringConfig:
        if self.routes != _EXPECTED_ROUTES:
            raise ValueError("protected scoring routes differ from the authorized seven-family map")
        return self


@dataclass(frozen=True, slots=True)
class ProtectedScoringTarget:
    output_set_id: str
    model_id: str
    benchmark_name: str
    route: ScorerRoute
    run_plan_path: Path
    run_plan_sha256: str
    output_directory: Path
    output_manifest_sha256: str
    record_count: int
    plan: FrozenProtectedRunPlan
    spec: ProtectedBenchmarkSpec


@dataclass(frozen=True, slots=True)
class ProtectedScoringPreflight:
    protocol_id: str
    matrix_sha256: str
    amendment_sha256: str
    authorization_sha256: str
    target_count: int
    routes: dict[str, int]
    targets: tuple[ProtectedScoringTarget, ...]

    def public_summary(self, *, dry_run: bool) -> dict[str, Any]:
        return {
            "status": "DRY_RUN_READY" if dry_run else "PREFLIGHT_READY",
            "protocol_id": self.protocol_id,
            "matrix_sha256": self.matrix_sha256,
            "recovery_amendment_sha256": self.amendment_sha256,
            "scoring_authorization_sha256": self.authorization_sha256,
            "target_count": self.target_count,
            "route_counts": dict(sorted(self.routes.items())),
            "output_set_ids": [target.output_set_id for target in self.targets],
            "all_complete_and_verified": True,
            "existing_rescore_count": 0,
            "semantic_outputs_logged": False,
            "scores_computed": False,
        }


def load_scoring_config(path: Path) -> ProtectedScoringConfig:
    return ProtectedScoringConfig.model_validate(yaml.safe_load(path.read_text("utf-8")))


def _verify_bound_path(value: object, *, label: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ProtectedOutputIntegrityError(f"scoring authorization has invalid {label} binding")
    path_value = value.get("path")
    digest = value.get("sha256")
    if not isinstance(path_value, str) or not isinstance(digest, str):
        raise ProtectedOutputIntegrityError(f"scoring authorization has invalid {label} identity")
    path = Path(path_value)
    if not path.is_file() or sha256_file(path) != digest:
        raise ProtectedOutputIntegrityError(f"scoring authorization {label} identity mismatch")
    return path


def verify_scoring_authorization(path: Path) -> dict[str, Any]:
    """Verify the additive, immutable authorization for the sole score pass."""

    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProtectedOutputIntegrityError(f"invalid scoring authorization: {error}") from error
    expected_fields = {
        "schema_version",
        "status",
        "authorization_id",
        "created_at",
        "parent_protocol_id",
        "recovery_protocol_id",
        "cp60_checkpoint_id",
        "expected_output_sets",
        "routes",
        "all_complete_before_any_score",
        "exactly_one_rescore_per_set",
        "refuse_existing_rescore_state",
        "semantic_output_logging",
        "score_values_logged",
        "generation_semantics_changed",
        "registry_mutated",
        "cuda_used",
        "scores_computed",
        "scoring_config",
        "recovery_config",
        "recovery_amendment",
        "frozen_matrix",
        "benchmark_registry",
        "cp60_manifest",
        "cp60_bindings",
        "code_inventory",
        "test_inventory",
        "documentation_inventory",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ProtectedOutputIntegrityError("scoring authorization fields differ from schema")
    if (
        value.get("schema_version") != 1
        or value.get("status") != "VERIFIED_SCORING_AUTHORIZATION"
        or value.get("authorization_id") != SCORING_AUTHORIZATION_ID
        or value.get("parent_protocol_id") != "RFPT-PROTECTED-v1"
        or value.get("recovery_protocol_id") != "RFPT-PROTECTED-RECOVERY-v2"
        or value.get("cp60_checkpoint_id") != "CP60_PRE_PROTECTED"
        or not isinstance(value.get("created_at"), str)
        or not value.get("created_at")
        or value.get("expected_output_sets") != 14
        or value.get("routes") != _EXPECTED_ROUTES
        or value.get("all_complete_before_any_score") is not True
        or value.get("exactly_one_rescore_per_set") is not True
        or value.get("refuse_existing_rescore_state") is not True
        or value.get("semantic_output_logging") is not False
        or value.get("score_values_logged") is not False
        or value.get("generation_semantics_changed") is not False
        or value.get("registry_mutated") is not False
        or value.get("cuda_used") is not False
        or value.get("scores_computed") is not False
    ):
        raise ProtectedOutputIntegrityError("scoring authorization policy is invalid")
    for label in (
        "scoring_config",
        "recovery_config",
        "recovery_amendment",
        "frozen_matrix",
        "benchmark_registry",
        "cp60_manifest",
        "cp60_bindings",
    ):
        _verify_bound_path(value[label], label=label)
    for inventory_name in ("code_inventory", "test_inventory", "documentation_inventory"):
        inventory = value.get(inventory_name)
        if not isinstance(inventory, Mapping) or not inventory:
            raise ProtectedOutputIntegrityError(f"scoring authorization lacks {inventory_name}")
        for raw_path, digest in inventory.items():
            candidate = Path(str(raw_path))
            if (
                not isinstance(digest, str)
                or not candidate.is_file()
                or sha256_file(candidate) != digest
            ):
                raise ProtectedOutputIntegrityError(
                    f"scoring authorization {inventory_name} identity mismatch: {candidate}"
                )
    return value


def _load_frozen_matrix(path: Path, expected_count: int) -> Mapping[str, Mapping[str, Any]]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProtectedOutputIntegrityError(f"invalid frozen plan matrix: {error}") from error
    if not isinstance(value, Mapping):
        raise ProtectedOutputIntegrityError("frozen plan matrix must be an object")
    plans = value.get("plans")
    if (
        value.get("schema_version") != 1
        or value.get("status") != "FROZEN"
        or value.get("scores_computed") is not False
        or value.get("plan_count") != expected_count
        or not isinstance(plans, Mapping)
        or len(plans) != expected_count
    ):
        raise ProtectedOutputIntegrityError(
            "frozen plan matrix is not the unscored 14-plan CP60 set"
        )
    normalized: dict[str, Mapping[str, Any]] = {}
    for output_set_id, raw in plans.items():
        if not isinstance(output_set_id, str) or not isinstance(raw, Mapping):
            raise ProtectedOutputIntegrityError("invalid frozen plan inventory entry")
        normalized[output_set_id] = raw
    return normalized


def _discover_output_directories(output_root: Path) -> dict[str, Path]:
    if not output_root.is_dir():
        raise ProtectedOutputIntegrityError("protected output root does not exist")
    discovered: dict[str, Path] = {}
    for model_directory in output_root.iterdir():
        if not model_directory.is_dir():
            raise ProtectedOutputIntegrityError(
                f"unexpected file in protected output root: {model_directory.name}"
            )
        for output_directory in model_directory.iterdir():
            if not output_directory.is_dir() or not (output_directory / "manifest.json").is_file():
                raise ProtectedOutputIntegrityError(
                    "unexpected non-output-set path: "
                    f"{model_directory.name}/{output_directory.name}"
                )
            if output_directory.name in discovered:
                raise ProtectedOutputIntegrityError(
                    f"duplicate output-set directory: {output_directory.name}"
                )
            discovered[output_directory.name] = output_directory
    return discovered


def _verify_writer_adoption(output_directory: Path, run_plan_sha256: str) -> None:
    pending = sorted(name for name in _PENDING_WRITER_FILES if (output_directory / name).exists())
    if pending:
        raise ProtectedOutputIntegrityError(
            f"output set has unresolved writer recovery state: {output_directory.name}: {pending}"
        )
    adoption_path = output_directory / "WRITER_V2_ADOPTION.json"
    try:
        adoption = json.loads(adoption_path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProtectedOutputIntegrityError(
            f"output set lacks valid writer-v2 adoption: {output_directory.name}"
        ) from error
    if (
        not isinstance(adoption, Mapping)
        or adoption.get("status") != "ADOPTED"
        or adoption.get("writer_protocol_id") != "RFPT-PROTECTED-WRITER-v2"
        or adoption.get("run_plan_sha256") != run_plan_sha256
    ):
        raise ProtectedOutputIntegrityError(
            f"writer-v2 adoption identity mismatch: {output_directory.name}"
        )


def _require_ifeval_backend() -> None:
    try:
        registry = importlib.import_module(
            "instruction_following_eval.instructions_registry"
        ).INSTRUCTION_DICT
    except (ImportError, AttributeError) as error:
        raise ProtectedOutputIntegrityError(
            "pinned official IFEval registry unavailable; refusing the whole score pass"
        ) from error
    if not isinstance(registry, Mapping) or not registry:
        raise ProtectedOutputIntegrityError("official IFEval instruction registry is invalid")


def preflight_protected_scoring(
    *,
    config_path: Path,
    matrix_path: Path,
    benchmark_registry_path: Path,
    recovery_amendment_path: Path,
    recovery_config_path: Path,
    scoring_authorization_path: Path,
    output_root: Path,
    validate_ifeval_backend: bool = True,
    authorization_verifier: Callable[[Path], dict[str, Any]] | None = None,
    allowed_empty_rescore_directories: frozenset[Path] = frozenset(),
) -> ProtectedScoringPreflight:
    """Validate the complete, exact matrix without producing any score artifact."""

    config = load_scoring_config(config_path)
    authorization = (authorization_verifier or verify_scoring_authorization)(
        scoring_authorization_path
    )
    amendment = verify_recovery_amendment(recovery_amendment_path)
    if (
        amendment.get("recovery_protocol_id") != config.recovery_protocol_id
        or amendment.get("parent_protocol_id") != config.parent_protocol_id
        or authorization.get("authorization_id") != config.protocol_id
        or authorization.get("recovery_protocol_id") != config.recovery_protocol_id
        or authorization.get("parent_protocol_id") != config.parent_protocol_id
    ):
        raise ProtectedOutputIntegrityError("scoring and recovery protocol ancestry differ")
    bound_inputs = {
        "scoring_config": config_path,
        "recovery_config": recovery_config_path,
        "recovery_amendment": recovery_amendment_path,
        "frozen_matrix": matrix_path,
        "benchmark_registry": benchmark_registry_path,
    }
    for label, observed_path in bound_inputs.items():
        bound = authorization.get(label)
        if not isinstance(bound, Mapping) or bound.get("sha256") != sha256_file(observed_path):
            raise ProtectedOutputIntegrityError(
                f"scoring invocation differs from authorized {label}"
            )
    inventory = _load_frozen_matrix(matrix_path, config.expected_output_sets)
    specs = load_benchmark_specs(benchmark_registry_path)
    if set(specs) != set(config.routes):
        raise ProtectedOutputIntegrityError("benchmark registry differs from scoring route map")
    discovered = _discover_output_directories(output_root)
    if set(discovered) != set(inventory):
        missing = sorted(set(inventory) - set(discovered))
        extra = sorted(set(discovered) - set(inventory))
        raise ProtectedOutputIntegrityError(
            f"protected output-set inventory mismatch; missing={missing}, extra={extra}"
        )
    if validate_ifeval_backend:
        _require_ifeval_backend()

    targets: list[ProtectedScoringTarget] = []
    route_counts = {"frozen_v1": 0, "recovery_v2": 0}
    for output_set_id in sorted(inventory):
        entry = inventory[output_set_id]
        plan_path_value = entry.get("path")
        expected_plan_sha256 = entry.get("sha256")
        if not isinstance(plan_path_value, str) or not isinstance(expected_plan_sha256, str):
            raise ProtectedOutputIntegrityError("frozen matrix plan identity is invalid")
        plan_path = Path(plan_path_value)
        if not plan_path.is_file() or sha256_file(plan_path) != expected_plan_sha256:
            raise ProtectedOutputIntegrityError(f"frozen run-plan hash mismatch: {output_set_id}")
        plan = load_run_plan(plan_path)
        if (
            plan.output_set_id != output_set_id
            or entry.get("benchmark_name") != plan.benchmark_name
        ):
            raise ProtectedOutputIntegrityError(f"frozen matrix/run-plan mismatch: {output_set_id}")
        spec = specs.get(plan.benchmark_name)
        if spec is None:
            raise ProtectedOutputIntegrityError(
                f"unknown protected benchmark: {plan.benchmark_name}"
            )
        verify_run_plan(plan, spec)
        output_directory = discovered[output_set_id]
        if output_directory.parent.name != plan.model_id:
            raise ProtectedOutputIntegrityError(f"output model directory mismatch: {output_set_id}")
        verification = ProtectedOutputWriter(output_directory).verify(require_complete=True)
        if verification.manifest.identity != plan.identity:
            raise ProtectedOutputIntegrityError(f"output identity mismatch: {output_set_id}")
        if len(verification.records) != len(plan.expected_example_ids):
            raise ProtectedOutputIntegrityError(f"output row count mismatch: {output_set_id}")
        output_plan = output_directory / "RUN_PLAN.json"
        if not output_plan.is_file() or sha256_file(output_plan) != expected_plan_sha256:
            raise ProtectedOutputIntegrityError(f"output run-plan mismatch: {output_set_id}")
        _verify_writer_adoption(output_directory, expected_plan_sha256)
        rescore_root = output_directory / "rescores"
        if rescore_root.exists():
            allowed = rescore_root.resolve() in allowed_empty_rescore_directories
            if not allowed or not rescore_root.is_dir() or any(rescore_root.iterdir()):
                raise ProtectedOutputIntegrityError(
                    f"prior, partial, or unauthorized rescore state exists: {output_set_id}"
                )
        route = config.routes[plan.benchmark_name]
        route_counts[route] += 1
        targets.append(
            ProtectedScoringTarget(
                output_set_id=output_set_id,
                model_id=plan.model_id,
                benchmark_name=plan.benchmark_name,
                route=route,
                run_plan_path=plan_path,
                run_plan_sha256=expected_plan_sha256,
                output_directory=output_directory,
                output_manifest_sha256=sha256_file(output_directory / "manifest.json"),
                record_count=len(verification.records),
                plan=plan,
                spec=spec,
            )
        )
    if len(targets) != config.expected_output_sets:
        raise ProtectedOutputIntegrityError(
            "protected scoring target count changed during preflight"
        )
    return ProtectedScoringPreflight(
        protocol_id=config.protocol_id,
        matrix_sha256=sha256_file(matrix_path),
        amendment_sha256=sha256_file(recovery_amendment_path),
        authorization_sha256=sha256_file(scoring_authorization_path),
        target_count=len(targets),
        routes=route_counts,
        targets=tuple(targets),
    )


V1Scorer = Callable[..., Path]
V2Scorer = Callable[..., Path]
RescoreVerifier = Callable[[Path | str, Path | str], tuple[dict[str, Any], tuple[Any, ...]]]
RescoreIdFactory = Callable[[ProtectedScoringTarget], str]


def execute_protected_scoring(
    *,
    preflight: ProtectedScoringPreflight,
    config: ProtectedScoringConfig,
    recovery: ProtectedRecoveryConfig,
    lock_path: Path,
    v1_scorer: V1Scorer = score_protected_outputs,
    v2_scorer: V2Scorer = score_protected_outputs_v2,
    rescore_verifier: RescoreVerifier = verify_deterministic_rescore,
    rescore_id_factory: RescoreIdFactory | None = None,
    allowed_empty_rescore_directories: frozenset[Path] = frozenset(),
) -> dict[str, Any]:
    """Publish exactly one deterministic rescore per preflighted output set."""

    if preflight.target_count != config.expected_output_sets:
        raise ProtectedOutputIntegrityError("preflight/config target count mismatch")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise ProtectedOutputIntegrityError("protected scoring lock already exists") from error
    completed: list[dict[str, str]] = []
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                canonical_json_bytes(
                    {
                        "schema_version": 1,
                        "status": "IN_PROGRESS",
                        "protocol_id": config.protocol_id,
                        "matrix_sha256": preflight.matrix_sha256,
                        "target_count": preflight.target_count,
                        "started_at": utc_now(),
                    }
                )
            )
            handle.flush()
            os.fsync(handle.fileno())
        # Recheck all no-rescore gates after exclusive lock acquisition.
        for target in preflight.targets:
            rescore_root = target.output_directory / "rescores"
            if rescore_root.exists():
                allowed = rescore_root.resolve() in allowed_empty_rescore_directories
                if not allowed or not rescore_root.is_dir() or any(rescore_root.iterdir()):
                    raise ProtectedOutputIntegrityError(
                        f"rescore state appeared after preflight: {target.output_set_id}"
                    )
            verification = ProtectedOutputWriter(target.output_directory).verify(
                require_complete=True
            )
            if (
                verification.manifest.identity != target.plan.identity
                or sha256_file(target.output_directory / "manifest.json")
                != target.output_manifest_sha256
                or sha256_file(target.output_directory / "RUN_PLAN.json") != target.run_plan_sha256
            ):
                raise ProtectedOutputIntegrityError(
                    f"output identity changed after preflight: {target.output_set_id}"
                )
            _verify_writer_adoption(target.output_directory, target.run_plan_sha256)
        for target in preflight.targets:
            rescore_id = (
                rescore_id_factory(target)
                if rescore_id_factory is not None
                else f"{target.output_set_id}-{config.rescore_id_suffix}"
            )
            if target.route == "recovery_v2":
                destination = v2_scorer(
                    plan=target.plan,
                    spec=target.spec,
                    output_directory=target.output_directory,
                    rescore_id=rescore_id,
                    recovery=recovery,
                )
            else:
                destination = v1_scorer(
                    plan=target.plan,
                    spec=target.spec,
                    output_directory=target.output_directory,
                    rescore_id=rescore_id,
                )
            manifest, _ = rescore_verifier(destination, target.output_directory)
            if manifest.get("rescore_id") != rescore_id:
                raise ProtectedOutputIntegrityError("published rescore identity mismatch")
            completed.append(
                {
                    "output_set_id": target.output_set_id,
                    "rescore_id": rescore_id,
                    "manifest_sha256": sha256_file(Path(destination) / "manifest.json"),
                }
            )
        for target in preflight.targets:
            expected = (
                rescore_id_factory(target)
                if rescore_id_factory is not None
                else f"{target.output_set_id}-{config.rescore_id_suffix}"
            )
            rescore_root = target.output_directory / "rescores"
            children = sorted(path.name for path in rescore_root.iterdir())
            if children != [expected]:
                raise ProtectedOutputIntegrityError(
                    f"output set lacks exactly one expected rescore: {target.output_set_id}"
                )
        lock_path.unlink()
    except BaseException:
        # Preserve the lock as interruption evidence. Any partial score namespace
        # also causes all subsequent invocations to fail closed.
        raise
    return {
        "status": "COMPLETE",
        "protocol_id": config.protocol_id,
        "target_count": len(completed),
        "rescores": completed,
        "semantic_outputs_logged": False,
        "score_values_logged": False,
    }
