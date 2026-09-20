"""Exact zero-publication E25 failure recovery, with immutable prior evidence."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from types import FunctionType
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict

from reasonforge.checkpoints.hashing import atomic_write_json, canonical_json_bytes, sha256_file

from .e25_protected_execution import (
    E25ScoringPreflight,
    E25ScoringTarget,
    _root_path,
    _verify_authorization,
    _verify_writer,
    load_e25_execution_config,
)
from .ifeval_backend import qualify_ifeval_backend
from .protected_orchestration_v2 import verify_recovery_amendment
from .protected_outputs import (
    ProtectedOutputIntegrityError,
    ProtectedOutputWriter,
    verify_deterministic_rescore,
)
from .protected_protocol import (
    load_benchmark_specs,
    load_run_plan,
    score_protected_outputs,
    verify_run_plan,
)
from .protected_recovery_v2 import load_recovery_config, score_protected_outputs_v2
from .short_rescore_publication import scorer_with_short_publication

RECOVERY_ID = "RFPT-E25-SCORING-RECOVERY-v1"
DEFAULT_CONFIG = Path("configs/evaluation/e25_scoring_recovery_v1.yaml")
CODE_PATHS = (
    "src/reasonforge/evaluation/e25_scoring_recovery_v1.py",
    "src/reasonforge/evaluation/short_rescore_publication.py",
    "src/reasonforge/evaluation/e25_protected_execution.py",
    "src/reasonforge/evaluation/e25_preprotected.py",
    "src/reasonforge/evaluation/e25_protected_replication.py",
    "src/reasonforge/evaluation/protected_outputs.py",
    "src/reasonforge/evaluation/protected_outputs_v2.py",
    "src/reasonforge/evaluation/protected_orchestration_v2.py",
    "src/reasonforge/evaluation/protected_protocol.py",
    "src/reasonforge/evaluation/protected_recovery_v2.py",
    "src/reasonforge/evaluation/ifeval_backend.py",
    "src/reasonforge/checkpoints/hashing.py",
    "scripts/recover_e25_scoring_v1.py",
)
TEST_PATHS = ("tests/test_e25_scoring_recovery_v1.py",)
DOC_PATHS = ("docs/E25_SCORING_RECOVERY_V1.md",)


class E25ScoringRecoveryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    authorization_id: Literal["RFPT-E25-SCORING-RECOVERY-v1"]
    failure: Literal["WINDOWS_PATH_LENGTH_BEFORE_FIRST_RESCORE_PUBLICATION"]
    execution_config: Literal["configs/evaluation/e25_protected_execution_v2.yaml"]
    execution_config_sha256: Literal[
        "a637097834b875934cd43fecbd81e83648107f0ab2a1a833bc33d33201564ddd"
    ]
    failed_lock: Path
    failed_lock_sha256: Literal["ece583e7f84cd5597822246922f59ba61d96c5f1151c5025929ff715bd5163c4"]
    failed_output_set_id: Literal["e25-e25_conventional_seed1702-ifeval-v1"]
    failed_directory: Path
    failed_results_sha256: Literal[
        "5b3a95cf77cacc62af76ccde82f20863dd779fb1c3afa0d3d512881ecd90685a"
    ]
    rescore_id: Literal["rfpt-e25-s1"]
    staging_root: Literal[".tmp/e25-score-v1"]
    recovery_lock: Literal[
        "evidence/protected_replication/RFPT-E25-SCORING-RECOVERY-v1/IN_PROGRESS.json"
    ]
    authorization: Literal[
        "evidence/protected_replication/RFPT-E25-SCORING-RECOVERY-v1/AUTHORIZATION.json"
    ]
    published_rescore_count_before_failure: Literal[0]
    generation_semantics_changed: Literal[False]
    scoring_routes_changed: Literal[False]
    semantic_output_logging: Literal[False]
    score_values_logging: Literal[False]


def load_config(path: Path) -> E25ScoringRecoveryConfig:
    return E25ScoringRecoveryConfig.model_validate(yaml.safe_load(path.read_text("utf-8")))


def verify_failure_namespace(
    root: Path,
    config: E25ScoringRecoveryConfig,
    output_directories: Mapping[str, Path],
    *,
    recovery_lock_acquired: bool = False,
) -> None:
    """Read only filenames and opaque hashes; never decode the failed results."""
    failed = _root_path(root, config.failed_directory, "failed namespace")
    expected_parent = output_directories[config.failed_output_set_id] / "rescores"
    if failed != expected_parent / "INVALID-rfpt-e25-s1-23a8c666":
        raise ProtectedOutputIntegrityError("E25 failed namespace path differs")
    if not failed.is_dir() or {p.name for p in failed.iterdir()} != {"results.jsonl"}:
        raise ProtectedOutputIntegrityError("E25 failed namespace contents differ")
    if sha256_file(failed / "results.jsonl") != config.failed_results_sha256:
        raise ProtectedOutputIntegrityError("E25 failed result artifact hash differs")
    lock = _root_path(root, config.failed_lock, "failed lock")
    if not lock.is_file() or sha256_file(lock) != config.failed_lock_sha256:
        raise ProtectedOutputIntegrityError("E25 failed lock hash differs")
    for output_id, directory in output_directories.items():
        rescores = directory / "rescores"
        if output_id == config.failed_output_set_id:
            if {p.name for p in rescores.iterdir()} != {failed.name}:
                raise ProtectedOutputIntegrityError("E25 prior or additional rescore state")
        elif rescores.exists():
            raise ProtectedOutputIntegrityError("E25 prior or additional rescore state")
    staging = _root_path(root, Path(config.staging_root), "staging")
    if staging.exists() and (not staging.is_dir() or any(staging.iterdir())):
        raise ProtectedOutputIntegrityError("E25 partial recovery staging exists")
    if not recovery_lock_acquired and (root / config.recovery_lock).exists():
        raise ProtectedOutputIntegrityError("E25 recovery lock already exists")


def recovery_preflight(root: Path, config: E25ScoringRecoveryConfig) -> E25ScoringPreflight:
    """The original v2 whole-matrix gates, with one exact invalid namespace exception."""
    original_path = root / config.execution_config
    if sha256_file(original_path) != config.execution_config_sha256:
        raise ProtectedOutputIntegrityError("E25 original execution configuration differs")
    original = load_e25_execution_config(original_path)
    _verify_authorization(root, original_path, original)
    verify_recovery_amendment(root / original.recovery_amendment)
    if config.failed_lock != original.scoring_lock or config.rescore_id != original.rescore_id:
        raise ProtectedOutputIntegrityError("E25 original lock/rescore identity differs")
    backend = qualify_ifeval_backend(root / original.ifeval_backend_config)
    if backend != json.loads((root / original.ifeval_backend_qualification).read_text("utf-8")):
        raise ProtectedOutputIntegrityError("E25 IFEval backend differs from qualification")
    matrix = json.loads((root / original.matrix).read_text("utf-8"))
    output_root = _root_path(root, original.output_root, "output root")
    directories = {
        output_id: output_root / entry["model_id"] / output_id
        for output_id, entry in matrix["plans"].items()
    }
    discovered: set[Path] = set()
    for model in output_root.iterdir():
        if not model.is_dir():
            raise ProtectedOutputIntegrityError("E25 output namespace has extra file")
        discovered.update(model.iterdir())
    if discovered != set(directories.values()):
        raise ProtectedOutputIntegrityError("E25 output namespace differs from frozen matrix")
    verify_failure_namespace(root, config, directories)
    specs = load_benchmark_specs(root / original.benchmark_registry)
    routes = {"frozen_v1": 0, "recovery_v2": 0}
    targets: list[E25ScoringTarget] = []
    for output_id, entry in sorted(matrix["plans"].items()):
        plan_path = _root_path(root, Path(entry["path"]), "plan")
        if sha256_file(plan_path) != entry["sha256"]:
            raise ProtectedOutputIntegrityError("E25 plan hash differs")
        plan = load_run_plan(plan_path)
        if (
            plan.output_set_id != output_id
            or plan.model_id != entry["model_id"]
            or plan.protocol_id != original.protocol_id
        ):
            raise ProtectedOutputIntegrityError("E25 plan identity differs")
        spec = specs[plan.benchmark_name]
        verify_run_plan(plan, spec)
        directory = directories[output_id]
        verified = ProtectedOutputWriter(directory).verify(require_complete=True)
        if verified.manifest.identity != plan.identity:
            raise ProtectedOutputIntegrityError("E25 output identity differs")
        if sha256_file(directory / "RUN_PLAN.json") != entry["sha256"]:
            raise ProtectedOutputIntegrityError("E25 output plan differs")
        _verify_writer(directory, plan_path)
        final_path = directory / "rescores" / config.rescore_id / "RESCORE_COMPLETE.json"
        if os.name == "nt" and len(str(final_path)) >= 260:
            raise ProtectedOutputIntegrityError("E25 final publication path exceeds MAX_PATH")
        route = original.routes[plan.benchmark_name]
        routes[route] += 1
        targets.append(
            E25ScoringTarget(
                output_id, plan, spec, route, directory, sha256_file(directory / "manifest.json")
            )
        )
    if len(targets) != 28 or targets[0].output_set_id != config.failed_output_set_id:
        raise ProtectedOutputIntegrityError("E25 recovery is not the exact first-of-28 failure")
    return E25ScoringPreflight(
        original.scoring_protocol_id,
        sha256_file(root / original.matrix),
        sha256_file(root / original.binding),
        routes,
        tuple(targets),
    )


def authorization_payload(
    root: Path,
    config_path: Path,
    config: E25ScoringRecoveryConfig,
    preflight: E25ScoringPreflight,
) -> dict[str, Any]:
    original = load_e25_execution_config(root / config.execution_config)
    bindings = {
        "recovery_config": config_path,
        "original_execution_config": root / config.execution_config,
        "original_binding": root / original.binding,
        "original_binding_config": root / original.binding_config,
        "frozen_matrix": root / original.matrix,
        "benchmark_registry": root / original.benchmark_registry,
        "generation_config": root / original.generation_config,
        "recovery_amendment": root / original.recovery_amendment,
        "recovery_scorer_config": root / original.recovery_scorer_config,
        "ifeval_backend_config": root / original.ifeval_backend_config,
        "ifeval_backend_qualification": root / original.ifeval_backend_qualification,
        "requirements_eval_lock": root / "requirements-eval-lock.txt",
        "failed_lock": root / config.failed_lock,
        "failed_results": root / config.failed_directory / "results.jsonl",
    }
    return {
        "schema_version": 1,
        "status": "VERIFIED_SCORING_RECOVERY_AUTHORIZATION",
        "authorization_id": RECOVERY_ID,
        "failure": config.failure,
        "published_rescore_count_before_failure": 0,
        "rescore_id": config.rescore_id,
        "generation_semantics_changed": False,
        "scoring_routes_changed": False,
        "semantic_outputs_inspected": False,
        "score_values_inspected": False,
        "scores_computed_by_recovery_authorization": False,
        "failed_results_exist_but_are_not_complete_or_reused": True,
        "bindings": {
            name: {"path": p.relative_to(root).as_posix(), "sha256": sha256_file(p)}
            for name, p in bindings.items()
        },
        "routes": original.routes,
        "output_manifest_inventory": {
            t.output_set_id: t.output_manifest_sha256 for t in preflight.targets
        },
        "code_inventory": {p: sha256_file(root / p) for p in CODE_PATHS},
        "test_inventory": {p: sha256_file(root / p) for p in TEST_PATHS},
        "documentation_inventory": {p: sha256_file(root / p) for p in DOC_PATHS},
    }


def prepare_recovery(
    root: Path,
    config_path: Path,
    *,
    publish: bool = False,
) -> tuple[E25ScoringRecoveryConfig, E25ScoringPreflight, dict[str, Any]]:
    root = root.resolve()
    config_path = _root_path(root, config_path, "recovery configuration")
    config = load_config(config_path)
    preflight = recovery_preflight(root, config)
    payload = authorization_payload(root, config_path, config, preflight)
    destination = root / config.authorization
    if publish:
        # Exclusive creation makes authorization immutable even under a racing invocation.
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
    if not destination.is_file() or json.loads(destination.read_text("utf-8")) != payload:
        raise ProtectedOutputIntegrityError("E25 recovery authorization differs from live evidence")
    return config, preflight, payload


def execute_recovery(root: Path, config_path: Path) -> dict[str, Any]:
    """One whole-matrix attempt. Any prior recovery attempt is refused, including partials."""
    config, preflight, _ = prepare_recovery(root, config_path)
    original = load_e25_execution_config(root / config.execution_config)
    recovery = load_recovery_config(root / original.recovery_scorer_config)
    lock_path = root / config.recovery_lock
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(
            canonical_json_bytes(
                {
                    "status": "IN_PROGRESS",
                    "authorization_id": RECOVERY_ID,
                    "authorization_sha256": sha256_file(root / config.authorization),
                }
            )
        )
        handle.flush()
        os.fsync(handle.fileno())
    verify_failure_namespace(
        root,
        config,
        {t.output_set_id: t.output_directory for t in preflight.targets},
        recovery_lock_acquired=True,
    )
    # Recheck every frozen source after acquiring the exclusive new recovery lock.
    for target in preflight.targets:
        verified = ProtectedOutputWriter(target.output_directory).verify(require_complete=True)
        if (
            verified.manifest.identity != target.plan.identity
            or sha256_file(target.output_directory / "manifest.json")
            != target.output_manifest_sha256
        ):
            raise ProtectedOutputIntegrityError("E25 source changed after recovery preflight")
        if json.loads((target.output_directory / "RUN_PLAN.json").read_text("utf-8")) != (
            target.plan.to_dict()
        ):
            raise ProtectedOutputIntegrityError("E25 output plan changed after recovery preflight")
        _verify_writer(target.output_directory, target.output_directory / "RUN_PLAN.json")
    completed: list[dict[str, Any]] = []
    for target in preflight.targets:
        rescores = target.output_directory / "rescores"
        expected = (
            {config.failed_directory.name}
            if target.output_set_id == config.failed_output_set_id
            else set()
        )
        observed = {p.name for p in rescores.iterdir()} if rescores.exists() else set()
        if observed != expected:
            raise ProtectedOutputIntegrityError("E25 rescore state appeared during recovery")
        scorer = (
            score_protected_outputs_v2 if target.route == "recovery_v2" else score_protected_outputs
        )
        short_scorer = scorer_with_short_publication(
            cast(FunctionType, scorer), root / config.staging_root
        )
        kwargs: dict[str, Any] = {
            "plan": target.plan,
            "spec": target.spec,
            "output_directory": target.output_directory,
            "rescore_id": config.rescore_id,
        }
        if target.route == "recovery_v2":
            kwargs["recovery"] = recovery
        destination = short_scorer(**kwargs)
        manifest, _ = verify_deterministic_rescore(destination, target.output_directory)
        if manifest["rescore_id"] != config.rescore_id:
            raise ProtectedOutputIntegrityError("E25 published rescore identity differs")
        completed.append(
            {
                "output_set_id": target.output_set_id,
                "rescore_id": config.rescore_id,
                "manifest_sha256": sha256_file(destination / "manifest.json"),
            }
        )
    for target, receipt in zip(preflight.targets, completed, strict=True):
        rescores = target.output_directory / "rescores"
        expected = {config.rescore_id}
        if target.output_set_id == config.failed_output_set_id:
            expected.add(config.failed_directory.name)
        if {p.name for p in rescores.iterdir()} != expected:
            raise ProtectedOutputIntegrityError("E25 final rescore namespace differs")
        destination = rescores / config.rescore_id
        verify_deterministic_rescore(destination, target.output_directory)
        if sha256_file(destination / "manifest.json") != receipt["manifest_sha256"]:
            raise ProtectedOutputIntegrityError("E25 rescore changed before matrix completion")
    if (
        sha256_file(root / config.failed_directory / "results.jsonl")
        != config.failed_results_sha256
        or sha256_file(root / config.failed_lock) != config.failed_lock_sha256
    ):
        raise ProtectedOutputIntegrityError("E25 original failure evidence changed")
    result = {
        "status": "COMPLETE",
        "authorization_id": RECOVERY_ID,
        "protocol_id": preflight.protocol_id,
        "target_count": len(completed),
        "rescores": completed,
        "semantic_outputs_logged": False,
        "score_values_logged": False,
        "original_failure_evidence_preserved": True,
    }
    atomic_write_json(lock_path.parent / "COMPLETE.json", result)
    # Retain both locks as durable attempt records; future invocations fail closed.
    return result
