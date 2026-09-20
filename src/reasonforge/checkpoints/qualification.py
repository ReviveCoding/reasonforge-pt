"""Fail-closed protocols and result verification for E27/E28 recovery tests."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from reasonforge.config import ExperimentStatus, file_sha256, identity_hash, load_registry
from reasonforge.training.runtime import validate_native_trainer_checkpoint

from .project import ProjectCheckpointStore


class QualificationConfigError(ValueError):
    """The frozen recovery protocol or one of its evidence bindings is invalid."""


class ComparisonSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    require_equal_global_step: bool = True
    require_equal_sample_count: bool = False
    require_equal_group_accounting: bool = False
    require_equal_task_sequence: bool = False
    require_partial_rollout_discard: bool = False
    require_equal_parameter_dtype: bool = False
    max_parameter_abs_difference: float = Field(ge=0.0)
    max_final_loss_abs_difference: float = Field(ge=0.0)


class SupersededAttemptSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence: Path
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    forbidden_run_ids: list[str] = Field(min_length=1)


class FailedAttemptSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence: Path
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_status: Literal["FAILED"]
    forbidden_run_ids: list[str] = Field(min_length=1)
    forensic_evidence: Path | None = None
    forensic_evidence_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_complete_forensic_binding(self) -> FailedAttemptSpec:
        if (self.forensic_evidence is None) != (self.forensic_evidence_sha256 is None):
            raise ValueError("failed-attempt forensic path and hash must be provided together")
        return self


class OperationalAttemptSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence: Path
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_status: Literal["INVALID_OPERATIONAL"]
    forbidden_run_ids: list[str] = Field(min_length=1)


class QualifiedAttemptSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence: Path
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    suite_config: Path
    suite_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_id: str
    expected_status: Literal["QUALIFIED"]
    reference_run_id: str
    recovery_run_id: str


class QualificationExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    source_experiment_id: Literal["E10", "E12"]
    source_config: Path
    source_config_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_execution_config_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_run_manifest: Path
    source_checkpoint: Path
    source_journal: Path | None = None
    reference_run_id: str
    recovery_run_id: str
    result_filename: str = Field(pattern=r"^E(27|28)_RESULT_v[0-9]+\.json$")
    seed: int
    max_steps: int = Field(gt=1)
    interruption_after_step: int = Field(gt=0)
    save_steps: int = Field(gt=0)
    save_total_limit: int = Field(gt=1)
    groups_per_step: int | None = Field(default=None, gt=0)
    generations_per_group: int | None = Field(default=None, gt=1)
    controlled_partial_exit_code: int | None = Field(default=None, ge=1, le=255)
    superseded_attempt: SupersededAttemptSpec | None = None
    failed_attempts: list[FailedAttemptSpec] = Field(default_factory=list)
    operational_attempts: list[OperationalAttemptSpec] = Field(default_factory=list)
    resume_adapter_dtype_policy: Literal["restore_pre_load_dtype"] | None = None
    sampler_checkpoint_policy: Literal["replay_completed_schedule_prefix"] | None = None
    qualified_attempt: QualifiedAttemptSpec | None = None
    comparison: ComparisonSpec

    @model_validator(mode="after")
    def validate_protocol(self) -> QualificationExperimentSpec:
        if self.interruption_after_step >= self.max_steps:
            raise ValueError("interruption step must precede the final optimizer step")
        if self.interruption_after_step % self.save_steps:
            raise ValueError("interruption step must be a durable save boundary")
        if self.max_steps % self.save_steps:
            raise ValueError("final step must be a durable save boundary")
        if self.reference_run_id == self.recovery_run_id:
            raise ValueError("reference and recovery run IDs must differ")
        if self.source_experiment_id == "E10":
            if self.source_config_sha256 is None or any(
                value is not None
                for value in (
                    self.source_execution_config_sha256,
                    self.source_journal,
                    self.groups_per_step,
                    self.generations_per_group,
                    self.controlled_partial_exit_code,
                )
            ):
                raise ValueError("E27 must bind only the Trainer-style source fields")
            if self.superseded_attempt is None:
                raise ValueError("E27 must bind its preserved invalid prequalification attempt")
            if self.resume_adapter_dtype_policy != "restore_pre_load_dtype":
                raise ValueError("E27 must freeze its adapter dtype restoration policy")
            if self.sampler_checkpoint_policy is not None:
                raise ValueError("E27 cannot define a GRPO sampler checkpoint policy")
            if self.qualified_attempt is not None and (
                self.qualified_attempt.reference_run_id != self.reference_run_id
                or self.qualified_attempt.recovery_run_id != self.recovery_run_id
            ):
                raise ValueError("E27 carried qualification run identities must remain frozen")
            if {
                self.reference_run_id,
                self.recovery_run_id,
            } & set(self.superseded_attempt.forbidden_run_ids):
                raise ValueError("E27 may not reuse a superseded run identity")
        else:
            required = (
                self.source_execution_config_sha256,
                self.source_journal,
                self.groups_per_step,
                self.generations_per_group,
                self.controlled_partial_exit_code,
            )
            if (
                any(value is None for value in required)
                or self.source_config_sha256 is not None
                or self.superseded_attempt is not None
            ):
                raise ValueError("E28 must bind all completed-step GRPO source fields")
            if self.resume_adapter_dtype_policy != "restore_pre_load_dtype":
                raise ValueError("E28 must freeze its adapter dtype restoration policy")
            if self.sampler_checkpoint_policy != "replay_completed_schedule_prefix":
                raise ValueError("E28 must freeze completed-prefix sampler checkpointing")
            if self.qualified_attempt is not None:
                raise ValueError("E28 cannot carry forward a prior qualified E27 attempt")
        forbidden = {
            run_id for attempt in self.failed_attempts for run_id in attempt.forbidden_run_ids
        }
        forbidden.update(
            run_id
            for operational in self.operational_attempts
            for run_id in operational.forbidden_run_ids
        )
        if {self.reference_run_id, self.recovery_run_id} & forbidden:
            raise ValueError("qualification may not reuse a failed run identity")
        return self


class QualificationSuite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1]
    protocol_id: str
    project_protocol_id: str
    protected_access_state: Literal["NOT_ACCESSED"]
    evidence_root: Path
    output_root: Path
    experiments: dict[Literal["E27", "E28"], QualificationExperimentSpec]

    @model_validator(mode="after")
    def require_both_experiments(self) -> QualificationSuite:
        if set(self.experiments) != {"E27", "E28"}:
            raise ValueError("qualification suite must define exactly E27 and E28")
        if self.experiments["E27"].source_experiment_id != "E10":
            raise ValueError("E27 must qualify E10-style Trainer recovery")
        if self.experiments["E28"].source_experiment_id != "E12":
            raise ValueError("E28 must qualify E12-style GRPO recovery")
        return self


@dataclass(frozen=True, slots=True)
class QualificationPlan:
    status: Literal["READY", "BLOCKED", "COMPLETE"]
    experiment_id: Literal["E27", "E28"]
    protocol_id: str
    suite_config_hash: str
    reference_run_id: str
    recovery_run_id: str
    output_root: str
    evidence_path: str
    max_steps: int
    interruption_after_step: int
    commands: tuple[str, ...]
    blockers: tuple[str, ...]
    gpu_touched: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_qualification_suite(path: Path) -> QualificationSuite:
    try:
        return QualificationSuite.model_validate(yaml.safe_load(path.read_text("utf-8")))
    except (OSError, UnicodeError, yaml.YAMLError, ValueError) as error:
        raise QualificationConfigError(
            f"invalid recovery qualification config {path}: {error}"
        ) from error


def _safe_path(workspace: Path, relative: Path, label: str) -> Path:
    path = (workspace / relative).resolve() if not relative.is_absolute() else relative.resolve()
    if path == workspace or workspace not in path.parents:
        raise QualificationConfigError(f"{label} escapes the workspace: {relative}")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QualificationConfigError(f"{label} is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise QualificationConfigError(f"{label} must be a JSON object")
    return value


def _validate_source(
    workspace: Path,
    experiment_id: Literal["E27", "E28"],
    spec: QualificationExperimentSpec,
) -> list[str]:
    blockers: list[str] = []
    config = _safe_path(workspace, spec.source_config, "source config")
    run_manifest_path = _safe_path(workspace, spec.source_run_manifest, "source run manifest")
    checkpoint = _safe_path(workspace, spec.source_checkpoint, "source checkpoint")
    if not config.is_file():
        blockers.append(f"source config is missing: {spec.source_config.as_posix()}")
    elif experiment_id == "E27" and file_sha256(config) != spec.source_config_sha256:
        blockers.append("E27 source configuration hash changed")
    if not run_manifest_path.is_file():
        blockers.append(f"source run manifest is missing: {spec.source_run_manifest.as_posix()}")
        return blockers
    try:
        manifest = _read_json(run_manifest_path, "source run manifest")
    except QualificationConfigError as error:
        blockers.append(str(error))
        return blockers
    checks = {
        "source run is COMPLETE": manifest.get("status") == "COMPLETE",
        "source experiment matches": manifest.get("experiment_id") == spec.source_experiment_id,
        "source did not access protected data": (
            manifest.get("run_identity", {}).get("protected_access_state", "NOT_ACCESSED")
            == "NOT_ACCESSED"
        ),
    }
    if experiment_id == "E28":
        checks["source execution configuration matches"] = (
            manifest.get("suite_config_hash") == spec.source_execution_config_sha256
        )
    blockers.extend(name for name, passed in checks.items() if not passed)
    expected_identity = manifest.get("run_identity")
    if experiment_id == "E27" and isinstance(expected_identity, dict):
        try:
            validate_native_trainer_checkpoint(checkpoint, expected_identity)
        except (OSError, ValueError) as error:
            blockers.append(f"source native checkpoint is invalid: {error}")
    elif experiment_id == "E28":
        identity_path = checkpoint.parent / "RUN_IDENTITY.json"
        try:
            execution_identity = _read_json(identity_path, "source GRPO run identity")
            validate_native_trainer_checkpoint(checkpoint, execution_identity)
        except (QualificationConfigError, OSError, ValueError) as error:
            blockers.append(f"source GRPO final native checkpoint is invalid: {error}")
            execution_identity = {}
        if manifest.get("optimizer_steps") != 32:
            blockers.append("source GRPO final step is not 32")
        assert spec.source_journal is not None
        journal_path = _safe_path(workspace, spec.source_journal, "source GRPO journal")
        try:
            journal = _read_json(journal_path, "source GRPO journal")
            if (
                journal.get("optimizer_step_complete") is not True
                or journal.get("completed_optimizer_step") != 32
                or journal.get("completed_rollout_groups") != 64
                or journal.get("completed_rollouts") != 512
            ):
                blockers.append("source GRPO journal accounting is invalid")
            from reasonforge.orchestration.rl import CompletedStepJournal

            source_journal = CompletedStepJournal(
                workspace / "artifacts/rl_orchestration",
                experiment_id="E12",
                run_id=str(manifest.get("run_id")),
                config_hash=identity_hash(execution_identity),
                groups_per_step=2,
                generations_per_group=8,
            )
            resume = source_journal.resume_plan(mutate=False)
            if resume.get("action") != "RESUME" or resume.get("completed_optimizer_step") != 32:
                blockers.append("source GRPO completed-step journal cannot resume at step 32")
        except QualificationConfigError as error:
            blockers.append(str(error))
    else:
        blockers.append("source Trainer run has no run_identity")
    return blockers


def plan_resume_qualification(
    workspace: Path | str,
    config_path: Path | str,
    experiment_id: Literal["E27", "E28"],
) -> QualificationPlan:
    """Return a read-only, fail-closed launch plan; this function never imports torch."""

    root = Path(workspace).resolve()
    config = _safe_path(root, Path(config_path), "qualification config")
    suite = load_qualification_suite(config)
    spec = suite.experiments[experiment_id]
    blockers: list[str] = []
    try:
        project, _ = ProjectCheckpointStore(root / "evidence/checkpoints").verify_latest()
        if project.protocol_id != suite.project_protocol_id:
            blockers.append("latest project checkpoint protocol differs from qualification")
        if project.protected_access_state not in {
            "NOT_ACCESSED",
            "ACQUIRED_REGISTERED_NOT_SCORED",
        }:
            blockers.append("protected outcomes have already been scored")
    except Exception as error:
        blockers.append(f"latest project checkpoint is not verified: {error}")
    registry = load_registry(root / "configs/EXPERIMENT_REGISTRY.yaml")
    by_id = {item.experiment_id: item for item in registry.experiments}
    registered = by_id.get(experiment_id)
    if registered is None:
        blockers.append(f"registry has no {experiment_id}")
    elif (
        registered.status is ExperimentStatus.COMPLETE
        and spec.qualified_attempt is None
        and not (root / suite.evidence_root / spec.result_filename).is_file()
    ):
        blockers.append(f"{experiment_id} is COMPLETE but qualification evidence is missing")
    source = by_id.get(spec.source_experiment_id)
    if source is None or source.status is not ExperimentStatus.COMPLETE:
        blockers.append(f"source dependency {spec.source_experiment_id} is not COMPLETE")
    blockers.extend(_validate_source(root, experiment_id, spec))
    if spec.superseded_attempt is not None:
        invalid_path = _safe_path(
            root, spec.superseded_attempt.evidence, "superseded-attempt evidence"
        )
        try:
            invalid = _read_json(invalid_path, "superseded-attempt evidence")
            if file_sha256(invalid_path) != spec.superseded_attempt.evidence_sha256:
                blockers.append("superseded-attempt evidence hash changed")
            if (
                invalid.get("status") != "INVALID_PREQUALIFICATION"
                or invalid.get("recovery_arm_started") is not False
                or invalid.get("completed_scientific_arms") != []
            ):
                blockers.append("superseded E27 attempt is not safely ineligible")
        except QualificationConfigError as error:
            blockers.append(str(error))
    for failed_attempt in spec.failed_attempts:
        failed_path = _safe_path(root, failed_attempt.evidence, "failed-attempt evidence")
        try:
            failed = _read_json(failed_path, "failed-attempt evidence")
            if file_sha256(failed_path) != failed_attempt.evidence_sha256:
                blockers.append("failed-attempt evidence hash changed")
            if failed.get("status") != failed_attempt.expected_status:
                blockers.append("prior failed qualification status changed")
            if failed_attempt.forensic_evidence is not None:
                forensic_path = _safe_path(
                    root, failed_attempt.forensic_evidence, "failed-attempt forensic evidence"
                )
                forensic = _read_json(forensic_path, "failed-attempt forensic evidence")
                if file_sha256(forensic_path) != failed_attempt.forensic_evidence_sha256:
                    blockers.append("failed-attempt forensic evidence hash changed")
                if forensic.get("status") != "FORENSIC_COMPLETE":
                    blockers.append("prior failed-attempt forensic status changed")
        except QualificationConfigError as error:
            blockers.append(str(error))
    for operational in spec.operational_attempts:
        invalid_path = _safe_path(root, operational.evidence, "operational-attempt evidence")
        try:
            invalid = _read_json(invalid_path, "operational-attempt evidence")
            if file_sha256(invalid_path) != operational.evidence_sha256:
                blockers.append("operational-attempt evidence hash changed")
            if invalid.get("status") != operational.expected_status:
                blockers.append("prior operational qualification status changed")
        except QualificationConfigError as error:
            blockers.append(str(error))
    carried_qualification = False
    if spec.qualified_attempt is not None:
        qualified_path = _safe_path(root, spec.qualified_attempt.evidence, "qualified evidence")
        try:
            qualified = _read_json(qualified_path, "qualified evidence")
            qualified_config_path = _safe_path(
                root, spec.qualified_attempt.suite_config, "qualified suite config"
            )
            qualified_suite = yaml.safe_load(qualified_config_path.read_text("utf-8"))
            if not isinstance(qualified_suite, Mapping):
                raise QualificationConfigError("qualified suite config must be a mapping")
            checks = {
                "hash": file_sha256(qualified_path) == spec.qualified_attempt.evidence_sha256,
                "suite config hash": file_sha256(qualified_config_path)
                == spec.qualified_attempt.suite_config_sha256,
                "status": qualified.get("status") == spec.qualified_attempt.expected_status,
                "protocol": qualified.get("protocol") == "TRAINER_NATIVE_NEW_PROCESS_RESUME",
                "suite": qualified.get("suite_config_hash")
                == spec.qualified_attempt.suite_config_sha256,
                "reference": qualified.get("reference_run_id")
                == spec.qualified_attempt.reference_run_id,
                "recovery": qualified.get("recovery_run_id")
                == spec.qualified_attempt.recovery_run_id,
                "result artifacts": qualified.get("recovered_result") is not None,
                "declared protocol": qualified_suite.get("protocol_id")
                == spec.qualified_attempt.protocol_id,
            }
            failed_checks = [name for name, passed in checks.items() if not passed]
            if failed_checks:
                blockers.append(f"carried qualified-attempt evidence failed: {failed_checks}")
            else:
                carried_qualification = True
        except QualificationConfigError as error:
            blockers.append(str(error))
    output = _safe_path(root, suite.output_root / experiment_id, "qualification output")
    evidence = _safe_path(
        root, suite.evidence_root / spec.result_filename, "qualification evidence"
    )
    status: Literal["READY", "BLOCKED", "COMPLETE"]
    if carried_qualification and evidence.is_file() and not blockers:
        status = "COMPLETE"
    elif evidence.is_file():
        try:
            existing = _read_json(evidence, "existing qualification result")
            if existing.get("status") == "QUALIFIED" and existing.get(
                "suite_config_hash"
            ) == file_sha256(config):
                status = "COMPLETE"
            else:
                blockers.append("an incompatible qualification result already exists")
                status = "BLOCKED"
        except QualificationConfigError as error:
            blockers.append(str(error))
            status = "BLOCKED"
    else:
        status = "BLOCKED" if blockers else "READY"
    if status != "COMPLETE":
        for run_id in (spec.reference_run_id, spec.recovery_run_id):
            target = output / run_id
            if target.is_dir() and any(target.iterdir()):
                blockers.append(
                    "target qualification run already has output and cannot be overwritten: "
                    f"{run_id}"
                )
                status = "BLOCKED"
        journal_root = root / "artifacts/rl_orchestration" / experiment_id
        if journal_root.is_dir() and any(
            (journal_root / run_id).is_dir() and any((journal_root / run_id).iterdir())
            for run_id in (spec.reference_run_id, spec.recovery_run_id)
        ):
            blockers.append("target qualification journal already exists and cannot be overwritten")
            status = "BLOCKED"
    disk = shutil.disk_usage(root)
    if disk.free < 2_000_000_000:
        blockers.append("less than 2 GB free disk remains for qualification checkpoints")
        status = "BLOCKED"
    script = ".\\.venv\\Scripts\\python.exe scripts/qualify_checkpoint_resume.py"
    commands = (
        f"{script} --experiment-id {experiment_id} --preflight-only",
        f"{script} --experiment-id {experiment_id} --execute",
    )
    return QualificationPlan(
        status=status,
        experiment_id=experiment_id,
        protocol_id=suite.protocol_id,
        suite_config_hash=file_sha256(config),
        reference_run_id=spec.reference_run_id,
        recovery_run_id=spec.recovery_run_id,
        output_root=output.relative_to(root).as_posix(),
        evidence_path=evidence.relative_to(root).as_posix(),
        max_steps=spec.max_steps,
        interruption_after_step=spec.interruption_after_step,
        commands=commands,
        blockers=tuple(blockers),
    )


def _require_fields(value: Mapping[str, Any], names: Sequence[str], label: str) -> None:
    missing = [name for name in names if name not in value]
    if missing:
        raise QualificationConfigError(f"{label} is missing fields: {missing}")


def verify_qualification_result(
    experiment_id: Literal["E27", "E28"],
    spec: QualificationExperimentSpec,
    reference: Mapping[str, Any],
    recovered: Mapping[str, Any],
    *,
    partial_exit: Mapping[str, Any] | None = None,
    discard_event: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify final parity and interruption provenance without weakening tolerances."""

    common = (
        "status",
        "run_id",
        "process_id",
        "final_global_step",
        "consumed_examples",
        "final_loss",
        "parameter_inventory_hash",
    )
    _require_fields(reference, common, "reference result")
    _require_fields(recovered, common, "recovered result")
    failures: list[str] = []
    if reference.get("status") != "COMPLETE" or recovered.get("status") != "COMPLETE":
        failures.append("both arms must be COMPLETE")
    if reference.get("run_id") != spec.reference_run_id:
        failures.append("reference run identity mismatch")
    if recovered.get("run_id") != spec.recovery_run_id:
        failures.append("recovery run identity mismatch")
    process_ids = recovered.get("process_ids")
    if not isinstance(process_ids, list) or len(set(process_ids)) < 2:
        failures.append("recovery did not span at least two distinct processes")
    if spec.comparison.require_equal_global_step and (
        reference.get("final_global_step") != recovered.get("final_global_step")
        or reference.get("final_global_step") != spec.max_steps
    ):
        failures.append("final global steps differ")
    if spec.comparison.require_equal_sample_count and (
        reference.get("consumed_examples") != recovered.get("consumed_examples")
    ):
        failures.append("sample consumption differs")
    if experiment_id == "E27":
        _require_fields(reference, ("initial_parameter_identity",), "reference result")
        _require_fields(
            recovered,
            ("initial_parameter_identity", "resume_process_initial_parameter_identity"),
            "recovered result",
        )
        if reference.get("initial_parameter_identity") != recovered.get(
            "initial_parameter_identity"
        ) or recovered.get("resume_process_initial_parameter_identity") != recovered.get(
            "initial_parameter_identity"
        ):
            failures.append("step-0 trainable adapter identity differs across processes")
    if spec.comparison.require_equal_parameter_dtype and (
        reference.get("final_parameter_dtypes") != recovered.get("final_parameter_dtypes")
        or recovered.get("resume_dtype_identity_preserved") is not True
    ):
        failures.append("adapter parameter dtype identity differs after resume")
    interruption_difference = recovered.get("interruption_checkpoint_max_abs_difference")
    if interruption_difference is None:
        failures.append("interruption-boundary parameter comparison is missing")
    else:
        try:
            if float(interruption_difference) > spec.comparison.max_parameter_abs_difference:
                failures.append("interruption-boundary parameters already differ")
        except (TypeError, ValueError) as error:
            raise QualificationConfigError(
                f"interruption-boundary parameter difference is invalid: {error}"
            ) from error
    if experiment_id == "E27" and recovered.get("batch_trace_parity") is not True:
        failures.append("post-resume microbatch order differs from uninterrupted reference")
    try:
        loss_difference = abs(float(reference["final_loss"]) - float(recovered["final_loss"]))
    except (TypeError, ValueError) as error:
        raise QualificationConfigError(f"final losses are not numeric: {error}") from error
    if loss_difference > spec.comparison.max_final_loss_abs_difference:
        failures.append("final loss difference exceeds the frozen tolerance")
    observed_difference = recovered.get("parameter_max_abs_difference_observed")
    if observed_difference is not None:
        try:
            parameter_max_abs = float(observed_difference)
        except (TypeError, ValueError) as error:
            raise QualificationConfigError(
                f"observed parameter difference is not numeric: {error}"
            ) from error
    else:
        left = reference.get("parameter_vector")
        right = recovered.get("parameter_vector")
        if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
            raise QualificationConfigError(
                "parameter vectors are missing or have different lengths"
            )
        try:
            parameter_max_abs = max(
                (abs(float(a) - float(b)) for a, b in zip(left, right, strict=True)),
                default=0.0,
            )
        except (TypeError, ValueError) as error:
            raise QualificationConfigError(f"parameter vectors are not numeric: {error}") from error
    if parameter_max_abs > spec.comparison.max_parameter_abs_difference:
        failures.append("parameter difference exceeds the frozen tolerance")
    accounting: dict[str, Any] = {}
    if experiment_id == "E28":
        if partial_exit is None or discard_event is None:
            failures.append("partial rollout interruption/discard evidence is missing")
        else:
            expected_partial: dict[str, object] = {
                "event": "PARTIAL_ROLLOUT_INTERRUPTED",
                "exit_code": spec.controlled_partial_exit_code,
                "excluded_from_gradient_accounting": True,
            }
            expected_discard: dict[str, object] = {
                "event": "PARTIAL_ROLLOUT_DISCARDED_AFTER_INTERRUPTION",
                "resume_from_completed_step": spec.interruption_after_step,
                "excluded_from_gradient_accounting": True,
                "must_not_double_count": True,
            }
            for name, expected in expected_partial.items():
                if partial_exit.get(name) != expected:
                    failures.append(f"partial interruption evidence mismatch: {name}")
            for name, expected in expected_discard.items():
                if discard_event.get(name) != expected:
                    failures.append(f"partial discard evidence mismatch: {name}")
        expected_groups = spec.max_steps * int(spec.groups_per_step or 0)
        expected_rollouts = expected_groups * int(spec.generations_per_group or 0)
        accounting = {
            "expected_completed_groups": expected_groups,
            "expected_completed_rollouts": expected_rollouts,
            "reference_completed_groups": reference.get("completed_rollout_groups"),
            "recovered_completed_groups": recovered.get("completed_rollout_groups"),
            "reference_completed_rollouts": reference.get("completed_rollouts"),
            "recovered_completed_rollouts": recovered.get("completed_rollouts"),
        }
        if spec.comparison.require_equal_group_accounting and any(
            value != expected
            for value, expected in (
                (reference.get("completed_rollout_groups"), expected_groups),
                (recovered.get("completed_rollout_groups"), expected_groups),
                (reference.get("completed_rollouts"), expected_rollouts),
                (recovered.get("completed_rollouts"), expected_rollouts),
            )
        ):
            failures.append("GRPO completed group/rollout accounting differs")
        if spec.comparison.require_equal_task_sequence and reference.get(
            "task_sequence"
        ) != recovered.get("task_sequence"):
            failures.append("GRPO task sequence differs after resume")
    return {
        "status": "QUALIFIED" if not failures else "FAILED",
        "experiment_id": experiment_id,
        "protocol": (
            "TRAINER_NATIVE_NEW_PROCESS_RESUME"
            if experiment_id == "E27"
            else "GRPO_COMPLETED_STEP_NEW_PROCESS_RESUME"
        ),
        "reference_run_id": spec.reference_run_id,
        "recovery_run_id": spec.recovery_run_id,
        "interruption_after_step": spec.interruption_after_step,
        "new_process_verified": isinstance(process_ids, list) and len(set(process_ids)) >= 2,
        "parameter_max_abs_difference": parameter_max_abs,
        "final_loss_abs_difference": loss_difference,
        "accounting": accounting,
        "failures": failures,
        "comparison_policy_hash": identity_hash(spec.comparison.model_dump(mode="json")),
    }
