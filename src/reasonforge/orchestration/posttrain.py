"""Fail-closed launch planning and execution for E09/E10/E11.

Importing and preflighting this module never loads a neural model or touches
CUDA. Actual training requires an explicit ``execute=True`` call.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from reasonforge.checkpoints import ProjectCheckpointStore, RunIdentity
from reasonforge.config import ExperimentStatus, file_sha256, identity_hash, load_registry
from reasonforge.models import AdapterTarget, LoRAAdapterConfig, QLoRAModelConfig
from reasonforge.preference import DPORunConfig, construct_dpo_trainer
from reasonforge.systems import GpuOwnerLease
from reasonforge.training import (
    NativeCheckpointIdentityCallback,
    SFTRunConfig,
    TelemetryCallback,
    TrainingRunConfig,
    construct_sft_trainer,
    train_with_native_resume,
    validate_native_trainer_checkpoint,
)


class PostTrainConfigError(ValueError):
    """A launch config or one of its bound artifacts is invalid."""


class ModelBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_id: str
    revision: str


class DatasetBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_role: Literal["TRAIN", "DEVELOPMENT"]
    format: Literal["sft_prompt_completion", "dpo_conversational"]


class AdapterBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    target: AdapterTarget
    rank: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0.0, lt=1.0)
    use_rslora: bool = False

    def runtime_config(self) -> LoRAAdapterConfig:
        return LoRAAdapterConfig(
            target=self.target,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            use_rslora=self.use_rslora,
        )


class DPOBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    beta: float = Field(gt=0.0)
    loss_type: str
    label_smoothing: float = Field(ge=0.0, lt=0.5)
    reference_semantics: Literal["frozen_copy_of_e10_starting_adapter"]
    implementation_revision: Literal["peft_named_ref_adapter_v2"]


class PostTrainSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    experiment_id: Literal["E09", "E10", "E11"]
    protocol_id: str
    scientific_state: Literal["DEVELOPMENT", "CANDIDATE_SELECTION"]
    objective: Literal["SFT_ADAPTER_PILOT", "SFT", "OFFLINE_DPO"]
    depends_on: list[str]
    model: ModelBinding
    dataset: DatasetBinding
    eval_dataset: DatasetBinding | None = None
    output_root: Path
    run_id: str | None = None
    seed: int
    data_seed: int
    max_steps: int = Field(gt=0)
    max_length: int = Field(gt=0)
    microbatch: int = Field(gt=0)
    gradient_accumulation: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    warmup_steps: int = Field(ge=0)
    estimated_step_seconds: float = Field(gt=0.0)
    estimated_checkpoint_bytes: int = Field(gt=0)
    minimum_free_disk_bytes: int = Field(gt=0)
    adapter_candidates: list[AdapterBinding] = Field(default_factory=list)
    selection_artifact: Path | None = None
    adapter_selection_artifact: Path | None = None
    starting_checkpoint: str | None = None
    starting_adapter_manifest: Path | None = None
    promotion_rule: str | None = None
    dpo: DPOBinding | None = None

    @model_validator(mode="after")
    def objective_contract(self) -> PostTrainSpec:
        expected = {
            "E09": "SFT_ADAPTER_PILOT",
            "E10": "SFT",
            "E11": "OFFLINE_DPO",
        }[self.experiment_id]
        if self.objective != expected:
            raise ValueError(f"{self.experiment_id} must use objective {expected}")
        if self.experiment_id == "E09":
            if not self.adapter_candidates or self.selection_artifact is None or self.run_id:
                raise ValueError("E09 requires candidates/selection artifact and dynamic run IDs")
        elif self.run_id is None:
            raise ValueError(f"{self.experiment_id} requires a frozen run_id")
        if self.experiment_id == "E10" and (
            self.adapter_selection_artifact is None
            or self.starting_checkpoint != "frozen_reference"
        ):
            raise ValueError("E10 must bind E09 selection and start from the frozen reference")
        if self.experiment_id == "E11" and (
            self.starting_adapter_manifest is None or self.dpo is None
        ):
            raise ValueError("E11 requires the completed E10 adapter and DPO config")
        if self.experiment_id != "E11" and self.dpo is not None:
            raise ValueError("DPO configuration is valid only for E11")
        return self


@dataclass(frozen=True)
class PostTrainPlan:
    status: Literal["READY", "BLOCKED"]
    experiment_id: str
    run_id: str
    config_path: str
    config_hash: str
    run_identity: dict[str, str]
    output_dir: str
    native_resume_from: str | None
    estimated_wall_seconds: float
    estimated_checkpoint_bytes: int
    free_disk_bytes: int
    checkpoint_save_steps: int
    blockers: tuple[str, ...]
    commands: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _ResolvedAdapter:
    config: AdapterBinding
    starting_adapter: Path | None = None
    starting_identity: str = "frozen_reference"


def load_posttrain_spec(path: Path) -> PostTrainSpec:
    try:
        value = yaml.safe_load(path.read_text("utf-8"))
        return PostTrainSpec.model_validate(value)
    except (OSError, UnicodeError, yaml.YAMLError, ValueError) as error:
        raise PostTrainConfigError(f"invalid post-training config {path}: {error}") from error


def _safe_path(workspace: Path, relative: Path, label: str) -> Path:
    path = (workspace / relative).resolve() if not relative.is_absolute() else relative.resolve()
    if path == workspace or workspace not in path.parents:
        raise PostTrainConfigError(f"{label} escapes the workspace: {relative}")
    return path


def _validate_dataset(workspace: Path, binding: DatasetBinding) -> tuple[Path, str | None]:
    path = _safe_path(workspace, binding.path, "dataset")
    if not path.is_file():
        return path, f"dataset is missing: {binding.path.as_posix()}"
    observed = file_sha256(path)
    if observed != binding.sha256:
        return path, f"dataset hash mismatch: expected {binding.sha256}, observed {observed}"
    if "split_role" not in pq.read_schema(path).names:
        return path, "dataset lacks split_role"
    roles: set[str] = set()
    for batch in pq.ParquetFile(path).iter_batches(columns=["split_role"], batch_size=8192):
        roles.update(str(item) for item in batch.column(0).to_pylist())
    if roles != {binding.split_role}:
        return (
            path,
            f"dataset role mismatch: expected {binding.split_role}, observed {sorted(roles)}",
        )
    if roles & {"LOCKED_FINAL", "EXTERNAL_PROTECTED"}:
        return path, "protected data cannot enter post-training"
    return path, None


def _load_adapter_selection(path: Path, workspace: Path) -> AdapterBinding:
    try:
        value = json.loads(path.read_text("utf-8"))
        if not isinstance(value, dict) or value.get("status") != "SELECTED":
            raise ValueError("status must be SELECTED")
        if value.get("experiment_id") != "E09":
            raise ValueError("selection must be produced by E09")
        evidence = value.get("development_evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("DEVELOPMENT evidence paths are required")
        ranking = value.get("candidate_ranking")
        if not isinstance(ranking, list) or not ranking or not isinstance(ranking[0], dict):
            raise ValueError("candidate ranking is required")
        if ranking[0].get("candidate") != value.get("selected_candidate"):
            raise ValueError("selected candidate is not first in the frozen ranking")
        adapter = AdapterBinding.model_validate(value.get("adapter"))
        if ranking[0].get("adapter") != adapter.model_dump(mode="json"):
            raise ValueError("selected adapter does not match the frozen ranking")
        config = _safe_path(workspace, Path(str(value["config_path"])), "E09 selection config")
        if file_sha256(config) != value.get("config_hash"):
            raise ValueError("E09 selection config hash mismatch")
        for item in ranking:
            if not isinstance(item, dict):
                raise ValueError("candidate ranking entries must be mappings")
            evaluation = _safe_path(
                workspace,
                Path(str(item["development_evaluation"])),
                "E09 development evidence",
            )
            if file_sha256(evaluation) != item.get("development_evaluation_sha256"):
                raise ValueError("E09 development evidence hash mismatch")
        return adapter
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise PostTrainConfigError(f"invalid adapter selection {path}: {error}") from error


def _load_starting_adapter(path: Path, workspace: Path) -> _ResolvedAdapter:
    try:
        value = json.loads(path.read_text("utf-8"))
        if not isinstance(value, dict) or value.get("status") != "COMPLETE":
            raise ValueError("E10 run manifest must be COMPLETE")
        if value.get("experiment_id") != "E10":
            raise ValueError("starting adapter must be produced by E10")
        adapter = AdapterBinding.model_validate(value.get("adapter"))
        relative = Path(str(value["final_adapter_path"]))
        final_adapter = _safe_path(workspace, relative, "starting adapter")
        if not final_adapter.is_dir():
            raise ValueError(f"starting adapter directory is missing: {relative}")
        expected_hash = value.get("final_adapter_identity")
        if not isinstance(expected_hash, str) or not expected_hash:
            raise ValueError("final_adapter_identity is missing")
        if _adapter_identity(final_adapter) != expected_hash:
            raise ValueError("starting adapter content identity mismatch")
        return _ResolvedAdapter(adapter, final_adapter, expected_hash)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, ValueError) as error:
        raise PostTrainConfigError(f"invalid starting adapter manifest {path}: {error}") from error


def _resolve_adapter(
    workspace: Path, spec: PostTrainSpec, candidate: str | None
) -> tuple[_ResolvedAdapter | None, str | None]:
    if spec.experiment_id == "E09":
        matches = [item for item in spec.adapter_candidates if item.name == candidate]
        if len(matches) != 1:
            names = ", ".join(item.name for item in spec.adapter_candidates)
            return None, f"--candidate must select exactly one of: {names}"
        return _ResolvedAdapter(matches[0]), None
    if spec.experiment_id == "E10":
        assert spec.adapter_selection_artifact is not None
        path = _safe_path(workspace, spec.adapter_selection_artifact, "adapter selection")
        if not path.is_file():
            return None, f"E09 adapter selection is pending: {spec.adapter_selection_artifact}"
        try:
            return _ResolvedAdapter(_load_adapter_selection(path, workspace)), None
        except PostTrainConfigError as error:
            return None, str(error)
    assert spec.starting_adapter_manifest is not None
    path = _safe_path(workspace, spec.starting_adapter_manifest, "starting adapter manifest")
    if not path.is_file():
        return None, f"E10 completed adapter manifest is pending: {spec.starting_adapter_manifest}"
    try:
        return _load_starting_adapter(path, workspace), None
    except PostTrainConfigError as error:
        return None, str(error)


def _run_identity(spec: PostTrainSpec, adapter: _ResolvedAdapter, config_hash: str) -> RunIdentity:
    optimizer = identity_hash(
        {
            "max_steps": spec.max_steps,
            "microbatch": spec.microbatch,
            "gradient_accumulation": spec.gradient_accumulation,
            "learning_rate": spec.learning_rate,
            "warmup_steps": spec.warmup_steps,
            "max_length": spec.max_length,
            "seed": spec.seed,
            "data_seed": spec.data_seed,
        }
    )
    return RunIdentity(
        model_family=f"{spec.model.model_id}@{spec.model.revision}",
        starting_checkpoint=adapter.starting_identity,
        adapter_architecture=identity_hash(adapter.config.model_dump(mode="json")),
        training_objective=spec.objective,
        optimizer_configuration=optimizer,
        dataset_fingerprint=spec.dataset.sha256,
        data_role=spec.dataset.split_role,
        reward_architecture="NOT_APPLICABLE",
        reward_weights="NOT_APPLICABLE",
        sampler_configuration="SEEDED_SHUFFLE",
        seed_semantics=f"train={spec.seed};data={spec.data_seed}",
        protected_access_state="NOT_ACCESSED",
        source_configuration=config_hash,
    )


def _latest_native_checkpoint(output_dir: Path, identity: RunIdentity) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    if not output_dir.is_dir():
        return None
    expected = asdict(identity)
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = validate_native_trainer_checkpoint(path, expected)
        except (OSError, ValueError):
            continue
        candidates.append((step, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def plan_posttrain(
    workspace: Path | str,
    config_path: Path | str,
    *,
    candidate: str | None = None,
    resume_from: Path | str | None = None,
    auto_resume: bool = False,
) -> PostTrainPlan:
    """Perform a read-only launch gate; no model/CUDA work occurs."""

    root = Path(workspace).resolve()
    config = _safe_path(root, Path(config_path), "config")
    spec = load_posttrain_spec(config)
    config_hash = file_sha256(config)
    blockers: list[str] = []
    registry = load_registry(root / "configs" / "EXPERIMENT_REGISTRY.yaml")
    matches = [item for item in registry.experiments if item.experiment_id == spec.experiment_id]
    if len(matches) != 1:
        blockers.append(f"experiment registry has no unique {spec.experiment_id}")
    else:
        registered = matches[0]
        if registered.depends_on != spec.depends_on:
            blockers.append("config dependencies differ from experiment registry")
        by_id = {item.experiment_id: item for item in registry.experiments}
        incomplete = [
            dependency
            for dependency in spec.depends_on
            if by_id[dependency].status is not ExperimentStatus.COMPLETE
        ]
        if incomplete:
            blockers.append(f"dependencies are not COMPLETE: {', '.join(incomplete)}")
    try:
        project, _ = ProjectCheckpointStore(root / "evidence" / "checkpoints").verify_latest()
        if project.protocol_id != spec.protocol_id:
            blockers.append("latest checkpoint protocol differs from launch config")
    except Exception as error:
        blockers.append(f"latest project checkpoint is not verified: {error}")
    _, dataset_error = _validate_dataset(root, spec.dataset)
    if dataset_error:
        blockers.append(dataset_error)
    if spec.eval_dataset is not None:
        _, eval_error = _validate_dataset(root, spec.eval_dataset)
        if eval_error:
            blockers.append(eval_error)
    adapter, adapter_error = _resolve_adapter(root, spec, candidate)
    if adapter_error:
        blockers.append(adapter_error)
    if adapter is None:
        adapter = _ResolvedAdapter(
            AdapterBinding(
                name="UNRESOLVED",
                target=AdapterTarget.QV,
                rank=1,
                alpha=1,
                dropout=0.0,
            ),
            starting_identity="UNRESOLVED",
        )
    run_id = spec.run_id or f"E09-{adapter.config.name}-seed{spec.seed}-v1"
    output_dir = _safe_path(root, spec.output_root / run_id, "output directory")
    run_identity = _run_identity(spec, adapter, config_hash)
    native_resume: Path | None = None
    if resume_from is not None:
        native_resume = _safe_path(root, Path(resume_from), "resume checkpoint")
        try:
            validate_native_trainer_checkpoint(native_resume, asdict(run_identity))
        except (OSError, ValueError) as error:
            blockers.append(f"resume checkpoint is incompatible: {error}")
    elif auto_resume:
        native_resume = _latest_native_checkpoint(output_dir, run_identity)
    complete_manifest = output_dir / "RUN_COMPLETE.json"
    if complete_manifest.exists():
        blockers.append("run is already COMPLETE and must not be rerun")
    disk = shutil.disk_usage(root)
    retention_need = spec.estimated_checkpoint_bytes * 4
    required_free = max(spec.minimum_free_disk_bytes, retention_need)
    if disk.free < required_free:
        blockers.append(f"insufficient disk: need {required_free}, have {disk.free}")
    cadence = TrainingRunConfig(
        experiment_id=spec.experiment_id,
        run_id=run_id,
        output_dir=output_dir,
        model=QLoRAModelConfig(spec.model.model_id, spec.model.revision),
        adapter=adapter.config.runtime_config(),
        dataset_fingerprint=spec.dataset.sha256,
        max_steps=spec.max_steps,
        median_optimizer_step_seconds=spec.estimated_step_seconds,
        per_device_train_batch_size=spec.microbatch,
        gradient_accumulation_steps=spec.gradient_accumulation,
        learning_rate=spec.learning_rate,
        warmup_steps=spec.warmup_steps,
        seed=spec.seed,
        data_seed=spec.data_seed,
        native_resume_from_checkpoint=native_resume,
    ).save_interval
    script = {
        "E09": "scripts/run_adapter_pilot.py",
        "E10": "scripts/run_sft.py",
        "E11": "scripts/run_dpo.py",
    }[spec.experiment_id]
    candidate_arg = f" --candidate {candidate}" if candidate else ""
    base = f".\\.venv\\Scripts\\python.exe {script}{candidate_arg}"
    commands = (f"{base} --preflight-only", f"{base} --execute")
    return PostTrainPlan(
        status="BLOCKED" if blockers else "READY",
        experiment_id=spec.experiment_id,
        run_id=run_id,
        config_path=config.relative_to(root).as_posix(),
        config_hash=config_hash,
        run_identity=asdict(run_identity),
        output_dir=output_dir.relative_to(root).as_posix(),
        native_resume_from=(
            native_resume.relative_to(root).as_posix() if native_resume is not None else None
        ),
        estimated_wall_seconds=spec.estimated_step_seconds * spec.max_steps,
        estimated_checkpoint_bytes=spec.estimated_checkpoint_bytes,
        free_disk_bytes=disk.free,
        checkpoint_save_steps=cadence.save_steps,
        blockers=tuple(blockers),
        commands=commands,
    )


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _assert_cuda_qualified(workspace: Path) -> None:
    environment_path = workspace / "evidence" / "environment" / "CP00_environment.json"
    environment = json.loads(environment_path.read_text("utf-8"))
    if not environment.get("cuda", {}).get("available"):
        raise RuntimeError("CUDA was not qualified in CP00")
    if environment.get("cuda", {}).get("device_count") != 1:
        raise RuntimeError("E09-E11 recipes are frozen for exactly one physical GPU")
    if not environment.get("cuda", {}).get("bf16_supported"):
        raise RuntimeError("BF16 support was not qualified")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("live CUDA state does not match the qualified single-GPU backend")


def _prepare_dataset(path: Path, objective: str) -> Any:
    from datasets import Dataset

    dataset = Dataset.from_parquet(str(path))
    columns = dataset.column_names
    if objective in {"SFT_ADAPTER_PILOT", "SFT"}:

        def convert(row: dict[str, Any]) -> dict[str, object]:
            reference = row.get("reference")
            if not isinstance(reference, str) or not reference:
                raise ValueError("SFT row has no deterministic reference")
            return {
                "prompt": [{"role": "user", "content": str(row["prompt"])}],
                "completion": [{"role": "assistant", "content": reference}],
            }
    else:

        def convert(row: dict[str, Any]) -> dict[str, object]:
            chosen, rejected = row.get("chosen"), row.get("rejected")
            if not isinstance(chosen, str) or not isinstance(rejected, str):
                raise ValueError("DPO row lacks chosen/rejected strings")
            return {
                "prompt": [{"role": "user", "content": str(row["prompt"])}],
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
            }

    return dataset.map(convert, remove_columns=columns, desc="canonical post-training format")


def _adapter_identity(path: Path) -> str:
    inventory = {
        item.relative_to(path).as_posix(): file_sha256(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }
    if not inventory:
        raise RuntimeError("final adapter directory is empty")
    return identity_hash(inventory)


def select_e09_adapter(
    workspace: Path | str,
    config_path: Path | str,
    evaluations: dict[str, Path],
    *,
    publish: bool = False,
) -> dict[str, object]:
    """Apply the frozen E09 DEVELOPMENT selection rule to all candidates."""

    root = Path(workspace).resolve()
    config = _safe_path(root, Path(config_path), "config")
    spec = load_posttrain_spec(config)
    if spec.experiment_id != "E09" or spec.eval_dataset is None:
        raise PostTrainConfigError("adapter selection requires the E09 config and eval dataset")
    expected_names = {item.name for item in spec.adapter_candidates}
    if set(evaluations) != expected_names:
        raise PostTrainConfigError(
            f"selection requires exactly {sorted(expected_names)}, received {sorted(evaluations)}"
        )
    records: list[dict[str, object]] = []
    for adapter in spec.adapter_candidates:
        run_id = f"E09-{adapter.name}-seed{spec.seed}-v1"
        run_manifest_path = _safe_path(
            root, spec.output_root / run_id / "RUN_COMPLETE.json", "E09 run manifest"
        )
        evaluation_path = _safe_path(root, evaluations[adapter.name], "E09 evaluation")
        try:
            run = json.loads(run_manifest_path.read_text("utf-8"))
            evaluation = json.loads(evaluation_path.read_text("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise PostTrainConfigError(f"cannot read E09 selection evidence: {error}") from error
        if not isinstance(run, dict) or run.get("status") != "COMPLETE":
            raise PostTrainConfigError(f"E09 run is not COMPLETE: {run_manifest_path}")
        if run.get("experiment_id") != "E09" or run.get("run_id") != run_id:
            raise PostTrainConfigError(f"E09 run identity mismatch: {run_manifest_path}")
        if run.get("adapter") != adapter.model_dump(mode="json"):
            raise PostTrainConfigError(f"E09 adapter config mismatch: {run_manifest_path}")
        final_adapter = _safe_path(
            root, Path(str(run.get("final_adapter_path"))), "E09 final adapter"
        )
        if not final_adapter.is_dir() or _adapter_identity(final_adapter) != run.get(
            "final_adapter_identity"
        ):
            raise PostTrainConfigError(f"E09 final adapter identity mismatch: {run_manifest_path}")
        if not isinstance(evaluation, dict) or evaluation.get("status") != "COMPLETE":
            raise PostTrainConfigError(f"E09 evaluation is not COMPLETE: {evaluation_path}")
        evaluation_run_id = evaluation.get("run_id")
        if evaluation.get("experiment_id") != "E09" or not isinstance(evaluation_run_id, str):
            raise PostTrainConfigError(f"E09 evaluation identity mismatch: {evaluation_path}")
        if not evaluation_run_id.startswith(f"E09-{adapter.name}-"):
            raise PostTrainConfigError(f"E09 evaluation candidate mismatch: {evaluation_path}")
        evaluation_adapter = _safe_path(
            root, Path(str(evaluation.get("adapter"))), "E09 evaluated adapter"
        )
        model_identity = evaluation.get("model_identity")
        if evaluation_adapter != final_adapter or not isinstance(model_identity, str):
            raise PostTrainConfigError(f"E09 evaluated adapter mismatch: {evaluation_path}")
        if str(run.get("final_adapter_identity"))[:8] not in model_identity:
            raise PostTrainConfigError(f"E09 evaluation model hash mismatch: {evaluation_path}")
        if evaluation.get("scientific_state") != "DEVELOPMENT":
            raise PostTrainConfigError(f"E09 evaluation is not DEVELOPMENT: {evaluation_path}")
        if (
            Path(str(evaluation.get("dataset_path"))).as_posix()
            != spec.eval_dataset.path.as_posix()
        ):
            raise PostTrainConfigError(f"E09 evaluation dataset mismatch: {evaluation_path}")
        try:
            mean_score = float(evaluation["mean_score"])
            pass_rate = float(evaluation["pass_rate"])
            peak_vram = int(evaluation["peak_allocated_bytes"])
            wall_seconds = float(evaluation["wall_seconds"])
            tokens = int(evaluation["tokens_generated"])
        except (KeyError, TypeError, ValueError) as error:
            raise PostTrainConfigError(
                f"invalid E09 evaluation metrics: {evaluation_path}"
            ) from error
        if wall_seconds <= 0 or peak_vram < 0 or tokens < 0:
            raise PostTrainConfigError(f"invalid E09 systems metrics: {evaluation_path}")
        generation_config_id = evaluation.get("generation_config_id")
        generation = evaluation.get("generation")
        if not isinstance(generation_config_id, str) or not isinstance(generation, dict):
            raise PostTrainConfigError(f"E09 generation identity is missing: {evaluation_path}")
        records.append(
            {
                "candidate": adapter.name,
                "adapter": adapter.model_dump(mode="json"),
                "run_manifest": run_manifest_path.relative_to(root).as_posix(),
                "run_manifest_sha256": file_sha256(run_manifest_path),
                "development_evaluation": evaluation_path.relative_to(root).as_posix(),
                "development_evaluation_sha256": file_sha256(evaluation_path),
                "evaluation_run_id": evaluation_run_id,
                "generation_config_id": generation_config_id,
                "generation": generation,
                "mean_score": mean_score,
                "pass_rate": pass_rate,
                "peak_allocated_bytes": peak_vram,
                "tokens_per_second": tokens / wall_seconds,
            }
        )
    protocol_identities = {
        identity_hash(
            {
                "generation_config_id": item["generation_config_id"],
                "generation": item["generation"],
            }
        )
        for item in records
    }
    if len(protocol_identities) != 1:
        raise PostTrainConfigError("E09 candidate evaluations do not share one protocol identity")
    ranked = sorted(
        records,
        key=lambda item: (
            -float(cast(Any, item["mean_score"])),
            -float(cast(Any, item["pass_rate"])),
            int(cast(Any, item["peak_allocated_bytes"])),
            -float(cast(Any, item["tokens_per_second"])),
            str(item["candidate"]),
        ),
    )
    selected = ranked[0]
    artifact: dict[str, object] = {
        "status": "SELECTED",
        "experiment_id": "E09",
        "protocol_id": spec.protocol_id,
        "config_path": config.relative_to(root).as_posix(),
        "config_hash": file_sha256(config),
        "selection_rule": spec.promotion_rule,
        "selected_candidate": selected["candidate"],
        "adapter": selected["adapter"],
        "development_evidence": [item["development_evaluation"] for item in ranked],
        "candidate_ranking": ranked,
        "training_loss_used_for_selection": False,
        "evaluation_protocol_identity": next(iter(protocol_identities)),
        "protected_data_accessed": False,
    }
    if publish:
        assert spec.selection_artifact is not None
        output = _safe_path(root, spec.selection_artifact, "adapter selection output")
        if output.exists():
            raise FileExistsError(f"adapter selection is immutable: {output}")
        _atomic_json(output, artifact)
    return artifact


def execute_posttrain(
    workspace: Path | str,
    config_path: Path | str,
    *,
    candidate: str | None = None,
    resume_from: Path | str | None = None,
    auto_resume: bool = False,
) -> dict[str, object]:
    """Acquire the single-GPU lease and execute an authorized post-training run."""

    root = Path(workspace).resolve()
    plan = plan_posttrain(
        root,
        config_path,
        candidate=candidate,
        resume_from=resume_from,
        auto_resume=auto_resume,
    )
    if plan.status != "READY":
        raise RuntimeError(f"post-training launch is blocked: {list(plan.blockers)}")
    with GpuOwnerLease(root, plan.experiment_id, plan.run_id):
        return _execute_posttrain_under_lease(
            root,
            config_path,
            candidate=candidate,
            resume_from=resume_from,
            auto_resume=auto_resume,
        )


def _execute_posttrain_under_lease(
    workspace: Path | str,
    config_path: Path | str,
    *,
    candidate: str | None = None,
    resume_from: Path | str | None = None,
    auto_resume: bool = False,
) -> dict[str, object]:
    """Execute an explicitly authorized CUDA run after the CPU launch gate passes."""

    root = Path(workspace).resolve()
    plan = plan_posttrain(
        root,
        config_path,
        candidate=candidate,
        resume_from=resume_from,
        auto_resume=auto_resume,
    )
    if plan.status != "READY":
        raise RuntimeError(f"post-training launch is blocked: {list(plan.blockers)}")
    _assert_cuda_qualified(root)
    spec = load_posttrain_spec(_safe_path(root, Path(config_path), "config"))
    adapter, error = _resolve_adapter(root, spec, candidate)
    if adapter is None or error:
        raise RuntimeError(error or "adapter resolution failed")
    output_dir = root / plan.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "RUN_IDENTITY.json", plan.run_identity)
    _atomic_json(output_dir / "PRELAUNCH.json", plan.to_dict())
    dataset_path = _safe_path(root, spec.dataset.path, "dataset")
    eval_path = (
        _safe_path(root, spec.eval_dataset.path, "evaluation dataset")
        if spec.eval_dataset is not None
        else None
    )
    train_dataset = _prepare_dataset(dataset_path, spec.objective)
    eval_dataset = _prepare_dataset(eval_path, spec.objective) if eval_path else None

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        spec.model.model_id,
        revision=spec.model.revision,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    native_resume = root / plan.native_resume_from if plan.native_resume_from else None
    run = TrainingRunConfig(
        experiment_id=spec.experiment_id,
        run_id=plan.run_id,
        output_dir=output_dir,
        model=QLoRAModelConfig(spec.model.model_id, spec.model.revision),
        adapter=adapter.config.runtime_config(),
        dataset_fingerprint=spec.dataset.sha256,
        max_steps=spec.max_steps,
        median_optimizer_step_seconds=spec.estimated_step_seconds,
        per_device_train_batch_size=spec.microbatch,
        gradient_accumulation_steps=spec.gradient_accumulation,
        learning_rate=spec.learning_rate,
        warmup_steps=spec.warmup_steps,
        seed=spec.seed,
        data_seed=spec.data_seed,
        native_resume_from_checkpoint=native_resume,
    )
    callbacks = [
        NativeCheckpointIdentityCallback(output_dir, plan.run_identity),
        TelemetryCallback(output_dir / "telemetry.jsonl"),
    ]
    if spec.objective == "OFFLINE_DPO":
        if adapter.starting_adapter is None or spec.dpo is None:
            raise RuntimeError("DPO requires a verified E10 starting adapter")
        from peft import PeftModel

        from reasonforge.models import build_qlora_model_kwargs

        model_kwargs = build_qlora_model_kwargs(run.model)
        quantization = model_kwargs.pop("quantization_config")
        base_model = AutoModelForCausalLM.from_pretrained(
            spec.model.model_id,
            quantization_config=quantization,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(base_model, adapter.starting_adapter, is_trainable=True)
        model.load_adapter(adapter.starting_adapter, adapter_name="ref", is_trainable=False)
        model.set_adapter("default")
        ref_trainable = [
            name
            for name, parameter in model.named_parameters()
            if ".ref." in name and parameter.requires_grad
        ]
        policy_trainable = [
            name
            for name, parameter in model.named_parameters()
            if ".default." in name and parameter.requires_grad
        ]
        if ref_trainable or not policy_trainable or "ref" not in model.peft_config:
            raise RuntimeError("DPO named reference adapter isolation failed")
        trainer = construct_dpo_trainer(
            DPORunConfig(
                run,
                beta=spec.dpo.beta,
                loss_type=spec.dpo.loss_type,
                max_length=spec.max_length,
                label_smoothing=spec.dpo.label_smoothing,
            ),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=callbacks,
            model_override=model,
            continue_existing_adapter=True,
        )
    else:
        trainer = construct_sft_trainer(
            SFTRunConfig(run, max_length=spec.max_length),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=callbacks,
        )
    result = train_with_native_resume(trainer, native_resume)
    final_adapter = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    tokenizer.save_pretrained(final_adapter)
    complete = {
        "status": "COMPLETE",
        "experiment_id": spec.experiment_id,
        "run_id": plan.run_id,
        "protocol_id": spec.protocol_id,
        "config_hash": plan.config_hash,
        "run_identity": plan.run_identity,
        "adapter": adapter.config.model_dump(mode="json"),
        "starting_identity": adapter.starting_identity,
        "final_adapter_path": final_adapter.relative_to(root).as_posix(),
        "final_adapter_identity": _adapter_identity(final_adapter),
        "native_resume_from": plan.native_resume_from,
        "train_result": getattr(result, "metrics", result),
    }
    if spec.dpo is not None:
        complete["dpo_reference_semantics"] = spec.dpo.reference_semantics
        complete["dpo_reference_implementation"] = spec.dpo.implementation_revision
        complete["dpo_reference_adapter_name"] = "ref"
        complete["dpo_reference_identity"] = adapter.starting_identity
    _atomic_json(output_dir / "RUN_COMPLETE.json", complete)
    return complete
