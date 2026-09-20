"""CPU-safe planning and completed-step accounting for GRPO-family runs."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq
import yaml

from reasonforge.config import file_sha256, identity_hash

SamplerName = Literal[
    "uniform",
    "difficulty",
    "reliability_only",
    "reward_variance",
    "rcac",
    "rcac_no_reliability",
    "rcac_no_difficulty",
]

MIN_QUALIFIED_RENDERED_PROMPT_TOKENS = 1_026


class RLPlanError(ValueError):
    """A plan would violate a frozen identity or comparison invariant."""


@dataclass(frozen=True, slots=True)
class SharedRolloutBudget:
    optimizer_steps: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_generations: int
    generation_batch_size: int
    max_prompt_tokens: int
    max_completion_tokens: int
    checkpoint_target_minutes: int
    checkpoint_retention: int

    def __post_init__(self) -> None:
        positive = asdict(self)
        if any(not isinstance(value, int) or value < 1 for value in positive.values()):
            raise RLPlanError("all shared rollout budget values must be positive integers")
        if self.global_generation_batch % self.num_generations:
            raise RLPlanError("global generation batch must be divisible by num_generations")
        if self.generation_batch_size % self.num_generations:
            raise RLPlanError("generation batch must contain only complete GRPO groups")
        if self.global_generation_batch % self.generation_batch_size:
            raise RLPlanError("optimizer rollout batch must contain complete generation batches")
        if self.generation_batch_size % self.per_device_train_batch_size:
            raise RLPlanError("generation batch must be divisible by the device microbatch")
        if self.max_prompt_tokens < MIN_QUALIFIED_RENDERED_PROMPT_TOKENS:
            raise RLPlanError("max_prompt_tokens is below the measured full-schema prompt maximum")

    @property
    def global_generation_batch(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    @property
    def groups_per_optimizer_step(self) -> int:
        return self.global_generation_batch // self.num_generations

    @property
    def steps_per_generation(self) -> int:
        return self.generation_batch_size // self.per_device_train_batch_size

    @property
    def rollout_groups(self) -> int:
        return self.optimizer_steps * self.groups_per_optimizer_step

    @property
    def rollouts(self) -> int:
        return self.rollout_groups * self.num_generations

    @property
    def maximum_completion_tokens(self) -> int:
        return self.rollouts * self.max_completion_tokens

    def fairness_identity(self) -> str:
        return identity_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class RLArm:
    arm_id: str
    sampler: SamplerName
    reward_architecture: str
    seed: int
    hidden_eval_every_steps: int | None = None

    def __post_init__(self) -> None:
        if not self.arm_id or self.seed < 0:
            raise RLPlanError("arm_id is required and seed cannot be negative")
        if self.hidden_eval_every_steps is not None and self.hidden_eval_every_steps < 1:
            raise RLPlanError("hidden evaluation interval must be positive")


@dataclass(frozen=True, slots=True)
class RLExperiment:
    experiment_id: str
    question: str
    dependencies: tuple[str, ...]
    arms: tuple[RLArm, ...]
    requires_resolution: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RLSuite:
    protocol_id: str
    model_id: str
    model_revision: str
    dataset_path: Path
    dataset_fingerprint: str
    dataset_file_sha256: str
    starting_checkpoint_ref: Path
    adapter_config_ref: Path
    reward_config_ref: Path
    reward_config_sha256: str
    sampler_config_ref: Path
    sampler_config_sha256: str
    adaptive_estimates_ref: Path
    sampler_implementation: str
    environment_mode: str
    reward_function_count: int
    trainer_semantics: Mapping[str, Any]
    budget: SharedRolloutBudget
    experiments: Mapping[str, RLExperiment]

    @classmethod
    def load(cls, path: Path) -> RLSuite:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise RLPlanError("RL suite must be a YAML mapping")
        common = value.get("common")
        if not isinstance(common, Mapping):
            raise RLPlanError("RL suite is missing common configuration")
        raw_budget = common.get("budget")
        if not isinstance(raw_budget, Mapping):
            raise RLPlanError("RL suite is missing a shared budget")
        budget = SharedRolloutBudget(**dict(raw_budget))
        raw_experiments = value.get("experiments")
        if not isinstance(raw_experiments, Mapping):
            raise RLPlanError("RL suite is missing experiments")
        experiments: dict[str, RLExperiment] = {}
        for experiment_id, raw in raw_experiments.items():
            if not isinstance(raw, Mapping):
                raise RLPlanError(f"{experiment_id} configuration must be a mapping")
            raw_arms = raw.get("arms")
            if not isinstance(raw_arms, list) or not raw_arms:
                raise RLPlanError(f"{experiment_id} requires at least one arm")
            arms = tuple(RLArm(**dict(arm)) for arm in raw_arms)
            experiments[str(experiment_id)] = RLExperiment(
                experiment_id=str(experiment_id),
                question=str(raw["question"]),
                dependencies=tuple(str(item) for item in raw.get("dependencies", [])),
                arms=arms,
                requires_resolution=tuple(str(item) for item in raw.get("requires_resolution", [])),
            )
        suite = cls(
            protocol_id=str(value["protocol_id"]),
            model_id=str(common["model_id"]),
            model_revision=str(common["model_revision"]),
            dataset_path=Path(str(common["dataset_path"])),
            dataset_fingerprint=str(common["dataset_fingerprint"]),
            dataset_file_sha256=str(common["dataset_file_sha256"]),
            starting_checkpoint_ref=Path(str(common["starting_checkpoint_ref"])),
            adapter_config_ref=Path(str(common["adapter_config_ref"])),
            reward_config_ref=Path(str(common["reward_config_ref"])),
            reward_config_sha256=str(common["reward_config_sha256"]),
            sampler_config_ref=Path(str(common["sampler_config_ref"])),
            sampler_config_sha256=str(common["sampler_config_sha256"]),
            adaptive_estimates_ref=Path(str(common["adaptive_estimates_ref"])),
            sampler_implementation=str(common["sampler_implementation"]),
            environment_mode=str(common["environment_mode"]),
            reward_function_count=int(common["reward_function_count"]),
            trainer_semantics=dict(common["trainer_semantics"]),
            budget=budget,
            experiments=experiments,
        )
        suite.validate()
        return suite

    def validate(self) -> None:
        if not self.model_id or len(self.model_revision) != 40:
            raise RLPlanError("RL model ID and immutable revision are required")
        identity_fields = {
            "dataset_fingerprint": self.dataset_fingerprint,
            "dataset_file_sha256": self.dataset_file_sha256,
            "reward_config_sha256": self.reward_config_sha256,
            "sampler_config_sha256": self.sampler_config_sha256,
        }
        invalid_identities = [
            name
            for name, value in identity_fields.items()
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        ]
        if invalid_identities:
            raise RLPlanError(f"invalid SHA-256 identities: {invalid_identities}")
        path_fields = {
            "dataset_path": self.dataset_path,
            "starting_checkpoint_ref": self.starting_checkpoint_ref,
            "adapter_config_ref": self.adapter_config_ref,
            "reward_config_ref": self.reward_config_ref,
            "sampler_config_ref": self.sampler_config_ref,
            "adaptive_estimates_ref": self.adaptive_estimates_ref,
        }
        absolute_paths = [name for name, path in path_fields.items() if path.is_absolute()]
        if absolute_paths:
            raise RLPlanError(
                f"RL artifact references must be workspace-relative: {absolute_paths}"
            )
        expected_ids = {f"E{number}" for number in range(12, 21)}
        if set(self.experiments) != expected_ids:
            difference = sorted(set(self.experiments) ^ expected_ids)
            raise RLPlanError(f"RL suite must define exactly E12-E20: {difference}")
        if self.reward_function_count != 1:
            raise RLPlanError(
                "canonical reward ownership requires exactly one scalar trainer reward function"
            )
        expected_trainer: dict[str, object] = {
            "loss_type": "grpo",
            "scale_rewards": "group",
            "multi_objective_aggregation": "sum_then_normalize",
            "use_vllm": False,
            "enable_jit_checkpoint": False,
            "learning_rate": 1e-6,
            "warmup_steps": 1,
            "weight_decay": 0.0,
            "optimizer": "adamw_torch_fused",
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "linear",
            "bf16": True,
            "tf32": True,
            "gradient_checkpointing": True,
            "gradient_checkpointing_use_reentrant": False,
            "auto_find_batch_size": False,
            "beta": 0.0,
            "epsilon": 0.2,
            "num_iterations": 1,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": None,
            "mask_truncated_completions": False,
            "shuffle_dataset": False,
            "enable_thinking": False,
        }
        if set(self.trainer_semantics) != set(expected_trainer):
            missing = sorted(set(expected_trainer) - set(self.trainer_semantics))
            extra = sorted(set(self.trainer_semantics) - set(expected_trainer))
            raise RLPlanError(
                f"RL trainer semantics keys are not exact: missing={missing}, extra={extra}"
            )
        for key, expected in expected_trainer.items():
            if self.trainer_semantics.get(key) != expected:
                raise RLPlanError(f"trainer semantic {key} must remain {expected!r}")
        if self.environment_mode != "dataset_verifier":
            raise RLPlanError(
                "stateful TRL environment_factory remains unqualified; use dataset_verifier"
            )
        if self.sampler_implementation != "reasonforge.curriculum.samplers:ComparisonSampler":
            raise RLPlanError("E12-E20 must use the qualified comparison sampler implementation")
        for experiment_id, experiment in self.experiments.items():
            if experiment.experiment_id != experiment_id:
                raise RLPlanError(f"experiment key/identity mismatch for {experiment_id}")
            arm_ids = [arm.arm_id for arm in experiment.arms]
            if len(arm_ids) != len(set(arm_ids)):
                raise RLPlanError(f"duplicate arm IDs in {experiment_id}")
        sampler_expectations: dict[str, set[str]] = {
            "E12": {"uniform"},
            "E13": {"difficulty"},
            "E14": {"reliability_only"},
            "E15": {"reward_variance"},
            "E16": {"uniform"},
            "E17": {"uniform"},
            "E18": {"rcac"},
            "E19": {"rcac_no_reliability"},
            "E20": {"rcac_no_difficulty"},
        }
        for experiment_id, expected in sampler_expectations.items():
            observed = {arm.sampler for arm in self.experiments[experiment_id].arms}
            if observed != expected:
                raise RLPlanError(
                    f"{experiment_id} sampler mismatch: expected {expected}, observed {observed}"
                )
        adaptive_ref = self.adaptive_estimates_ref.as_posix()
        for experiment_id in (f"E{number}" for number in range(13, 21)):
            if adaptive_ref not in self.experiments[experiment_id].requires_resolution:
                raise RLPlanError(
                    f"{experiment_id} must require the qualified adaptive-estimate artifact"
                )
        reward_arms = {arm.reward_architecture for arm in self.experiments["E16"].arms}
        if reward_arms != {"terminal", "structured"}:
            raise RLPlanError("E16 must isolate terminal versus structured reward arms")

    @property
    def config_hash(self) -> str:
        return identity_hash(
            {
                "protocol_id": self.protocol_id,
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "dataset_path": self.dataset_path.as_posix(),
                "dataset_fingerprint": self.dataset_fingerprint,
                "dataset_file_sha256": self.dataset_file_sha256,
                "starting_checkpoint_ref": self.starting_checkpoint_ref.as_posix(),
                "adapter_config_ref": self.adapter_config_ref.as_posix(),
                "reward_config_ref": self.reward_config_ref.as_posix(),
                "reward_config_sha256": self.reward_config_sha256,
                "sampler_config_ref": self.sampler_config_ref.as_posix(),
                "sampler_config_sha256": self.sampler_config_sha256,
                "adaptive_estimates_ref": self.adaptive_estimates_ref.as_posix(),
                "sampler_implementation": self.sampler_implementation,
                "environment_mode": self.environment_mode,
                "reward_function_count": self.reward_function_count,
                "trainer_semantics": dict(self.trainer_semantics),
                "budget": asdict(self.budget),
                "experiments": {
                    key: {
                        "question": value.question,
                        "dependencies": value.dependencies,
                        "arms": [asdict(arm) for arm in value.arms],
                        "requires_resolution": value.requires_resolution,
                    }
                    for key, value in sorted(self.experiments.items())
                },
            }
        )

    def preflight(self, experiment_id: str, *, workspace: Path) -> dict[str, Any]:
        experiment = self.experiments[experiment_id]
        required_paths = {
            "dataset": workspace / self.dataset_path,
            "starting_checkpoint": workspace / self.starting_checkpoint_ref,
            "adapter_config": workspace / self.adapter_config_ref,
            "reward_config": workspace / self.reward_config_ref,
            "sampler_config": workspace / self.sampler_config_ref,
        }
        absent = [name for name, path in required_paths.items() if not path.is_file()]
        identity_errors: list[str] = []
        if "dataset" not in absent and file_sha256(required_paths["dataset"]) != (
            self.dataset_file_sha256
        ):
            identity_errors.append("dataset file SHA-256 mismatch")
        if "dataset" not in absent:
            identity_errors.extend(
                _validate_train_dataset(
                    required_paths["dataset"], expected_fingerprint=self.dataset_fingerprint
                )
            )
        if "reward_config" not in absent and file_sha256(required_paths["reward_config"]) != (
            self.reward_config_sha256
        ):
            identity_errors.append("reward config SHA-256 mismatch")
        if "sampler_config" not in absent and file_sha256(required_paths["sampler_config"]) != (
            self.sampler_config_sha256
        ):
            identity_errors.append("sampler config SHA-256 mismatch")
        if "adapter_config" not in absent:
            identity_errors.extend(_validate_adapter_selection(required_paths["adapter_config"]))
        starting_adapter: str | None = None
        starting_adapter_identity: str | None = None
        if "starting_checkpoint" not in absent:
            manifest_errors, starting_adapter, starting_adapter_identity = (
                _validate_starting_checkpoint_manifest(
                    required_paths["starting_checkpoint"], workspace=workspace
                )
            )
            identity_errors.extend(manifest_errors)
        unresolved = [
            item for item in experiment.requires_resolution if not (workspace / item).is_file()
        ]
        if experiment_id != "E12" and not unresolved:
            identity_errors.extend(
                _validate_adaptive_estimates(
                    workspace / self.adaptive_estimates_ref,
                    expected_dataset_fingerprint=self.dataset_fingerprint,
                    expected_starting_adapter_identity=starting_adapter_identity,
                )
            )
        return {
            "experiment_id": experiment_id,
            "protocol_id": self.protocol_id,
            "suite_config_hash": self.config_hash,
            "fairness_budget_hash": self.budget.fairness_identity(),
            "budget": {
                **asdict(self.budget),
                "global_generation_batch": self.budget.global_generation_batch,
                "steps_per_generation": self.budget.steps_per_generation,
                "groups_per_optimizer_step": self.budget.groups_per_optimizer_step,
                "rollout_groups": self.budget.rollout_groups,
                "rollouts": self.budget.rollouts,
                "maximum_completion_tokens": self.budget.maximum_completion_tokens,
            },
            "arms": [asdict(arm) for arm in experiment.arms],
            "sampler_implementation": self.sampler_implementation,
            "missing_required_paths": absent,
            "identity_errors": identity_errors,
            "unresolved_material_decisions": unresolved,
            "starting_adapter": starting_adapter,
            "starting_adapter_identity": starting_adapter_identity,
            "launch_authorized": not absent and not identity_errors and not unresolved,
            "gpu_touched": False,
        }


def _validate_adapter_selection(path: Path) -> list[str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return [f"adapter selection is unreadable: {error}"]
    if not isinstance(value, Mapping):
        return ["adapter selection must be a JSON object"]
    errors: list[str] = []
    if value.get("status") != "SELECTED":
        errors.append("adapter selection status is not SELECTED")
    if value.get("experiment_id") != "E09":
        errors.append("adapter selection was not produced by E09")
    if not isinstance(value.get("adapter"), Mapping):
        errors.append("adapter selection has no adapter configuration")
    return errors


def _validate_train_dataset(path: Path, *, expected_fingerprint: str) -> list[str]:
    try:
        table = pq.read_table(path, columns=["content_hash", "split_role"])
        hashes = [str(value) for value in table.column("content_hash").to_pylist()]
        roles = {str(value) for value in table.column("split_role").to_pylist()}
    except (OSError, ValueError, KeyError) as error:
        return [f"TRAIN dataset identity is unreadable: {error}"]
    errors: list[str] = []
    if not hashes:
        errors.append("TRAIN dataset is empty")
    if roles != {"TRAIN"}:
        errors.append(f"TRAIN dataset contains disallowed split roles: {sorted(roles)}")
    digest = hashlib.sha256()
    for value in sorted(hashes):
        digest.update(value.encode())
        digest.update(b"\n")
    if digest.hexdigest() != expected_fingerprint:
        errors.append("TRAIN dataset semantic fingerprint mismatch")
    return errors


def _validate_adaptive_estimates(
    path: Path,
    *,
    expected_dataset_fingerprint: str,
    expected_starting_adapter_identity: str | None,
) -> list[str]:
    """Validate the frozen model-outcome estimates required by adaptive arms."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return [f"adaptive estimates are unreadable: {error}"]
    if not isinstance(value, Mapping):
        return ["adaptive estimates must be a JSON object"]
    errors: list[str] = []
    if value.get("status") != "QUALIFIED":
        errors.append("adaptive estimates status is not QUALIFIED")
    if value.get("data_role") != "DEVELOPMENT":
        errors.append("adaptive estimates were not derived from DEVELOPMENT evidence")
    if value.get("dataset_fingerprint") != expected_dataset_fingerprint:
        errors.append("adaptive estimates dataset fingerprint mismatch")
    if value.get("starting_adapter_identity") != expected_starting_adapter_identity:
        errors.append("adaptive estimates starting-adapter identity mismatch")
    estimates = value.get("task_estimates")
    if not isinstance(estimates, list) or not estimates:
        errors.append("adaptive estimates contain no task estimates")
        return errors
    task_ids: list[str] = []
    for estimate in estimates:
        if not isinstance(estimate, Mapping):
            errors.append("adaptive estimate row is not an object")
            continue
        task_id = estimate.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            errors.append("adaptive estimate task_id is missing")
            continue
        task_ids.append(task_id)
        for name in ("p", "q", "u"):
            observed = estimate.get(name)
            if not isinstance(observed, (int, float)) or not 0.0 <= float(observed) <= 1.0:
                errors.append(f"adaptive estimate {task_id} has invalid {name}")
        if not isinstance(estimate.get("provenance"), str) or not estimate["provenance"]:
            errors.append(f"adaptive estimate {task_id} lacks provenance")
    if len(task_ids) != len(set(task_ids)):
        errors.append("adaptive estimates contain duplicate task IDs")
    return errors


def _validate_starting_checkpoint_manifest(
    path: Path, *, workspace: Path
) -> tuple[list[str], str | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return [f"starting checkpoint manifest is unreadable: {error}"], None, None
    if not isinstance(value, Mapping):
        return ["starting checkpoint manifest must be a JSON object"], None, None
    errors: list[str] = []
    if value.get("status") != "COMPLETE" or value.get("experiment_id") != "E10":
        errors.append("starting checkpoint is not a COMPLETE E10 manifest")
    raw_adapter = value.get("final_adapter_path")
    expected_identity = value.get("final_adapter_identity")
    if not isinstance(raw_adapter, str) or not isinstance(expected_identity, str):
        errors.append("starting checkpoint manifest lacks final adapter identity")
        return errors, None, None
    adapter_path = (workspace / raw_adapter).resolve()
    workspace_resolved = workspace.resolve()
    if workspace_resolved not in adapter_path.parents or not adapter_path.is_dir():
        errors.append("starting adapter is missing or escapes the workspace")
        return errors, adapter_path.as_posix(), expected_identity
    inventory = {
        item.relative_to(adapter_path).as_posix(): file_sha256(item)
        for item in sorted(adapter_path.rglob("*"))
        if item.is_file()
    }
    if not inventory or identity_hash(inventory) != expected_identity:
        errors.append("starting adapter inventory identity mismatch")
    return errors, adapter_path.as_posix(), expected_identity


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class CompletedStepJournal:
    """Bind rollout accounting to native checkpoints at completed steps only."""

    def __init__(
        self,
        root: Path,
        *,
        experiment_id: str,
        run_id: str,
        config_hash: str,
        groups_per_step: int,
        generations_per_group: int,
    ) -> None:
        if not experiment_id or not run_id or not config_hash:
            raise RLPlanError("journal identities must be non-empty")
        if groups_per_step < 1 or generations_per_group < 2:
            raise RLPlanError("invalid GRPO group accounting")
        self.root = root / experiment_id / run_id
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.config_hash = config_hash
        self.groups_per_step = groups_per_step
        self.generations_per_group = generations_per_group
        self.completed_step = 0
        self.completed_groups = 0
        self.completed_rollouts = 0
        self.sampler_state: dict[str, Any] = {}
        self._active_group: str | None = None

    @property
    def partial_path(self) -> Path:
        return self.root / "IN_PROGRESS" / "PARTIAL_ROLLOUT.json"

    def begin_group(
        self, group_id: str, *, task_id: str, artifact_paths: Sequence[str] = ()
    ) -> None:
        if not group_id or not task_id:
            raise RLPlanError("group and task identities are required")
        if self._active_group is not None or self.partial_path.exists():
            raise RLPlanError("a partial rollout group is already active")
        self._active_group = group_id
        _atomic_json(
            self.partial_path,
            {
                "experiment_id": self.experiment_id,
                "run_id": self.run_id,
                "config_hash": self.config_hash,
                "group_id": group_id,
                "task_id": task_id,
                "artifact_paths": list(artifact_paths),
                "excluded_from_gradient_accounting": True,
                "status": "IN_PROGRESS",
            },
        )

    def finish_group(self, group_id: str) -> None:
        if group_id != self._active_group:
            raise RLPlanError("rollout group completion does not match active group")
        self.completed_groups += 1
        self.completed_rollouts += self.generations_per_group
        self._active_group = None
        self.partial_path.unlink(missing_ok=True)

    def publish_completed_step(
        self, step: int, *, native_checkpoint: Path, sampler_state: Mapping[str, Any]
    ) -> Path:
        if self._active_group is not None or self.partial_path.exists():
            raise RLPlanError("cannot checkpoint a partial rollout group")
        # Durable native saves are intentionally less frequent than optimizer
        # steps.  The rollout counts below prove that every intervening step is
        # complete; requiring a native checkpoint for step N-1 would defeat the
        # timing-derived checkpoint cadence.
        if step <= self.completed_step:
            raise RLPlanError("completed checkpoint steps must increase monotonically")
        expected_groups = step * self.groups_per_step
        if self.completed_groups != expected_groups:
            observed = self.completed_groups
            raise RLPlanError(
                f"step {step} requires {expected_groups} complete groups; observed {observed}"
            )
        if not native_checkpoint.is_dir() or not any(
            path.is_file() for path in native_checkpoint.rglob("*")
        ):
            raise RLPlanError("native Trainer checkpoint is missing or empty")
        required_sampler_state = {
            "mode",
            "config",
            "candidate_pool",
            "tasks",
            "family_counts",
            "total_selections",
            "curriculum_phase",
            "seed",
            "rng_state_b64",
        }
        missing_sampler_state = required_sampler_state - set(sampler_state)
        if missing_sampler_state:
            raise RLPlanError(
                f"completed-step sampler state is incomplete: {sorted(missing_sampler_state)}"
            )
        next_sampler_state = copy.deepcopy(dict(sampler_state))
        state = {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "completed_optimizer_step": step,
            "optimizer_step_complete": True,
            "completed_rollout_groups": self.completed_groups,
            "completed_rollouts": self.completed_rollouts,
            "groups_per_step": self.groups_per_step,
            "generations_per_group": self.generations_per_group,
            "native_checkpoint": str(native_checkpoint.resolve()),
            "native_checkpoint_inventory_hash": _directory_inventory_hash(native_checkpoint),
            "sampler_state": next_sampler_state,
            "sampler_state_hash": identity_hash(next_sampler_state),
        }
        destination = self.root / "COMPLETE" / f"step_{step:08d}.json"
        if destination.exists():
            raise FileExistsError(f"completed-step journal is immutable: {destination}")
        _atomic_json(destination, state)
        self.completed_step = step
        self.sampler_state = next_sampler_state
        return destination

    def resume_plan(self, *, mutate: bool = False) -> dict[str, Any]:
        states = sorted((self.root / "COMPLETE").glob("step_*.json"))
        compatible: list[tuple[int, Path, dict[str, Any]]] = []
        identity_mismatches: list[str] = []
        invalid_checkpoints: list[str] = []
        for path in states:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                invalid_checkpoints.append(path.as_posix())
                continue
            if (
                value.get("experiment_id") != self.experiment_id
                or value.get("run_id") != self.run_id
                or value.get("config_hash") != self.config_hash
                or value.get("groups_per_step") != self.groups_per_step
                or value.get("generations_per_group") != self.generations_per_group
            ):
                identity_mismatches.append(path.as_posix())
                continue
            step = value.get("completed_optimizer_step")
            completed_groups = value.get("completed_rollout_groups")
            completed_rollouts = value.get("completed_rollouts")
            sampler_state = value.get("sampler_state")
            accounting_valid = (
                value.get("optimizer_step_complete") is True
                and isinstance(step, int)
                and step >= 1
                and completed_groups == step * self.groups_per_step
                and completed_rollouts == completed_groups * self.generations_per_group
            )
            sampler_valid = isinstance(sampler_state, Mapping) and identity_hash(
                dict(sampler_state) if isinstance(sampler_state, Mapping) else {}
            ) == value.get("sampler_state_hash")
            native_value = value.get("native_checkpoint")
            native = Path(native_value) if isinstance(native_value, str) else Path()
            native_valid = (
                isinstance(native_value, str)
                and native.is_dir()
                and _directory_inventory_hash(native)
                == value.get("native_checkpoint_inventory_hash")
            )
            if not accounting_valid or not sampler_valid or not native_valid:
                invalid_checkpoints.append(path.as_posix())
                continue
            compatible.append((step, path, value))
        latest = max(compatible, default=None, key=lambda item: item[0])
        partial = (
            json.loads(self.partial_path.read_text(encoding="utf-8"))
            if self.partial_path.is_file()
            else None
        )
        partial_identity_mismatch = bool(
            partial is not None
            and (
                partial.get("experiment_id") != self.experiment_id
                or partial.get("run_id") != self.run_id
                or partial.get("config_hash") != self.config_hash
            )
        )
        if identity_mismatches or partial_identity_mismatch:
            action = "START_NEW_RUN_REQUIRED"
        elif invalid_checkpoints and latest is None:
            action = "CHECKPOINT_INVALID"
        else:
            action = "RESUME" if latest is not None else "START_NEW"
        plan: dict[str, Any] = {
            "action": action,
            "mutates_state": mutate,
            "completed_optimizer_step": latest[0] if latest else 0,
            "native_checkpoint": latest[2]["native_checkpoint"] if latest else None,
            "partial_rollout": partial,
            "partial_rollout_action": (
                "PARTIAL_ROLLOUT_DISCARDED_AFTER_INTERRUPTION" if partial else None
            ),
            "excluded_from_gradient_accounting": bool(partial),
            "identity_mismatches": identity_mismatches,
            "invalid_checkpoints": invalid_checkpoints,
        }
        if not mutate:
            return plan
        if action in {"START_NEW_RUN_REQUIRED", "CHECKPOINT_INVALID"}:
            raise RLPlanError(
                f"unsafe resume action {action}; preserve this run and use a new run_id"
            )
        if latest is not None:
            value = latest[2]
            self.completed_step = latest[0]
            self.completed_groups = int(value["completed_rollout_groups"])
            self.completed_rollouts = int(value["completed_rollouts"])
            self.sampler_state = copy.deepcopy(value["sampler_state"])
        if partial is not None:
            discard = {
                **partial,
                "event": "PARTIAL_ROLLOUT_DISCARDED_AFTER_INTERRUPTION",
                "resume_from_completed_step": self.completed_step,
                "excluded_from_gradient_accounting": True,
                "must_not_double_count": True,
                "status": "DISCARDED",
            }
            destination = (
                self.root / "DISCARDED" / f"partial_{partial['group_id']}_{uuid.uuid4().hex}.json"
            )
            _atomic_json(destination, discard)
            self.partial_path.unlink()
        self._active_group = None
        return plan


def _directory_inventory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    files = (candidate for candidate in path.rglob("*") if candidate.is_file())
    for item in sorted(files):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()
