"""Executable, checkpoint-aware E12/E13 GRPO integration.

Heavy dependencies are imported only inside :func:`execute_grpo`, so schedule
qualification and CLI preflight remain CPU-only.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pyarrow.parquet as pq
import yaml

from reasonforge.config import file_sha256, identity_hash
from reasonforge.curriculum import (
    AdaptiveGroupRepeatSampler,
    ComparisonMode,
    ComparisonSampler,
    RCACConfig,
    TaskEstimate,
)
from reasonforge.environments import GroupedEnvironmentFactory
from reasonforge.models import AdapterTarget, LoRAAdapterConfig, QLoRAModelConfig
from reasonforge.orchestration.reward_selection import SELECTION_POLICY_ID
from reasonforge.orchestration.rl import CompletedStepJournal, RLSuite, _atomic_json
from reasonforge.rewards import (
    CanonicalBatchReward,
    CanonicalRewardPolicy,
    canonical_reward_functions,
)
from reasonforge.rewards.completion import TOOL_REWARD_IMPLEMENTATION
from reasonforge.systems import GpuOwnerLease
from reasonforge.training import (
    GRPORunConfig,
    NativeCheckpointIdentityCallback,
    TelemetryCallback,
    TrainingRunConfig,
    construct_grpo_trainer,
    derive_save_interval,
    train_with_native_resume,
    validate_native_trainer_checkpoint,
)

_QUALIFIED_GRPO_RUNS: dict[str, tuple[ComparisonMode, str, int, str]] = {
    "E12": ("uniform", "E12-uniform-grpo", 3, "structured"),
    "E13": ("difficulty", "E13-difficulty-grpo", 1, "structured"),
    "E15": ("reward_variance", "E15-reward-variance-grpo", 1, "structured"),
    # v1 completed with an inaccurate retention value in its published identity;
    # it is preserved as invalid evidence and must never gate the full run.
    "E16": ("uniform", "E16-terminal-uniform-grpo", 2, "terminal"),
    "E18": ("rcac", "E18-rcac-grpo", 1, "selected_from_E16"),
    "E20": (
        "rcac_no_difficulty",
        "E20-rcac-no-difficulty-grpo",
        1,
        "selected_from_E16",
    ),
}
_ADAPTIVE_ESTIMATE_EXPERIMENTS = {"E13", "E15", "E18", "E20"}
_SELECTED_REWARD_EXPERIMENTS = {"E18", "E20"}


@dataclass(frozen=True, slots=True)
class PreparedUniformSchedule:
    """Frozen prompt pool and deterministic E12/E13 task-selection schedule."""

    rows: tuple[dict[str, Any], ...]
    task_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    num_generations: int
    schedule_hash: str
    rendered_pool_hash: str
    maximum_rendered_prompt_tokens: int
    initial_sampler_state: Mapping[str, Any]
    final_planning_sampler_state: Mapping[str, Any]
    sampler_mode: ComparisonMode = "uniform"
    estimate_artifact_sha256: str | None = None

    def manifest(self) -> dict[str, Any]:
        manifest = {
            "status": "PREPARED",
            "sampler": self.sampler_mode,
            "estimate_provenance": (
                "STRUCTURAL_PLACEHOLDER_UNUSED_BY_UNIFORM"
                if self.sampler_mode == "uniform"
                else "QUALIFIED_DEVELOPMENT_ADAPTIVE_ESTIMATES"
            ),
            "group_count": len(self.task_ids),
            "num_generations": self.num_generations,
            "task_ids": list(self.task_ids),
            "group_ids": list(self.group_ids),
            "schedule_hash": self.schedule_hash,
            "rendered_pool_hash": self.rendered_pool_hash,
            "maximum_rendered_prompt_tokens": self.maximum_rendered_prompt_tokens,
            "initial_sampler_state_hash": identity_hash(dict(self.initial_sampler_state)),
            "final_planning_sampler_state_hash": identity_hash(
                dict(self.final_planning_sampler_state)
            ),
        }
        if self.estimate_artifact_sha256 is not None:
            manifest["estimate_artifact_sha256"] = self.estimate_artifact_sha256
        return manifest


def _load_rcac_config(path: Path) -> RCACConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("RCAC configuration must be a mapping")
    return RCACConfig(
        tau_q=float(raw["tau_q"]),
        p_target=float(raw["p_target"]),
        sigma=float(raw["sigma"]),
        lambda_u=float(raw["lambda_u"]),
    )


def _task_estimates(rows: Sequence[Mapping[str, Any]]) -> list[TaskEstimate]:
    estimates: list[TaskEstimate] = []
    for row in rows:
        if row.get("split_role") != "TRAIN":
            raise ValueError("E12 prompt pool may contain only TRAIN rows")
        difficulty = row.get("difficulty")
        reliability = row.get("reliability")
        if not isinstance(difficulty, (int, float)) or not isinstance(reliability, (int, float)):
            raise ValueError("uniform prompt-pool rows require bounded metadata estimates")
        estimates.append(
            TaskEstimate(
                task_id=str(row["example_id"]),
                family=str(row["task_family"]),
                # These two values are state-schema placeholders only. Uniform
                # sampling never reads them; adaptive arms require a separate
                # qualified model-outcome estimate artifact.
                p=1.0 - float(difficulty),
                q=float(reliability),
                u=0.0,
            )
        )
    return estimates


def prepare_uniform_schedule(
    rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    rcac_config: RCACConfig,
    seed: int,
    num_groups: int,
    num_generations: int,
    max_prompt_tokens: int,
    run_id: str,
    sampler_mode: ComparisonMode = "uniform",
    task_estimates: Sequence[TaskEstimate] | None = None,
    estimate_artifact_sha256: str | None = None,
    curriculum_phase: str = "E12",
) -> PreparedUniformSchedule:
    """Render the full prompt pool and hash a deterministic GRPO schedule."""

    if len({str(row["example_id"]) for row in rows}) != len(rows):
        raise ValueError("E12 prompt-pool example IDs must be unique")
    rendered_rows: list[dict[str, Any]] = []
    token_lengths: list[int] = []
    for source in rows:
        row = copy.deepcopy(dict(source))
        template_kwargs: dict[str, Any] = {}
        if row.get("tool_schema"):
            template_kwargs["tools"] = json.loads(str(row["tool_schema"]))
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": str(row["prompt"])}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            **template_kwargs,
        )
        encoded = tokenizer(str(rendered), add_special_tokens=False)["input_ids"]
        token_count = len(encoded)
        if token_count > max_prompt_tokens:
            raise ValueError(
                f"rendered prompt {row['example_id']} has {token_count} tokens, "
                f"exceeding the frozen cap {max_prompt_tokens}"
            )
        row["source_prompt"] = row["prompt"]
        row["prompt"] = str(rendered)
        row["rendered_prompt_tokens"] = token_count
        rendered_rows.append(row)
        token_lengths.append(token_count)

    estimates = (
        list(task_estimates) if task_estimates is not None else _task_estimates(rendered_rows)
    )
    row_families = {str(row["example_id"]): str(row["task_family"]) for row in rendered_rows}
    estimate_families = {estimate.task_id: estimate.family for estimate in estimates}
    if estimate_families != row_families:
        missing = sorted(set(row_families) - set(estimate_families))
        extra = sorted(set(estimate_families) - set(row_families))
        mismatched = sorted(
            task_id
            for task_id in set(row_families) & set(estimate_families)
            if row_families[task_id] != estimate_families[task_id]
        )
        raise ValueError(
            "task-estimate coverage/family mismatch: "
            f"missing={missing}, extra={extra}, mismatched={mismatched}"
        )
    if sampler_mode != "uniform" and estimate_artifact_sha256 is None:
        raise ValueError("adaptive schedules require a qualified estimate-artifact identity")
    comparison = ComparisonSampler(
        estimates,
        rcac_config,
        mode=sampler_mode,
        seed=seed,
        curriculum_phase=curriculum_phase,
    )
    initial_state = comparison.state_dict()
    task_to_index = {str(row["example_id"]): index for index, row in enumerate(rendered_rows)}
    planner = AdaptiveGroupRepeatSampler(
        comparison,
        task_to_index,
        num_groups=num_groups,
        num_generations=num_generations,
    )
    groups = [planner.select_group() for _ in range(num_groups)]
    task_ids = tuple(group.task_id for group in groups)
    group_ids = tuple(f"{run_id}-group-{index:05d}" for index in range(num_groups))
    rendered_pool_identity = [
        {
            "example_id": row["example_id"],
            "content_hash": row["content_hash"],
            "prompt_sha256": identity_hash(str(row["prompt"])),
            "rendered_prompt_tokens": row["rendered_prompt_tokens"],
        }
        for row in rendered_rows
    ]
    rendered_pool_hash = identity_hash(rendered_pool_identity)
    schedule_identity: dict[str, Any] = {
        "run_id": run_id,
        "seed": seed,
        "num_generations": num_generations,
        "task_ids": task_ids,
        "group_ids": group_ids,
        "rendered_pool_hash": rendered_pool_hash,
    }
    if sampler_mode != "uniform":
        schedule_identity.update(
            {
                "sampler_mode": sampler_mode,
                "estimate_artifact_sha256": estimate_artifact_sha256,
            }
        )
    schedule_hash = identity_hash(schedule_identity)
    return PreparedUniformSchedule(
        rows=tuple(rendered_rows),
        task_ids=task_ids,
        group_ids=group_ids,
        num_generations=num_generations,
        schedule_hash=schedule_hash,
        rendered_pool_hash=rendered_pool_hash,
        maximum_rendered_prompt_tokens=max(token_lengths),
        initial_sampler_state=initial_state,
        final_planning_sampler_state=planner.sampler.state_dict(),
        sampler_mode=sampler_mode,
        estimate_artifact_sha256=estimate_artifact_sha256,
    )


def load_qualified_adaptive_estimates(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_dataset_fingerprint: str,
    expected_starting_adapter_identity: str,
) -> tuple[list[TaskEstimate], str]:
    """Load a complete identity-bound p/q/u table for the TRAIN candidate pool."""

    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"adaptive estimate artifact is unreadable: {error}") from error
    if not isinstance(artifact, Mapping):
        raise ValueError("adaptive estimate artifact must be a JSON object")
    expected_identity = {
        "status": "QUALIFIED",
        "data_role": "DEVELOPMENT",
        "dataset_fingerprint": expected_dataset_fingerprint,
        "starting_adapter_identity": expected_starting_adapter_identity,
    }
    mismatched = {
        name: {"expected": expected, "observed": artifact.get(name)}
        for name, expected in expected_identity.items()
        if artifact.get(name) != expected
    }
    if mismatched:
        raise ValueError(f"adaptive estimate artifact identity mismatch: {mismatched}")
    raw_estimates = artifact.get("task_estimates")
    if not isinstance(raw_estimates, list):
        raise ValueError("adaptive estimate artifact has no task_estimates list")
    row_families = {str(row["example_id"]): str(row["task_family"]) for row in rows}
    estimates_by_id: dict[str, TaskEstimate] = {}
    for raw in raw_estimates:
        if not isinstance(raw, Mapping):
            raise ValueError("adaptive estimate rows must be JSON objects")
        task_id = raw.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("adaptive estimate task_id must be non-empty text")
        if task_id in estimates_by_id:
            raise ValueError(f"duplicate adaptive estimate task_id: {task_id}")
        if not isinstance(raw.get("provenance"), str) or not raw["provenance"]:
            raise ValueError(f"adaptive estimate {task_id} has no provenance")
        values: dict[str, float] = {}
        for name in ("p", "q", "u"):
            observed = raw.get(name)
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or not 0.0 <= float(observed) <= 1.0
            ):
                raise ValueError(f"adaptive estimate {task_id} has invalid {name}")
            values[name] = float(observed)
        family = raw.get("task_family", row_families.get(task_id))
        estimates_by_id[task_id] = TaskEstimate(
            task_id=task_id,
            family=str(family),
            p=values["p"],
            q=values["q"],
            u=values["u"],
        )
    if set(estimates_by_id) != set(row_families):
        missing = sorted(set(row_families) - set(estimates_by_id))
        extra = sorted(set(estimates_by_id) - set(row_families))
        raise ValueError(
            f"adaptive estimate coverage must exactly match TRAIN: missing={missing}, extra={extra}"
        )
    family_mismatches = sorted(
        task_id
        for task_id, estimate in estimates_by_id.items()
        if estimate.family != row_families[task_id]
    )
    if family_mismatches:
        raise ValueError(f"adaptive estimate task-family mismatch: {family_mismatches}")
    return [estimates_by_id[str(row["example_id"])] for row in rows], file_sha256(path)


def load_selected_reward_architecture(
    path: Path,
    *,
    workspace: Path,
    suite: RLSuite,
) -> tuple[Literal["structured", "terminal"], str, str]:
    """Validate E16 selection and every hash-bound source before E18 may plan."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"selected reward artifact is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("selected reward artifact must be a JSON object")
    identity = value.pop("selection_identity", None)
    failures = []
    if identity != identity_hash(value):
        failures.append("selection identity mismatch")
    expected = {
        "status": "SELECTED",
        "experiment_id": "E16",
        "protocol_id": suite.protocol_id,
        "selection_policy_id": SELECTION_POLICY_ID,
        "scientific_state": "DEVELOPMENT",
        "selected_reward_config_sha256": suite.reward_config_sha256,
        "selected_reward_implementation": TOOL_REWARD_IMPLEMENTATION,
        "protected_access_state": "NOT_ACCESSED",
        "training_seed_count": 1,
    }
    failures.extend(
        f"{key} mismatch"
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    )
    architecture = value.get("selected_reward_architecture")
    if architecture not in ("structured", "terminal"):
        failures.append("unsupported selected reward architecture")
    sources = value.get("source_artifacts")
    if not isinstance(sources, Mapping):
        failures.append("source artifact inventory missing")
    else:
        for name, record in sources.items():
            if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
                failures.append(f"invalid source record {name}")
                continue
            source = (workspace / str(record["path"])).resolve()
            try:
                source.relative_to(workspace.resolve())
            except ValueError:
                failures.append(f"source escapes workspace: {name}")
                continue
            if not source.is_file() or record.get("sha256") != file_sha256(source):
                failures.append(f"source hash mismatch: {name}")
    if failures:
        raise ValueError(f"selected reward artifact failed validation: {sorted(failures)}")
    return cast(Literal["structured", "terminal"], architecture), file_sha256(path), str(identity)


def _validate_e18_dependencies(
    root: Path,
    *,
    suite: RLSuite,
    estimate_sha256: str,
    starting_adapter_identity: str,
) -> dict[str, str]:
    paths = {
        "E13": root / "artifacts/training/E13/E13-difficulty-grpo-seed1701-v1/RUN_COMPLETE.json",
        "E14": root / "evidence/training/E14/POLICY_EQUIVALENCE.json",
        "E15": root
        / "artifacts/training/E15/E15-reward-variance-grpo-seed1701-v1/RUN_COMPLETE.json",
    }
    hashes: dict[str, str] = {}
    for experiment_id, path in paths.items():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"E18 dependency {experiment_id} is unreadable: {error}") from error
        if experiment_id == "E14":
            valid = (
                value.get("status") == "MEASURED_POLICY_EQUIVALENCE"
                and value.get("computed_seeded_sequences_equal") is True
                and value.get("cuda_run_executed_for_e14") is False
            )
        else:
            valid = (
                value.get("status") == "COMPLETE"
                and value.get("execution_mode") == "FULL"
                and value.get("experiment_id") == experiment_id
                and value.get("completed_rollouts") == suite.budget.rollouts
            )
        if (
            not valid
            or value.get("suite_config_hash") != suite.config_hash
            or value.get("adaptive_estimates_sha256") != estimate_sha256
            or value.get("starting_adapter_identity") != starting_adapter_identity
        ):
            raise ValueError(f"E18 dependency {experiment_id} identity is incompatible")
        hashes[experiment_id] = file_sha256(path)
    return hashes


def _validate_e20_dependencies(
    root: Path,
    *,
    suite: RLSuite,
    estimate_sha256: str,
    starting_adapter_identity: str,
    selected_reward_sha256: str,
    selected_reward_identity: str,
) -> dict[str, str]:
    paths = {
        "E18": root / "artifacts/training/E18/E18-rcac-grpo-seed1701-v1/RUN_COMPLETE.json",
        "E19_equivalence": root / "evidence/training/E19/POLICY_EQUIVALENCE.json",
        "E20_qualification": root / "evidence/training/E20/POLICY_QUALIFICATION.json",
    }
    try:
        values = {
            name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()
        }
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"E20 dependency evidence is unreadable: {error}") from error
    e18 = values["E18"]
    e19 = dict(values["E19_equivalence"])
    e20 = dict(values["E20_qualification"])
    e19_identity = e19.pop("evidence_identity", None)
    e20_identity = e20.pop("evidence_identity", None)
    valid = (
        e18.get("status") == "COMPLETE"
        and e18.get("experiment_id") == "E18"
        and e18.get("execution_mode") == "FULL"
        and e18.get("suite_config_hash") == suite.config_hash
        and e18.get("completed_rollouts") == suite.budget.rollouts
        and e18.get("adaptive_estimates_sha256") == estimate_sha256
        and e18.get("starting_adapter_identity") == starting_adapter_identity
        and e18.get("selected_reward_sha256") == selected_reward_sha256
        and e18.get("selected_reward_identity") == selected_reward_identity
        and e18.get("reward_architecture") == "terminal"
        and e19_identity == identity_hash(e19)
        and e19.get("status") == "MEASURED_POLICY_EQUIVALENCE"
        and e19.get("computed_sequence_equals_frozen_e18_schedule") is True
        and e19.get("source_run_complete_sha256") == file_sha256(paths["E18"])
        and e20_identity == identity_hash(e20)
        and e20.get("status") == "QUALIFIED_DISTINCT_POLICY"
        and e20.get("cuda_reuse_authorized") is False
        and e20.get("distinct_cuda_run_required") is True
        and e20.get("computed_seeded_sequences_equal") is False
        and e20.get("selected_reward_sha256") == selected_reward_sha256
        and e20.get("selected_reward_identity") == selected_reward_identity
    )
    if not valid:
        raise ValueError("E20 dependency/equivalence identities are incompatible")
    return {name: file_sha256(path) for name, path in paths.items()}


class GroupValidatedReward:
    """Assert GRPO group identity before delegating to the sole reward owner."""

    def __init__(
        self,
        reward: CanonicalBatchReward,
        *,
        schedule: PreparedUniformSchedule,
        journal: CompletedStepJournal,
        num_generations: int,
        evidence_writer: DurableGroupEvidenceWriter,
    ) -> None:
        self.reward = reward
        self.schedule = schedule
        self.journal = journal
        self.num_generations = num_generations
        self.evidence_writer = evidence_writer

    def __call__(self, *, completions: Sequence[Any], **columns: Any) -> list[float]:
        if len(completions) != self.num_generations:
            raise RuntimeError("reward batch does not contain exactly one complete GRPO group")
        ordinal = self.journal.completed_groups
        if ordinal >= len(self.schedule.task_ids):
            raise RuntimeError("reward batch exceeds the frozen rollout schedule")
        example_ids = columns.get("example_id")
        if not isinstance(example_ids, Sequence) or isinstance(example_ids, (str, bytes)):
            raise RuntimeError("reward batch has no repeated example identity")
        expected = self.schedule.task_ids[ordinal]
        if list(example_ids) != [expected] * self.num_generations:
            raise RuntimeError("GRPO group members do not share the frozen task identity")
        metadata_values = columns.get("metadata")
        if not isinstance(metadata_values, Sequence) or isinstance(metadata_values, (str, bytes)):
            raise RuntimeError("reward batch has no state metadata")
        initial_identities: set[str] = set()
        for raw in metadata_values:
            metadata = json.loads(raw) if isinstance(raw, str) else dict(raw)
            group_key = metadata.get("group_key")
            initial_identity = metadata.get("initial_state_identity")
            if group_key is not None:
                environment = GroupedEnvironmentFactory(
                    seed=int(metadata["generator_seed"])
                ).create(str(group_key))
                if environment.initial_state_identity != initial_identity:
                    raise RuntimeError("tool environment initial-state identity mismatch")
                initial_identities.add(environment.initial_state_identity)
        if len(initial_identities) > 1:
            raise RuntimeError("one GRPO group received inconsistent initial tool states")
        delegated = dict(columns)
        delegated["rollout_group_id"] = [self.schedule.group_ids[ordinal]] * self.num_generations
        captured: dict[str, list[Any]] = {}
        upstream_log_extra = delegated.get("log_extra")

        def capture_log_extra(name: str, values: list[Any]) -> None:
            captured[name] = list(values)
            if callable(upstream_log_extra):
                upstream_log_extra(name, values)

        delegated["log_extra"] = capture_log_extra
        rewards = self.reward(completions=completions, **delegated)
        self.evidence_writer.append_group(
            ordinal=ordinal,
            group_id=self.schedule.group_ids[ordinal],
            example_id=expected,
            completions=completions,
            rewards=rewards,
            reward_logs=captured,
            task_family=columns["task_family"],
            reference=columns["reference"],
            metadata=metadata_values,
        )
        return rewards


class DurableGroupEvidenceWriter:
    """Append and fsync one complete rollout/reward record before journal commit."""

    def __init__(self, path: Path, *, run_id: str, resume_group: int | None = None) -> None:
        self.path = path
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._attempts: dict[int, int] = {}
        existing_groups: list[int] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        "rollout evidence JSONL contains a partial record"
                    ) from error
                if event.get("run_id") != run_id:
                    raise RuntimeError("rollout evidence belongs to a different run")
                if event.get("event") == "GROUP_REWARD_SCORED":
                    ordinal = int(event["group_ordinal"])
                    existing_groups.append(ordinal)
                    self._attempts[ordinal] = self._attempts.get(ordinal, 0) + 1
        if existing_groups and resume_group is None:
            raise RuntimeError("existing rollout evidence requires explicit checkpoint resume")
        if resume_group is not None:
            self._append(
                {
                    "event": "RESUME_BOUNDARY",
                    "run_id": run_id,
                    "resume_completed_groups": resume_group,
                    "prior_scored_groups_not_checkpointed": sorted(
                        ordinal for ordinal in existing_groups if ordinal >= resume_group
                    ),
                }
            )

    def _append(self, event: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            json.dump(event, handle, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def append_group(
        self,
        *,
        ordinal: int,
        group_id: str,
        example_id: str,
        completions: Sequence[Any],
        rewards: Sequence[float],
        reward_logs: Mapping[str, Sequence[Any]],
        task_family: Sequence[Any],
        reference: Sequence[Any],
        metadata: Sequence[Any],
    ) -> None:
        attempt = self._attempts.get(ordinal, 0) + 1
        event = {
            "event": "GROUP_REWARD_SCORED",
            "run_id": self.run_id,
            "group_ordinal": ordinal,
            "group_id": group_id,
            "attempt": attempt,
            "example_id": example_id,
            "completion_count": len(completions),
            "completions": list(completions),
            "rewards": list(rewards),
            "reward_logs": {name: list(values) for name, values in reward_logs.items()},
            "task_family": list(task_family),
            "reference": list(reference),
            "metadata": list(metadata),
        }
        self._append(event)
        self._attempts[ordinal] = attempt


class TrainerGroupRepeatSampler:
    """Repeat each selected generation batch for TRL's buffered microsteps."""

    def __init__(self, groups: AdaptiveGroupRepeatSampler, *, repeat_count: int) -> None:
        if repeat_count < 1:
            raise ValueError("TRL generation-group repeat count must be positive")
        self.groups = groups
        self.repeat_count = repeat_count

    def __len__(self) -> int:
        return len(self.groups) * self.repeat_count

    def __iter__(self) -> Iterable[int]:
        while self.groups.completed_groups < self.groups.num_groups:
            group = self.groups.select_group()
            for _ in range(self.repeat_count):
                yield from group.indices


def _sampler_from_schedule(
    schedule: PreparedUniformSchedule,
    *,
    completed_groups: int = 0,
    restored_comparison_state: Mapping[str, Any] | None = None,
) -> AdaptiveGroupRepeatSampler:
    state = restored_comparison_state or schedule.initial_sampler_state
    comparison = ComparisonSampler.from_state_dict(state)
    mapping = {str(row["example_id"]): index for index, row in enumerate(schedule.rows)}
    return AdaptiveGroupRepeatSampler(
        comparison,
        mapping,
        num_groups=len(schedule.task_ids),
        num_generations=schedule.num_generations,
        completed_groups=completed_groups,
    )


def _completed_prefix_sampler_state(
    schedule: PreparedUniformSchedule, completed_groups: int
) -> dict[str, Any]:
    """Reconstruct state at the consumed-group boundary, excluding prefetch.

    A DataLoader may ask the sampler for the next generation group before the
    current optimizer step checkpoint is published. The live sampler is then
    ahead of reward/journal accounting and is not a resumable state. Replaying
    the immutable schedule from its initial state creates the exact state after
    only the groups that completed reward scoring and gradient accounting.
    """

    if not 0 <= completed_groups <= len(schedule.task_ids):
        raise RuntimeError("completed group count is outside the frozen schedule")
    sampler = ComparisonSampler.from_state_dict(schedule.initial_sampler_state)
    observed = tuple(sampler.select(1)[0] for _ in range(completed_groups))
    expected = schedule.task_ids[:completed_groups]
    if observed != expected:
        raise RuntimeError("completed-prefix sampler replay differs from frozen task schedule")
    state = sampler.state_dict()
    if state.get("total_selections") != completed_groups:
        raise RuntimeError("completed-prefix sampler state has incorrect selection accounting")
    return state


def _read_training_rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in pq.read_table(path).to_pylist()]


def _publish_or_validate_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish an identity once, or require byte-semantic equality on resume."""

    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"existing run identity is unreadable: {path}") from error
        if existing != dict(value):
            raise RuntimeError(f"existing run identity does not match resumed execution: {path}")
        return
    _atomic_json(path, value)


def plan_e12(
    workspace: Path,
    *,
    config_path: Path,
    tokenizer: Any,
    pilot: bool,
    pilot_manifest: Path | None = None,
) -> tuple[dict[str, Any], PreparedUniformSchedule]:
    """Build the complete CPU-side E12 launch plan without touching CUDA."""

    return plan_grpo(
        workspace,
        config_path=config_path,
        tokenizer=tokenizer,
        experiment_id="E12",
        pilot=pilot,
        pilot_manifest=pilot_manifest,
    )


def plan_grpo(
    workspace: Path,
    *,
    config_path: Path,
    tokenizer: Any,
    experiment_id: str,
    pilot: bool,
    pilot_manifest: Path | None = None,
) -> tuple[dict[str, Any], PreparedUniformSchedule]:
    """Build a qualified CPU-side GRPO launch plan without touching CUDA."""

    root = workspace.resolve()
    suite = RLSuite.load(config_path)
    if experiment_id not in _QUALIFIED_GRPO_RUNS:
        raise ValueError("qualified GRPO runner supports E12, E13, E15, E16, E18, and E20")
    preflight = suite.preflight(experiment_id, workspace=root)
    mode, run_stem, pilot_version, reward_architecture = _QUALIFIED_GRPO_RUNS[experiment_id]
    run_id = f"{run_stem}-pilot-seed1701-v{pilot_version}" if pilot else f"{run_stem}-seed1701-v1"
    groups = suite.budget.groups_per_optimizer_step if pilot else suite.budget.rollout_groups
    rows = _read_training_rows(root / suite.dataset_path)
    adaptive_estimates: list[TaskEstimate] | None = None
    estimate_sha256: str | None = None
    selected_reward_sha256: str | None = None
    selected_reward_identity: str | None = None
    dependency_hashes: dict[str, str] | None = None
    if experiment_id in _ADAPTIVE_ESTIMATE_EXPERIMENTS:
        if not preflight["launch_authorized"]:
            raise RuntimeError(f"{experiment_id} launch blocked by preflight: {preflight}")
        starting_identity = preflight.get("starting_adapter_identity")
        if not isinstance(starting_identity, str):
            raise RuntimeError(
                f"{experiment_id} preflight did not resolve the frozen E10 adapter identity"
            )
        adaptive_estimates, estimate_sha256 = load_qualified_adaptive_estimates(
            root / suite.adaptive_estimates_ref,
            rows,
            expected_dataset_fingerprint=suite.dataset_fingerprint,
            expected_starting_adapter_identity=starting_identity,
        )
    if experiment_id in _SELECTED_REWARD_EXPERIMENTS:
        selection_path = root / "evidence" / "training" / "E16" / "SELECTED_REWARD.json"
        reward_architecture, selected_reward_sha256, selected_reward_identity = (
            load_selected_reward_architecture(
                selection_path,
                workspace=root,
                suite=suite,
            )
        )
        if experiment_id == "E18":
            dependency_hashes = _validate_e18_dependencies(
                root,
                suite=suite,
                estimate_sha256=str(estimate_sha256),
                starting_adapter_identity=str(preflight["starting_adapter_identity"]),
            )
        else:
            dependency_hashes = _validate_e20_dependencies(
                root,
                suite=suite,
                estimate_sha256=str(estimate_sha256),
                starting_adapter_identity=str(preflight["starting_adapter_identity"]),
                selected_reward_sha256=str(selected_reward_sha256),
                selected_reward_identity=str(selected_reward_identity),
            )
    schedule = prepare_uniform_schedule(
        rows,
        tokenizer=tokenizer,
        rcac_config=_load_rcac_config(root / suite.sampler_config_ref),
        seed=suite.experiments[experiment_id].arms[0].seed,
        num_groups=groups,
        num_generations=suite.budget.num_generations,
        max_prompt_tokens=suite.budget.max_prompt_tokens,
        run_id=run_id,
        sampler_mode=mode,
        task_estimates=adaptive_estimates,
        estimate_artifact_sha256=estimate_sha256,
        curriculum_phase=experiment_id,
    )
    plan = {
        **preflight,
        "run_id": run_id,
        "execution_mode": "PILOT" if pilot else "FULL",
        "scientific_state": "PILOT" if pilot else "CANDIDATE_SELECTION",
        "optimizer_steps": 1 if pilot else suite.budget.optimizer_steps,
        "rollout_groups": groups,
        "rollouts": groups * suite.budget.num_generations,
        "schedule_hash": schedule.schedule_hash,
        "rendered_pool_hash": schedule.rendered_pool_hash,
        "maximum_rendered_prompt_tokens": schedule.maximum_rendered_prompt_tokens,
        "generation_batch_size": suite.budget.generation_batch_size,
        "steps_per_generation": suite.budget.steps_per_generation,
        "gpu_touched": False,
    }
    if experiment_id == "E16":
        plan.update(
            {
                "reward_architecture": reward_architecture,
                "executed_arm_id": "E16_terminal",
                "reused_arm_id": "E16_structured",
                "structured_control": _validate_e12_structured_control(
                    root, suite=suite, candidate_schedule=schedule, pilot=pilot
                ),
                "checkpoint_evidence_steps": ([] if pilot else list(range(4, 33, 4))),
                "checkpoint_save_total_limit": (suite.budget.checkpoint_retention if pilot else 8),
            }
        )
    if experiment_id in _ADAPTIVE_ESTIMATE_EXPERIMENTS:
        plan.update(
            {
                "sampler_mode": mode,
                "adaptive_estimates_path": suite.adaptive_estimates_ref.as_posix(),
                "adaptive_estimates_sha256": estimate_sha256,
                "adaptive_estimate_count": len(adaptive_estimates or ()),
            }
        )
    if experiment_id in _SELECTED_REWARD_EXPERIMENTS:
        plan.update(
            {
                "reward_architecture": reward_architecture,
                "selected_reward_path": "evidence/training/E16/SELECTED_REWARD.json",
                "selected_reward_sha256": selected_reward_sha256,
                "selected_reward_identity": selected_reward_identity,
                "dependency_artifact_sha256": dependency_hashes,
            }
        )
    if not pilot:
        timing_path = pilot_manifest or Path(
            f"artifacts/training/{experiment_id}/{run_stem}-pilot-seed1701-v{pilot_version}/"
            "RUN_COMPLETE.json"
        )
        if not timing_path.is_absolute():
            timing_path = root / timing_path
        plan["timing_pilot_manifest"] = timing_path.as_posix()
        try:
            plan["estimated_optimizer_step_seconds"] = _pilot_step_seconds(
                timing_path,
                suite_hash=suite.config_hash,
                experiment_id=experiment_id,
                sampler_mode=(mode if experiment_id in _ADAPTIVE_ESTIMATE_EXPERIMENTS else None),
                adaptive_estimates_sha256=(
                    estimate_sha256 if experiment_id in _ADAPTIVE_ESTIMATE_EXPERIMENTS else None
                ),
                reward_architecture=(
                    reward_architecture
                    if experiment_id == "E16" or experiment_id in _SELECTED_REWARD_EXPERIMENTS
                    else None
                ),
            )
        except RuntimeError as error:
            plan["launch_authorized"] = False
            unresolved = list(plan["unresolved_material_decisions"])
            unresolved.append(str(error))
            plan["unresolved_material_decisions"] = unresolved
    return plan, schedule


def _pilot_step_seconds(
    path: Path,
    *,
    suite_hash: str,
    experiment_id: str = "E12",
    sampler_mode: ComparisonMode | None = None,
    adaptive_estimates_sha256: str | None = None,
    reward_architecture: str | None = None,
) -> float:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"{experiment_id} timing pilot evidence is unreadable: {error}"
        ) from error
    if (
        value.get("status") != "COMPLETE"
        or value.get("experiment_id") != experiment_id
        or value.get("execution_mode") != "PILOT"
        or value.get("suite_config_hash") != suite_hash
    ):
        raise RuntimeError(
            f"{experiment_id} timing pilot identity is incompatible with the full run"
        )
    if sampler_mode is not None and value.get("sampler_mode") != sampler_mode:
        raise RuntimeError(f"{experiment_id} timing pilot sampler identity is incompatible")
    if (
        adaptive_estimates_sha256 is not None
        and value.get("adaptive_estimates_sha256") != adaptive_estimates_sha256
    ):
        raise RuntimeError(
            f"{experiment_id} timing pilot adaptive-estimate identity is incompatible"
        )
    if reward_architecture is not None and value.get("reward_architecture") != reward_architecture:
        raise RuntimeError(f"{experiment_id} timing pilot reward architecture is incompatible")
    seconds = value.get("optimizer_step_seconds")
    if not isinstance(seconds, (int, float)) or float(seconds) <= 0.0:
        raise RuntimeError(f"{experiment_id} timing pilot has no positive optimizer-step duration")
    return float(seconds)


def _validate_e12_structured_control(
    root: Path,
    *,
    suite: RLSuite,
    candidate_schedule: PreparedUniformSchedule,
    pilot: bool,
) -> dict[str, Any]:
    """Bind E16's reused structured arm to the immutable measured E12 run."""

    e12 = suite.experiments["E12"].arms
    e16 = suite.experiments["E16"].arms
    if len(e12) != 1 or e12[0].sampler != "uniform" or e12[0].reward_architecture != "structured":
        raise RuntimeError("E12 is not the frozen uniform/structured control")
    if {(arm.sampler, arm.reward_architecture, arm.seed) for arm in e16} != {
        ("uniform", "terminal", e12[0].seed),
        ("uniform", "structured", e12[0].seed),
    }:
        raise RuntimeError("E16 arms do not isolate reward architecture from E12")
    control_dir = root / "artifacts" / "training" / "E12" / "E12-uniform-grpo-seed1701-v1"
    schedule_path = control_dir / "ROLLOUT_SCHEDULE.json"
    complete_path = control_dir / "RUN_COMPLETE.json"
    identity_path = control_dir / "RUN_IDENTITY.json"
    try:
        frozen = json.loads(schedule_path.read_text(encoding="utf-8"))
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"E12 structured-control evidence is unreadable: {error}") from error
    expected_groups = (
        suite.budget.groups_per_optimizer_step if pilot else suite.budget.rollout_groups
    )
    control_ids = frozen.get("task_ids")
    if not isinstance(control_ids, list) or control_ids[:expected_groups] != list(
        candidate_schedule.task_ids
    ):
        raise RuntimeError("E16 task sequence does not match the frozen E12 structured control")
    checks = {
        "schedule_status": frozen.get("status") == "PREPARED",
        "sampler": frozen.get("sampler") == "uniform",
        "suite": frozen.get("suite_config_hash") == suite.config_hash,
        "rendered_pool": frozen.get("rendered_pool_hash") == candidate_schedule.rendered_pool_hash,
        "complete": complete.get("status") == "COMPLETE"
        and complete.get("experiment_id") == "E12"
        and complete.get("execution_mode") == "FULL",
        "complete_schedule": complete.get("schedule_hash") == frozen.get("schedule_hash"),
        "budget": complete.get("optimizer_steps") == suite.budget.optimizer_steps
        and complete.get("completed_rollouts") == suite.budget.rollouts,
        "fairness": identity.get("fairness_budget_hash") == suite.budget.fairness_identity(),
        "start": identity.get("starting_adapter_identity")
        == complete.get("starting_adapter_identity"),
        "reward": identity.get("reward_config_hash") == suite.reward_config_sha256
        and identity.get("reward_implementation") == TOOL_REWARD_IMPLEMENTATION,
        "protected": identity.get("protected_access_state") == "NOT_ACCESSED",
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise RuntimeError(f"E12 structured-control identity failed: {failed}")
    return {
        "disposition": "REUSE_MEASURED_E12_STRUCTURED_ARM",
        "source_experiment_id": "E12",
        "source_run_id": complete.get("run_id"),
        "source_schedule_hash": frozen.get("schedule_hash"),
        "source_schedule_sha256": file_sha256(schedule_path),
        "source_run_complete_sha256": file_sha256(complete_path),
        "source_run_identity_sha256": file_sha256(identity_path),
        "task_sequence_prefix_groups": expected_groups,
        "task_sequence_equal": True,
        "new_cuda_arm": "E16_terminal",
        "redundant_cuda_arm_omitted": "E16_structured",
    }


def _final_checkpoint_cadence(step_seconds: float, optimizer_steps: int) -> int:
    """Choose a timing-derived cadence that always lands on the final step."""

    proposed = derive_save_interval(
        median_optimizer_step_seconds=step_seconds,
        total_optimizer_steps=optimizer_steps,
    ).save_steps
    divisors = [step for step in range(1, optimizer_steps + 1) if optimizer_steps % step == 0]
    return min(divisors, key=lambda step: (abs(step - proposed), -step))


def _publish_completed_step_after_native(
    journal: CompletedStepJournal,
    *,
    checkpoint: Path,
    step: int,
    sampler_state: Mapping[str, Any],
    execution_identity: Mapping[str, object],
) -> Path:
    """Bind sampler state only after the native checkpoint marker verifies."""

    validate_native_trainer_checkpoint(checkpoint, execution_identity)
    return journal.publish_completed_step(
        step,
        native_checkpoint=checkpoint,
        sampler_state=sampler_state,
    )


def execute_e12(
    workspace: Path,
    *,
    config_path: Path,
    pilot: bool,
    resume: bool = False,
    pilot_manifest: Path | None = None,
) -> dict[str, Any]:
    """Execute E12 on CUDA after all CPU identities pass qualification."""

    return execute_grpo(
        workspace,
        config_path=config_path,
        experiment_id="E12",
        pilot=pilot,
        resume=resume,
        pilot_manifest=pilot_manifest,
    )


def execute_grpo(
    workspace: Path,
    *,
    config_path: Path,
    experiment_id: str,
    pilot: bool,
    resume: bool = False,
    pilot_manifest: Path | None = None,
) -> dict[str, Any]:
    """Execute a qualified GRPO arm on the single CUDA GPU."""

    root = workspace.resolve()
    suite = RLSuite.load(config_path)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        suite.model_id,
        revision=suite.model_revision,
        local_files_only=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    plan, schedule = plan_grpo(
        root,
        config_path=config_path,
        tokenizer=tokenizer,
        experiment_id=experiment_id,
        pilot=pilot,
        pilot_manifest=pilot_manifest,
    )
    if not plan["launch_authorized"]:
        raise RuntimeError(f"{experiment_id} launch blocked by preflight: {plan}")
    run_id = str(plan["run_id"])
    with GpuOwnerLease(root, experiment_id, run_id):
        return _execute_grpo_under_lease(
            root,
            suite=suite,
            plan=plan,
            schedule=schedule,
            tokenizer=tokenizer,
            resume=resume,
        )


def execute_e28_qualification_worker(
    workspace: Path,
    *,
    qualification_config: Path,
    arm: Literal["reference", "recovery"],
    stage: Literal["full", "controlled-stop", "resume"],
) -> dict[str, Any]:
    """Run one compact, actual GRPO recovery stage under a distinct process."""

    from transformers import AutoTokenizer

    from reasonforge.checkpoints.qualification import load_qualification_suite

    if (arm, stage) not in {
        ("reference", "full"),
        ("recovery", "controlled-stop"),
        ("recovery", "resume"),
    }:
        raise ValueError(f"invalid E28 arm/stage combination: {arm}/{stage}")
    root = workspace.resolve()
    qualification_suite = load_qualification_suite(qualification_config)
    qualification_spec = qualification_suite.experiments["E28"]
    suite = RLSuite.load(root / qualification_spec.source_config)
    tokenizer = AutoTokenizer.from_pretrained(
        suite.model_id,
        revision=suite.model_revision,
        local_files_only=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    source_plan, schedule = plan_grpo(
        root,
        config_path=root / qualification_spec.source_config,
        tokenizer=tokenizer,
        experiment_id="E12",
        pilot=False,
    )
    if not source_plan["launch_authorized"]:
        raise RuntimeError(f"E28 source GRPO plan failed qualification: {source_plan}")
    run_id = (
        qualification_spec.reference_run_id
        if arm == "reference"
        else qualification_spec.recovery_run_id
    )
    plan = {
        **source_plan,
        "experiment_id": "E28",
        "run_id": run_id,
        "execution_mode": "QUALIFICATION",
        "optimizer_steps": qualification_spec.max_steps,
        "completed_rollout_groups": (
            qualification_spec.max_steps * int(qualification_spec.groups_per_step or 0)
        ),
        "completed_rollouts": (
            qualification_spec.max_steps
            * int(qualification_spec.groups_per_step or 0)
            * int(qualification_spec.generations_per_group or 0)
        ),
        "reward_architecture": "structured",
    }
    output_dir = qualification_suite.output_root / "E28" / run_id
    qualification = {
        "source_experiment_id": "E12",
        "output_dir": output_dir.as_posix(),
        "save_steps": qualification_spec.save_steps,
        "save_total_limit": qualification_spec.save_total_limit,
        "seed": qualification_spec.seed,
        "protocol_id": qualification_suite.protocol_id,
        "suite_config_hash": file_sha256(qualification_config),
        "source_execution_config_sha256": qualification_spec.source_execution_config_sha256,
        "interruption_after_step": qualification_spec.interruption_after_step,
        "resume_adapter_dtype_policy": qualification_spec.resume_adapter_dtype_policy,
        "sampler_checkpoint_policy": qualification_spec.sampler_checkpoint_policy,
        "controlled_stop_after": (
            qualification_spec.interruption_after_step if stage == "controlled-stop" else None
        ),
    }
    with GpuOwnerLease(root, "E28", f"{run_id}-{stage}"):
        return _execute_grpo_under_lease(
            root,
            suite=suite,
            plan=plan,
            schedule=schedule,
            tokenizer=tokenizer,
            resume=stage == "resume",
            qualification=qualification,
        )


def _execute_grpo_under_lease(
    root: Path,
    *,
    suite: RLSuite,
    plan: Mapping[str, Any],
    schedule: PreparedUniformSchedule,
    tokenizer: Any,
    resume: bool,
    qualification: Mapping[str, Any] | None = None,
    replication: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a qualified heavy GRPO path while the caller holds the GPU lease."""

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, TrainerCallback
    from trl import GRPOTrainer

    experiment_id = str(plan["experiment_id"])
    if qualification is not None and replication is not None:
        raise RuntimeError("GRPO execution cannot be both qualification and replication")
    source_experiment_id = experiment_id
    if qualification is not None:
        source_experiment_id = str(qualification["source_experiment_id"])
    elif replication is not None:
        source_experiment_id = str(replication["source_experiment_id"])
    e25_replication = (
        experiment_id == "E25"
        and replication is not None
        and source_experiment_id in {"E16", "E18"}
    )
    if experiment_id not in _QUALIFIED_GRPO_RUNS and not (
        (experiment_id == "E28" and source_experiment_id == "E12") or e25_replication
    ):
        raise RuntimeError("unqualified GRPO experiment reached the heavy execution path")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"{experiment_id} requires the qualified single CUDA GPU")
    run_id = str(plan["run_id"])
    external_output = qualification if qualification is not None else replication
    output_dir = (
        root / Path(str(external_output["output_dir"]))
        if external_output is not None
        else root / "artifacts" / "training" / experiment_id / run_id
    )
    if (output_dir / "RUN_COMPLETE.json").exists():
        raise RuntimeError(
            f"the requested {experiment_id} run is already complete and must not be overwritten"
        )
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise RuntimeError(
            f"an incomplete {experiment_id} output exists; preserve it and use explicit resume"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule_manifest = {**schedule.manifest(), "suite_config_hash": suite.config_hash}
    _publish_or_validate_json(output_dir / "ROLLOUT_SCHEDULE.json", schedule_manifest)

    optimizer_steps = int(plan["optimizer_steps"])
    pilot = plan["execution_mode"] == "PILOT"
    estimated_step_seconds = 60.0 if pilot else float(plan["estimated_optimizer_step_seconds"])
    semantics = suite.trainer_semantics
    checkpoint_save_steps = (
        int(external_output["save_steps"])
        if external_output is not None
        else (
            4
            if experiment_id == "E16" and not pilot
            else _final_checkpoint_cadence(estimated_step_seconds, optimizer_steps)
        )
    )
    selected = json.loads((root / suite.adapter_config_ref).read_text(encoding="utf-8"))
    adapter_value = selected["adapter"]
    adapter = LoRAAdapterConfig(
        target=AdapterTarget(str(adapter_value["target"])),
        rank=int(adapter_value["rank"]),
        alpha=int(adapter_value["alpha"]),
        dropout=float(adapter_value["dropout"]),
        use_rslora=bool(adapter_value["use_rslora"]),
    )
    training_run = TrainingRunConfig(
        experiment_id=experiment_id,
        run_id=run_id,
        output_dir=output_dir,
        model=QLoRAModelConfig(suite.model_id, suite.model_revision),
        adapter=adapter,
        dataset_fingerprint=schedule.schedule_hash,
        max_steps=optimizer_steps,
        median_optimizer_step_seconds=estimated_step_seconds,
        per_device_train_batch_size=suite.budget.per_device_train_batch_size,
        gradient_accumulation_steps=suite.budget.gradient_accumulation_steps,
        learning_rate=float(semantics["learning_rate"]),
        warmup_steps=int(semantics["warmup_steps"]),
        weight_decay=float(semantics["weight_decay"]),
        seed=(
            int(external_output["seed"])
            if external_output is not None
            else suite.experiments[source_experiment_id].arms[0].seed
        ),
        data_seed=(
            int(external_output["seed"])
            if external_output is not None
            else suite.experiments[source_experiment_id].arms[0].seed
        ),
    )
    execution_identity = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "execution_mode": (
            "QUALIFICATION"
            if qualification is not None
            else (
                "LOCKED_FINAL_REPLICATION"
                if replication is not None
                else ("PILOT" if pilot else "FULL")
            )
        ),
        "protocol_id": (
            str(external_output["protocol_id"])
            if external_output is not None
            else suite.protocol_id
        ),
        "suite_config_hash": suite.config_hash,
        "fairness_budget_hash": suite.budget.fairness_identity(),
        "schedule_hash": schedule.schedule_hash,
        "starting_adapter_identity": plan["starting_adapter_identity"],
        "reward_config_hash": suite.reward_config_sha256,
        "reward_implementation": TOOL_REWARD_IMPLEMENTATION,
        "sampler_config_hash": suite.sampler_config_sha256,
        "trainer_config_hash": training_run.config_hash,
        "checkpoint_save_steps": checkpoint_save_steps,
        "trainer_generation_group_repeat_count": suite.budget.steps_per_generation,
        "resume_data_skip_policy": "RESTORED_SAMPLER_AND_IGNORE_DATA_SKIP",
        "protected_access_state": "NOT_ACCESSED",
    }
    if qualification is not None:
        execution_identity.update(
            {
                "qualification_suite_config_hash": qualification["suite_config_hash"],
                "source_experiment_id": source_experiment_id,
                "source_execution_config_sha256": qualification["source_execution_config_sha256"],
                "interruption_after_step": qualification["interruption_after_step"],
                "completed_step_resume_only": True,
                "partial_rollout_policy": "PRESERVE_EXCLUDE_AND_DISCARD",
                "resume_adapter_dtype_policy": qualification["resume_adapter_dtype_policy"],
                "sampler_checkpoint_policy": qualification["sampler_checkpoint_policy"],
            }
        )
    if replication is not None:
        execution_identity.update(
            {
                "replication_design_identity": replication["design_identity"],
                "replication_design_sha256": replication["design_sha256"],
                "replication_config_sha256": replication["config_sha256"],
                "source_experiment_id": source_experiment_id,
                "seed": replication["seed"],
                "checkpoint_save_total_limit": replication["save_total_limit"],
                "required_complete_checkpoint_steps": replication[
                    "required_complete_checkpoint_steps"
                ],
                "frozen_schedule_artifact_sha256": replication["schedule_artifact_sha256"],
                "completed_step_resume_only": True,
                "partial_rollout_policy": replication["partial_rollout_policy"],
            }
        )
    adaptive_execution = experiment_id in _ADAPTIVE_ESTIMATE_EXPERIMENTS or (
        e25_replication and source_experiment_id == "E18"
    )
    selected_reward_execution = experiment_id in _SELECTED_REWARD_EXPERIMENTS or e25_replication
    if adaptive_execution:
        execution_identity.update(
            {
                "sampler_mode": schedule.sampler_mode,
                "adaptive_estimates_sha256": plan["adaptive_estimates_sha256"],
            }
        )
    if selected_reward_execution:
        execution_identity.update(
            {
                "reward_architecture": plan["reward_architecture"],
                "selected_reward_sha256": plan["selected_reward_sha256"],
                "selected_reward_identity": plan["selected_reward_identity"],
                "dependency_artifact_sha256": plan["dependency_artifact_sha256"],
            }
        )
    if experiment_id == "E16":
        execution_identity.update(
            {
                "reward_architecture": plan["reward_architecture"],
                "structured_control": plan["structured_control"],
                "checkpoint_evidence_steps": list(range(4, optimizer_steps + 1, 4)),
                "checkpoint_save_total_limit": (suite.budget.checkpoint_retention if pilot else 8),
            }
        )
    _publish_or_validate_json(output_dir / "RUN_IDENTITY.json", execution_identity)
    _publish_or_validate_json(output_dir / "PRELAUNCH.json", plan)
    journal = CompletedStepJournal(
        root / "artifacts" / "rl_orchestration",
        experiment_id=experiment_id,
        run_id=run_id,
        config_hash=identity_hash(execution_identity),
        groups_per_step=suite.budget.groups_per_optimizer_step,
        generations_per_group=suite.budget.num_generations,
    )
    native_resume: Path | None = None
    restored_state: Mapping[str, Any] | None = None
    completed_groups = 0
    if resume:
        resume_plan = journal.resume_plan(mutate=True)
        if resume_plan["action"] != "RESUME":
            raise RuntimeError(
                f"no compatible {experiment_id} checkpoint can resume: {resume_plan}"
            )
        native_resume = Path(str(resume_plan["native_checkpoint"]))
        restored_state = journal.sampler_state
        completed_groups = journal.completed_groups
    group_sampler = _sampler_from_schedule(
        schedule,
        completed_groups=completed_groups,
        restored_comparison_state=restored_state,
    )
    trainer_sampler = TrainerGroupRepeatSampler(
        group_sampler, repeat_count=suite.budget.steps_per_generation
    )
    rollout_evidence = output_dir / "ROLLOUT_REWARD_EVIDENCE.jsonl"
    evidence_writer = DurableGroupEvidenceWriter(
        rollout_evidence,
        run_id=run_id,
        resume_group=completed_groups if resume else None,
    )
    policy = CanonicalRewardPolicy.from_yaml(root / suite.reward_config_ref)
    reward_architecture = cast(
        Literal["structured", "terminal"], str(plan.get("reward_architecture", "structured"))
    )
    base_reward = canonical_reward_functions(
        policy, reward_architecture, num_generations=suite.budget.num_generations
    )[0]
    reward = GroupValidatedReward(
        base_reward,
        schedule=schedule,
        journal=journal,
        num_generations=suite.budget.num_generations,
        evidence_writer=evidence_writer,
    )

    class JournaledGRPOTrainer(GRPOTrainer):  # type: ignore[misc]
        _rf_sampler: TrainerGroupRepeatSampler
        _rf_schedule: PreparedUniformSchedule
        _rf_journal: CompletedStepJournal
        _rf_resume_dtype_audit: dict[str, Any] | None = None

        def _load_from_checkpoint(self, resume_from_checkpoint: str, model: Any = None) -> None:
            """Preserve adapter dtype identity across native PEFT resume."""

            if qualification is None:
                super()._load_from_checkpoint(resume_from_checkpoint, model)
                return
            from reasonforge.checkpoints.qualification_execution import (
                trainable_parameter_dtype_inventory,
            )

            active_model = self.model if model is None else model
            before = trainable_parameter_dtype_inventory(active_model)
            expected_dtypes = {
                name: parameter.dtype
                for name, parameter in active_model.named_parameters()
                if parameter.requires_grad
            }
            if self.optimizer is not None:
                raise RuntimeError("E28 dtype correction must precede optimizer construction")
            super()._load_from_checkpoint(resume_from_checkpoint, model)
            native_loaded = trainable_parameter_dtype_inventory(active_model)
            observed_names = {
                name
                for name, parameter in active_model.named_parameters()
                if parameter.requires_grad
            }
            if observed_names != set(expected_dtypes):
                raise RuntimeError("E28 native load changed the trainable parameter set")
            restored = 0
            for name, parameter in active_model.named_parameters():
                if parameter.requires_grad and parameter.dtype != expected_dtypes[name]:
                    parameter.data = parameter.data.to(dtype=expected_dtypes[name])
                    restored += 1
            corrected = trainable_parameter_dtype_inventory(active_model)
            if corrected != before:
                raise RuntimeError("E28 failed to restore pre-load adapter dtype identity")
            if restored == 0:
                raise RuntimeError("E28 v7 did not exercise the frozen PEFT dtype correction")
            self._rf_resume_dtype_audit = {
                "status": "CORRECTED_BEFORE_OPTIMIZER_CONSTRUCTION",
                "process_id": os.getpid(),
                "policy": qualification["resume_adapter_dtype_policy"],
                "native_resume_from": str(resume_from_checkpoint),
                "optimizer_existed_during_correction": False,
                "before_native_load": before,
                "after_native_load_before_correction": native_loaded,
                "after_correction": corrected,
                "restored_tensor_count": restored,
                "native_load_changed_dtype": native_loaded != before,
            }
            _atomic_json(output_dir / "RESUME_DTYPE_AUDIT.json", self._rf_resume_dtype_audit)

        def _get_train_sampler(self, dataset: Any | None = None) -> Any:
            return self._rf_sampler

        def _generate_and_score_completions(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
            ordinal = self._rf_journal.completed_groups
            if ordinal >= len(self._rf_schedule.task_ids):
                raise RuntimeError(f"generation exceeded the frozen {experiment_id} schedule")
            expected = self._rf_schedule.task_ids[ordinal]
            observed = [str(item.get("example_id")) for item in inputs]
            if observed != [expected] * suite.budget.num_generations:
                raise RuntimeError("trainer sampler broke same-task GRPO grouping")
            group_id = self._rf_schedule.group_ids[ordinal]
            self._rf_journal.begin_group(
                group_id,
                task_id=expected,
                artifact_paths=[rollout_evidence.relative_to(root).as_posix()],
            )
            result = super()._generate_and_score_completions(inputs)
            self._rf_journal.finish_group(group_id)
            return cast(dict[str, Any], result)

    class JournalCheckpointCallback(TrainerCallback):  # type: ignore[misc]
        def on_step_end(self, _args: Any, state: Any, control: Any, **_kwargs: Any) -> Any:
            expected_groups = int(state.global_step) * suite.budget.groups_per_optimizer_step
            if journal.completed_groups != expected_groups:
                raise RuntimeError("optimizer-step and rollout-group accounting diverged")
            return control

    class QualificationStopCallback(TrainerCallback):  # type: ignore[misc]
        def on_step_end(self, _args: Any, state: Any, control: Any, **_kwargs: Any) -> Any:
            stop_after = (
                int(qualification["controlled_stop_after"])
                if qualification is not None
                and qualification.get("controlled_stop_after") is not None
                else None
            )
            if stop_after is not None and int(state.global_step) >= stop_after:
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_save(self, args: Any, state: Any, control: Any, **_kwargs: Any) -> Any:
            checkpoint = Path(str(args.output_dir)) / f"checkpoint-{int(state.global_step)}"
            # Callback order is scientifically material: the native identity
            # callback must have published and verified CHECKPOINT_COMPLETE
            # before the custom sampler/journal state can bind to this save.
            completed_groups = journal.completed_groups
            canonical_sampler_state = _completed_prefix_sampler_state(schedule, completed_groups)
            live_sampler_state = group_sampler.sampler.state_dict()
            prefetch_audit = {
                "status": "COMPLETED_PREFIX_BOUND",
                "optimizer_step": int(state.global_step),
                "completed_rollout_groups": completed_groups,
                "policy": (
                    qualification["sampler_checkpoint_policy"]
                    if qualification is not None
                    else "replay_completed_schedule_prefix"
                ),
                "canonical_total_selections": canonical_sampler_state["total_selections"],
                "canonical_sampler_state_hash": identity_hash(canonical_sampler_state),
                "live_total_selections": live_sampler_state["total_selections"],
                "live_sampler_state_hash": identity_hash(live_sampler_state),
                "prefetch_ahead_selections": (
                    int(live_sampler_state["total_selections"]) - completed_groups
                ),
            }
            _atomic_json(
                output_dir / f"SAMPLER_CHECKPOINT_AUDIT_step_{int(state.global_step):08d}.json",
                prefetch_audit,
            )
            _publish_completed_step_after_native(
                journal,
                checkpoint=checkpoint,
                step=int(state.global_step),
                sampler_state=canonical_sampler_state,
                execution_identity=execution_identity,
            )
            return control

    from reasonforge.models import build_qlora_model_kwargs

    model_kwargs = build_qlora_model_kwargs(training_run.model)
    quantization = model_kwargs.pop("quantization_config")
    base_model = AutoModelForCausalLM.from_pretrained(
        training_run.model.model_id,
        quantization_config=quantization,
        local_files_only=True,
        **model_kwargs,
    )
    starting_adapter = Path(str(plan["starting_adapter"]))
    model = PeftModel.from_pretrained(base_model, starting_adapter, is_trainable=True)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any(".default." not in name for name in trainable):
        raise RuntimeError("E10 continuation did not isolate the default trainable adapter")
    callbacks = [
        NativeCheckpointIdentityCallback(output_dir, execution_identity),
        JournalCheckpointCallback(),
        QualificationStopCallback(),
        TelemetryCallback(output_dir / "telemetry.jsonl"),
    ]
    trainer = construct_grpo_trainer(
        GRPORunConfig(
            training_run,
            num_generations=suite.budget.num_generations,
            max_completion_length=suite.budget.max_completion_tokens,
            beta=float(semantics["beta"]),
            temperature=float(semantics["temperature"]),
            top_p=float(semantics["top_p"]),
            top_k=int(semantics["top_k"]),
            min_p=None,
            epsilon=float(semantics["epsilon"]),
            num_iterations=int(semantics["num_iterations"]),
            generation_batch_size=suite.budget.generation_batch_size,
            mask_truncated_completions=bool(semantics["mask_truncated_completions"]),
            shuffle_dataset=bool(semantics["shuffle_dataset"]),
            enable_jit_checkpoint=bool(semantics["enable_jit_checkpoint"]),
            enable_thinking=bool(semantics["enable_thinking"]),
            checkpoint_save_steps=checkpoint_save_steps,
            checkpoint_save_total_limit=(
                int(external_output["save_total_limit"])
                if external_output is not None
                else (8 if experiment_id == "E16" and not pilot else None)
            ),
            adam_beta1=float(semantics["adam_beta1"]),
            adam_beta2=float(semantics["adam_beta2"]),
            adam_epsilon=float(semantics["adam_epsilon"]),
            max_grad_norm=float(semantics["max_grad_norm"]),
            lr_scheduler_type=str(semantics["lr_scheduler_type"]),
            use_vllm=False,
            loss_type="grpo",
        ),
        reward_funcs=[reward],
        train_dataset=Dataset.from_list(list(schedule.rows)),
        processing_class=tokenizer,
        callbacks=callbacks,
        model_override=model,
        continue_existing_adapter=True,
        trainer_cls=JournaledGRPOTrainer,
    )
    trainer._rf_sampler = trainer_sampler
    trainer._rf_schedule = schedule
    trainer._rf_journal = journal
    if native_resume is not None:
        trainer.args.ignore_data_skip = True
    result = train_with_native_resume(trainer, native_resume)
    completed_global_step = int(trainer.state.global_step)
    if qualification is not None and completed_global_step < optimizer_steps:
        expected_stop = int(qualification["interruption_after_step"])
        if completed_global_step != expected_stop:
            raise RuntimeError("E28 controlled stop missed the frozen completed-step boundary")
        checkpoint = output_dir / f"checkpoint-{completed_global_step}"
        validate_native_trainer_checkpoint(checkpoint, execution_identity)
        stopped = {
            "status": "CONTROLLED_STOP",
            "experiment_id": experiment_id,
            "run_id": run_id,
            "process_id": os.getpid(),
            "global_step": completed_global_step,
            "checkpoint": checkpoint.relative_to(root).as_posix(),
            "completed_rollout_groups": journal.completed_groups,
            "completed_rollouts": journal.completed_rollouts,
        }
        _atomic_json(output_dir / "CONTROLLED_STOP.json", stopped)
        return stopped
    if completed_global_step != optimizer_steps:
        raise RuntimeError("GRPO run did not reach its frozen final optimizer step")
    if replication is not None:
        for checkpoint_step in replication["required_complete_checkpoint_steps"]:
            checkpoint = output_dir / f"checkpoint-{int(checkpoint_step)}"
            observed_step = validate_native_trainer_checkpoint(checkpoint, execution_identity)
            if observed_step != int(checkpoint_step):
                raise RuntimeError("E25 required checkpoint step is internally inconsistent")
    final_adapter = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    tokenizer.save_pretrained(final_adapter)
    metrics = dict(getattr(result, "metrics", result))
    runtime = float(metrics.get("train_runtime", estimated_step_seconds))
    complete = {
        "status": "COMPLETE",
        "experiment_id": experiment_id,
        "run_id": run_id,
        "execution_mode": (
            "QUALIFICATION"
            if qualification is not None
            else (
                "LOCKED_FINAL_REPLICATION"
                if replication is not None
                else ("PILOT" if pilot else "FULL")
            )
        ),
        "protocol_id": (
            str(external_output["protocol_id"])
            if external_output is not None
            else suite.protocol_id
        ),
        "suite_config_hash": suite.config_hash,
        "execution_identity_hash": identity_hash(execution_identity),
        "schedule_hash": schedule.schedule_hash,
        "starting_adapter_identity": plan["starting_adapter_identity"],
        "final_adapter_path": final_adapter.relative_to(root).as_posix(),
        "final_adapter_identity": identity_hash(
            {
                path.relative_to(final_adapter).as_posix(): file_sha256(path)
                for path in sorted(final_adapter.rglob("*"))
                if path.is_file()
            }
        ),
        "optimizer_steps": optimizer_steps,
        "optimizer_step_seconds": runtime / optimizer_steps,
        "completed_rollout_groups": journal.completed_groups,
        "completed_rollouts": journal.completed_rollouts,
        "rollout_reward_evidence": rollout_evidence.relative_to(root).as_posix(),
        "rollout_reward_evidence_sha256": file_sha256(rollout_evidence),
        "train_result": metrics,
    }
    if qualification is not None:
        from reasonforge.checkpoints.qualification_execution import (
            trainable_parameter_dtype_inventory,
        )

        complete.update(
            {
                "process_id": os.getpid(),
                "process_ids": [os.getpid()],
                "final_global_step": completed_global_step,
                "consumed_examples": journal.completed_rollouts,
                "final_loss": float(metrics.get("train_loss", 0.0)),
                "parameter_inventory_hash": complete["final_adapter_identity"],
                "task_sequence": list(schedule.task_ids[: journal.completed_groups]),
                "final_parameter_dtypes": trainable_parameter_dtype_inventory(trainer.model),
                "resume_dtype_audit": (
                    (output_dir / "RESUME_DTYPE_AUDIT.json").relative_to(root).as_posix()
                    if trainer._rf_resume_dtype_audit is not None
                    else None
                ),
                "resume_dtype_audit_sha256": (
                    file_sha256(output_dir / "RESUME_DTYPE_AUDIT.json")
                    if trainer._rf_resume_dtype_audit is not None
                    else None
                ),
            }
        )
    if adaptive_execution:
        complete.update(
            {
                "sampler_mode": schedule.sampler_mode,
                "adaptive_estimates_sha256": plan["adaptive_estimates_sha256"],
            }
        )
    if selected_reward_execution:
        complete.update(
            {
                "reward_architecture": reward_architecture,
                "selected_reward_sha256": plan["selected_reward_sha256"],
                "selected_reward_identity": plan["selected_reward_identity"],
                "dependency_artifact_sha256": plan["dependency_artifact_sha256"],
            }
        )
    if replication is not None:
        complete.update(
            {
                "source_experiment_id": source_experiment_id,
                "seed": replication["seed"],
                "replication_design_identity": replication["design_identity"],
                "replication_design_sha256": replication["design_sha256"],
                "replication_config_sha256": replication["config_sha256"],
                "frozen_schedule_artifact_sha256": replication["schedule_artifact_sha256"],
                "required_complete_checkpoint_steps": replication[
                    "required_complete_checkpoint_steps"
                ],
                "checkpoint_save_total_limit": replication["save_total_limit"],
            }
        )
    if experiment_id == "E16":
        complete.update(
            {
                "reward_architecture": reward_architecture,
                "structured_control": plan["structured_control"],
            }
        )
    _atomic_json(output_dir / "RUN_COMPLETE.json", complete)
    return complete
