"""Executable task-selection policies for the E12-E20 sampler comparison.

The policies in this module change only which task is selected. They do not
modify GRPO's policy-gradient estimator, reward aggregation, or rollout budget.
"""

from __future__ import annotations

import base64
import math
import pickle
import random
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

from reasonforge.curriculum.rcac import RCACConfig, TaskEstimate

ComparisonMode = Literal[
    "uniform",
    "difficulty",
    "reliability_only",
    "reward_variance",
    "rcac",
    "rcac_no_reliability",
    "rcac_no_difficulty",
]

_MODES = {
    "uniform",
    "difficulty",
    "reliability_only",
    "reward_variance",
    "rcac",
    "rcac_no_reliability",
    "rcac_no_difficulty",
}


class ComparisonSampler:
    """Deterministic weighted sampler with serializable scientific state.

    ``p`` is estimated student success, ``q`` is supervision reliability, and
    ``u`` is the normalized reward-variance/uncertainty signal. Only RCAC and
    its ablations use the dynamic inverse-square-root family balance term.
    """

    def __init__(
        self,
        tasks: Iterable[TaskEstimate],
        config: RCACConfig,
        *,
        mode: ComparisonMode,
        seed: int,
        curriculum_phase: str = "development",
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"unknown comparison sampler mode: {mode}")
        source = list(tasks)
        if not source:
            raise ValueError("comparison sampling requires at least one task")
        if len({task.task_id for task in source}) != len(source):
            raise ValueError("task_id values must be unique")
        for task in source:
            task.validate()
        # Do not mutate estimates owned by a caller or another experimental arm.
        task_list = [TaskEstimate(**asdict(task)) for task in source]
        self.tasks = {task.task_id: task for task in task_list}
        self.config = config
        self.mode = mode
        self.seed = seed
        self.rng = random.Random(seed)
        self.curriculum_phase = curriculum_phase
        self.family_counts: Counter[str] = Counter()
        self.total_selections = 0

    def family_balance(self, family: str) -> float:
        return 1.0 / math.sqrt(1.0 + self.family_counts[family])

    def score(self, task: TaskEstimate) -> float:
        difficulty = math.exp(
            -((task.p - self.config.p_target) ** 2) / (2.0 * self.config.sigma**2)
        )
        reliable = task.q >= self.config.tau_q
        information = 1.0 + self.config.lambda_u * task.u
        balance = self.family_balance(task.family)
        if self.mode == "uniform":
            return 1.0
        if self.mode == "difficulty":
            return difficulty
        if self.mode == "reliability_only":
            return task.q if reliable else 0.0
        if self.mode == "reward_variance":
            return max(task.u, self.config.floor)
        if self.mode == "rcac_no_reliability":
            return difficulty * information * balance
        if self.mode == "rcac_no_difficulty":
            return information * balance if reliable else 0.0
        return difficulty * information * balance if reliable else 0.0

    def distribution(self) -> dict[str, float]:
        scores = {task_id: self.score(task) for task_id, task in self.tasks.items()}
        total = sum(scores.values())
        if total <= self.config.floor:
            raise RuntimeError(f"no task is eligible under sampler mode {self.mode}")
        return {task_id: value / total for task_id, value in scores.items()}

    def select(self, count: int = 1) -> list[str]:
        if count < 1:
            raise ValueError("count must be positive")
        selected: list[str] = []
        for _ in range(count):
            probabilities = self.distribution()
            task_id = self.rng.choices(
                list(probabilities), weights=list(probabilities.values()), k=1
            )[0]
            task = self.tasks[task_id]
            task.selected += 1
            self.family_counts[task.family] += 1
            self.total_selections += 1
            selected.append(task_id)
        return selected

    def update(
        self,
        task_id: str,
        *,
        p: float | None = None,
        q: float | None = None,
        u: float | None = None,
    ) -> None:
        task = self.tasks[task_id]
        for field_name, value in (("p", p), ("q", q), ("u", u)):
            if value is not None:
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{field_name} must be in [0, 1]")
                setattr(task, field_name, value)

    def state_dict(self) -> dict[str, Any]:
        # Candidate order is scientifically material because ``random.choices``
        # maps its seeded draw onto the population by position. Preserve the
        # original insertion order across checkpoint/resume; sorting here would
        # produce a different task sequence after restoration.
        candidate_pool = list(self.tasks)
        return {
            "version": 1,
            "mode": self.mode,
            "config": asdict(self.config),
            "candidate_pool": candidate_pool,
            "tasks": {key: asdict(self.tasks[key]) for key in candidate_pool},
            "family_counts": dict(sorted(self.family_counts.items())),
            "family_quotas": {},
            "total_selections": self.total_selections,
            "curriculum_phase": self.curriculum_phase,
            "seed": self.seed,
            "rng_state_b64": base64.b64encode(pickle.dumps(self.rng.getstate())).decode("ascii"),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> ComparisonSampler:
        required = {
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
        missing = required - set(state)
        if missing:
            raise ValueError(f"incomplete comparison sampler state: {sorted(missing)}")
        raw_mode = str(state["mode"])
        if raw_mode not in _MODES:
            raise ValueError(f"unknown comparison sampler mode: {raw_mode}")
        raw_tasks = state["tasks"]
        raw_config = state["config"]
        if not isinstance(raw_tasks, Mapping) or not isinstance(raw_config, Mapping):
            raise ValueError("invalid comparison sampler task/config state")
        raw_pool = state["candidate_pool"]
        if not isinstance(raw_pool, list) or any(not isinstance(item, str) for item in raw_pool):
            raise ValueError("invalid comparison sampler candidate-pool order")
        if set(raw_pool) != {str(key) for key in raw_tasks} or len(raw_pool) != len(raw_tasks):
            raise ValueError("candidate pool identity mismatch")
        tasks = [TaskEstimate(**dict(raw_tasks[task_id])) for task_id in raw_pool]
        sampler = cls(
            tasks,
            RCACConfig(**dict(raw_config)),
            mode=cast(ComparisonMode, raw_mode),
            seed=int(state["seed"]),
            curriculum_phase=str(state["curriculum_phase"]),
        )
        if list(sampler.tasks) != raw_pool:
            raise ValueError("candidate pool identity mismatch")
        raw_family_counts = state["family_counts"]
        if not isinstance(raw_family_counts, Mapping):
            raise ValueError("invalid family count state")
        sampler.family_counts = Counter(
            {str(key): int(value) for key, value in raw_family_counts.items()}
        )
        sampler.total_selections = int(state["total_selections"])
        sampler.rng.setstate(pickle.loads(base64.b64decode(str(state["rng_state_b64"]))))
        return sampler


@dataclass(frozen=True, slots=True)
class SelectedTaskGroup:
    """One immutable GRPO group backed by one canonical dataset row."""

    ordinal: int
    task_id: str
    dataset_index: int
    indices: tuple[int, ...]


class AdaptiveGroupRepeatSampler:
    """Draw tasks adaptively and repeat each index as one contiguous GRPO group.

    The wrapped :class:`ComparisonSampler` is advanced exactly once per group.
    This object deliberately owns selection order, so a trainer integration must
    not wrap it in TRL's ``RepeatSampler`` a second time.
    """

    def __init__(
        self,
        sampler: ComparisonSampler,
        task_to_index: Mapping[str, int],
        *,
        num_groups: int,
        num_generations: int,
        completed_groups: int = 0,
    ) -> None:
        if num_groups < 1:
            raise ValueError("num_groups must be positive")
        if num_generations < 2:
            raise ValueError("num_generations must be at least two")
        mapping = {str(key): int(value) for key, value in task_to_index.items()}
        if set(mapping) != set(sampler.tasks):
            raise ValueError("task-to-index mapping must exactly match the candidate pool")
        if any(index < 0 for index in mapping.values()) or len(set(mapping.values())) != len(
            mapping
        ):
            raise ValueError("dataset indices must be unique non-negative integers")
        if not 0 <= completed_groups <= num_groups:
            raise ValueError("completed_groups must be within the configured group budget")
        self.sampler = sampler
        self.task_to_index = mapping
        self.num_groups = num_groups
        self.num_generations = num_generations
        self.completed_groups = completed_groups

    def __len__(self) -> int:
        return (self.num_groups - self.completed_groups) * self.num_generations

    def select_group(self) -> SelectedTaskGroup:
        """Select one task and atomically publish its immutable repeat group."""

        if self.completed_groups >= self.num_groups:
            raise StopIteration("the configured rollout-group budget is exhausted")
        task_id = self.sampler.select(1)[0]
        dataset_index = self.task_to_index[task_id]
        group = SelectedTaskGroup(
            ordinal=self.completed_groups,
            task_id=task_id,
            dataset_index=dataset_index,
            indices=(dataset_index,) * self.num_generations,
        )
        self.completed_groups += 1
        return group

    def __iter__(self) -> Iterable[int]:
        while self.completed_groups < self.num_groups:
            yield from self.select_group().indices

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "num_groups": self.num_groups,
            "num_generations": self.num_generations,
            "completed_groups": self.completed_groups,
            "task_to_index": dict(sorted(self.task_to_index.items())),
            "comparison_sampler": self.sampler.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> AdaptiveGroupRepeatSampler:
        required = {
            "version",
            "num_groups",
            "num_generations",
            "completed_groups",
            "task_to_index",
            "comparison_sampler",
        }
        missing = required - set(state)
        if missing:
            raise ValueError(f"incomplete adaptive group sampler state: {sorted(missing)}")
        if state["version"] != 1:
            raise ValueError(f"unsupported adaptive group sampler version: {state['version']!r}")
        mapping = state["task_to_index"]
        comparison_state = state["comparison_sampler"]
        if not isinstance(mapping, Mapping) or not isinstance(comparison_state, Mapping):
            raise ValueError("adaptive group sampler mappings are invalid")
        return cls(
            ComparisonSampler.from_state_dict(comparison_state),
            {str(key): int(value) for key, value in mapping.items()},
            num_groups=int(state["num_groups"]),
            num_generations=int(state["num_generations"]),
            completed_groups=int(state["completed_groups"]),
        )
