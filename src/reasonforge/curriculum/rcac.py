"""Reliability-Constrained Adaptive Curriculum (RCAC).

RCAC changes task selection, not the policy-gradient estimator. Its state is
scientifically material and therefore serializable at completed optimizer steps.
"""

from __future__ import annotations

import base64
import math
import pickle
import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RCACConfig:
    tau_q: float = 0.75
    p_target: float = 0.55
    sigma: float = 0.20
    lambda_u: float = 0.50
    floor: float = 1e-12

    def __post_init__(self) -> None:
        if not 0.0 <= self.tau_q <= 1.0:
            raise ValueError("tau_q must be in [0, 1]")
        if not 0.0 <= self.p_target <= 1.0:
            raise ValueError("p_target must be in [0, 1]")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if self.lambda_u < 0.0:
            raise ValueError("lambda_u must be nonnegative")


@dataclass(slots=True)
class TaskEstimate:
    task_id: str
    family: str
    p: float
    q: float
    u: float
    selected: int = 0

    def validate(self) -> None:
        for label, value in (("p", self.p), ("q", self.q), ("u", self.u)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be in [0, 1]")


class RCACSampler:
    """Stateful weighted sampler with reliability gating and family balancing."""

    def __init__(
        self,
        tasks: Iterable[TaskEstimate],
        config: RCACConfig,
        *,
        seed: int,
        curriculum_phase: str = "development",
    ) -> None:
        task_list = list(tasks)
        if not task_list:
            raise ValueError("RCAC requires at least one task")
        if len({task.task_id for task in task_list}) != len(task_list):
            raise ValueError("task_id values must be unique")
        for task in task_list:
            task.validate()
        self.tasks = {task.task_id: task for task in task_list}
        self.config = config
        self.rng = random.Random(seed)
        self.seed = seed
        self.curriculum_phase = curriculum_phase
        self.family_counts: Counter[str] = Counter()
        self.total_selections = 0

    def family_balance(self, family: str) -> float:
        return 1.0 / math.sqrt(1.0 + self.family_counts[family])

    def score(self, task: TaskEstimate) -> float:
        if task.q < self.config.tau_q:
            return 0.0
        difficulty = math.exp(
            -((task.p - self.config.p_target) ** 2) / (2.0 * self.config.sigma**2)
        )
        return difficulty * (1.0 + self.config.lambda_u * task.u) * self.family_balance(task.family)

    def distribution(self) -> dict[str, float]:
        scores = {task_id: self.score(task) for task_id, task in self.tasks.items()}
        total = sum(scores.values())
        if total <= self.config.floor:
            raise RuntimeError("no task passes the reliability gate")
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
        return {
            "version": 1,
            "config": asdict(self.config),
            "candidate_pool": sorted(self.tasks),
            "tasks": {key: asdict(value) for key, value in sorted(self.tasks.items())},
            "family_counts": dict(sorted(self.family_counts.items())),
            "family_quotas": {},
            "total_selections": self.total_selections,
            "curriculum_phase": self.curriculum_phase,
            "seed": self.seed,
            "rng_state_b64": base64.b64encode(pickle.dumps(self.rng.getstate())).decode("ascii"),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> RCACSampler:
        required = {
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
            raise ValueError(f"incomplete RCAC state: {sorted(missing)}")
        tasks = [TaskEstimate(**item) for item in state["tasks"].values()]
        sampler = cls(
            tasks,
            RCACConfig(**state["config"]),
            seed=int(state["seed"]),
            curriculum_phase=str(state["curriculum_phase"]),
        )
        if sorted(sampler.tasks) != sorted(state["candidate_pool"]):
            raise ValueError("candidate pool identity mismatch")
        sampler.family_counts = Counter(state["family_counts"])
        sampler.total_selections = int(state["total_selections"])
        sampler.rng.setstate(pickle.loads(base64.b64decode(state["rng_state_b64"])))
        return sampler
