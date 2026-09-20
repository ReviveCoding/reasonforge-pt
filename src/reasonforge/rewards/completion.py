"""Canonical deterministic completion scoring and single-owner RL rewards.

The scoring dictionaries in this module are also used by the DEVELOPMENT
evaluator.  Keeping evaluation and training on one implementation prevents a
reward parser from silently disagreeing with the reported diagnostic score.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from reasonforge.environments import GroupedEnvironmentFactory
from reasonforge.synthetic import Constraint, ConstraintKind

from .aggregation import CanonicalRewardAggregator, RewardComponents, RewardWeights
from .instruction import verify_instruction
from .tool import ToolExpectation, verify_tool_trajectory

RewardArchitecture = Literal["structured", "terminal"]
TOOL_REWARD_IMPLEMENTATION = "reasonforge.rewards.completion:multi_tool_trace_v2"
TRAILING_CONTROL_TOKENS = re.compile(r"(?:<\|im_end\|>|<\|endoftext\|>)+\s*$")
SUPPORTED_HARD_CONSTRAINTS = frozenset(
    {"execution_validity", "authorization", "deterministic_correctness"}
)


def strip_trailing_control_tokens(completion: str) -> str:
    """Remove generation terminators while preserving tool-call markup."""

    return TRAILING_CONTROL_TOKENS.sub("", completion)


def _metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("metadata")
    if isinstance(raw, str):
        parsed = json.loads(raw)
    elif isinstance(raw, Mapping):
        parsed = dict(raw)
    else:
        raise ValueError("canonical reward row has invalid metadata")
    if not isinstance(parsed, dict):
        raise ValueError("canonical reward metadata must be an object")
    return parsed


def _score_instruction(completion: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    constraints = tuple(
        Constraint(ConstraintKind(item["kind"]), item["parameters"])
        for item in metadata["constraints"]
    )
    verification = verify_instruction(strip_trailing_control_tokens(completion), constraints)
    return {
        "score": verification.score,
        "passed": verification.passed,
        "components": dict(verification.checks),
        "details": dict(verification.details),
    }


def _last_number(text: str) -> str | None:
    matches = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None


def _score_reasoning(completion: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    expected = str(metadata["answer"])
    observed = _last_number(completion)
    passed = observed is not None and float(observed) == float(expected)
    return {"score": float(passed), "passed": passed, "expected": expected, "observed": observed}


def _parse_tool_calls(completion: str) -> list[dict[str, Any]] | None:
    """Parse every emitted tool call, failing closed on malformed call markup."""

    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", completion, re.DOTALL)
    opening_tags = completion.count("<tool_call>")
    closing_tags = completion.count("</tool_call>")
    if not blocks or opening_tags != closing_tags or len(blocks) != opening_tags:
        return None
    parsed_calls: list[dict[str, Any]] = []
    for block in blocks:
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict) or not isinstance(parsed.get("arguments"), dict):
            return None
        parsed_calls.append(parsed)
    return parsed_calls


def _score_tool(
    completion: str, row: Mapping[str, Any], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    expected = json.loads(str(row["reference"]))
    parsed_calls = _parse_tool_calls(completion)
    if parsed_calls is None:
        return {
            "score": 0.0,
            "passed": False,
            "parseable": False,
            "expected": expected,
            "observed": None,
        }
    factory = GroupedEnvironmentFactory(seed=int(metadata["generator_seed"]))
    environment = factory.create(str(metadata["group_key"]))
    expected_initial_state = metadata.get("initial_state_identity")
    if (
        expected_initial_state is not None
        and environment.initial_state_identity != expected_initial_state
    ):
        raise ValueError(
            "reconstructed tool initial-state identity does not match the canonical row"
        )
    if expected.get("authorization_required"):
        environment.authorize(str(expected["name"]))
    for parsed in parsed_calls:
        environment.call(str(parsed.get("name")), parsed["arguments"])
    expectation = ToolExpectation(
        calls=(str(expected["name"]),),
        arguments=(expected["arguments"],),
        max_calls=1,
        clarification_required=expected["name"] == "assistant.request_clarification",
    )
    verification = verify_tool_trajectory(environment.trace, environment.state, expectation)
    return {
        "score": verification.score,
        "passed": verification.score == 1.0,
        "parseable": True,
        "expected": expected,
        "observed": parsed_calls[0] if len(parsed_calls) == 1 else parsed_calls,
        "dimensions": {
            "function_selection": verification.function_selection,
            "argument_values": verification.argument_values,
            "tool_order": verification.tool_order,
            "final_state_correctness": verification.final_state_correctness,
            "clarification": verification.clarification,
            "recovery": verification.recovery,
            "unnecessary_calls": verification.unnecessary_calls,
            "authorization_compliance": verification.authorization_compliance,
        },
    }


def score_completion(completion: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable deterministic diagnostic score for one canonical row."""

    metadata = _metadata(row)
    family = str(row["task_family"])
    completion = strip_trailing_control_tokens(completion)
    if family == "instruction":
        return _score_instruction(completion, metadata)
    if family.startswith("reasoning/"):
        return _score_reasoning(completion, metadata)
    if family == "stateful_tool":
        return _score_tool(completion, row, metadata)
    raise ValueError(f"unsupported family: {family}")


@dataclass(frozen=True, slots=True)
class CanonicalRewardPolicy:
    """Frozen scalar reward policy loaded from ``configs/reward.yaml``."""

    identity: str
    state: str
    weights: RewardWeights
    hard_constraints: tuple[str, ...]

    @classmethod
    def from_yaml(cls, path: Path) -> CanonicalRewardPolicy:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("reward configuration must be a mapping")
        raw_weights = value.get("weights")
        raw_constraints = value.get("hard_constraints", [])
        if not isinstance(raw_weights, Mapping):
            raise ValueError("reward configuration is missing weights")
        if not isinstance(raw_constraints, list) or any(
            not isinstance(item, str) or not item for item in raw_constraints
        ):
            raise ValueError("hard_constraints must be a string list")
        unsupported = sorted(set(raw_constraints) - SUPPORTED_HARD_CONSTRAINTS)
        if unsupported:
            raise ValueError(f"unsupported hard reward constraints: {unsupported}")
        identity = value.get("identity")
        state = value.get("state")
        if not isinstance(identity, str) or not identity or not isinstance(state, str) or not state:
            raise ValueError("reward configuration identity and state are required")
        return cls(
            identity=identity,
            state=state,
            weights=RewardWeights(**dict(raw_weights)),
            hard_constraints=tuple(raw_constraints),
        )


@dataclass(frozen=True, slots=True)
class CompletionReward:
    """One scored completion with independently loggable reward components."""

    architecture: RewardArchitecture
    scoring: Mapping[str, Any]
    components: RewardComponents
    contributions: Mapping[str, float]
    hard_checks: Mapping[str, bool | None]
    hard_vetoed: bool
    pre_veto_scalar: float
    scalar: float


def reward_completion(
    completion: str,
    row: Mapping[str, Any],
    *,
    policy: CanonicalRewardPolicy,
    architecture: RewardArchitecture,
) -> CompletionReward:
    """Compute task-local components and the sole canonical scalar reward."""

    if architecture not in ("structured", "terminal"):
        raise ValueError(f"unsupported reward architecture: {architecture}")
    scoring = score_completion(completion, row)
    signal = float(scoring["score"] if architecture == "structured" else scoring["passed"])
    family = str(row["task_family"])
    if family == "instruction":
        components = RewardComponents(instruction=signal)
    elif family.startswith("reasoning/"):
        components = RewardComponents(reason=signal)
    elif family == "stateful_tool":
        expected = json.loads(str(row["reference"]))
        dimensions = scoring.get("dimensions", {})
        authorization = (
            float(dimensions.get("authorization_compliance", 0.0))
            if expected.get("authorization_required")
            else 0.0
        )
        efficiency_cost = (
            1.0 - float(dimensions.get("unnecessary_calls", 0.0))
            if scoring.get("parseable")
            else 0.0
        )
        components = RewardComponents(
            tool=signal,
            safety=authorization,
            efficiency_cost=efficiency_cost,
        )
    else:
        raise ValueError(f"unsupported family: {family}")
    hard_checks: dict[str, bool | None] = {
        constraint: None for constraint in policy.hard_constraints
    }
    hard_vetoed = False
    if family == "stateful_tool":
        dimensions = scoring.get("dimensions", {})
        tool_checks = {
            "execution_validity": bool(
                scoring.get("parseable") and dimensions.get("recovery") == 1.0
            ),
            "authorization": bool(
                scoring.get("parseable") and dimensions.get("authorization_compliance") == 1.0
            ),
            "deterministic_correctness": bool(scoring["passed"]),
        }
        hard_checks.update(
            {name: tool_checks[name] for name in policy.hard_constraints if name in tool_checks}
        )
        hard_vetoed = any(value is False for value in hard_checks.values())
    aggregator = CanonicalRewardAggregator(policy.weights)
    pre_veto_scalar = aggregator.aggregate(components)
    return CompletionReward(
        architecture=architecture,
        scoring=scoring,
        components=components,
        contributions=aggregator.contributions(components),
        hard_checks=hard_checks,
        hard_vetoed=hard_vetoed,
        pre_veto_scalar=pre_veto_scalar,
        scalar=0.0 if hard_vetoed else pre_veto_scalar,
    )


def _completion_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = [item.get("content", "") for item in value if isinstance(item, Mapping)]
        return "".join(str(part) for part in parts)
    raise ValueError("completion must be text or a sequence of message mappings")


class CanonicalBatchReward:
    """TRL-compatible callable that owns the only scalar reward channel."""

    def __init__(
        self,
        policy: CanonicalRewardPolicy,
        architecture: RewardArchitecture,
        *,
        num_generations: int | None = None,
    ) -> None:
        if num_generations is not None and num_generations < 2:
            raise ValueError("GRPO reward grouping requires at least two generations")
        self.policy = policy
        self.architecture = architecture
        self.num_generations = num_generations

    def __call__(
        self,
        *,
        completions: Sequence[Any],
        log_extra: Callable[[str, list[Any]], None] | None = None,
        log_metric: Callable[[str, float], None] | None = None,
        **columns: Any,
    ) -> list[float]:
        count = len(completions)
        required = ("task_family", "reference", "metadata")
        for name in required:
            values = columns.get(name)
            if (
                not isinstance(values, Sequence)
                or isinstance(values, (str, bytes))
                or len(values) != count
            ):
                raise ValueError(f"reward column {name!r} must contain one value per completion")
        group_ids, example_ids, tool_state_ids = self._validate_grpo_groups(count, columns)
        results: list[CompletionReward] = []
        for index, completion in enumerate(completions):
            row = {name: columns[name][index] for name in required}
            if "tool_schema" in columns:
                row["tool_schema"] = columns["tool_schema"][index]
            results.append(
                reward_completion(
                    _completion_text(completion),
                    row,
                    policy=self.policy,
                    architecture=self.architecture,
                )
            )
        if log_extra is not None:
            log_extra("rf_reward_architecture", [self.architecture] * count)
            for name in RewardComponents.__dataclass_fields__:
                log_extra(
                    f"rf_reward_component_{name}",
                    [result.components.as_dict()[name] for result in results],
                )
                log_extra(
                    f"rf_reward_contribution_{name}",
                    [result.contributions[name] for result in results],
                )
            log_extra("rf_reward_scalar", [result.scalar for result in results])
            log_extra("rf_reward_pre_veto_scalar", [result.pre_veto_scalar for result in results])
            log_extra("rf_reward_hard_vetoed", [result.hard_vetoed for result in results])
            for name in self.policy.hard_constraints:
                log_extra(
                    f"rf_reward_hard_check_{name}",
                    [result.hard_checks[name] for result in results],
                )
            if self.num_generations is not None:
                log_extra("rf_rollout_group_id", group_ids)
                log_extra("rf_source_example_id", example_ids)
                log_extra("rf_tool_initial_state_identity", tool_state_ids)
        if log_metric is not None and results:
            log_metric("rf_reward/scalar", sum(result.scalar for result in results) / len(results))
        return [result.scalar for result in results]

    def _validate_grpo_groups(
        self, count: int, columns: Mapping[str, Any]
    ) -> tuple[list[str], list[str], list[str | None]]:
        if self.num_generations is None:
            return [], [], []
        if count == 0 or count % self.num_generations:
            raise ValueError("GRPO reward batch must contain complete contiguous generation groups")
        raw_group_ids = columns.get("rollout_group_id")
        raw_example_ids = columns.get("example_id")
        if (
            not isinstance(raw_group_ids, Sequence)
            or isinstance(raw_group_ids, (str, bytes))
            or len(raw_group_ids) != count
            or not isinstance(raw_example_ids, Sequence)
            or isinstance(raw_example_ids, (str, bytes))
            or len(raw_example_ids) != count
        ):
            raise ValueError(
                "GRPO reward batches require rollout_group_id and example_id per completion"
            )
        group_ids = [str(value) for value in raw_group_ids]
        example_ids = [str(value) for value in raw_example_ids]
        if any(not value for value in group_ids) or any(not value for value in example_ids):
            raise ValueError("GRPO group and source-example identities must be non-empty")
        metadata_values = columns["metadata"]
        tool_state_ids: list[str | None] = []
        for metadata in metadata_values:
            value = _metadata({"metadata": metadata}).get("initial_state_identity")
            tool_state_ids.append(str(value) if value is not None else None)
        seen_groups: set[str] = set()
        for start in range(0, count, self.num_generations):
            stop = start + self.num_generations
            identity_columns: dict[str, Sequence[Any]] = {
                "rollout_group_id": group_ids[start:stop],
                "example_id": example_ids[start:stop],
                "reference": columns["reference"][start:stop],
                "task_family": columns["task_family"][start:stop],
                "tool_initial_state_identity": tool_state_ids[start:stop],
            }
            mismatched = [
                name
                for name, values in identity_columns.items()
                if len({json.dumps(value, sort_keys=True, default=str) for value in values}) != 1
            ]
            if mismatched:
                raise ValueError(
                    "GRPO group members do not share one underlying task/state: "
                    + ", ".join(mismatched)
                )
            group_id = group_ids[start]
            if group_id in seen_groups:
                raise ValueError("rollout_group_id is reused across unrelated GRPO groups")
            seen_groups.add(group_id)
        return group_ids, example_ids, tool_state_ids


def canonical_reward_functions(
    policy: CanonicalRewardPolicy,
    architecture: RewardArchitecture,
    *,
    num_generations: int | None = None,
) -> tuple[CanonicalBatchReward]:
    """Return exactly one reward callable, preventing trainer-side re-aggregation."""

    return (CanonicalBatchReward(policy, architecture, num_generations=num_generations),)
