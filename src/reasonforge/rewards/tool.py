"""Multi-dimensional scoring for stateful tool trajectories."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from reasonforge.environments.stateful import SandboxState, Transition


@dataclass(frozen=True)
class ToolExpectation:
    calls: tuple[str, ...]
    arguments: tuple[Mapping[str, Any], ...] = ()
    final_state: Mapping[str, Any] = field(default_factory=dict)
    max_calls: int | None = None
    clarification_required: bool = False
    recovery_required: bool = False


@dataclass(frozen=True)
class ToolVerification:
    score: float
    function_selection: float
    argument_values: float
    tool_order: float
    final_state_correctness: float
    clarification: float
    recovery: float
    unnecessary_calls: float
    authorization_compliance: float
    details: Mapping[str, Any]


def _nested_value(state: SandboxState, dotted_path: str) -> Any:
    value: Any = state
    for part in dotted_path.split("."):
        value = getattr(value, part) if hasattr(value, part) else value[part]
    return value


def _state_matches(state: SandboxState, dotted_path: str, expected: Any) -> bool:
    try:
        return bool(_nested_value(state, dotted_path) == expected)
    except (AttributeError, KeyError, TypeError):
        return False


def verify_tool_trajectory(
    trace: Sequence[Transition], state: SandboxState, expectation: ToolExpectation
) -> ToolVerification:
    """Score behavior beyond function-name accuracy with equal visible dimensions."""

    actual_calls = [transition.call.name for transition in trace]
    expected_calls = list(expectation.calls)
    if expected_calls:
        matched = sum((Counter(actual_calls) & Counter(expected_calls)).values())
        selection = matched / len(expected_calls)
        order = (
            1.0
            if actual_calls == expected_calls
            else _ordered_fraction(actual_calls, expected_calls)
        )
    else:
        selection = 1.0 if not actual_calls else 0.0
        order = selection

    arg_scores: list[float] = []
    for index, expected in enumerate(expectation.arguments):
        if index >= len(trace):
            arg_scores.append(0.0)
            continue
        actual = trace[index].call.arguments
        arg_scores.append(
            sum(actual.get(key) == value for key, value in expected.items()) / len(expected)
            if expected
            else 1.0
        )
    argument_values = sum(arg_scores) / len(arg_scores) if arg_scores else 1.0

    state_checks = {
        path: _state_matches(state, path, expected)
        for path, expected in expectation.final_state.items()
    }
    final_state = sum(state_checks.values()) / len(state_checks) if state_checks else 1.0
    failures = [transition for transition in trace if not transition.result.ok]
    successes_after_failure = any(
        transition.result.ok and any(prior.index < transition.index for prior in failures)
        for transition in trace
    )
    recovery = (
        float(successes_after_failure) if expectation.recovery_required else float(not failures)
    )
    clarification = (
        float(any(t.call.name == "assistant.request_clarification" and t.result.ok for t in trace))
        if expectation.clarification_required
        else 1.0
    )
    allowed = expectation.max_calls if expectation.max_calls is not None else len(expected_calls)
    unnecessary = 1.0 if len(trace) <= allowed else max(0.0, allowed / len(trace))
    unauthorized = [t for t in trace if t.result.error_code == "AUTHORIZATION_REQUIRED"]
    authorization = 1.0 if not unauthorized else 0.0
    dimensions = (
        selection,
        argument_values,
        order,
        final_state,
        clarification,
        recovery,
        unnecessary,
        authorization,
    )
    return ToolVerification(
        score=sum(dimensions) / len(dimensions),
        function_selection=selection,
        argument_values=argument_values,
        tool_order=order,
        final_state_correctness=final_state,
        clarification=clarification,
        recovery=recovery,
        unnecessary_calls=unnecessary,
        authorization_compliance=authorization,
        details={
            "actual_calls": actual_calls,
            "expected_calls": expected_calls,
            "state_checks": state_checks,
            "failed_call_count": len(failures),
            "unauthorized_attempt_count": len(unauthorized),
        },
    )


def _ordered_fraction(actual: Sequence[str], expected: Sequence[str]) -> float:
    """Fraction of expected sequence found as an ordered subsequence."""

    cursor = 0
    for name in actual:
        if cursor < len(expected) and name == expected[cursor]:
            cursor += 1
    return cursor / len(expected) if expected else 1.0
