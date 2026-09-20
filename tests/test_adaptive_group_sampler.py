from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from reasonforge.curriculum import (
    AdaptiveGroupRepeatSampler,
    ComparisonSampler,
    RCACConfig,
    TaskEstimate,
)


def _comparison(seed: int = 1701) -> ComparisonSampler:
    return ComparisonSampler(
        [
            TaskEstimate("task-a", "reasoning", p=0.5, q=0.9, u=0.2),
            TaskEstimate("task-b", "tool", p=0.6, q=0.95, u=0.8),
            TaskEstimate("task-c", "instruction", p=0.4, q=0.85, u=0.5),
        ],
        RCACConfig(),
        mode="rcac",
        seed=seed,
    )


def _group_sampler(seed: int = 1701) -> AdaptiveGroupRepeatSampler:
    return AdaptiveGroupRepeatSampler(
        _comparison(seed),
        {"task-a": 4, "task-b": 8, "task-c": 12},
        num_groups=7,
        num_generations=4,
    )


def test_each_selected_task_is_one_contiguous_grpo_group() -> None:
    sampler = _group_sampler()
    indices = list(sampler)
    groups = [indices[start : start + 4] for start in range(0, len(indices), 4)]

    assert len(indices) == 28
    assert len(groups) == 7
    assert all(len(set(group)) == 1 for group in groups)
    assert sampler.completed_groups == 7
    assert sampler.sampler.total_selections == 7


def test_group_sampler_state_restores_exact_future_selection_sequence() -> None:
    uninterrupted = _group_sampler(seed=44)
    prefix = [uninterrupted.select_group() for _ in range(3)]
    restored = AdaptiveGroupRepeatSampler.from_state_dict(uninterrupted.state_dict())

    expected_suffix = list(uninterrupted)
    recovered_suffix = list(restored)
    assert len(prefix) == 3
    assert recovered_suffix == expected_suffix
    assert restored.completed_groups == uninterrupted.completed_groups == 7
    assert restored.sampler.state_dict() == uninterrupted.sampler.state_dict()


def test_groups_are_immutable_and_do_not_leak_selection_state() -> None:
    sampler = _group_sampler()
    first = sampler.select_group()
    frozen_indices = first.indices
    sampler.sampler.update(first.task_id, p=0.0, q=0.0, u=0.0)
    second = sampler.select_group()

    assert first.indices == frozen_indices == (first.dataset_index,) * 4
    assert second.ordinal == first.ordinal + 1
    with pytest.raises(FrozenInstanceError):
        first.task_id = "changed"  # type: ignore[misc]


def test_group_sampler_rejects_candidate_pool_or_index_aliasing() -> None:
    with pytest.raises(ValueError, match="exactly match"):
        AdaptiveGroupRepeatSampler(
            _comparison(), {"task-a": 0, "task-b": 1}, num_groups=2, num_generations=4
        )
    with pytest.raises(ValueError, match="unique non-negative"):
        AdaptiveGroupRepeatSampler(
            _comparison(),
            {"task-a": 0, "task-b": 0, "task-c": 2},
            num_groups=2,
            num_generations=4,
        )
