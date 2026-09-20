from __future__ import annotations

import math

import pytest

from reasonforge.statistics import (
    benjamini_hochberg,
    binary_paired_analysis,
    continuous_paired_analysis,
    holm_adjust,
    paired_bootstrap,
    paired_sign_flip_test,
    seed_aware_summary,
)


def test_continuous_analysis_collapses_repeated_cluster_observations() -> None:
    result = continuous_paired_analysis(
        baseline=[0.0, 1.0, 10.0, 10.0],
        candidate=[2.0, 1.0, 11.0, 13.0],
        cluster_ids=["task-a", "task-a", "task-b", "task-b"],
        bootstrap_replicates=500,
        permutation_replicates=500,
        seed=7,
    )
    assert result.observation_count == 4
    assert result.analysis_unit_count == 2
    assert result.baseline_mean == pytest.approx(5.25)
    assert result.candidate_mean == pytest.approx(6.75)
    assert result.absolute_difference == pytest.approx(1.5)
    assert result.permutation_exact is True
    assert result.permutation_replicates == 4


def test_exact_sign_flip_known_result() -> None:
    p_value, permutations, exact = paired_sign_flip_test([1.0, 1.0, 1.0])
    assert exact
    assert permutations == 8
    assert p_value == 0.25


def test_monte_carlo_sign_flip_is_reproducible_and_nonzero() -> None:
    first = paired_sign_flip_test(
        [float(index) for index in range(1, 19)],
        replicates=1_000,
        seed=99,
        max_exact_permutations=2,
    )
    second = paired_sign_flip_test(
        [float(index) for index in range(1, 19)],
        replicates=1_000,
        seed=99,
        max_exact_permutations=2,
    )
    assert first == second
    assert first[0] >= 1 / 1_001
    assert first[1:] == (1_000, False)


def test_binary_requires_real_booleans_and_unique_units() -> None:
    with pytest.raises(ValueError, match="bool"):
        binary_paired_analysis([1], [0], bootstrap_replicates=10)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="unique"):
        binary_paired_analysis(
            [True, False],
            [False, True],
            unit_ids=["same-task", "same-task"],
            bootstrap_replicates=10,
        )


def test_resampling_rejects_nonfinite_or_invalid_configuration() -> None:
    for values in ([float("nan")], [float("inf")]):
        with pytest.raises(ValueError, match="finite"):
            paired_bootstrap(values, replicates=10)
    with pytest.raises(ValueError, match="positive integer"):
        paired_bootstrap([1.0], replicates=0)
    with pytest.raises(ValueError, match="strictly between"):
        paired_bootstrap([1.0], alpha=1.0)


def test_multiplicity_adjustments_match_known_values_and_validate_inputs() -> None:
    p_values = [0.01, 0.04, 0.03, 0.002]
    assert holm_adjust(p_values) == pytest.approx([0.03, 0.06, 0.06, 0.008])
    assert benjamini_hochberg(p_values) == pytest.approx([0.02, 0.04, 0.04, 0.008])
    assert holm_adjust([]) == []
    assert benjamini_hochberg([]) == []
    for invalid in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="p-values"):
            holm_adjust([invalid])


def test_seed_aware_summary_reports_every_seed_and_is_reproducible() -> None:
    differences = {
        "seed-17": [0.1, 0.2, 0.0],
        "seed-23": [-0.1, 0.1, 0.0],
        "seed-41": [0.3, 0.2, 0.1],
    }
    first = seed_aware_summary(differences, bootstrap_replicates=500, seed=5)
    second = seed_aware_summary(differences, bootstrap_replicates=500, seed=5)
    assert first == second
    assert first.seed_count == 3
    assert first.observation_count == 9
    assert first.unit_ids_verified is False
    assert [item.seed_id for item in first.per_seed] == ["seed-17", "seed-23", "seed-41"]
    assert first.mean_difference == pytest.approx(0.1)
    assert first.ci_low <= first.mean_difference <= first.ci_high
    assert "Few seeds" in first.qualifier
    assert math.isfinite(first.to_dict()["mean_difference"])


def test_seed_aware_summary_rejects_missing_or_misaligned_units() -> None:
    with pytest.raises(ValueError, match="same number"):
        seed_aware_summary({"seed-1": [0.1, 0.2], "seed-2": [0.3]}, bootstrap_replicates=10)
    with pytest.raises(ValueError, match="same protected"):
        seed_aware_summary(
            {"seed-1": [0.1, 0.2], "seed-2": [0.3, 0.4]},
            unit_ids_by_seed={"seed-1": ["a", "b"], "seed-2": ["a", "c"]},
            bootstrap_replicates=10,
        )
    verified = seed_aware_summary(
        {"seed-1": [0.1, 0.2], "seed-2": [0.3, 0.4]},
        unit_ids_by_seed={"seed-1": ["a", "b"], "seed-2": ["b", "a"]},
        bootstrap_replicates=10,
    )
    assert verified.unit_ids_verified is True
