"""Paired, multiplicity, and seed-aware analyses for protected evaluations."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.stats import binomtest


@dataclass(frozen=True, slots=True)
class PairedBinaryResult:
    n: int
    baseline_rate: float
    candidate_rate: float
    absolute_difference: float
    both_fail: int
    baseline_only: int
    candidate_only: int
    both_pass: int
    mcnemar_exact_p: float
    ci_low: float
    ci_high: float
    bootstrap_replicates: int
    seed_count: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairedContinuousResult:
    observation_count: int
    analysis_unit_count: int
    baseline_mean: float
    candidate_mean: float
    absolute_difference: float
    ci_low: float
    ci_high: float
    permutation_p: float
    permutation_replicates: int
    permutation_exact: bool
    bootstrap_replicates: int
    seed_count: int

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PerSeedEstimate:
    seed_id: str
    observation_count: int
    mean_difference: float

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SeedAwareResult:
    seed_count: int
    observation_count: int
    unit_ids_verified: bool
    mean_difference: float
    ci_low: float
    ci_high: float
    bootstrap_replicates: int
    per_seed: tuple[PerSeedEstimate, ...]
    qualifier: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["per_seed"] = [item.to_dict() for item in self.per_seed]
        return value


def _validate_resampling(*, replicates: int, alpha: float) -> None:
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")


def _finite_vector(values: Sequence[float], *, name: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values") from error
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _bootstrap_means(
    values: np.ndarray, *, replicates: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw bootstrap means while bounding the temporary index allocation."""

    batch_size = max(1, min(replicates, (32 * 1024 * 1024) // (8 * values.size)))
    means = np.empty(replicates, dtype=float)
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    return means


def paired_bootstrap(
    differences: Sequence[float],
    *,
    replicates: int = 10_000,
    seed: int = 2026,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile interval for paired, independent-unit differences."""

    _validate_resampling(replicates=replicates, alpha=alpha)
    values = _finite_vector(differences, name="differences")
    means = _bootstrap_means(values, replicates=replicates, rng=np.random.default_rng(seed))
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def _validate_unit_ids(unit_ids: Sequence[Hashable] | None, length: int) -> None:
    if unit_ids is None:
        return
    if len(unit_ids) != length:
        raise ValueError("unit_ids must align one-to-one with paired outcomes")
    try:
        unique = set(unit_ids)
    except TypeError as error:
        raise ValueError("unit_ids must be hashable") from error
    if len(unique) != length:
        raise ValueError("unit_ids must be unique; aggregate repeated observations first")


def binary_paired_analysis(
    baseline: Sequence[bool],
    candidate: Sequence[bool],
    *,
    unit_ids: Sequence[Hashable] | None = None,
    bootstrap_replicates: int = 10_000,
    seed: int = 2026,
    seed_count: int = 1,
) -> PairedBinaryResult:
    """Exact McNemar and paired bootstrap for unique binary units."""

    if len(baseline) != len(candidate) or len(baseline) == 0:
        raise ValueError("paired vectors must have equal nonzero length")
    if any(type(value) is not bool for value in [*baseline, *candidate]):
        raise ValueError("binary paired outcomes must be bool values")
    if isinstance(seed_count, bool) or not isinstance(seed_count, int) or seed_count < 1:
        raise ValueError("seed_count must be a positive integer")
    _validate_unit_ids(unit_ids, len(baseline))
    _validate_resampling(replicates=bootstrap_replicates, alpha=0.05)
    pairs = list(zip(baseline, candidate, strict=True))
    both_fail = sum(not left and not right for left, right in pairs)
    baseline_only = sum(left and not right for left, right in pairs)
    candidate_only = sum(not left and right for left, right in pairs)
    both_pass = sum(left and right for left, right in pairs)
    discordant = baseline_only + candidate_only
    mcnemar_p = (
        float(binomtest(candidate_only, discordant, 0.5, alternative="two-sided").pvalue)
        if discordant
        else 1.0
    )
    differences = [float(right) - float(left) for left, right in pairs]
    ci_low, ci_high = paired_bootstrap(differences, replicates=bootstrap_replicates, seed=seed)
    return PairedBinaryResult(
        n=len(pairs),
        baseline_rate=sum(baseline) / len(pairs),
        candidate_rate=sum(candidate) / len(pairs),
        absolute_difference=sum(differences) / len(pairs),
        both_fail=both_fail,
        baseline_only=baseline_only,
        candidate_only=candidate_only,
        both_pass=both_pass,
        mcnemar_exact_p=mcnemar_p,
        ci_low=ci_low,
        ci_high=ci_high,
        bootstrap_replicates=bootstrap_replicates,
        seed_count=seed_count,
    )


def _collapse_clusters(
    baseline: np.ndarray,
    candidate: np.ndarray,
    cluster_ids: Sequence[Hashable] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if cluster_ids is None:
        return baseline, candidate
    if len(cluster_ids) != baseline.size:
        raise ValueError("cluster_ids must align one-to-one with paired observations")
    grouped_baseline: dict[Hashable, list[float]] = defaultdict(list)
    grouped_candidate: dict[Hashable, list[float]] = defaultdict(list)
    for cluster, left, right in zip(cluster_ids, baseline, candidate, strict=True):
        try:
            hash(cluster)
        except TypeError as error:
            raise ValueError("cluster_ids must be hashable") from error
        grouped_baseline[cluster].append(float(left))
        grouped_candidate[cluster].append(float(right))
    return (
        np.asarray([np.mean(values) for values in grouped_baseline.values()]),
        np.asarray([np.mean(values) for values in grouped_candidate.values()]),
    )


def paired_sign_flip_test(
    differences: Sequence[float],
    *,
    replicates: int = 10_000,
    seed: int = 2026,
    max_exact_permutations: int = 65_536,
) -> tuple[float, int, bool]:
    """Two-sided paired sign-flip test, exact for sufficiently small samples."""

    _validate_resampling(replicates=replicates, alpha=0.05)
    if (
        isinstance(max_exact_permutations, bool)
        or not isinstance(max_exact_permutations, int)
        or max_exact_permutations < 2
    ):
        raise ValueError("max_exact_permutations must be an integer of at least two")
    values = _finite_vector(differences, name="differences")
    observed = abs(float(values.mean()))
    exact_count = 2 ** int(values.size)
    tolerance = np.finfo(float).eps * max(1.0, observed) * 8
    if exact_count <= max_exact_permutations:
        exceed = 0
        for exact_signs in itertools.product((-1.0, 1.0), repeat=int(values.size)):
            statistic = abs(float(np.dot(values, np.asarray(exact_signs)) / values.size))
            exceed += int(statistic + tolerance >= observed)
        return exceed / exact_count, exact_count, True
    rng = np.random.default_rng(seed)
    exceed = 0
    batch_size = max(1, min(replicates, (32 * 1024 * 1024) // (8 * values.size)))
    for start in range(0, replicates, batch_size):
        count = min(batch_size, replicates - start)
        random_signs = rng.integers(0, 2, size=(count, values.size), dtype=np.int8) * 2 - 1
        statistics = np.abs((random_signs * values).mean(axis=1))
        exceed += int(np.count_nonzero(statistics + tolerance >= observed))
    return (exceed + 1) / (replicates + 1), replicates, False


def continuous_paired_analysis(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    cluster_ids: Sequence[Hashable] | None = None,
    bootstrap_replicates: int = 10_000,
    permutation_replicates: int = 10_000,
    seed: int = 2026,
    alpha: float = 0.05,
    seed_count: int = 1,
) -> PairedContinuousResult:
    """Analyze continuous pairs after equal-weight task/scenario aggregation."""

    left = _finite_vector(baseline, name="baseline")
    right = _finite_vector(candidate, name="candidate")
    if left.size != right.size:
        raise ValueError("paired vectors must have equal nonzero length")
    if isinstance(seed_count, bool) or not isinstance(seed_count, int) or seed_count < 1:
        raise ValueError("seed_count must be a positive integer")
    _validate_resampling(replicates=bootstrap_replicates, alpha=alpha)
    _validate_resampling(replicates=permutation_replicates, alpha=alpha)
    unit_left, unit_right = _collapse_clusters(left, right, cluster_ids)
    differences = unit_right - unit_left
    ci_low, ci_high = paired_bootstrap(
        differences, replicates=bootstrap_replicates, seed=seed, alpha=alpha
    )
    permutation_p, actual_permutations, exact = paired_sign_flip_test(
        differences, replicates=permutation_replicates, seed=seed + 1
    )
    return PairedContinuousResult(
        observation_count=int(left.size),
        analysis_unit_count=int(unit_left.size),
        baseline_mean=float(unit_left.mean()),
        candidate_mean=float(unit_right.mean()),
        absolute_difference=float(differences.mean()),
        ci_low=ci_low,
        ci_high=ci_high,
        permutation_p=permutation_p,
        permutation_replicates=actual_permutations,
        permutation_exact=exact,
        bootstrap_replicates=bootstrap_replicates,
        seed_count=seed_count,
    )


def _validated_p_values(p_values: Sequence[float]) -> list[float]:
    values: list[float] = []
    for value in p_values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("p-values must be numeric")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError("p-values must be finite and in [0, 1]")
        values.append(number)
    return values


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm family-wise-error adjusted p-values in original order."""

    values = _validated_p_values(p_values)
    count = len(values)
    order = sorted(range(count), key=values.__getitem__)
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Return BH-FDR adjusted p-values in original order."""

    values = _validated_p_values(p_values)
    count = len(values)
    order = sorted(range(count), key=values.__getitem__, reverse=True)
    adjusted = [0.0] * count
    running = 1.0
    for reverse_rank, index in enumerate(order, start=1):
        rank = count - reverse_rank + 1
        running = min(running, values[index] * count / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def seed_aware_summary(
    differences_by_seed: Mapping[str, Sequence[float]],
    *,
    unit_ids_by_seed: Mapping[str, Sequence[Hashable]] | None = None,
    bootstrap_replicates: int = 10_000,
    seed: int = 2026,
    alpha: float = 0.05,
) -> SeedAwareResult:
    """Equal-seed hierarchical bootstrap over seeds and task differences."""

    _validate_resampling(replicates=bootstrap_replicates, alpha=alpha)
    if not differences_by_seed:
        raise ValueError("at least one seed is required")
    seed_ids = sorted(differences_by_seed)
    if any(not isinstance(seed_id, str) or not seed_id for seed_id in seed_ids):
        raise ValueError("seed IDs must be non-empty strings")
    vectors = [
        _finite_vector(differences_by_seed[seed_id], name=f"seed {seed_id} differences")
        for seed_id in seed_ids
    ]
    if len({int(vector.size) for vector in vectors}) != 1:
        raise ValueError("every seed must contain the same number of protected analysis units")
    if unit_ids_by_seed is not None:
        if set(unit_ids_by_seed) != set(seed_ids):
            raise ValueError("unit_ids_by_seed must contain exactly the analyzed seeds")
        expected_units: set[Hashable] | None = None
        for seed_id, vector in zip(seed_ids, vectors, strict=True):
            unit_ids = unit_ids_by_seed[seed_id]
            _validate_unit_ids(unit_ids, int(vector.size))
            current_units = set(unit_ids)
            if expected_units is None:
                expected_units = current_units
            elif current_units != expected_units:
                raise ValueError("every seed must contain the same protected analysis unit IDs")
    per_seed = tuple(
        PerSeedEstimate(seed_id, int(values.size), float(values.mean()))
        for seed_id, values in zip(seed_ids, vectors, strict=True)
    )
    observed = float(np.mean([item.mean_difference for item in per_seed]))
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_replicates, dtype=float)
    for replicate in range(bootstrap_replicates):
        selected = rng.integers(0, len(vectors), size=len(vectors))
        seed_means = [
            float(vectors[index][rng.integers(0, vectors[index].size, vectors[index].size)].mean())
            for index in selected
        ]
        bootstrap[replicate] = np.mean(seed_means)
    low, high = np.quantile(bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0])
    identity_qualifier = (
        " Exact protected unit identities matched across seeds."
        if unit_ids_by_seed is not None
        else " Only equal unit counts were verified because unit IDs were not supplied."
    )
    return SeedAwareResult(
        seed_count=len(vectors),
        observation_count=sum(item.observation_count for item in per_seed),
        unit_ids_verified=unit_ids_by_seed is not None,
        mean_difference=observed,
        ci_low=float(low),
        ci_high=float(high),
        bootstrap_replicates=bootstrap_replicates,
        per_seed=per_seed,
        qualifier=(
            "Hierarchical bootstrap over observed seeds and task units; each seed is reported "
            "explicitly. Few seeds do not support high-powered population inference."
            f"{identity_qualifier}"
        ),
    )
