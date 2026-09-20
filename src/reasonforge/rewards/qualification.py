"""Qualification statistics for binary rewards/verifiers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BinaryQualification:
    sample_size: int
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    accuracy: float
    sensitivity: float | None
    specificity: float | None
    perturbation_stability: float | None
    judge_agreement: float | None


def qualify_binary_verifier(
    truth: Sequence[bool],
    predictions: Sequence[bool],
    *,
    perturbed_predictions: Sequence[bool] | None = None,
    second_judge_predictions: Sequence[bool] | None = None,
) -> BinaryQualification:
    """Calculate confusion metrics plus optional stability/agreement rates."""

    if len(truth) != len(predictions):
        raise ValueError("truth and predictions must have equal length")
    if not truth:
        raise ValueError("qualification requires at least one example")
    if perturbed_predictions is not None and len(perturbed_predictions) != len(predictions):
        raise ValueError("perturbed_predictions must have equal length")
    if second_judge_predictions is not None and len(second_judge_predictions) != len(predictions):
        raise ValueError("second_judge_predictions must have equal length")

    tp = sum(expected and predicted for expected, predicted in zip(truth, predictions, strict=True))
    tn = sum(
        not expected and not predicted
        for expected, predicted in zip(truth, predictions, strict=True)
    )
    fp = sum(
        not expected and predicted for expected, predicted in zip(truth, predictions, strict=True)
    )
    fn = sum(
        expected and not predicted for expected, predicted in zip(truth, predictions, strict=True)
    )
    positives = tp + fn
    negatives = tn + fp
    stability = (
        sum(a == b for a, b in zip(predictions, perturbed_predictions, strict=True))
        / len(predictions)
        if perturbed_predictions is not None
        else None
    )
    agreement = (
        sum(a == b for a, b in zip(predictions, second_judge_predictions, strict=True))
        / len(predictions)
        if second_judge_predictions is not None
        else None
    )
    return BinaryQualification(
        sample_size=len(truth),
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        accuracy=(tp + tn) / len(truth),
        sensitivity=tp / positives if positives else None,
        specificity=tn / negatives if negatives else None,
        perturbation_stability=stability,
        judge_agreement=agreement,
    )
