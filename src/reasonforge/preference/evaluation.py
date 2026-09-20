"""Pure response-token likelihood metrics for preference evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True, slots=True)
class ResponseLogProb:
    """Conditional log-probability statistics for one response."""

    total: float
    mean: float
    token_count: int


def masked_response_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> list[ResponseLogProb]:
    """Score only response tokens using causal next-token probabilities.

    ``response_mask`` is aligned to ``input_ids``: prompt and padding positions
    are zero and response positions are one. The first sequence position cannot
    be scored by a causal model and is therefore removed with the logits shift.
    The row loop mirrors TRL's memory-conscious selective-log-softmax behavior.
    """

    if logits.ndim != 3 or input_ids.ndim != 2 or response_mask.ndim != 2:
        raise ValueError("expected logits [batch, sequence, vocabulary] and rank-2 ids/mask")
    if logits.shape[:2] != input_ids.shape or input_ids.shape != response_mask.shape:
        raise ValueError("logits, input IDs, and response mask shapes are inconsistent")
    if input_ids.shape[1] < 2:
        raise ValueError("sequences must contain at least two tokens")
    if response_mask.dtype is not torch.bool:
        response_mask = response_mask.bool()

    labels = input_ids[:, 1:]
    shifted_mask = response_mask[:, 1:]
    scores: list[ResponseLogProb] = []
    for row_logits, row_labels, row_mask in zip(
        logits[:, :-1, :], labels, shifted_mask, strict=True
    ):
        count = int(row_mask.sum().item())
        if count < 1:
            raise ValueError("a sequence has no scoreable response tokens after truncation")
        # Avoid materializing a full-batch float32 log-softmax. This follows the
        # pinned TRL implementation's row-wise path for BF16 model logits.
        row_log_probs = functional.log_softmax(row_logits, dim=-1)
        selected = row_log_probs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
        selected = selected[row_mask].float()
        total = float(selected.sum().item())
        scores.append(ResponseLogProb(total=total, mean=total / count, token_count=count))
    return scores


def preference_pair_record(
    chosen: ResponseLogProb,
    rejected: ResponseLogProb,
) -> dict[str, float | int | bool]:
    """Create both sequence-sum and length-normalized preference margins."""

    sum_margin = chosen.total - rejected.total
    mean_margin = chosen.mean - rejected.mean
    return {
        "chosen_logprob_sum": chosen.total,
        "rejected_logprob_sum": rejected.total,
        "sum_logprob_margin": sum_margin,
        "chosen_logprob_mean": chosen.mean,
        "rejected_logprob_mean": rejected.mean,
        "mean_logprob_margin": mean_margin,
        "chosen_response_token_count": chosen.token_count,
        "rejected_response_token_count": rejected.token_count,
        "chosen_preferred_by_sum": sum_margin > 0.0,
        "chosen_preferred_by_mean": mean_margin > 0.0,
    }


def aggregate_preference_records(
    records: Sequence[Mapping[str, object]],
) -> dict[str, float | int]:
    """Aggregate paired preference records without treating tokens as units."""

    if not records:
        raise ValueError("at least one paired preference record is required")

    def numeric(record: Mapping[str, object], key: str) -> float:
        value = record[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be numeric")
        return float(value)

    sum_margins = [numeric(record, "sum_logprob_margin") for record in records]
    mean_margins = [numeric(record, "mean_logprob_margin") for record in records]
    count = len(records)
    return {
        "pair_count": count,
        "mean_sum_logprob_margin": sum(sum_margins) / count,
        "mean_length_normalized_logprob_margin": sum(mean_margins) / count,
        "chosen_preferred_rate_by_sum": sum(margin > 0.0 for margin in sum_margins) / count,
        "chosen_preferred_rate_by_mean": sum(margin > 0.0 for margin in mean_margins) / count,
        "sum_margin_ties": sum(margin == 0.0 for margin in sum_margins),
        "mean_margin_ties": sum(margin == 0.0 for margin in mean_margins),
    }
