"""Single-owner canonical reward scaling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from math import isfinite


@dataclass(frozen=True)
class RewardComponents:
    reason: float = 0.0
    instruction: float = 0.0
    tool: float = 0.0
    safety: float = 0.0
    factual: float = 0.0
    efficiency_cost: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {field.name: float(getattr(self, field.name)) for field in fields(self)}


@dataclass(frozen=True)
class RewardWeights:
    reason: float = 1.0
    instruction: float = 1.0
    tool: float = 1.0
    safety: float = 1.0
    factual: float = 1.0
    efficiency_cost: float = 1.0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = float(getattr(self, field.name))
            if not isfinite(value) or value < 0:
                raise ValueError(f"weight {field.name} must be finite and non-negative")

    def as_dict(self) -> dict[str, float]:
        return {field.name: float(getattr(self, field.name)) for field in fields(self)}


class CanonicalRewardAggregator:
    """Compute the sole scalar reward and preserve component contributions.

    Positive component values are constrained to [0, 1].  Efficiency is a
    non-negative cost and is the only subtracted component.  The class rejects
    accidental unknown or duplicate environment-owned scalar rewards.
    """

    def __init__(self, weights: RewardWeights) -> None:
        self.weights = weights

    def aggregate(self, components: RewardComponents) -> float:
        values = components.as_dict()
        for name in ("reason", "instruction", "tool", "safety", "factual"):
            value = values[name]
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"component {name} must be finite and in [0, 1]")
        cost = values["efficiency_cost"]
        if not isfinite(cost) or cost < 0.0:
            raise ValueError("efficiency_cost must be finite and non-negative")
        weights = self.weights.as_dict()
        positive = sum(weights[name] * values[name] for name in values if name != "efficiency_cost")
        return positive - weights["efficiency_cost"] * cost

    def contributions(self, components: RewardComponents) -> Mapping[str, float]:
        """Return independently loggable signed weighted components."""

        self.aggregate(components)  # validation
        values = components.as_dict()
        weights = self.weights.as_dict()
        return {
            name: (-1.0 if name == "efficiency_cost" else 1.0) * weights[name] * value
            for name, value in values.items()
        }
