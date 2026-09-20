"""Pre-protected decision-margin proposals from POLICY/DEVELOPMENT variance.

This module deliberately accepts variance summaries rather than model outcomes.  It is
therefore suitable for defining a decision policy before LOCKED_FINAL scoring, but it
cannot evaluate or promote a model.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum

from reasonforge.config import SplitRole


class EndpointRule(StrEnum):
    """How a candidate-minus-baseline endpoint is used in the decision gate."""

    MATERIAL_IMPROVEMENT = "MATERIAL_IMPROVEMENT"
    NONINFERIORITY = "NONINFERIORITY"


class PrecisionStatus(StrEnum):
    """Whether the planned sample resolves the operational margin."""

    ADEQUATE = "ADEQUATE"
    UNDERPOWERED_WITHIN_BUDGET = "UNDERPOWERED_WITHIN_BUDGET"


_POLICY_ROLES = frozenset({SplitRole.DEVELOPMENT, SplitRole.POLICY})


@dataclass(frozen=True, slots=True)
class PilotVariance:
    """Outcome-blind paired-variance summary used to propose one endpoint rule.

    ``paired_sd`` is the pilot standard deviation of paired candidate-minus-baseline
    differences.  It is not the standard deviation of either model's unpaired scores.
    ``planned_pairs`` is the intended final evaluation size, bounded by
    ``max_pairs_from_budget``.
    """

    endpoint: str
    family: str
    role: SplitRole
    rule: EndpointRule
    paired_sd: float
    pilot_pairs: int
    planned_pairs: int
    max_pairs_from_budget: int
    operational_margin: float
    z_value: float = 1.96

    def __post_init__(self) -> None:
        if self.role not in _POLICY_ROLES:
            raise ValueError(
                "decision margins may use only DEVELOPMENT or POLICY variance summaries"
            )
        if not self.endpoint.strip() or not self.family.strip():
            raise ValueError("endpoint and family must be nonempty")
        if not math.isfinite(self.paired_sd) or self.paired_sd < 0:
            raise ValueError("paired_sd must be finite and nonnegative")
        if self.pilot_pairs < 2:
            raise ValueError("pilot_pairs must be at least 2")
        if self.planned_pairs < 1 or self.max_pairs_from_budget < 1:
            raise ValueError("planned and budgeted pair counts must be positive")
        if self.planned_pairs > self.max_pairs_from_budget:
            raise ValueError("planned_pairs exceeds max_pairs_from_budget")
        if not math.isfinite(self.operational_margin) or self.operational_margin <= 0:
            raise ValueError("operational_margin must be finite and positive")
        if not math.isfinite(self.z_value) or self.z_value <= 0:
            raise ValueError("z_value must be finite and positive")


@dataclass(frozen=True, slots=True)
class MarginProposal:
    """An unfrozen, outcome-blind decision-margin proposal."""

    endpoint: str
    family: str
    source_role: SplitRole
    rule: EndpointRule
    pilot_pairs: int
    planned_pairs: int
    max_pairs_from_budget: int
    paired_sd: float
    z_value: float
    uncertainty_resolution: float
    operational_margin: float
    decision_margin: float
    required_pairs_for_operational_margin: int
    precision_status: PrecisionStatus
    policy_status: str = "PROPOSED_UNFROZEN"

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


def propose_margin(summary: PilotVariance) -> MarginProposal:
    """Derive a conservative margin without inspecting endpoint point estimates.

    For a primary endpoint, the materiality bar is the larger of operational meaning
    and the planned sample's uncertainty resolution.  For a guardrail, pilot noise
    never relaxes the operational noninferiority tolerance; inadequate resolution is
    instead surfaced as an underpowered status.
    """

    resolution = summary.z_value * summary.paired_sd / math.sqrt(summary.planned_pairs)
    required_pairs = math.ceil(
        (summary.z_value * summary.paired_sd / summary.operational_margin) ** 2
    )
    required_pairs = max(1, required_pairs)
    precision_status = (
        PrecisionStatus.ADEQUATE
        if resolution <= summary.operational_margin
        else PrecisionStatus.UNDERPOWERED_WITHIN_BUDGET
    )
    decision_margin = (
        max(summary.operational_margin, resolution)
        if summary.rule is EndpointRule.MATERIAL_IMPROVEMENT
        else summary.operational_margin
    )
    return MarginProposal(
        endpoint=summary.endpoint,
        family=summary.family,
        source_role=summary.role,
        rule=summary.rule,
        pilot_pairs=summary.pilot_pairs,
        planned_pairs=summary.planned_pairs,
        max_pairs_from_budget=summary.max_pairs_from_budget,
        paired_sd=summary.paired_sd,
        z_value=summary.z_value,
        uncertainty_resolution=resolution,
        operational_margin=summary.operational_margin,
        decision_margin=decision_margin,
        required_pairs_for_operational_margin=required_pairs,
        precision_status=precision_status,
    )


def propose_policy(summaries: list[PilotVariance]) -> tuple[MarginProposal, ...]:
    """Build unique endpoint proposals in stable input order."""

    if not summaries:
        raise ValueError("at least one pilot variance summary is required")
    endpoints = [summary.endpoint for summary in summaries]
    if len(endpoints) != len(set(endpoints)):
        raise ValueError("endpoint names must be unique")
    return tuple(propose_margin(summary) for summary in summaries)
