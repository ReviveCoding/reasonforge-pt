"""Paired, seed-aware, and pre-protected policy analyses."""

from reasonforge.statistics.paired import (
    PairedBinaryResult,
    PairedContinuousResult,
    PerSeedEstimate,
    SeedAwareResult,
    benjamini_hochberg,
    binary_paired_analysis,
    continuous_paired_analysis,
    holm_adjust,
    paired_bootstrap,
    paired_sign_flip_test,
    seed_aware_summary,
)
from reasonforge.statistics.policy import (
    EndpointRule,
    MarginProposal,
    PilotVariance,
    PrecisionStatus,
    propose_margin,
    propose_policy,
)

__all__ = [
    "EndpointRule",
    "MarginProposal",
    "PairedBinaryResult",
    "PairedContinuousResult",
    "PerSeedEstimate",
    "PilotVariance",
    "PrecisionStatus",
    "SeedAwareResult",
    "benjamini_hochberg",
    "binary_paired_analysis",
    "continuous_paired_analysis",
    "holm_adjust",
    "paired_bootstrap",
    "paired_sign_flip_test",
    "propose_margin",
    "propose_policy",
    "seed_aware_summary",
]
