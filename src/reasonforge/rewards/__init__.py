"""Transparent reward components, aggregation, and verifier qualification."""

from .aggregation import CanonicalRewardAggregator, RewardComponents, RewardWeights
from .completion import (
    CanonicalBatchReward,
    CanonicalRewardPolicy,
    CompletionReward,
    RewardArchitecture,
    canonical_reward_functions,
    reward_completion,
    score_completion,
    strip_trailing_control_tokens,
)
from .instruction import InstructionVerification, verify_instruction
from .qualification import BinaryQualification, qualify_binary_verifier
from .tool import ToolExpectation, ToolVerification, verify_tool_trajectory

__all__ = [
    "BinaryQualification",
    "CanonicalBatchReward",
    "CanonicalRewardAggregator",
    "CanonicalRewardPolicy",
    "CompletionReward",
    "InstructionVerification",
    "RewardArchitecture",
    "RewardComponents",
    "RewardWeights",
    "ToolExpectation",
    "ToolVerification",
    "canonical_reward_functions",
    "qualify_binary_verifier",
    "reward_completion",
    "score_completion",
    "strip_trailing_control_tokens",
    "verify_instruction",
    "verify_tool_trajectory",
]
