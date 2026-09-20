"""CPU-safe public curriculum interface.

The full research runtime also contains evidence-backed estimate construction.
That path requires local rollout artifacts and is not imported by this interface.
"""

from reasonforge.curriculum.rcac import RCACConfig, RCACSampler, TaskEstimate
from reasonforge.curriculum.samplers import (
    AdaptiveGroupRepeatSampler,
    ComparisonMode,
    ComparisonSampler,
    SelectedTaskGroup,
)

__all__ = [
    "AdaptiveGroupRepeatSampler",
    "ComparisonMode",
    "ComparisonSampler",
    "RCACConfig",
    "RCACSampler",
    "SelectedTaskGroup",
    "TaskEstimate",
]
