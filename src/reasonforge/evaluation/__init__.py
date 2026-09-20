"""Protected evaluation evidence contracts."""

from .overoptimization import E17Protocol, analyze_e17, preflight_e17
from .protected_outputs import (
    DeterministicGraderResult,
    ProtectedOutputIdentity,
    ProtectedOutputIntegrityError,
    ProtectedOutputRecord,
    ProtectedOutputVerification,
    ProtectedOutputWriter,
    publish_deterministic_rescore,
    verify_deterministic_rescore,
)
from .protected_protocol import (
    FrozenProtectedRunPlan,
    ProtectedBenchmarkSpec,
    ProtectedGenerationConfig,
    ProtectedGenerationExample,
    freeze_run_plan,
    load_benchmark_specs,
    load_run_plan,
    run_protected_predictions,
    score_protected_outputs,
    verify_run_plan,
)

__all__ = [
    "DeterministicGraderResult",
    "E17Protocol",
    "FrozenProtectedRunPlan",
    "ProtectedBenchmarkSpec",
    "ProtectedGenerationConfig",
    "ProtectedGenerationExample",
    "ProtectedOutputIdentity",
    "ProtectedOutputIntegrityError",
    "ProtectedOutputRecord",
    "ProtectedOutputVerification",
    "ProtectedOutputWriter",
    "analyze_e17",
    "freeze_run_plan",
    "load_benchmark_specs",
    "load_run_plan",
    "preflight_e17",
    "publish_deterministic_rescore",
    "run_protected_predictions",
    "score_protected_outputs",
    "verify_deterministic_rescore",
    "verify_run_plan",
]
