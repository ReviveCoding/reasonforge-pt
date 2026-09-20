"""Fail-closed E16 reward-architecture selection on DEVELOPMENT evidence."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reasonforge.config import file_sha256, identity_hash
from reasonforge.orchestration.rl import RLSuite, _atomic_json
from reasonforge.statistics import binary_paired_analysis, continuous_paired_analysis

SELECTION_POLICY_ID = "E16-STRONGEST-ELIGIBLE-DEVELOPMENT-v1"
_CONTROL_RUN = "E12-uniform-grpo-seed1701-v1"
_CANDIDATE_RUN = "E16-terminal-uniform-grpo-seed1701-v1"
_CONTROL_EVAL = "E12-uniform-grpo-dev-eval-v1"
_CANDIDATE_EVAL = "E16-terminal-uniform-grpo-dev-eval-v1"


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"required E16 evidence is unreadable: {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"required E16 evidence is not a JSON object: {path}")
    return value


def _predictions(path: Path, *, expected_experiment: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"required E16 predictions are unreadable: {path}: {error}") from error
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid prediction JSONL in {path}: {error}") from error
        if not isinstance(row, dict):
            raise RuntimeError(f"prediction row is not an object: {path}")
        example_id = row.get("example_id")
        scoring = row.get("scoring")
        if (
            not isinstance(example_id, str)
            or not isinstance(scoring, Mapping)
            or row.get("split_role") != "DEVELOPMENT"
            or row.get("experiment_id") != expected_experiment
            or example_id in records
        ):
            raise RuntimeError(f"prediction identity/role violation: {path}")
        score = scoring.get("score")
        passed = scoring.get("passed")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not 0.0 <= float(score) <= 1.0
            or type(passed) is not bool
        ):
            raise RuntimeError(f"prediction scoring violation: {path}")
        records[example_id] = row
    if len(records) != 36:
        raise RuntimeError(f"E16 selection requires exactly 36 DEVELOPMENT units: {path}")
    return records


def _family_means(records: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in records.values():
        values[str(row["task_family"])].append(float(row["scoring"]["score"]))
    return {family: sum(scores) / len(scores) for family, scores in sorted(values.items())}


def _validate_summary(
    summary: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]], *, label: str
) -> None:
    scores = [float(row["scoring"]["score"]) for row in records.values()]
    passes = [bool(row["scoring"]["passed"]) for row in records.values()]
    if (
        abs(float(summary.get("mean_score", -1.0)) - sum(scores) / len(scores)) > 1e-12
        or abs(float(summary.get("pass_rate", -1.0)) - sum(passes) / len(passes)) > 1e-12
    ):
        raise RuntimeError(f"{label} summary metrics do not reproduce from predictions")


def build_e16_reward_selection(workspace: Path, *, suite_path: Path) -> dict[str, Any]:
    """Select the strongest eligible E16 architecture without a superiority claim."""

    root = workspace.resolve()
    suite = RLSuite.load(suite_path)
    control_dir = root / "artifacts" / "training" / "E12" / _CONTROL_RUN
    candidate_dir = root / "artifacts" / "training" / "E16" / _CANDIDATE_RUN
    control_eval = root / "outputs" / _CONTROL_EVAL
    candidate_eval = root / "outputs" / _CANDIDATE_EVAL
    paths = {
        "paired_comparison": root
        / "evidence"
        / "evaluation"
        / "E16_TERMINAL_VS_STRUCTURED_DEVELOPMENT_COMPARISON.json",
        "control_run_complete": control_dir / "RUN_COMPLETE.json",
        "control_run_identity": control_dir / "RUN_IDENTITY.json",
        "control_schedule": control_dir / "ROLLOUT_SCHEDULE.json",
        "candidate_run_complete": candidate_dir / "RUN_COMPLETE.json",
        "candidate_run_identity": candidate_dir / "RUN_IDENTITY.json",
        "candidate_schedule": candidate_dir / "ROLLOUT_SCHEDULE.json",
        "control_eval_summary": control_eval / "summary.json",
        "control_predictions": control_eval / "predictions.jsonl",
        "candidate_eval_summary": candidate_eval / "summary.json",
        "candidate_predictions": candidate_eval / "predictions.jsonl",
    }
    values = {name: _json(path) for name, path in paths.items() if "predictions" not in name}
    control_complete = values["control_run_complete"]
    candidate_complete = values["candidate_run_complete"]
    control_identity = values["control_run_identity"]
    candidate_identity = values["candidate_run_identity"]
    control_schedule = values["control_schedule"]
    candidate_schedule = values["candidate_schedule"]
    failures = []
    if (
        control_complete.get("status") != "COMPLETE"
        or candidate_complete.get("status") != "COMPLETE"
    ):
        failures.append("training run incomplete")
    for identity in (control_identity, candidate_identity):
        if identity.get("suite_config_hash") != suite.config_hash:
            failures.append("suite identity mismatch")
        if identity.get("fairness_budget_hash") != suite.budget.fairness_identity():
            failures.append("fairness budget mismatch")
        if identity.get("protected_access_state") != "NOT_ACCESSED":
            failures.append("protected-access boundary violation")
    if candidate_identity.get("reward_architecture") != "terminal":
        failures.append("candidate is not terminal reward")
    if candidate_identity.get("starting_adapter_identity") != control_identity.get(
        "starting_adapter_identity"
    ):
        failures.append("starting adapter mismatch")
    if candidate_schedule.get("task_ids") != control_schedule.get("task_ids"):
        failures.append("training task sequence mismatch")
    if candidate_complete.get("completed_rollouts") != control_complete.get("completed_rollouts"):
        failures.append("rollout budget mismatch")
    if failures:
        raise RuntimeError(f"E16 reward selection identity failed: {sorted(set(failures))}")

    control_summary = values["control_eval_summary"]
    candidate_summary = values["candidate_eval_summary"]
    if any(summary.get("status") != "COMPLETE" for summary in (control_summary, candidate_summary)):
        raise RuntimeError("E16 selection requires complete evaluation summaries")
    fixed_eval_fields = ("scientific_state", "generation_config_id", "dataset_path", "dataset_rows")
    if any(control_summary.get(key) != candidate_summary.get(key) for key in fixed_eval_fields):
        raise RuntimeError("E16 evaluation identities are not matched")
    if control_summary.get("scientific_state") != "DEVELOPMENT":
        raise RuntimeError("E16 reward selection may use DEVELOPMENT only")

    control = _predictions(paths["control_predictions"], expected_experiment="E12")
    candidate = _predictions(paths["candidate_predictions"], expected_experiment="E16")
    if set(control) != set(candidate):
        raise RuntimeError("E16 evaluation example IDs are not exactly paired")
    _validate_summary(control_summary, control, label="structured")
    _validate_summary(candidate_summary, candidate, label="terminal")
    unit_ids = sorted(control)
    control_scores = [float(control[key]["scoring"]["score"]) for key in unit_ids]
    candidate_scores = [float(candidate[key]["scoring"]["score"]) for key in unit_ids]
    control_pass = [bool(control[key]["scoring"]["passed"]) for key in unit_ids]
    candidate_pass = [bool(candidate[key]["scoring"]["passed"]) for key in unit_ids]
    continuous = continuous_paired_analysis(
        control_scores,
        candidate_scores,
        bootstrap_replicates=10_000,
        permutation_replicates=10_000,
        seed=2112,
        seed_count=1,
    )
    binary = binary_paired_analysis(
        control_pass,
        candidate_pass,
        unit_ids=unit_ids,
        bootstrap_replicates=10_000,
        seed=2112,
        seed_count=1,
    )
    control_families = _family_means(control)
    candidate_families = _family_means(candidate)
    if set(control_families) != set(candidate_families):
        raise RuntimeError("E16 evaluation family coverage mismatch")
    family_differences = {
        family: candidate_families[family] - control_families[family] for family in control_families
    }
    eligible = all(delta >= -1e-12 for delta in family_differences.values())
    # Predeclared operational selection: maximize paired DEVELOPMENT mean among
    # eligible arms; exact ties retain structured. This is not an inferential
    # superiority rule and cannot support a protected or multi-seed claim.
    selected = (
        "terminal"
        if eligible and continuous.candidate_mean > continuous.baseline_mean
        else "structured"
    )
    comparison = values["paired_comparison"]
    expected_comparison = comparison.get("comparisons", {}).get("E12_STRUCTURED", {})
    if (
        comparison.get("status") != "COMPLETE"
        or comparison.get("scientific_state") != "DEVELOPMENT"
        or comparison.get("protected_data_accessed") is not False
        or expected_comparison.get("continuous_score") != continuous.to_dict()
        or expected_comparison.get("binary_pass") != binary.to_dict()
    ):
        raise RuntimeError("published E16 paired comparison does not reproduce exactly")
    artifact: dict[str, Any] = {
        "status": "SELECTED",
        "experiment_id": "E16",
        "protocol_id": suite.protocol_id,
        "selection_policy_id": SELECTION_POLICY_ID,
        "selection_rule": {
            "basis": "MASTER_PROTOCOL_STRONGEST_ELIGIBLE_REWARD_RULE",
            "eligibility": "NO_DEVELOPMENT_TASK_FAMILY_MEAN_DROP",
            "ranking": "HIGHEST_PAIRED_DEVELOPMENT_MEAN_SCORE",
            "tie_break": "STRUCTURED",
        },
        "selected_reward_architecture": selected,
        "selected_arm_id": "E16_terminal" if selected == "terminal" else "E16_structured",
        "selected_reward_config_sha256": suite.reward_config_sha256,
        "selected_reward_implementation": candidate_identity.get("reward_implementation"),
        "sample_size": len(unit_ids),
        "training_seed_count": 1,
        "scientific_state": "DEVELOPMENT",
        "paired_score": continuous.to_dict(),
        "paired_pass": binary.to_dict(),
        "family_mean_differences_terminal_minus_structured": family_differences,
        "terminal_eligible": eligible,
        "inferential_qualifier": (
            "Terminal had the higher eligible DEVELOPMENT point estimate, but the paired 95% "
            "bootstrap interval includes zero; this single-seed result does not establish "
            "terminal superiority."
        ),
        "h4_development_result": "STRUCTURED_REWARD_IMPROVEMENT_NOT_OBSERVED",
        "protected_access_state": "NOT_ACCESSED",
        "source_artifacts": {
            name: {"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)}
            for name, path in sorted(paths.items())
        },
    }
    artifact["selection_identity"] = identity_hash(artifact)
    return artifact


def publish_e16_reward_selection(workspace: Path, *, suite_path: Path) -> Path:
    """Atomically publish the deterministic selection, or validate an existing copy."""

    root = workspace.resolve()
    artifact = build_e16_reward_selection(root, suite_path=suite_path)
    destination = root / "evidence" / "training" / "E16" / "SELECTED_REWARD.json"
    if destination.exists():
        if _json(destination) != artifact:
            raise RuntimeError("existing E16 reward selection does not match source evidence")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(destination, artifact)
    if _json(destination) != artifact:
        raise RuntimeError("published E16 reward selection failed verification")
    return destination
