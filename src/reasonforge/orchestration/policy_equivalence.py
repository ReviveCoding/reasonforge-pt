"""CPU-only proof for E14 reliability-policy equivalence to E12 uniform."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

from reasonforge.config import file_sha256, identity_hash
from reasonforge.curriculum import ComparisonSampler, RCACConfig
from reasonforge.orchestration.grpo_execution import load_qualified_adaptive_estimates
from reasonforge.orchestration.rl import RLSuite, _atomic_json


def _rcac_config(path: Path) -> RCACConfig:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("RCAC configuration must be a mapping")
    return RCACConfig(
        tau_q=float(value["tau_q"]),
        p_target=float(value["p_target"]),
        sigma=float(value["sigma"]),
        lambda_u=float(value["lambda_u"]),
    )


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def analyze_e14_policy_equivalence(
    workspace: Path,
    *,
    config_path: Path,
    publish: bool = False,
    output_path: Path = Path("evidence/training/E14/POLICY_EQUIVALENCE.json"),
    e12_schedule_path: Path = Path(
        "artifacts/training/E12/E12-uniform-grpo-seed1701-v1/ROLLOUT_SCHEDULE.json"
    ),
    e12_complete_path: Path = Path(
        "artifacts/training/E12/E12-uniform-grpo-seed1701-v1/RUN_COMPLETE.json"
    ),
) -> dict[str, Any]:
    """Prove the frozen E14 policy produces E12's exact task schedule."""

    root = workspace.resolve()
    suite = RLSuite.load(config_path)
    preflight = suite.preflight("E14", workspace=root)
    if not preflight["launch_authorized"]:
        raise RuntimeError(f"E14 equivalence blocked by preflight: {preflight}")
    starting_identity = preflight.get("starting_adapter_identity")
    if not isinstance(starting_identity, str):
        raise RuntimeError("E14 preflight did not resolve the frozen E10 adapter identity")
    rows = [dict(row) for row in pq.read_table(root / suite.dataset_path).to_pylist()]
    estimates, estimates_hash = load_qualified_adaptive_estimates(
        root / suite.adaptive_estimates_ref,
        rows,
        expected_dataset_fingerprint=suite.dataset_fingerprint,
        expected_starting_adapter_identity=starting_identity,
    )
    q_values = [estimate.q for estimate in estimates]
    if len(q_values) != 96 or any(value != 1.0 for value in q_values):
        raise RuntimeError("E14 equivalence requires exactly 96 q values equal to 1.0")

    e12_arm = suite.experiments["E12"].arms[0]
    e14_arm = suite.experiments["E14"].arms[0]
    if e12_arm.seed != e14_arm.seed:
        raise RuntimeError("E12 and E14 seeds differ; seeded policy equivalence is impossible")
    config = _rcac_config(root / suite.sampler_config_ref)
    if any(estimate.q < config.tau_q for estimate in estimates):
        raise RuntimeError("at least one E14 task is ineligible under the reliability threshold")
    uniform = ComparisonSampler(
        estimates,
        config,
        mode="uniform",
        seed=e12_arm.seed,
        curriculum_phase="E12_EQUIVALENCE_REPLAY",
    )
    reliability = ComparisonSampler(
        estimates,
        config,
        mode="reliability_only",
        seed=e14_arm.seed,
        curriculum_phase="E14_POLICY_EQUIVALENCE",
    )
    uniform_distribution = uniform.distribution()
    reliability_distribution = reliability.distribution()
    if uniform_distribution != reliability_distribution:
        raise RuntimeError("E14 reliability distribution is not exactly uniform")
    group_count = suite.budget.rollout_groups
    uniform_tasks = uniform.select(group_count)
    reliability_tasks = reliability.select(group_count)
    if uniform_tasks != reliability_tasks:
        raise RuntimeError("seeded E14 reliability schedule differs from computed E12 uniform")

    schedule_path = root / e12_schedule_path
    complete_path = root / e12_complete_path
    schedule = _load_json_object(schedule_path, "E12 frozen rollout schedule")
    complete = _load_json_object(complete_path, "E12 run completion manifest")
    schedule_errors = []
    if schedule.get("status") != "PREPARED" or schedule.get("sampler") != "uniform":
        schedule_errors.append("not a PREPARED uniform schedule")
    if schedule.get("suite_config_hash") != suite.config_hash:
        schedule_errors.append("suite identity mismatch")
    if schedule.get("group_count") != group_count:
        schedule_errors.append("group count mismatch")
    if schedule.get("num_generations") != suite.budget.num_generations:
        schedule_errors.append("generation count mismatch")
    if complete.get("status") != "COMPLETE" or complete.get("experiment_id") != "E12":
        schedule_errors.append("E12 source run is not complete")
    if complete.get("schedule_hash") != schedule.get("schedule_hash"):
        schedule_errors.append("E12 completion/schedule hash mismatch")
    frozen_tasks = schedule.get("task_ids")
    if frozen_tasks != uniform_tasks:
        schedule_errors.append("computed uniform tasks differ from frozen executed E12 tasks")
    if schedule_errors:
        raise RuntimeError(f"E12 frozen schedule validation failed: {schedule_errors}")

    distribution_hash = identity_hash(uniform_distribution)
    task_sequence_hash = identity_hash(uniform_tasks)
    artifact: dict[str, Any] = {
        "status": "MEASURED_POLICY_EQUIVALENCE",
        "evidence_class": "MEASURED",
        "experiment_id": "E14",
        "source_experiment_id": "E12",
        "method": "EXACT_POLICY_AND_SEEDED_TASK_SEQUENCE_EQUIVALENCE",
        "disposition": "NO_REDUNDANT_CUDA_RUN_REQUIRED",
        "cuda_run_executed_for_e14": False,
        "protected_access_state": "NOT_ACCESSED",
        "protocol_id": suite.protocol_id,
        "suite_config_hash": suite.config_hash,
        "fairness_budget_hash": suite.budget.fairness_identity(),
        "starting_adapter_identity": starting_identity,
        "reward_config_hash": suite.reward_config_sha256,
        "sampler_config_hash": suite.sampler_config_sha256,
        "adaptive_estimates_path": suite.adaptive_estimates_ref.as_posix(),
        "adaptive_estimates_sha256": estimates_hash,
        "candidate_task_count": len(estimates),
        "candidate_pool_identity": identity_hash([estimate.task_id for estimate in estimates]),
        "q_value_count": len(q_values),
        "q_unique_values": sorted(set(q_values)),
        "tau_q": config.tau_q,
        "all_tasks_reliability_eligible": True,
        "uniform_distribution_equals_reliability_distribution": True,
        "distribution_identity": distribution_hash,
        "seed": e12_arm.seed,
        "rollout_group_count": group_count,
        "computed_seeded_sequences_equal": True,
        "computed_sequence_equals_frozen_e12_schedule": True,
        "task_sequence_identity": task_sequence_hash,
        "task_sequence": uniform_tasks,
        "e12_schedule_path": e12_schedule_path.as_posix(),
        "e12_schedule_file_sha256": file_sha256(schedule_path),
        "e12_schedule_hash": schedule["schedule_hash"],
        "e12_run_complete_path": e12_complete_path.as_posix(),
        "e12_run_complete_sha256": file_sha256(complete_path),
        "authorized_wording": (
            "Under the qualified q_i=1 estimates, E14 reliability-only sampling is exactly "
            "policy- and seeded-task-sequence-equivalent to the measured E12 uniform run."
        ),
        "prohibited_stronger_wording": (
            "E14 was independently trained, or a second CUDA run numerically replicated E12."
        ),
        "limitations": [
            "This proves policy and seeded task-selection equivalence, not an independent GPU "
            "replication.",
            "The disposition becomes invalid if q, threshold, seed, candidate order, or frozen "
            "budget changes.",
        ],
    }
    destination = root / output_path
    if publish:
        if destination.exists():
            existing = _load_json_object(destination, "existing E14 equivalence artifact")
            if existing != artifact:
                raise FileExistsError(
                    f"E14 policy-equivalence evidence is immutable and differs: {destination}"
                )
        else:
            _atomic_json(destination, artifact)
    return artifact
