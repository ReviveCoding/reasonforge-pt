"""Exact CPU qualification of the E19/E20 RCAC ablation policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from reasonforge.config import file_sha256, identity_hash
from reasonforge.curriculum import ComparisonSampler
from reasonforge.orchestration.grpo_execution import (
    _load_rcac_config,
    load_qualified_adaptive_estimates,
    load_selected_reward_architecture,
)
from reasonforge.orchestration.rl import RLSuite, _atomic_json


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"ablation evidence is unreadable: {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"ablation evidence is not a JSON object: {path}")
    return value


def _validate_source_run(
    root: Path,
    *,
    suite: RLSuite,
    experiment_id: str,
    run_id: str,
    sampler_mode: str,
    reward_architecture: str,
    estimate_sha256: str | None,
) -> tuple[Path, Path]:
    directory = root / "artifacts" / "training" / experiment_id / run_id
    complete_path = directory / "RUN_COMPLETE.json"
    schedule_path = directory / "ROLLOUT_SCHEDULE.json"
    complete = _json(complete_path)
    schedule = _json(schedule_path)
    required = {
        "status": "COMPLETE",
        "experiment_id": experiment_id,
        "run_id": run_id,
        "execution_mode": "FULL",
        "suite_config_hash": suite.config_hash,
        "completed_rollouts": suite.budget.rollouts,
        "starting_adapter_identity": _json(root / suite.starting_checkpoint_ref).get(
            "final_adapter_identity"
        ),
    }
    failures = [key for key, expected in required.items() if complete.get(key) != expected]
    if sampler_mode != "uniform" and complete.get("sampler_mode") != sampler_mode:
        failures.append("sampler_mode")
    if complete.get("reward_architecture", "structured") != reward_architecture:
        failures.append("reward_architecture")
    if estimate_sha256 is not None and complete.get("adaptive_estimates_sha256") != estimate_sha256:
        failures.append("adaptive_estimates_sha256")
    if (
        schedule.get("status") != "PREPARED"
        or schedule.get("sampler") != sampler_mode
        or schedule.get("schedule_hash") != complete.get("schedule_hash")
        or len(schedule.get("task_ids", [])) != suite.budget.rollout_groups
    ):
        failures.append("schedule")
    if failures:
        raise RuntimeError(f"{experiment_id} ablation source identity failed: {sorted(failures)}")
    return complete_path, schedule_path


def qualify_e19_e20_ablations(workspace: Path, *, suite_path: Path) -> dict[str, Path]:
    """Publish E19 equivalence and E20 non-equivalence under frozen identities."""

    root = workspace.resolve()
    suite = RLSuite.load(suite_path)
    rows = [dict(row) for row in pq.read_table(root / suite.dataset_path).to_pylist()]
    start_identity = _json(root / suite.starting_checkpoint_ref).get("final_adapter_identity")
    if not isinstance(start_identity, str):
        raise RuntimeError("frozen E10 starting adapter identity is missing")
    estimates, estimate_sha256 = load_qualified_adaptive_estimates(
        root / suite.adaptive_estimates_ref,
        rows,
        expected_dataset_fingerprint=suite.dataset_fingerprint,
        expected_starting_adapter_identity=start_identity,
    )
    if {estimate.q for estimate in estimates} != {1.0}:
        raise RuntimeError("E19/E20 equivalence audit requires every qualified q_i to equal 1")
    selected_path = root / "evidence/training/E16/SELECTED_REWARD.json"
    architecture, selection_sha256, selection_identity = load_selected_reward_architecture(
        selected_path, workspace=root, suite=suite
    )
    if architecture != "terminal":
        raise RuntimeError("the current E19/E20 audit is bound to selected terminal reward")
    e18_complete, e18_schedule_path = _validate_source_run(
        root,
        suite=suite,
        experiment_id="E18",
        run_id="E18-rcac-grpo-seed1701-v1",
        sampler_mode="rcac",
        reward_architecture="terminal",
        estimate_sha256=estimate_sha256,
    )
    e16_complete, e16_schedule_path = _validate_source_run(
        root,
        suite=suite,
        experiment_id="E16",
        run_id="E16-terminal-uniform-grpo-seed1701-v1",
        sampler_mode="uniform",
        reward_architecture="terminal",
        estimate_sha256=None,
    )
    e18_schedule = _json(e18_schedule_path)
    e16_schedule = _json(e16_schedule_path)
    config = _load_rcac_config(root / suite.sampler_config_ref)
    samplers = {
        mode: ComparisonSampler(estimates, config, mode=mode, seed=1701, curriculum_phase=mode)
        for mode in ("uniform", "rcac", "rcac_no_reliability", "rcac_no_difficulty")
    }
    sequences: dict[str, list[str]] = {mode: [] for mode in samplers}
    e19_distributions_equal = True
    e20_distributions_equal = True
    maximum_e20_probability_difference = 0.0
    for _ in range(suite.budget.rollout_groups):
        distributions = {mode: sampler.distribution() for mode, sampler in samplers.items()}
        e19_distributions_equal &= distributions["rcac"] == distributions["rcac_no_reliability"]
        e20_distributions_equal &= distributions["uniform"] == distributions["rcac_no_difficulty"]
        maximum_e20_probability_difference = max(
            maximum_e20_probability_difference,
            max(
                abs(
                    distributions["uniform"][task_id] - distributions["rcac_no_difficulty"][task_id]
                )
                for task_id in distributions["uniform"]
            ),
        )
        for mode, sampler in samplers.items():
            sequences[mode].extend(sampler.select())
    frozen_e18 = list(e18_schedule["task_ids"])
    frozen_e16 = list(e16_schedule["task_ids"])
    e19_equivalent = (
        e19_distributions_equal
        and sequences["rcac"] == sequences["rcac_no_reliability"] == frozen_e18
    )
    e20_equivalent = (
        e20_distributions_equal
        and sequences["uniform"] == sequences["rcac_no_difficulty"] == frozen_e16
    )
    if not e19_equivalent:
        raise RuntimeError("E19 no-reliability policy is not exactly equivalent to frozen E18")
    common = {
        "protocol_id": suite.protocol_id,
        "suite_config_hash": suite.config_hash,
        "fairness_budget_hash": suite.budget.fairness_identity(),
        "dataset_fingerprint": suite.dataset_fingerprint,
        "starting_adapter_identity": start_identity,
        "adaptive_estimates_sha256": estimate_sha256,
        "selected_reward_architecture": architecture,
        "selected_reward_sha256": selection_sha256,
        "selected_reward_identity": selection_identity,
        "seed": 1701,
        "rollout_group_count": suite.budget.rollout_groups,
        "all_q_exactly_one": True,
        "protected_access_state": "NOT_ACCESSED",
    }
    e19: dict[str, Any] = {
        **common,
        "status": "MEASURED_POLICY_EQUIVALENCE",
        "experiment_id": "E19",
        "method": "EXACT_SCORE_DISTRIBUTION_AND_SEEDED_SEQUENCE_EQUIVALENCE",
        "source_experiment_id": "E18",
        "source_run_complete_sha256": file_sha256(e18_complete),
        "source_schedule_sha256": file_sha256(e18_schedule_path),
        "source_schedule_hash": e18_schedule["schedule_hash"],
        "score_distributions_exactly_equal_at_every_selection": e19_distributions_equal,
        "computed_seeded_sequences_equal": sequences["rcac"] == sequences["rcac_no_reliability"],
        "computed_sequence_equals_frozen_e18_schedule": sequences["rcac"] == frozen_e18,
        "cuda_run_executed_for_e19": False,
        "disposition": "NO_REDUNDANT_CUDA_RUN_REQUIRED",
        "authorized_wording": (
            "With every qualified q_i=1, E19 RCAC without reliability gating is exactly "
            "policy- and seeded-task-sequence-equivalent to the measured E18 RCAC run."
        ),
        "prohibited_stronger_wording": "E19 was independently trained or replicated on CUDA.",
    }
    e19["evidence_identity"] = identity_hash(e19)
    mismatch_indices = [
        index
        for index, (uniform, no_difficulty) in enumerate(
            zip(sequences["uniform"], sequences["rcac_no_difficulty"], strict=True)
        )
        if uniform != no_difficulty
    ]
    e20: dict[str, Any] = {
        **common,
        "status": "QUALIFIED_DISTINCT_POLICY",
        "experiment_id": "E20",
        "method": "EXACT_NON_EQUIVALENCE_AUDIT",
        "candidate_sampler": "rcac_no_difficulty",
        "proposed_reuse_experiment_id": "E16",
        "proposed_reuse_run_complete_sha256": file_sha256(e16_complete),
        "proposed_reuse_schedule_sha256": file_sha256(e16_schedule_path),
        "proposed_reuse_schedule_hash": e16_schedule["schedule_hash"],
        "score_distributions_exactly_equal_at_every_selection": e20_distributions_equal,
        "computed_seeded_sequences_equal": sequences["uniform"] == sequences["rcac_no_difficulty"],
        "computed_sequence_equals_frozen_e16_schedule": sequences["rcac_no_difficulty"]
        == frozen_e16,
        "first_sequence_mismatch_index": mismatch_indices[0] if mismatch_indices else None,
        "sequence_mismatch_count": len(mismatch_indices),
        "maximum_probability_difference": maximum_e20_probability_difference,
        "cuda_reuse_authorized": e20_equivalent,
        "distinct_cuda_run_required": not e20_equivalent,
        "reason": (
            "Removing difficulty leaves informativeness and dynamic family balancing; this is "
            "not the uniform E16 policy under the qualified estimates."
        ),
    }
    if e20_equivalent:
        raise RuntimeError("unexpected E20 equivalence requires explicit protocol review")
    e20["evidence_identity"] = identity_hash(e20)
    outputs = {
        "E19": root / "evidence/training/E19/POLICY_EQUIVALENCE.json",
        "E20": root / "evidence/training/E20/POLICY_QUALIFICATION.json",
    }
    for experiment_id, artifact in (("E19", e19), ("E20", e20)):
        path = outputs[experiment_id]
        if path.exists() and _json(path) != artifact:
            raise RuntimeError(f"existing {experiment_id} qualification does not match")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_json(path, artifact)
    return outputs
