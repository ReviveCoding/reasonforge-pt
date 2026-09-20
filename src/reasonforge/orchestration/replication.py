"""CPU-only, fail-closed planning for E25 finalist replication."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import yaml

from reasonforge.config import file_sha256, identity_hash
from reasonforge.curriculum import ComparisonMode
from reasonforge.orchestration.grpo_execution import (
    PreparedUniformSchedule,
    _execute_grpo_under_lease,
    _load_rcac_config,
    _read_training_rows,
    load_qualified_adaptive_estimates,
    prepare_uniform_schedule,
)
from reasonforge.orchestration.rl import RLSuite, _atomic_json
from reasonforge.systems import GpuOwnerLease

_ARM_NAMES = ("conventional", "rcac")
_EXPECTED_SEEDS = (1701, 1702, 1703)
_EXPECTED_NEW_SEEDS = (1702, 1703)
_FROZEN_CONFIG_SHA256 = "50ed5c6e389dfc6985aeaf5dc97677bb477ab7ff24faecf1f84486ccbec78d1c"
_FROZEN_DESIGN_SHA256 = "c7f7b55b13d6e167000df1a52a64dac04b343ac202df6948ca065d30cefea4bb"
_EXECUTOR_QUALIFICATION = Path("evidence/training/E25/EXECUTOR_QUALIFICATION.json")
_SELECTED_REWARD_IDENTITY = "babd51451067128fceb58cc77f66cde1ad7dc9837f0ddcc51cc38c97e022f0ae"


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"E25 {name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _exact_hash(root: Path, relative: object, expected: object, name: str) -> str:
    path = root / Path(str(relative))
    if not path.is_file():
        raise ValueError(f"E25 frozen {name} is missing: {path}")
    observed = file_sha256(path)
    if observed != str(expected):
        raise ValueError(f"E25 frozen {name} hash mismatch: {observed}")
    return observed


def _load_complete_run(root: Path, arm: Mapping[str, Any]) -> Mapping[str, Any]:
    path = root / Path(str(arm["imported_run_manifest"]))
    value = _mapping(json.loads(path.read_text(encoding="utf-8")), "imported run")
    if value.get("status") != "COMPLETE":
        raise ValueError(f"E25 imported run is not COMPLETE: {path}")
    if value.get("run_id") != arm.get("imported_run_id"):
        raise ValueError(f"E25 imported run identity mismatch: {path}")
    if value.get("final_adapter_identity") != arm.get("imported_final_adapter_identity"):
        raise ValueError(f"E25 imported adapter identity mismatch: {path}")
    checkpoint = path.parent / "checkpoint-32" / "CHECKPOINT_COMPLETE.json"
    if file_sha256(checkpoint) != arm.get("imported_checkpoint_completion_sha256"):
        raise ValueError(f"E25 imported checkpoint completion hash mismatch: {checkpoint}")
    return value


def _validate_budget(config: Mapping[str, Any], suite: RLSuite) -> None:
    frozen = dict(_mapping(config.get("budget"), "budget"))
    derived = {
        **asdict(suite.budget),
        "rollout_groups": suite.budget.rollout_groups,
        "rollouts": suite.budget.rollouts,
    }
    # E25 intentionally replaces the source suite's operational checkpoint
    # targets with its separately bound replication checkpoint policy.
    derived.pop("checkpoint_target_minutes")
    derived.pop("checkpoint_retention")
    if frozen != derived:
        raise ValueError("E25 budget is not identical to the frozen E16/E18 scientific budget")


def _schedule_for_run(
    *,
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    suite: RLSuite,
    seed: int,
    run_id: str,
    mode: ComparisonMode,
    estimates: Sequence[Any],
    estimate_sha256: str,
) -> PreparedUniformSchedule:
    return prepare_uniform_schedule(
        rows,
        tokenizer=tokenizer,
        rcac_config=_load_rcac_config(root / suite.sampler_config_ref),
        seed=seed,
        num_groups=suite.budget.rollout_groups,
        num_generations=suite.budget.num_generations,
        max_prompt_tokens=suite.budget.max_prompt_tokens,
        run_id=run_id,
        sampler_mode=mode,
        task_estimates=(estimates if mode == "rcac" else None),
        estimate_artifact_sha256=(estimate_sha256 if mode == "rcac" else None),
        curriculum_phase="E25",
    )


def build_e25_replication_design(
    workspace: Path,
    *,
    config_path: Path,
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, PreparedUniformSchedule]]:
    """Validate frozen sources and materialize four deterministic new-seed schedules."""

    root = workspace.resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = _mapping(raw, "configuration")
    if config.get("schema_version") != 1 or config.get("experiment_id") != "E25":
        raise ValueError("E25 configuration identity is invalid")
    if config.get("scientific_state") != "LOCKED_FINAL":
        raise ValueError("E25 must remain LOCKED_FINAL")

    frozen = _mapping(config.get("frozen_sources"), "frozen_sources")
    suite_path = root / Path(str(frozen["rl_suite"]))
    suite = RLSuite.load(suite_path)
    if suite.config_hash != frozen.get("rl_suite_semantic_identity"):
        raise ValueError("E25 source RL-suite semantic identity mismatch")
    for stem in ("cp50_manifest", "cp60_manifest", "decision_policy", "selected_reward"):
        _exact_hash(root, frozen[stem], frozen[f"{stem}_sha256"], stem)
    estimate_sha256 = _exact_hash(
        root,
        frozen["adaptive_estimates"],
        frozen["adaptive_estimates_sha256"],
        "adaptive_estimates",
    )
    _validate_budget(config, suite)

    seed_policy = _mapping(config.get("seed_policy"), "seed_policy")
    seeds = tuple(int(item) for item in cast(Sequence[Any], seed_policy["matched_training_seeds"]))
    new_seeds = tuple(int(item) for item in cast(Sequence[Any], seed_policy["new_training_seeds"]))
    if seeds != _EXPECTED_SEEDS or new_seeds != _EXPECTED_NEW_SEEDS:
        raise ValueError("E25 requires exact matched seeds 1701/1702/1703 with only 1702/1703 new")
    if int(seed_policy["imported_seed"]) != 1701:
        raise ValueError("E25 imported seed must be 1701")

    rows = _read_training_rows(root / suite.dataset_path)
    preflight = suite.preflight("E18", workspace=root)
    if preflight.get("starting_adapter_identity") != frozen.get("starting_adapter_identity"):
        raise ValueError("E25 frozen E10 starting adapter identity mismatch")
    estimates, observed_estimate_sha256 = load_qualified_adaptive_estimates(
        root / suite.adaptive_estimates_ref,
        rows,
        expected_dataset_fingerprint=suite.dataset_fingerprint,
        expected_starting_adapter_identity=str(frozen["starting_adapter_identity"]),
    )
    if observed_estimate_sha256 != estimate_sha256:
        raise ValueError("E25 adaptive estimate loader identity mismatch")

    arms = _mapping(config.get("arms"), "arms")
    if tuple(arms) != _ARM_NAMES:
        raise ValueError("E25 must contain exactly conventional and rcac arms")
    imported: dict[str, Any] = {}
    schedules: dict[str, PreparedUniformSchedule] = {}
    runs: list[dict[str, Any]] = []
    for arm_name in _ARM_NAMES:
        arm = _mapping(arms[arm_name], f"arm {arm_name}")
        expected = {
            "conventional": ("E16", "uniform", "terminal"),
            "rcac": ("E18", "rcac", "terminal"),
        }[arm_name]
        if (
            arm.get("source_experiment_id"),
            arm.get("sampler"),
            arm.get("reward_architecture"),
        ) != expected:
            raise ValueError(f"E25 {arm_name} recipe differs from its frozen finalist")
        complete = _load_complete_run(root, arm)
        imported[arm_name] = {
            "seed": 1701,
            "run_id": complete["run_id"],
            "run_manifest": arm["imported_run_manifest"],
            "run_manifest_sha256": file_sha256(root / Path(str(arm["imported_run_manifest"]))),
            "final_adapter_identity": complete["final_adapter_identity"],
        }
        template = str(arm["run_id_template"])
        if template.count("{seed}") != 1:
            raise ValueError(f"E25 {arm_name} run template must contain one seed placeholder")
        mode = cast(ComparisonMode, str(arm["sampler"]))
        for seed in new_seeds:
            run_id = template.format(seed=seed)
            schedule = _schedule_for_run(
                root=root,
                rows=rows,
                tokenizer=tokenizer,
                suite=suite,
                seed=seed,
                run_id=run_id,
                mode=mode,
                estimates=estimates,
                estimate_sha256=estimate_sha256,
            )
            schedules[run_id] = schedule
            runs.append(
                {
                    "arm": arm_name,
                    "source_experiment_id": arm["source_experiment_id"],
                    "run_id": run_id,
                    "seed": seed,
                    "trainer_seed": seed,
                    "data_seed": seed,
                    "task_selection_seed": seed,
                    "generation_seed": seed,
                    "sampler": mode,
                    "reward_architecture": "terminal",
                    "optimizer_steps": suite.budget.optimizer_steps,
                    "rollout_groups": suite.budget.rollout_groups,
                    "rollouts": suite.budget.rollouts,
                    "schedule_hash": schedule.schedule_hash,
                    "output_dir": f"artifacts/training/E25/{run_id}",
                    "development_eval_run_id": f"{run_id}-dev-eval-v1",
                }
            )

    checkpoint_policy = dict(_mapping(config.get("checkpoint_policy"), "checkpoint_policy"))
    if checkpoint_policy != {
        "save_steps": 16,
        "save_total_limit": 2,
        "required_complete_steps": [16, 32],
        "resumable_boundary": "completed_optimizer_step_only",
        "partial_rollout_policy": "PRESERVE_EXCLUDE_AND_DISCARD",
    }:
        raise ValueError("E25 checkpoint policy identity mismatch")

    conventional_seconds = float(
        _load_complete_run(root, _mapping(arms["conventional"], "conventional"))["train_result"][
            "train_runtime"
        ]
    )
    rcac_seconds = float(
        _load_complete_run(root, _mapping(arms["rcac"], "rcac"))["train_result"]["train_runtime"]
    )
    checkpoint_bytes = max(
        int(
            json.loads(
                (
                    root
                    / Path(str(_mapping(arms[name], name)["imported_run_manifest"])).parent
                    / "checkpoint-32"
                    / "CHECKPOINT_COMPLETE.json"
                ).read_text(encoding="utf-8")
            )["checkpoint_byte_size"]
        )
        for name in _ARM_NAMES
    )
    compute = {
        "new_training_run_count": len(runs),
        "estimated_training_seconds": 2.0 * (conventional_seconds + rcac_seconds),
        "estimated_training_gpu_hours": 2.0 * (conventional_seconds + rcac_seconds) / 3600.0,
        "estimate_basis": "measured seed1701 E16/E18 full-run wall times",
        "checkpoint_bytes_each_upper_basis": checkpoint_bytes,
        "maximum_retained_checkpoints_per_run": 2,
        "estimated_checkpoint_retention_bytes": checkpoint_bytes * 2 * len(runs),
        "disk_safety_reserve_bytes": 20 * 1024**3,
    }
    launch_gate = _mapping(config.get("launch_gate"), "launch_gate")

    protected = dict(_mapping(config.get("protected_evaluation_policy"), "protected policy"))
    if protected.get("current_cp60_authorizes_new_seed_models") is not False:
        raise ValueError("E25 must not claim that CP60 authorizes new-seed model hashes")
    design = {
        "schema_version": 1,
        "status": "FROZEN_CPU_DESIGN",
        "experiment_id": "E25",
        "protocol_id": config["protocol_id"],
        "scientific_state": config["scientific_state"],
        "configuration_path": config_path.resolve().relative_to(root).as_posix(),
        "configuration_sha256": file_sha256(config_path),
        "source_rl_suite_semantic_identity": suite.config_hash,
        "frozen_source_hashes": {
            key: frozen[key]
            for key in (
                "cp50_manifest_sha256",
                "cp60_manifest_sha256",
                "decision_policy_sha256",
                "selected_reward_sha256",
                "adaptive_estimates_sha256",
            )
        },
        "starting_adapter_identity": frozen["starting_adapter_identity"],
        "matched_training_seeds": list(seeds),
        "imported_seed1701": imported,
        "new_runs": runs,
        "fairness_budget": dict(_mapping(config["budget"], "budget")),
        "fairness_budget_hash": suite.budget.fairness_identity(),
        "checkpoint_policy": checkpoint_policy,
        "compute_budget": compute,
        "launch_gate": {
            "state": "CONDITIONAL_FAIL_CLOSED",
            "required_milestone": launch_gate["required_milestone"],
            "required_completed_experiments": ["E21", "E22", "E23", "E24"],
            "requires_fresh_preflight": True,
        },
        "protected_evaluation_policy": protected,
        "implementation_status": {
            "cpu_schedule_preflight": "IMPLEMENTED",
            "cuda_executor": "NOT_YET_WIRED_FOR_E25",
            "development_evaluation": "REUSE_36_ROW_DEVELOPMENT_ADAPTER_WITH_NEW_RUN_IDS",
            "protected_new_seed_generation": "BLOCKED_UNTIL_VERSIONED_E25_PRE_PROTECTED_BINDING",
        },
    }
    design["design_identity"] = identity_hash(design)
    return design, schedules


def preflight_e25_replication(workspace: Path, design: Mapping[str, Any]) -> dict[str, Any]:
    """Check mutable launch conditions without changing the immutable design."""

    root = workspace.resolve()
    frozen_config = root / "configs/replication/e25_finalist_multiseed.yaml"
    frozen_design = root / "evidence/training/E25/EXECUTION_DESIGN.json"
    cp70 = root / "evidence/checkpoints/CP70_PROTECTED_COMPLETE/CHECKPOINT_MANIFEST.json"
    latest_path = root / "evidence/checkpoints/LATEST.json"
    reasons: list[str] = []
    cp70_sha256: str | None = None
    if file_sha256(frozen_config) != _FROZEN_CONFIG_SHA256:
        reasons.append("frozen E25 configuration hash mismatch")
    if file_sha256(frozen_design) != _FROZEN_DESIGN_SHA256:
        reasons.append("frozen E25 execution-design hash mismatch")
    qualification_path = root / _EXECUTOR_QUALIFICATION
    qualification_sha256: str | None = None
    if not qualification_path.is_file():
        reasons.append("E25 executor qualification artifact is missing")
    else:
        qualification = _mapping(
            json.loads(qualification_path.read_text(encoding="utf-8")),
            "executor qualification",
        )
        qualification_sha256 = file_sha256(qualification_path)
        if qualification.get("status") != "QUALIFIED_CPU_NO_CUDA":
            reasons.append("E25 executor is not CPU-qualified")
        inventory = _mapping(qualification.get("code_inventory"), "executor code inventory")
        for relative, expected in inventory.items():
            path = root / Path(str(relative))
            if not path.is_file() or file_sha256(path) != expected:
                reasons.append(f"E25 executor code identity mismatch: {relative}")
    if not cp70.is_file():
        reasons.append("required verified milestone missing: CP70_PROTECTED_COMPLETE")
    else:
        cp70_value = _mapping(json.loads(cp70.read_text(encoding="utf-8")), "CP70 manifest")
        complete_ids = set(cast(Sequence[str], cp70_value.get("completed_experiment_ids", [])))
        required = set(cast(Sequence[str], design["launch_gate"]["required_completed_experiments"]))
        if cp70_value.get("checkpoint_id") != "CP70_PROTECTED_COMPLETE":
            reasons.append("CP70 checkpoint identity mismatch")
        if not required.issubset(complete_ids):
            reasons.append("CP70 does not bind E21-E24 as complete")
        cp70_sha256 = file_sha256(cp70)
        if not latest_path.is_file():
            reasons.append("verified LATEST checkpoint pointer is missing")
        else:
            latest = _mapping(json.loads(latest_path.read_text(encoding="utf-8")), "LATEST")
            if latest.get("checkpoint_id") != "CP70_PROTECTED_COMPLETE":
                reasons.append("LATEST does not point to verified CP70")
            if latest.get("manifest_sha256") != cp70_sha256:
                reasons.append("LATEST does not hash-bind the CP70 manifest")
    disk = shutil.disk_usage(root)
    compute = _mapping(design["compute_budget"], "design compute budget")
    required_free = int(compute["disk_safety_reserve_bytes"]) + int(
        compute["estimated_checkpoint_retention_bytes"]
    )
    if disk.free < required_free:
        reasons.append("free disk cannot preserve the frozen checkpoint retention plus reserve")
    return {
        "experiment_id": "E25",
        "design_identity": design["design_identity"],
        "cpu_preflight_valid": not reasons,
        "cuda_launch_authorized": not reasons,
        "gpu_touched": False,
        "cp70_manifest_sha256": cp70_sha256,
        "executor_qualification_sha256": qualification_sha256,
        "free_disk_bytes": disk.free,
        "required_free_disk_bytes": required_free,
        "blocking_reasons": reasons,
    }


def _published_e25_design(root: Path) -> Mapping[str, Any]:
    path = root / "evidence/training/E25/EXECUTION_DESIGN.json"
    if file_sha256(path) != _FROZEN_DESIGN_SHA256:
        raise ValueError("frozen E25 execution-design hash mismatch")
    design = _mapping(json.loads(path.read_text(encoding="utf-8")), "published design")
    if (
        design.get("design_identity")
        != "e54002d4333418574bfbcadad047fc7090628ffb6f93eee55b6b61221853c83f"
    ):
        raise ValueError("frozen E25 design identity mismatch")
    return design


def plan_e25_replication_run(
    workspace: Path,
    *,
    config_path: Path,
    tokenizer: Any,
    arm: str,
    seed: int,
    resume: bool = False,
) -> tuple[dict[str, Any], PreparedUniformSchedule]:
    """Create one exact E25 run plan and check every mutable launch gate."""

    root = workspace.resolve()
    if arm not in _ARM_NAMES or seed not in _EXPECTED_NEW_SEEDS:
        raise ValueError("E25 run must be conventional/rcac at seed 1702 or 1703")
    if file_sha256(config_path) != _FROZEN_CONFIG_SHA256:
        raise ValueError("frozen E25 configuration hash mismatch")
    published = _published_e25_design(root)
    rebuilt, schedules = build_e25_replication_design(
        root, config_path=config_path, tokenizer=tokenizer
    )
    if rebuilt != published:
        raise ValueError("rebuilt E25 design differs from the immutable published design")
    candidates = [
        item
        for item in cast(Sequence[Mapping[str, Any]], published["new_runs"])
        if item.get("arm") == arm and item.get("seed") == seed
    ]
    if len(candidates) != 1:
        raise ValueError("frozen E25 design does not contain exactly one requested run")
    frozen_run = candidates[0]
    run_id = str(frozen_run["run_id"])
    schedule = schedules[run_id]
    schedule_path = root / f"evidence/training/E25/{run_id}-ROLLOUT_SCHEDULE.json"
    schedule_value = {
        **schedule.manifest(),
        "experiment_id": "E25",
        "protocol_id": published["protocol_id"],
        "run_id": run_id,
        "suite_config_hash": published["source_rl_suite_semantic_identity"],
    }
    if (
        _mapping(json.loads(schedule_path.read_text(encoding="utf-8")), "schedule")
        != schedule_value
    ):
        raise ValueError("E25 frozen rollout schedule artifact mismatch")
    if schedule.schedule_hash != frozen_run["schedule_hash"]:
        raise ValueError("E25 frozen rollout schedule identity mismatch")

    suite = RLSuite.load(root / "configs/rl/e12_e20.yaml")
    source_experiment_id = str(frozen_run["source_experiment_id"])
    source_preflight = suite.preflight(source_experiment_id, workspace=root)
    preflight = preflight_e25_replication(root, published)
    reasons = list(cast(Sequence[str], preflight["blocking_reasons"]))
    if not source_preflight["launch_authorized"]:
        reasons.append(f"frozen {source_experiment_id} source preflight failed")
    output_dir = root / Path(str(frozen_run["output_dir"]))
    if (output_dir / "RUN_COMPLETE.json").is_file():
        reasons.append("requested E25 run is already complete and must not be rerun")
    elif output_dir.is_dir() and any(output_dir.iterdir()) and not resume:
        reasons.append("incomplete E25 output exists; explicit --resume is required")
    elif resume and (not output_dir.is_dir() or not any(output_dir.iterdir())):
        reasons.append("--resume requested but no incomplete E25 output exists")
    imported = _mapping(published["imported_seed1701"], "imported runs")
    source_runtime = _mapping(imported[arm], f"imported {arm}")
    source_complete = _mapping(
        json.loads((root / Path(str(source_runtime["run_manifest"]))).read_text(encoding="utf-8")),
        f"imported {arm} run",
    )
    estimated_step_seconds = float(source_complete["train_result"]["train_runtime"]) / 32.0
    plan = {
        **source_preflight,
        "experiment_id": "E25",
        "protocol_id": published["protocol_id"],
        "scientific_state": "LOCKED_FINAL",
        "execution_mode": "LOCKED_FINAL_REPLICATION",
        "run_id": run_id,
        "arm": arm,
        "source_experiment_id": source_experiment_id,
        "seed": seed,
        "trainer_seed": seed,
        "data_seed": seed,
        "task_selection_seed": seed,
        "generation_seed": seed,
        "optimizer_steps": 32,
        "rollout_groups": 64,
        "rollouts": 512,
        "reward_architecture": "terminal",
        "schedule_hash": schedule.schedule_hash,
        "schedule_artifact": schedule_path.relative_to(root).as_posix(),
        "schedule_artifact_sha256": file_sha256(schedule_path),
        "adaptive_estimates_sha256": published["frozen_source_hashes"]["adaptive_estimates_sha256"],
        "selected_reward_sha256": published["frozen_source_hashes"]["selected_reward_sha256"],
        "selected_reward_identity": _SELECTED_REWARD_IDENTITY,
        "dependency_artifact_sha256": {
            "CP50": published["frozen_source_hashes"]["cp50_manifest_sha256"],
            "CP60": published["frozen_source_hashes"]["cp60_manifest_sha256"],
        },
        "estimated_optimizer_step_seconds": estimated_step_seconds,
        "checkpoint_save_steps": 16,
        "checkpoint_save_total_limit": 2,
        "resume": resume,
        "launch_authorized": not reasons,
        "unresolved_material_decisions": reasons,
        "current_preflight": {**preflight, "blocking_reasons": reasons},
        "gpu_touched": False,
        "protected_access_state": "NOT_ACCESSED",
        "post_training_development_command": (
            ".\\.venv\\Scripts\\python.exe scripts/evaluate_development.py "
            "--dataset data/curated/reasonforge_synthetic_v2/development.parquet "
            f"--output-dir outputs/{frozen_run['development_eval_run_id']} "
            f"--adapter artifacts/training/E25/{run_id}/final_adapter "
            "--batch-size 1 --max-new-tokens 192 --experiment-id E25 "
            f"--run-id {frozen_run['development_eval_run_id']} "
            "--generation-config-id qwen3-dev-greedy-v3-bs1-bounded --bounded-memory "
            "--model-identity <RUN_COMPLETE.final_adapter_identity>"
        ),
    }
    return plan, schedule


def execute_e25_replication_run(
    workspace: Path,
    *,
    config_path: Path,
    arm: str,
    seed: int,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute one frozen E25 run on the qualified single GPU."""

    from transformers import AutoTokenizer

    root = workspace.resolve()
    suite = RLSuite.load(root / "configs/rl/e12_e20.yaml")
    tokenizer = AutoTokenizer.from_pretrained(
        suite.model_id,
        revision=suite.model_revision,
        local_files_only=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    plan, schedule = plan_e25_replication_run(
        root,
        config_path=config_path,
        tokenizer=tokenizer,
        arm=arm,
        seed=seed,
        resume=resume,
    )
    if not plan["launch_authorized"]:
        raise RuntimeError(
            f"E25 launch blocked by preflight: {plan['unresolved_material_decisions']}"
        )
    run_id = str(plan["run_id"])
    replication = {
        "source_experiment_id": plan["source_experiment_id"],
        "output_dir": f"artifacts/training/E25/{run_id}",
        "save_steps": 16,
        "save_total_limit": 2,
        "required_complete_checkpoint_steps": [16, 32],
        "seed": seed,
        "protocol_id": plan["protocol_id"],
        "design_identity": plan["current_preflight"]["design_identity"],
        "design_sha256": _FROZEN_DESIGN_SHA256,
        "config_sha256": _FROZEN_CONFIG_SHA256,
        "schedule_artifact_sha256": plan["schedule_artifact_sha256"],
        "partial_rollout_policy": "PRESERVE_EXCLUDE_AND_DISCARD",
    }
    with GpuOwnerLease(root, "E25", run_id):
        return _execute_grpo_under_lease(
            root,
            suite=suite,
            plan=plan,
            schedule=schedule,
            tokenizer=tokenizer,
            resume=resume,
            replication=replication,
        )


def publish_e25_replication_design(
    workspace: Path,
    *,
    config_path: Path,
    tokenizer: Any,
) -> dict[str, Any]:
    """Atomically publish immutable schedules and the E25 CPU design."""

    root = workspace.resolve()
    design, schedules = build_e25_replication_design(
        root, config_path=config_path, tokenizer=tokenizer
    )
    output = root / "evidence/training/E25"
    output.mkdir(parents=True, exist_ok=True)
    for run_id, schedule in schedules.items():
        path = output / f"{run_id}-ROLLOUT_SCHEDULE.json"
        value = {
            **schedule.manifest(),
            "experiment_id": "E25",
            "protocol_id": design["protocol_id"],
            "run_id": run_id,
            "suite_config_hash": design["source_rl_suite_semantic_identity"],
        }
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"refusing to overwrite mismatched frozen E25 schedule: {path}")
        if not path.exists():
            _atomic_json(path, value)
    design_path = output / "EXECUTION_DESIGN.json"
    if design_path.exists() and json.loads(design_path.read_text(encoding="utf-8")) != design:
        raise RuntimeError(f"refusing to overwrite mismatched frozen E25 design: {design_path}")
    if not design_path.exists():
        _atomic_json(design_path, design)
    return design
