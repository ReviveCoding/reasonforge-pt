"""CPU-only evidence contract for the E17 reward-overoptimization analysis."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from reasonforge.statistics import binary_paired_analysis, continuous_paired_analysis


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class E17Protocol:
    protocol_id: str
    run_id: str
    run_directory: Path
    rollout_trace: Path
    reward_architecture: str
    source_suite_config_hash: str
    source_schedule_hash: str
    starting_adapter_identity: str
    optimizer_steps: int
    groups_per_step: int
    generations_per_group: int
    checkpoint_steps: tuple[int, ...]
    baseline_directory: Path
    baseline_adapter: Path
    checkpoint_directory_pattern: str
    expected_rows: int
    dataset_path: str
    generation_config_id: str
    bootstrap_replicates: int
    analysis_seed: int
    config_sha256: str

    @classmethod
    def from_yaml(cls, path: Path) -> E17Protocol:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping) or value.get("experiment_id") != "E17":
            raise ValueError("E17 protocol must be a mapping with experiment_id E17")
        source = value.get("source_training")
        hidden = value.get("hidden_evaluation")
        analysis = value.get("analysis")
        if not all(isinstance(item, Mapping) for item in (source, hidden, analysis)):
            raise ValueError("E17 protocol is missing source, hidden, or analysis configuration")
        assert isinstance(source, Mapping)
        assert isinstance(hidden, Mapping)
        assert isinstance(analysis, Mapping)
        steps = tuple(int(step) for step in source["checkpoint_steps"])
        protocol = cls(
            protocol_id=str(value["protocol_id"]),
            run_id=str(source["run_id"]),
            run_directory=Path(str(source["run_directory"])),
            rollout_trace=Path(str(source["rollout_trace"])),
            reward_architecture=str(source["reward_architecture"]),
            source_suite_config_hash=str(source["suite_config_hash"]),
            source_schedule_hash=str(source["schedule_hash"]),
            starting_adapter_identity=str(source["starting_adapter_identity"]),
            optimizer_steps=int(source["optimizer_steps"]),
            groups_per_step=int(source["groups_per_optimizer_step"]),
            generations_per_group=int(source["generations_per_group"]),
            checkpoint_steps=steps,
            baseline_directory=Path(str(hidden["baseline_directory"])),
            baseline_adapter=Path(str(hidden["baseline_adapter"])),
            checkpoint_directory_pattern=str(hidden["checkpoint_directory_pattern"]),
            expected_rows=int(hidden["expected_rows"]),
            dataset_path=str(hidden["dataset_path"]),
            generation_config_id=str(hidden["generation_config_id"]),
            bootstrap_replicates=int(analysis["bootstrap_replicates"]),
            analysis_seed=int(analysis["analysis_seed"]),
            config_sha256=_sha256(path),
        )
        protocol.validate()
        return protocol

    def validate(self) -> None:
        if not self.protocol_id or not self.run_id:
            raise ValueError("E17 protocol and run identities are required")
        if self.reward_architecture != "terminal":
            raise ValueError("this E17 revision is scoped only to terminal-reward overoptimization")
        if self.optimizer_steps < 8 or self.groups_per_step < 1 or self.generations_per_group < 2:
            raise ValueError("E17 training budget must be positive")
        expected_steps = tuple(range(4, self.optimizer_steps + 1, 4))
        if self.checkpoint_steps != expected_steps or self.checkpoint_steps[-1] != (
            self.optimizer_steps
        ):
            raise ValueError("E17 requires checkpoints every four steps through the final step")
        if self.expected_rows < 1 or self.bootstrap_replicates < 1:
            raise ValueError("E17 hidden rows and bootstrap replicates must be positive")
        if (
            self.run_directory.is_absolute()
            or self.rollout_trace.is_absolute()
            or self.baseline_adapter.is_absolute()
        ):
            raise ValueError("E17 evidence paths must be workspace-relative")

    def hidden_directory(self, step: int) -> Path:
        if step == 0:
            return self.baseline_directory
        if step not in self.checkpoint_steps:
            raise ValueError(f"step {step} is not an E17 hidden-evaluation checkpoint")
        return Path(self.checkpoint_directory_pattern.format(step=step))


def preflight_e17(protocol: E17Protocol, *, workspace: Path) -> dict[str, Any]:
    """Report missing unprotected evidence without mutating experiment state."""

    root = workspace.resolve()
    run = root / protocol.run_directory
    run_complete = run / "RUN_COMPLETE.json"
    trace = root / protocol.rollout_trace
    checkpoints = {
        step: run / f"checkpoint-{step}" / "CHECKPOINT_COMPLETE.json"
        for step in protocol.checkpoint_steps
    }
    hidden = {
        step: root / protocol.hidden_directory(step) for step in (0, *protocol.checkpoint_steps)
    }
    invalid_run_complete = False
    run_manifest: dict[str, Any] = {}
    if run_complete.is_file():
        try:
            run_manifest = json.loads(run_complete.read_text("utf-8"))
            invalid_run_complete = not (
                run_manifest.get("status") == "COMPLETE"
                and run_manifest.get("experiment_id") == "E16"
                and run_manifest.get("run_id") == protocol.run_id
                and run_manifest.get("protocol_id") == "RFPT-RL-DEVELOPMENT-v2"
                and run_manifest.get("suite_config_hash") == protocol.source_suite_config_hash
                and run_manifest.get("schedule_hash") == protocol.source_schedule_hash
                and run_manifest.get("starting_adapter_identity")
                == protocol.starting_adapter_identity
                and run_manifest.get("optimizer_steps") == protocol.optimizer_steps
                and run_manifest.get("completed_rollout_groups")
                == protocol.optimizer_steps * protocol.groups_per_step
                and run_manifest.get("completed_rollouts")
                == (
                    protocol.optimizer_steps
                    * protocol.groups_per_step
                    * protocol.generations_per_group
                )
            )
        except (OSError, UnicodeError, json.JSONDecodeError):
            invalid_run_complete = True
    missing_checkpoints: list[int] = []
    invalid_checkpoints: list[int] = []
    for step, path in checkpoints.items():
        if not path.is_file():
            missing_checkpoints.append(step)
            continue
        try:
            marker = json.loads(path.read_text("utf-8"))
            if (
                marker.get("status") != "COMPLETE"
                or marker.get("checkpoint_step") != step
                or (
                    run_complete.is_file()
                    and marker.get("run_identity_hash")
                    != run_manifest.get("execution_identity_hash")
                )
            ):
                invalid_checkpoints.append(step)
        except (OSError, UnicodeError, json.JSONDecodeError):
            invalid_checkpoints.append(step)
    trace_invalid_reason: str | None = None
    if run_complete.is_file() and trace.is_file() and not invalid_run_complete:
        try:
            _load_trace(protocol, trace)
            if _sha256(trace) != run_manifest.get("rollout_reward_evidence_sha256"):
                raise ValueError("rollout trace hash does not match RUN_COMPLETE")
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            trace_invalid_reason = str(exc)
    missing_hidden = [
        step
        for step, directory in hidden.items()
        if not (directory / "summary.json").is_file()
        or not (directory / "predictions.jsonl").is_file()
    ]
    invalid_hidden: list[int] = []
    hidden_ids: list[str] | None = None
    for step, directory in hidden.items():
        if step in missing_hidden:
            continue
        try:
            rows, _ = _load_hidden(protocol, directory, step=step)
            ids = sorted(rows)
            if hidden_ids is None:
                hidden_ids = ids
            elif ids != hidden_ids:
                raise ValueError("hidden DEVELOPMENT example IDs are not matched")
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            invalid_hidden.append(step)
    if not run_complete.is_file():
        action = "WAIT_FOR_INSTRUMENTED_E16_TERMINAL_RUN"
    elif (
        invalid_run_complete
        or not trace.is_file()
        or trace_invalid_reason is not None
        or missing_checkpoints
        or invalid_checkpoints
    ):
        action = "E16_EVIDENCE_INCOMPLETE_NEW_TRAINING_REQUIRED"
    elif invalid_hidden:
        action = "INVALID_HIDDEN_EVIDENCE_NEW_EVALUATION_ID_REQUIRED"
    elif missing_hidden:
        action = "GENERATE_MISSING_DEVELOPMENT_HIDDEN_EVALUATIONS"
    else:
        action = "ANALYSIS_READY"
    evaluation_commands = {
        step: [
            r".\.venv\Scripts\python.exe",
            "scripts/evaluate_development.py",
            "--dataset",
            protocol.dataset_path,
            "--output-dir",
            protocol.hidden_directory(step).as_posix(),
            "--model-identity",
            f"Qwen3-1.7B+E16-terminal@step{step:03d}",
            "--adapter",
            (protocol.run_directory / f"checkpoint-{step}").as_posix(),
            "--batch-size",
            "1",
            "--max-new-tokens",
            "192",
            "--experiment-id",
            "E17",
            "--run-id",
            f"E17-terminal-step{step:03d}-dev-eval-v1",
            "--generation-config-id",
            protocol.generation_config_id,
            "--bounded-memory",
        ]
        for step in missing_hidden
        if step != 0
    }
    return {
        "experiment_id": "E17",
        "protocol_id": protocol.protocol_id,
        "reward_architecture": protocol.reward_architecture,
        "action": action,
        "run_complete_present": run_complete.is_file(),
        "run_complete_valid": run_complete.is_file() and not invalid_run_complete,
        "rollout_trace_present": trace.is_file(),
        "rollout_trace_invalid_reason": trace_invalid_reason,
        "missing_checkpoint_steps": missing_checkpoints,
        "invalid_checkpoint_steps": invalid_checkpoints,
        "missing_hidden_steps": missing_hidden,
        "invalid_hidden_steps": invalid_hidden,
        "missing_hidden_evaluation_argv": evaluation_commands,
        "separate_e17_training_required": action == "E16_EVIDENCE_INCOMPLETE_NEW_TRAINING_REQUIRED",
        "protected_data_accessed": False,
        "mutates_state": False,
    }


def _load_trace(protocol: E17Protocol, path: Path) -> tuple[dict[int, float], dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
    expected_groups = protocol.optimizer_steps * protocol.groups_per_step
    if len(records) != expected_groups:
        raise ValueError("rollout trace does not contain the complete E16 group budget")
    by_ordinal: dict[int, list[float]] = {}
    group_ids: set[str] = set()
    for record in records:
        if record.get("event") != "GROUP_REWARD_SCORED" or record.get("run_id") != protocol.run_id:
            raise ValueError("rollout trace event or run identity mismatch")
        ordinal = record.get("group_ordinal")
        rewards = record.get("rewards")
        reward_logs = record.get("reward_logs")
        group_id = record.get("group_id")
        if (
            not isinstance(ordinal, int)
            or not isinstance(rewards, list)
            or len(rewards) != protocol.generations_per_group
            or record.get("completion_count") != protocol.generations_per_group
            or not isinstance(group_id, str)
            or not group_id
            or not isinstance(reward_logs, Mapping)
        ):
            raise ValueError("rollout trace group is malformed")
        numeric = [float(reward) for reward in rewards]
        if not all(math.isfinite(reward) for reward in numeric):
            raise ValueError("rollout rewards must be finite")
        if ordinal in by_ordinal or group_id in group_ids:
            raise ValueError("rollout group ordinal and ID must be unique")
        architectures = reward_logs.get("rf_reward_architecture")
        logged_scalars = reward_logs.get("rf_reward_scalar")
        if architectures != ["terminal"] * protocol.generations_per_group:
            raise ValueError("rollout trace is not canonical terminal-reward evidence")
        if (
            not isinstance(logged_scalars, list)
            or [float(value) for value in logged_scalars] != numeric
        ):
            raise ValueError("rollout rewards do not match canonical scalar reward logs")
        by_ordinal[ordinal] = numeric
        group_ids.add(group_id)
    if sorted(by_ordinal) != list(range(expected_groups)):
        raise ValueError("rollout group ordinals are incomplete or noncontiguous")
    interval_means: dict[int, float] = {}
    previous_step = 0
    for step in protocol.checkpoint_steps:
        start = previous_step * protocol.groups_per_step
        stop = step * protocol.groups_per_step
        interval = [reward for ordinal in range(start, stop) for reward in by_ordinal[ordinal]]
        interval_means[step] = sum(interval) / len(interval)
        previous_step = step
    return interval_means, {
        "path": path.as_posix(),
        "sha256": _sha256(path),
        "group_count": len(records),
        "rollout_count": len(records) * protocol.generations_per_group,
    }


def _load_hidden(
    protocol: E17Protocol, directory: Path, *, step: int
) -> tuple[dict[str, tuple[float, bool]], dict[str, Any]]:
    summary_path = directory / "summary.json"
    predictions_path = directory / "predictions.jsonl"
    summary = json.loads(summary_path.read_text("utf-8"))
    expected_adapter = (
        protocol.baseline_adapter if step == 0 else protocol.run_directory / f"checkpoint-{step}"
    )
    observed_adapter = Path(str(summary.get("adapter", "")))
    if (
        summary.get("status") != "COMPLETE"
        or summary.get("scientific_state") != "DEVELOPMENT"
        or summary.get("dataset_rows") != protocol.expected_rows
        or summary.get("dataset_path") != protocol.dataset_path
        or summary.get("generation_config_id") != protocol.generation_config_id
        or observed_adapter.as_posix().replace("\\", "/")
        != expected_adapter.as_posix().replace("\\", "/")
    ):
        raise ValueError(f"ineligible E17 hidden summary: {summary_path}")
    records: dict[str, tuple[float, bool]] = {}
    for line in predictions_path.read_text("utf-8").splitlines():
        record = json.loads(line)
        example_id = str(record["example_id"])
        if record.get("split_role") != "DEVELOPMENT" or example_id in records:
            raise ValueError(
                f"hidden predictions are not unique DEVELOPMENT rows: {predictions_path}"
            )
        if (
            record.get("run_id") != summary.get("run_id")
            or record.get("dataset_path") != protocol.dataset_path
            or record.get("generation_config_id") != protocol.generation_config_id
        ):
            raise ValueError(f"hidden prediction identity mismatch: {predictions_path}")
        score = float(record["scoring"]["score"])
        if not math.isfinite(score):
            raise ValueError("hidden scores must be finite")
        records[example_id] = (score, bool(record["scoring"]["passed"]))
    if len(records) != protocol.expected_rows:
        raise ValueError("hidden prediction count does not match the protocol")
    return records, {
        "run_id": summary["run_id"],
        "summary": summary_path.as_posix(),
        "summary_sha256": _sha256(summary_path),
        "predictions": predictions_path.as_posix(),
        "predictions_sha256": _sha256(predictions_path),
    }


def _slope(points: Mapping[int, float]) -> float:
    xs = [float(step) for step in sorted(points)]
    ys = [float(points[int(step)]) for step in xs]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)) / denominator


def analyze_e17(protocol: E17Protocol, *, workspace: Path) -> dict[str, Any]:
    """Analyze an already-complete E16 terminal trajectory and hidden evaluations."""

    preflight = preflight_e17(protocol, workspace=workspace)
    if preflight["action"] != "ANALYSIS_READY":
        raise ValueError(f"E17 analysis is not ready: {preflight['action']}")
    root = workspace.resolve()
    train_points, trace_evidence = _load_trace(protocol, root / protocol.rollout_trace)
    checkpoint_evidence = {
        step: {
            "path": (
                root / protocol.run_directory / f"checkpoint-{step}" / "CHECKPOINT_COMPLETE.json"
            ).as_posix(),
            "sha256": _sha256(
                root / protocol.run_directory / f"checkpoint-{step}" / "CHECKPOINT_COMPLETE.json"
            ),
        }
        for step in protocol.checkpoint_steps
    }
    hidden_rows: dict[int, dict[str, tuple[float, bool]]] = {}
    hidden_evidence: dict[int, dict[str, Any]] = {}
    expected_ids: list[str] | None = None
    for step in (0, *protocol.checkpoint_steps):
        rows, evidence = _load_hidden(protocol, root / protocol.hidden_directory(step), step=step)
        ids = sorted(rows)
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError(f"hidden DEVELOPMENT example IDs differ at step {step}")
        hidden_rows[step] = rows
        hidden_evidence[step] = evidence
    assert expected_ids is not None
    hidden_means = {
        step: sum(score for score, _ in rows.values()) / len(rows)
        for step, rows in hidden_rows.items()
    }
    aligned_hidden = {step: hidden_means[step] for step in protocol.checkpoint_steps}
    train_slope = _slope(train_points)
    hidden_slope = _slope(aligned_hidden)
    divergence = train_slope > 0.0 and hidden_slope < 0.0
    baseline_scores = [hidden_rows[0][example_id][0] for example_id in expected_ids]
    final_scores = [
        hidden_rows[protocol.optimizer_steps][example_id][0] for example_id in expected_ids
    ]
    baseline_pass = [hidden_rows[0][example_id][1] for example_id in expected_ids]
    final_pass = [
        hidden_rows[protocol.optimizer_steps][example_id][1] for example_id in expected_ids
    ]
    return {
        "artifact_type": "E17_TERMINAL_REWARD_OVEROPTIMIZATION_ANALYSIS",
        "experiment_id": "E17",
        "protocol_id": protocol.protocol_id,
        "protocol_config_sha256": protocol.config_sha256,
        "scientific_state": "QUALIFICATION",
        "status": "COMPLETE",
        "source_experiment_id": "E16",
        "source_run_id": protocol.run_id,
        "reward_architecture": "terminal",
        "train_interval_mean_by_step": train_points,
        "hidden_mean_by_step": hidden_means,
        "train_slope_per_step": train_slope,
        "hidden_slope_per_step": hidden_slope,
        "classification": (
            "TERMINAL_PROXY_DIVERGENCE_OBSERVED"
            if divergence
            else "NO_TERMINAL_PROXY_DIVERGENCE_OBSERVED"
        ),
        "baseline_to_final_hidden_continuous": continuous_paired_analysis(
            baseline_scores,
            final_scores,
            cluster_ids=expected_ids,
            bootstrap_replicates=protocol.bootstrap_replicates,
            seed=protocol.analysis_seed,
        ).to_dict(),
        "baseline_to_final_hidden_binary": binary_paired_analysis(
            baseline_pass,
            final_pass,
            unit_ids=expected_ids,
            bootstrap_replicates=protocol.bootstrap_replicates,
            seed=protocol.analysis_seed,
        ).to_dict(),
        "trace_evidence": trace_evidence,
        "checkpoint_evidence": checkpoint_evidence,
        "hidden_evidence": hidden_evidence,
        "analysis_units": {
            "train": "rollouts nested in task groups; interval means are descriptive",
            "hidden": f"{len(expected_ids)} unique matched DEVELOPMENT example_ids",
        },
        "qualifier": (
            "Single-seed terminal-reward qualification. A sign divergence is proxy "
            "overoptimization evidence, not proof of reward hacking, and does not test "
            "structured-reward or protected performance."
        ),
        "protected_data_accessed": False,
    }
