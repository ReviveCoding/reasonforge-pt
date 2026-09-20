"""Fail-closed, protected-free systems evidence extraction for E26."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

from reasonforge.checkpoints.hashing import atomic_write_bytes, atomic_write_json
from reasonforge.config import file_sha256, identity_hash

UNAVAILABLE: Final = "UNAVAILABLE_NOT_MEASURED"
MEASURED: Final = "MEASURED"
_ALLOWED_EXPERIMENTS: Final = {
    *(f"E{number:02d}" for number in range(8, 21)),
    "E25",
    "E27",
    "E28",
}
_FORBIDDEN_PATH_PARTS: Final = {"protected_outputs", "locked_final", "external_protected"}


class SystemsEvidenceError(ValueError):
    """Raised when E26 source evidence is absent, unsafe, or internally inconsistent."""


@dataclass(frozen=True)
class SourceBinding:
    """A source file bound into the generated E26 manifest."""

    path: str
    sha256: str
    bytes: int

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemsEvidenceError(f"cannot read JSON evidence {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemsEvidenceError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise SystemsEvidenceError(f"non-object JSONL row {path}:{line_number}")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise SystemsEvidenceError(f"cannot read JSONL evidence {path}: {error}") from error
    if not rows:
        raise SystemsEvidenceError(f"empty JSONL evidence: {path}")
    return rows


def _safe_source(root: Path, relative: str) -> Path:
    normalized_relative = relative.replace("\\", "/").lower()
    if any(part in normalized_relative for part in _FORBIDDEN_PATH_PARTS):
        raise SystemsEvidenceError(f"protected path is prohibited in E26: {relative}")
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise SystemsEvidenceError(f"source escapes repository: {relative}") from error
    lowered = {part.lower() for part in candidate.parts}
    forbidden = sorted(lowered & _FORBIDDEN_PATH_PARTS)
    if forbidden:
        raise SystemsEvidenceError(f"protected path is prohibited in E26: {relative}")
    if not candidate.is_file():
        raise SystemsEvidenceError(f"missing E26 source: {relative}")
    return candidate


def _binding(root: Path, relative: str) -> SourceBinding:
    path = _safe_source(root, relative)
    return SourceBinding(relative, file_sha256(path), path.stat().st_size)


def _number(value: object, *, field: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemsEvidenceError(f"{field} must be numeric in {path}")
    result = float(value)
    if result < 0:
        raise SystemsEvidenceError(f"{field} must be nonnegative in {path}")
    return result


def _integer(value: object, *, field: str, path: Path) -> int:
    number = _number(value, field=field, path=path)
    result = int(number)
    if float(result) != number:
        raise SystemsEvidenceError(f"{field} must be an integer in {path}")
    return result


def _optional_metric(value: float | int | None) -> tuple[float | int | None, str]:
    return (value, MEASURED) if value is not None else (None, UNAVAILABLE)


def _max_nested_metric(rows: Sequence[Mapping[str, Any]], metric_name: str) -> float | None:
    values: list[float] = []
    for row in rows:
        metrics = row.get("metrics")
        if isinstance(metrics, dict):
            value = metrics.get(metric_name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
    return max(values) if values else None


def _last_nested_metric(rows: Sequence[Mapping[str, Any]], metric_name: str) -> float | None:
    for row in reversed(rows):
        metrics = row.get("metrics")
        if isinstance(metrics, dict):
            value = metrics.get(metric_name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


def _same_step_save_intervals(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    """Return observed callback intervals; these are not claimed as isolated I/O latency."""
    intervals: list[float] = []
    for index, row in enumerate(rows):
        if row.get("event") != "save" or index == 0:
            continue
        prior = rows[index - 1]
        if prior.get("event") != "log" or prior.get("global_step") != row.get("global_step"):
            continue
        before = prior.get("wall_seconds")
        after = row.get("wall_seconds")
        if (
            isinstance(before, (int, float))
            and not isinstance(before, bool)
            and isinstance(after, (int, float))
            and not isinstance(after, bool)
            and float(after) >= float(before)
        ):
            intervals.append(float(after) - float(before))
    return intervals


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _extract_training_run(
    root: Path, entry: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[SourceBinding]]:
    experiment_id = str(entry.get("experiment_id", ""))
    run_id = str(entry.get("run_id", ""))
    if experiment_id not in _ALLOWED_EXPERIMENTS or not run_id.startswith(f"{experiment_id}-"):
        raise SystemsEvidenceError(f"invalid training source identity: {experiment_id}/{run_id}")
    completion_relative = str(entry.get("completion", ""))
    telemetry_relative = entry.get("telemetry")
    completion_path = _safe_source(root, completion_relative)
    completion = _load_json(completion_path)
    if completion.get("status") != "COMPLETE":
        raise SystemsEvidenceError(f"run is not COMPLETE: {completion_relative}")
    if completion.get("experiment_id") != experiment_id or completion.get("run_id") != run_id:
        raise SystemsEvidenceError(f"run identity mismatch: {completion_relative}")

    bindings = [_binding(root, completion_relative)]
    telemetry: list[dict[str, Any]] = []
    if telemetry_relative is not None:
        telemetry_path = _safe_source(root, str(telemetry_relative))
        telemetry = _load_jsonl(telemetry_path)
        if telemetry[0].get("event") != "train_begin" or telemetry[-1].get("event") != "train_end":
            raise SystemsEvidenceError(f"telemetry is not a complete train trace: {telemetry_path}")
        begin_metrics = telemetry[0].get("metrics")
        end_metrics = telemetry[-1].get("metrics")
        if (
            not isinstance(begin_metrics, dict)
            or begin_metrics.get("cuda_available") is not True
            or not isinstance(end_metrics, dict)
            or end_metrics.get("cuda_available") is not True
        ):
            raise SystemsEvidenceError(f"training telemetry does not verify CUDA: {telemetry_path}")
        bindings.append(_binding(root, str(telemetry_relative)))

    train_result = completion.get("train_result")
    if train_result is not None and not isinstance(train_result, dict):
        raise SystemsEvidenceError(f"train_result must be an object: {completion_relative}")
    train_result = train_result if isinstance(train_result, dict) else {}
    runtime: float | None = None
    if "train_runtime" in train_result:
        runtime = _number(
            train_result["train_runtime"], field="train_runtime", path=completion_path
        )
    telemetry_runtime: float | None = None
    if telemetry:
        telemetry_runtime = _number(
            telemetry[-1].get("wall_seconds"), field="wall_seconds", path=telemetry_path
        )
    if runtime is not None and telemetry_runtime is not None:
        tolerance = max(0.1, runtime * 0.01)
        if abs(runtime - telemetry_runtime) > tolerance:
            raise SystemsEvidenceError(
                f"runtime disagreement exceeds 1%/0.1s: {completion_relative}"
            )
    wall_seconds = runtime if runtime is not None else telemetry_runtime
    tokens = _last_nested_metric(telemetry, "num_tokens") if telemetry else None
    tokens_per_second = (
        tokens / wall_seconds if tokens is not None and wall_seconds and wall_seconds > 0 else None
    )
    step_count: int | None = None
    for key in ("optimizer_steps", "final_global_step"):
        if key in completion:
            step_count = _integer(completion[key], field=key, path=completion_path)
            break
    if step_count is None and telemetry:
        step_count = _integer(
            telemetry[-1].get("global_step"), field="global_step", path=telemetry_path
        )
    save_intervals = _same_step_save_intervals(telemetry)
    wall_value, wall_status = _optional_metric(wall_seconds)
    token_value, token_status = _optional_metric(tokens)
    token_rate, token_rate_status = _optional_metric(tokens_per_second)
    peak_allocated, peak_allocated_status = _optional_metric(
        _max_nested_metric(telemetry, "peak_allocated_bytes")
    )
    peak_reserved, peak_reserved_status = _optional_metric(
        _max_nested_metric(telemetry, "peak_reserved_bytes")
    )
    sample_rate: float | None = None
    if "train_samples_per_second" in train_result:
        sample_rate = _number(
            train_result["train_samples_per_second"],
            field="train_samples_per_second",
            path=completion_path,
        )
    step_rate: float | None = None
    if "train_steps_per_second" in train_result:
        step_rate = _number(
            train_result["train_steps_per_second"],
            field="train_steps_per_second",
            path=completion_path,
        )
    sample_value, sample_status = _optional_metric(sample_rate)
    step_rate_value, step_rate_status = _optional_metric(step_rate)
    callback_median, callback_status = _optional_metric(_median(save_intervals))
    row: dict[str, Any] = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "workload_kind": str(entry.get("workload_kind", "training")),
        "execution_scope": str(entry.get("execution_scope", "FULL")),
        "status": "COMPLETE",
        "wall_seconds": wall_value,
        "wall_seconds_status": wall_status,
        "gpu_time_seconds": None,
        "gpu_time_seconds_status": UNAVAILABLE,
        "tokens": token_value,
        "tokens_status": token_status,
        "tokens_per_wall_second": token_rate,
        "tokens_per_wall_second_status": token_rate_status,
        "train_samples_per_second": sample_value,
        "train_samples_per_second_status": sample_status,
        "train_steps_per_second": step_rate_value,
        "train_steps_per_second_status": step_rate_status,
        "optimizer_steps": step_count,
        "peak_allocated_bytes": peak_allocated,
        "peak_allocated_bytes_status": peak_allocated_status,
        "peak_reserved_bytes": peak_reserved,
        "peak_reserved_bytes_status": peak_reserved_status,
        "gpu_utilization_status": UNAVAILABLE,
        "power_status": UNAVAILABLE,
        "temperature_status": UNAVAILABLE,
        "save_event_count": len(save_intervals),
        "same_step_log_to_save_median_seconds": callback_median,
        "same_step_log_to_save_median_seconds_status": callback_status,
        "completion_path": completion_relative,
        "telemetry_path": str(telemetry_relative) if telemetry_relative is not None else None,
    }
    checkpoints, checkpoint_bindings = _extract_checkpoints(
        root, completion_path.parent, experiment_id, run_id
    )
    bindings.extend(checkpoint_bindings)
    return row, checkpoints, bindings


def _extract_checkpoints(
    root: Path, run_directory: Path, experiment_id: str, run_id: str
) -> tuple[list[dict[str, Any]], list[SourceBinding]]:
    rows: list[dict[str, Any]] = []
    bindings: list[SourceBinding] = []
    identity_path = run_directory / "RUN_IDENTITY.json"
    if not identity_path.is_file():
        raise SystemsEvidenceError(f"run lacks identity manifest: {run_directory}")
    identity_payload = _load_json(identity_path)
    identity_run_id = identity_payload.get("run_id")
    identity_experiment_id = identity_payload.get("experiment_id")
    if identity_run_id is not None and identity_run_id != run_id:
        raise SystemsEvidenceError(f"run identity mismatch: {identity_path}")
    if identity_experiment_id is not None and identity_experiment_id != experiment_id:
        raise SystemsEvidenceError(f"experiment identity mismatch: {identity_path}")
    identity_digest = identity_hash(identity_payload)
    bindings.append(_binding(root, identity_path.relative_to(root).as_posix()))
    for directory in sorted(run_directory.glob("checkpoint-*"), key=lambda path: path.name):
        marker = directory / "CHECKPOINT_COMPLETE.json"
        if not marker.is_file():
            raise SystemsEvidenceError(f"checkpoint lacks completion marker: {directory}")
        relative = marker.relative_to(root).as_posix()
        payload = _load_json(marker)
        if (
            payload.get("status") != "COMPLETE"
            or payload.get("run_identity_hash") != identity_digest
        ):
            raise SystemsEvidenceError(f"checkpoint completion identity mismatch: {relative}")
        step = _integer(payload.get("checkpoint_step"), field="checkpoint_step", path=marker)
        size = _integer(
            payload.get("checkpoint_byte_size"), field="checkpoint_byte_size", path=marker
        )
        actual_size = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
        marker_size = marker.stat().st_size
        # checkpoint_byte_size is computed before publishing its own completion marker.
        if actual_size - marker_size != size:
            raise SystemsEvidenceError(f"checkpoint byte-size mismatch: {relative}")
        rows.append(
            {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "checkpoint_step": step,
                "checkpoint_byte_size": size,
                "completion_marker_bytes": marker_size,
                "total_directory_bytes": actual_size,
                "checkpoint_save_seconds": None,
                "checkpoint_save_seconds_status": UNAVAILABLE,
                "completion_marker": relative,
                "completion_timestamp": payload.get("completion_timestamp"),
            }
        )
        bindings.append(_binding(root, relative))
    return rows, bindings


def _extract_evaluation_memory(
    root: Path, entry: Mapping[str, Any]
) -> tuple[dict[str, Any], SourceBinding]:
    experiment_id = str(entry.get("experiment_id", ""))
    run_id = str(entry.get("run_id", ""))
    if experiment_id not in _ALLOWED_EXPERIMENTS or not run_id.startswith(f"{experiment_id}-"):
        raise SystemsEvidenceError(f"invalid evaluation source identity: {experiment_id}/{run_id}")
    relative = str(entry.get("memory_telemetry", ""))
    path = _safe_source(root, relative)
    rows = _load_jsonl(path)
    completed_rows: list[int] = []
    allocated: list[float] = []
    reserved: list[float] = []
    bounded: list[bool] = []
    for row in rows:
        completed_rows.append(
            _integer(row.get("completed_rows"), field="completed_rows", path=path)
        )
        bounded_value = row.get("bounded_memory")
        if not isinstance(bounded_value, bool):
            raise SystemsEvidenceError(f"bounded_memory must be boolean in {relative}")
        bounded.append(bounded_value)
        for snapshot in ("before_generate", "after_generate", "after_cleanup"):
            value = row.get(snapshot)
            if not isinstance(value, dict):
                raise SystemsEvidenceError(f"missing {snapshot} in {relative}")
            allocated.append(
                _number(value.get("max_allocated_bytes"), field="max_allocated_bytes", path=path)
            )
            reserved.append(
                _number(value.get("max_reserved_bytes"), field="max_reserved_bytes", path=path)
            )
    if completed_rows != sorted(completed_rows) or len(set(completed_rows)) != len(completed_rows):
        raise SystemsEvidenceError(
            f"evaluation completion counter is not strictly increasing: {relative}"
        )
    expected_rows = _integer(entry.get("expected_rows"), field="expected_rows", path=path)
    if completed_rows[-1] != expected_rows:
        raise SystemsEvidenceError(
            f"evaluation telemetry is partial: {relative} ({completed_rows[-1]}/{expected_rows})"
        )
    if not all(bounded):
        raise SystemsEvidenceError(f"evaluation telemetry reports unbounded memory: {relative}")
    row_out: dict[str, Any] = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "workload_kind": "development_evaluation",
        "execution_scope": str(entry.get("execution_scope", "DEVELOPMENT")),
        "status": "COMPLETE_TELEMETRY",
        "wall_seconds": None,
        "wall_seconds_status": UNAVAILABLE,
        "gpu_time_seconds": None,
        "gpu_time_seconds_status": UNAVAILABLE,
        "tokens": None,
        "tokens_status": UNAVAILABLE,
        "tokens_per_wall_second": None,
        "tokens_per_wall_second_status": UNAVAILABLE,
        "train_samples_per_second": None,
        "train_samples_per_second_status": UNAVAILABLE,
        "train_steps_per_second": None,
        "train_steps_per_second_status": UNAVAILABLE,
        "optimizer_steps": None,
        "peak_allocated_bytes": max(allocated),
        "peak_allocated_bytes_status": MEASURED,
        "peak_reserved_bytes": max(reserved),
        "peak_reserved_bytes_status": MEASURED,
        "gpu_utilization_status": UNAVAILABLE,
        "power_status": UNAVAILABLE,
        "temperature_status": UNAVAILABLE,
        "save_event_count": 0,
        "same_step_log_to_save_median_seconds": None,
        "same_step_log_to_save_median_seconds_status": UNAVAILABLE,
        "completed_evaluation_rows": completed_rows[-1],
        "bounded_memory_all_batches": all(bounded),
        "completion_path": None,
        "telemetry_path": relative,
    }
    return row_out, _binding(root, relative)


def _extract_quality_efficiency(
    root: Path,
    entry: Mapping[str, Any],
    training_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[SourceBinding]]:
    """Derive DEVELOPMENT score gain per exclusive-GPU wall hour.

    This deliberately does not label wall time as device-active GPU time. The
    metric is eligible only for one-GPU, one-heavy-process runs whose candidate
    and baseline use the same deterministic DEVELOPMENT fixture.
    """
    experiment_id = str(entry.get("experiment_id", ""))
    run_id = str(entry.get("run_id", ""))
    if experiment_id not in _ALLOWED_EXPERIMENTS or not run_id.startswith(
        f"{experiment_id}-"
    ):
        raise SystemsEvidenceError(
            f"invalid quality-efficiency source identity: {experiment_id}/{run_id}"
        )
    training = training_rows.get(run_id)
    if training is None or training.get("wall_seconds_status") != MEASURED:
        raise SystemsEvidenceError(f"quality-efficiency run lacks measured wall time: {run_id}")
    if entry.get("exclusive_single_gpu_process") is not True:
        raise SystemsEvidenceError(
            f"quality-efficiency run lacks exclusive single-GPU attestation: {run_id}"
        )

    baseline_relative = str(entry.get("baseline_summary", ""))
    candidate_relative = str(entry.get("candidate_summary", ""))
    baseline_path = _safe_source(root, baseline_relative)
    candidate_path = _safe_source(root, candidate_relative)
    baseline = _load_json(baseline_path)
    candidate = _load_json(candidate_path)
    expected_evaluation_run_id = str(entry.get("evaluation_run_id", ""))
    if (
        baseline.get("status") != "COMPLETE"
        or baseline.get("scientific_state") != "DEVELOPMENT"
        or candidate.get("status") != "COMPLETE"
        or candidate.get("scientific_state") != "DEVELOPMENT"
    ):
        raise SystemsEvidenceError(
            f"quality-efficiency summaries are not COMPLETE DEVELOPMENT: {run_id}"
        )
    if (
        candidate.get("experiment_id") != experiment_id
        or candidate.get("run_id") != expected_evaluation_run_id
    ):
        raise SystemsEvidenceError(f"quality-efficiency evaluation identity mismatch: {run_id}")
    comparison_fields = ("dataset_path", "dataset_rows", "generation_config_id")
    for field in comparison_fields:
        if candidate.get(field) != baseline.get(field):
            raise SystemsEvidenceError(
                f"quality-efficiency DEVELOPMENT comparison differs on {field}: {run_id}"
            )
    baseline_score = _number(
        baseline.get("mean_score"), field="baseline.mean_score", path=baseline_path
    )
    candidate_score = _number(
        candidate.get("mean_score"), field="candidate.mean_score", path=candidate_path
    )
    if baseline_score > 1 or candidate_score > 1:
        raise SystemsEvidenceError(f"quality-efficiency score outside [0, 1]: {run_id}")
    wall_seconds = float(training["wall_seconds"])
    if wall_seconds <= 0:
        raise SystemsEvidenceError(f"quality-efficiency wall time must be positive: {run_id}")
    gain = candidate_score - baseline_score
    wall_hours = wall_seconds / 3600.0
    row = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "evaluation_run_id": expected_evaluation_run_id,
        "baseline_evaluation_run_id": baseline.get("run_id"),
        "metric_name": "development_mean_score",
        "candidate_score": candidate_score,
        "baseline_score": baseline_score,
        "absolute_gain": gain,
        "training_wall_seconds": wall_seconds,
        "training_wall_hours": wall_hours,
        "development_score_gain_per_exclusive_gpu_wall_hour": gain / wall_hours,
        "exclusive_single_gpu_process": True,
        "gpu_active_time_measured": False,
        "status": MEASURED,
        "candidate_summary_path": candidate_relative,
        "baseline_summary_path": baseline_relative,
    }
    return row, [_binding(root, baseline_relative), _binding(root, candidate_relative)]


def _load_config(root: Path, config_path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise SystemsEvidenceError(f"cannot load E26 config: {error}") from error
    if not isinstance(payload, dict):
        raise SystemsEvidenceError("E26 config must be an object")
    if (
        payload.get("experiment_id") != "E26"
        or payload.get("protected_data_policy") != "PROHIBITED"
    ):
        raise SystemsEvidenceError("E26 config identity or protected-data policy is invalid")
    config_resolved = config_path.resolve()
    try:
        config_resolved.relative_to(root.resolve())
    except ValueError as error:
        raise SystemsEvidenceError("E26 config must be inside the repository") from error
    return payload


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise SystemsEvidenceError("refusing to publish an empty canonical table")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _recovery_events(
    root: Path, entries: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[SourceBinding]]:
    rows: list[dict[str, Any]] = []
    bindings: list[SourceBinding] = []
    for entry in entries:
        relative = str(entry.get("path", ""))
        path = _safe_source(root, relative)
        payload = _load_json(path)
        experiment_id = str(entry.get("experiment_id", ""))
        if experiment_id not in _ALLOWED_EXPERIMENTS:
            raise SystemsEvidenceError(f"recovery event outside E26 scope: {experiment_id}")
        status_field = str(entry.get("status_field", "status"))
        observed_status = payload.get(status_field)
        expected_status = entry.get("expected_status")
        if observed_status != expected_status:
            raise SystemsEvidenceError(
                f"recovery status mismatch in {relative}: "
                f"{observed_status!r} != {expected_status!r}"
            )
        rows.append(
            {
                "experiment_id": experiment_id,
                "event_id": str(entry.get("event_id", "")),
                "event_type": str(entry.get("event_type", "")),
                "status": observed_status,
                "run_id": payload.get("run_id")
                or payload.get("recovery_run_id")
                or payload.get("experiment_id"),
                "new_process_verified": payload.get("new_process_verified"),
                "excluded_from_gradient_accounting": payload.get(
                    "excluded_from_gradient_accounting"
                )
                or (
                    payload.get("discard_event", {}).get("excluded_from_gradient_accounting")
                    if isinstance(payload.get("discard_event"), dict)
                    else None
                ),
                "completed_rollout_groups": (
                    payload.get("accounting", {}).get("recovered_completed_groups")
                    if isinstance(payload.get("accounting"), dict)
                    else None
                ),
                "completed_rollouts": (
                    payload.get("accounting", {}).get("recovered_completed_rollouts")
                    if isinstance(payload.get("accounting"), dict)
                    else None
                ),
                "partial_completed_units": payload.get("completed_predictions")
                or payload.get("completed_rows"),
                "expected_units": payload.get("expected_predictions")
                or payload.get("expected_rows"),
                "preserved_failed_history_entries": (
                    len(payload.get("failed_and_invalid_history", {}))
                    if isinstance(payload.get("failed_and_invalid_history"), dict)
                    else None
                ),
                "evidence_path": relative,
            }
        )
        bindings.append(_binding(root, relative))
    return rows, bindings


def _render_report(
    manifest: Mapping[str, Any],
    run_rows: Sequence[Mapping[str, Any]],
    checkpoint_rows: Sequence[Mapping[str, Any]],
    quality_rows: Sequence[Mapping[str, Any]],
) -> bytes:
    training = [row for row in run_rows if row["workload_kind"] != "development_evaluation"]
    measured_runtime = [row for row in training if row["wall_seconds_status"] == MEASURED]
    measured_memory = [row for row in run_rows if row["peak_allocated_bytes_status"] == MEASURED]
    total_wall = sum(float(row["wall_seconds"]) for row in measured_runtime)
    total_checkpoint_bytes = sum(int(row["checkpoint_byte_size"]) for row in checkpoint_rows)
    checkpoints_by_run: dict[str, int] = {}
    for checkpoint in checkpoint_rows:
        run_id = str(checkpoint["run_id"])
        checkpoints_by_run[run_id] = checkpoints_by_run.get(run_id, 0) + 1
    lines = [
        "# E26 GPU and Systems Analysis",
        "",
        "Status: deterministic infrastructure output from non-protected runtime evidence.",
        "",
        "## Qualified hardware",
        "",
        f"- GPU: {manifest['hardware']['gpu_name']}",
        f"- CUDA devices: {manifest['hardware']['cuda_device_count']}",
        f"- Torch / CUDA: {manifest['hardware']['torch_version']} / "
        f"{manifest['hardware']['torch_cuda_version']}",
        f"- BF16 supported: {manifest['hardware']['bf16_supported']}",
        "",
        "## Evidence boundary",
        "",
        "This analysis reads only explicitly allowlisted E08-E20, E25, and E27/E28 "
        "operational artifacts plus aggregate DEVELOPMENT summaries used for the E25 efficiency "
        "comparison. It does not read model predictions, "
        "LOCKED_FINAL outputs, or "
        "protected-output shards. Metrics absent from source telemetry remain unavailable.",
        "",
        "## Measured coverage",
        "",
        f"- Training/qualification rows: {len(training)}",
        f"- Rows with measured wall time: {len(measured_runtime)}",
        f"- Rows with measured CUDA peak allocation: {len(measured_memory)}",
        f"- Sum of measured per-run wall time (not GPU-active time): {total_wall:.3f} seconds",
        f"- Verified checkpoint completion markers: {len(checkpoint_rows)}",
        f"- Sum of measured checkpoint payload bytes: {total_checkpoint_bytes}",
        "",
        "## Availability and interpretation",
        "",
        "- `wall_seconds` is Trainer/runtime or complete telemetry elapsed wall time.",
        "- `tokens_per_wall_second` is derived only when cumulative `num_tokens` and wall time "
        "both exist.",
        "- CUDA peak allocated/reserved bytes are process-local PyTorch counters, not total "
        "board usage.",
        "- `same_step_log_to_save_median_seconds` is an observed callback interval. It is not "
        "isolated "
        "checkpoint-storage latency and must not be described as such.",
        "- Device-active GPU time, utilization, temperature, and power are unavailable because no "
        "time-series source recorded them. Wall time is not substituted for GPU time.",
        "- Checkpoint payload footprint is measured; isolated checkpoint save time is unavailable.",
        "- True quality-per-device-active-GPU-hour is unavailable because device-active time was "
        "not measured. For E25, the table below reports DEVELOPMENT mean-score gain per "
        "exclusive-GPU wall hour; it is not a protected endpoint and does not substitute wall time "
        "for measured GPU-active time.",
        "",
        "## Workload comparability",
        "",
        "Full training, development evaluation, and checkpoint-recovery qualification are labeled "
        "separately. Recovery arms may contain only the post-resume process duration and are not "
        "treated "
        "as end-to-end workload timing unless their source records the complete interval.",
        "Token counters also have objective-specific semantics, so token rates are valid within "
        "the "
        "recorded workload but are not normalized cross-objective hardware benchmarks.",
        "",
        "## Measured training and qualification runs",
        "",
        "| Experiment | Run | Scope | Wall s | Tokens/s (wall) | Peak alloc GiB | "
        "Peak reserved GiB | Checkpoints |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in training:
        wall = f"{float(row['wall_seconds']):.3f}" if row["wall_seconds"] is not None else "NA"
        token_rate = (
            f"{float(row['tokens_per_wall_second']):.3f}"
            if row["tokens_per_wall_second"] is not None
            else "NA"
        )
        allocated = (
            f"{float(row['peak_allocated_bytes']) / (1024**3):.3f}"
            if row["peak_allocated_bytes"] is not None
            else "NA"
        )
        reserved = (
            f"{float(row['peak_reserved_bytes']) / (1024**3):.3f}"
            if row["peak_reserved_bytes"] is not None
            else "NA"
        )
        lines.append(
            f"| {row['experiment_id']} | `{row['run_id']}` | {row['execution_scope']} | "
            f"{wall} | {token_rate} | {allocated} | {reserved} | "
            f"{checkpoints_by_run.get(str(row['run_id']), 0)} |"
        )
    if quality_rows:
        lines.extend(
            [
                "",
                "## E25 DEVELOPMENT quality versus exclusive-GPU wall time",
                "",
                "The baseline is the E10 starting checkpoint on the same 36-row deterministic "
                "DEVELOPMENT fixture and generation configuration. Negative values are retained.",
                "",
                "| Run | Candidate score | E10 score | Gain | Training wall h | "
                "Gain / wall-GPU h |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in quality_rows:
            lines.append(
                f"| `{row['run_id']}` | {float(row['candidate_score']):.6f} | "
                f"{float(row['baseline_score']):.6f} | {float(row['absolute_gain']):.6f} | "
                f"{float(row['training_wall_hours']):.6f} | "
                f"{float(row['development_score_gain_per_exclusive_gpu_wall_hour']):.6f} |"
            )
    lines.extend(
        [
            "",
            "## Canonical artifacts",
            "",
            "- `reports/tables/e26_run_metrics.csv` and `.json`",
            "- `reports/tables/e26_checkpoint_metrics.csv` and `.json`",
            "- `reports/tables/e26_recovery_events.csv` and `.json`",
            "- `reports/tables/e26_experiment_coverage.csv` and `.json`",
            "- `reports/tables/e26_quality_efficiency.csv` and `.json`",
            "- `evidence/systems/E26/SYSTEMS_ANALYSIS.json`",
            "",
            f"Analysis identity: `{manifest['analysis_identity']}`",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def build_e26_systems_analysis(
    root: Path,
    *,
    config_path: Path,
    output_directory: Path | None = None,
    table_directory: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Validate allowlisted runtime evidence and atomically publish canonical E26 artifacts."""
    root = root.resolve()
    config_path = config_path.resolve()
    config = _load_config(root, config_path)
    output_directory = (output_directory or root / "evidence/systems/E26").resolve()
    table_directory = (table_directory or root / "reports/tables").resolve()
    report_path = (report_path or root / "reports/GPU_SYSTEMS_ANALYSIS.md").resolve()
    for output in (output_directory, table_directory, report_path.parent):
        try:
            output.relative_to(root)
        except ValueError as error:
            raise SystemsEvidenceError(f"output escapes repository: {output}") from error

    run_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    environment_relative = str(config["environment_snapshot"])
    environment_path = _safe_source(root, environment_relative)
    environment = _load_json(environment_path)
    cuda = environment.get("cuda")
    packages = environment.get("packages")
    if not isinstance(cuda, dict) or cuda.get("available") is not True:
        raise SystemsEvidenceError("environment snapshot does not qualify CUDA")
    if not isinstance(packages, dict):
        raise SystemsEvidenceError("environment snapshot lacks package identities")
    device_count = _integer(
        cuda.get("device_count"), field="cuda.device_count", path=environment_path
    )
    if device_count < 1:
        raise SystemsEvidenceError("environment snapshot has no CUDA devices")
    hardware = {
        "gpu_name": str(cuda.get("name", "UNKNOWN")),
        "cuda_device_count": device_count,
        "bf16_supported": bool(cuda.get("bf16_supported", False)),
        "torch_cuda_version": str(cuda.get("torch_cuda", "UNKNOWN")),
        "torch_version": str(packages.get("torch", "UNKNOWN")),
        "platform": str(environment.get("platform", "UNKNOWN")),
    }
    source_bindings: list[SourceBinding] = [
        _binding(root, config_path.relative_to(root).as_posix()),
        _binding(root, environment_relative),
    ]
    for raw in config.get("training_runs", []):
        if not isinstance(raw, dict):
            raise SystemsEvidenceError("training_runs entries must be objects")
        row, checkpoints, bindings = _extract_training_run(root, raw)
        run_rows.append(row)
        checkpoint_rows.extend(checkpoints)
        source_bindings.extend(bindings)
    for raw in config.get("evaluation_memory_runs", []):
        if not isinstance(raw, dict):
            raise SystemsEvidenceError("evaluation_memory_runs entries must be objects")
        row, binding = _extract_evaluation_memory(root, raw)
        run_rows.append(row)
        source_bindings.append(binding)

    training_rows_by_id = {
        str(row["run_id"]): row
        for row in run_rows
        if row["workload_kind"] != "development_evaluation"
    }
    quality_rows: list[dict[str, Any]] = []
    for raw in config.get("quality_efficiency_runs", []):
        if not isinstance(raw, dict):
            raise SystemsEvidenceError("quality_efficiency_runs entries must be objects")
        row, bindings = _extract_quality_efficiency(root, raw, training_rows_by_id)
        quality_rows.append(row)
        source_bindings.extend(bindings)

    recovery_entries = config.get("recovery_events", [])
    if not isinstance(recovery_entries, list):
        raise SystemsEvidenceError("recovery_events must be a list")
    recovery_rows, recovery_bindings = _recovery_events(root, recovery_entries)
    source_bindings.extend(recovery_bindings)

    coverage = config.get("experiment_coverage")
    if not isinstance(coverage, list) or not coverage:
        raise SystemsEvidenceError("experiment_coverage must be a nonempty list")
    coverage_rows: list[dict[str, Any]] = []
    covered_ids: set[str] = set()
    for raw in coverage:
        if not isinstance(raw, dict):
            raise SystemsEvidenceError("experiment_coverage entries must be objects")
        experiment_id = str(raw.get("experiment_id", ""))
        if experiment_id not in _ALLOWED_EXPERIMENTS or experiment_id in covered_ids:
            raise SystemsEvidenceError(f"invalid or duplicate coverage identity: {experiment_id}")
        covered_ids.add(experiment_id)
        evidence_path = str(raw.get("evidence_path", ""))
        source_bindings.append(_binding(root, evidence_path))
        coverage_rows.append(
            {
                "experiment_id": experiment_id,
                "disposition": str(raw.get("disposition", "")),
                "independent_gpu_run": bool(raw.get("independent_gpu_run", False)),
                "evidence_path": evidence_path,
            }
        )
    required_coverage = {
        *(f"E{number:02d}" for number in range(8, 21)),
        "E25",
        "E27",
        "E28",
    }
    if covered_ids != required_coverage:
        raise SystemsEvidenceError(
            f"experiment coverage mismatch: missing={sorted(required_coverage - covered_ids)}, "
            f"extra={sorted(covered_ids - required_coverage)}"
        )

    run_rows.sort(key=lambda row: (str(row["experiment_id"]), str(row["run_id"])))
    checkpoint_rows.sort(
        key=lambda row: (str(row["experiment_id"]), str(row["run_id"]), int(row["checkpoint_step"]))
    )
    recovery_rows.sort(key=lambda row: (str(row["experiment_id"]), str(row["event_id"])))
    coverage_rows.sort(key=lambda row: str(row["experiment_id"]))
    quality_rows.sort(key=lambda row: (str(row["experiment_id"]), str(row["run_id"])))
    unique_bindings = {binding.path: binding for binding in source_bindings}
    if len(unique_bindings) != len(source_bindings):
        # Duplicate bindings are benign across tables, but canonicalize them by path.
        source_bindings = list(unique_bindings.values())
    source_bindings.sort(key=lambda binding: binding.path)

    run_json = {"schema_version": 1, "rows": run_rows}
    checkpoint_json = {"schema_version": 1, "rows": checkpoint_rows}
    recovery_json = {"schema_version": 1, "rows": recovery_rows}
    coverage_json = {"schema_version": 1, "rows": coverage_rows}
    quality_json = {"schema_version": 1, "rows": quality_rows}
    output_directory.mkdir(parents=True, exist_ok=True)
    table_directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "run_json": table_directory / "e26_run_metrics.json",
        "run_csv": table_directory / "e26_run_metrics.csv",
        "checkpoint_json": table_directory / "e26_checkpoint_metrics.json",
        "checkpoint_csv": table_directory / "e26_checkpoint_metrics.csv",
        "recovery_json": table_directory / "e26_recovery_events.json",
        "recovery_csv": table_directory / "e26_recovery_events.csv",
        "coverage_json": table_directory / "e26_experiment_coverage.json",
        "coverage_csv": table_directory / "e26_experiment_coverage.csv",
        "quality_json": table_directory / "e26_quality_efficiency.json",
        "quality_csv": table_directory / "e26_quality_efficiency.csv",
    }
    atomic_write_json(outputs["run_json"], run_json)
    atomic_write_bytes(outputs["run_csv"], _csv_bytes(run_rows))
    atomic_write_json(outputs["checkpoint_json"], checkpoint_json)
    atomic_write_bytes(outputs["checkpoint_csv"], _csv_bytes(checkpoint_rows))
    atomic_write_json(outputs["recovery_json"], recovery_json)
    atomic_write_bytes(outputs["recovery_csv"], _csv_bytes(recovery_rows))
    atomic_write_json(outputs["coverage_json"], coverage_json)
    atomic_write_bytes(outputs["coverage_csv"], _csv_bytes(coverage_rows))
    atomic_write_json(outputs["quality_json"], quality_json)
    atomic_write_bytes(outputs["quality_csv"], _csv_bytes(quality_rows))

    manifest_core: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "E26",
        "status": "ANALYSIS_ARTIFACT_GENERATED",
        "protected_access_state": "NOT_ACCESSED_BY_E26",
        "config": _binding(root, config_path.relative_to(root).as_posix()).to_dict(),
        "environment": _binding(root, environment_relative).to_dict(),
        "hardware": hardware,
        "source_bindings": [binding.to_dict() for binding in source_bindings],
        "row_counts": {
            "run_metrics": len(run_rows),
            "checkpoint_metrics": len(checkpoint_rows),
            "recovery_events": len(recovery_rows),
            "experiment_coverage": len(coverage_rows),
            "quality_efficiency": len(quality_rows),
        },
        "availability": {
            "wall_time": "PARTIALLY_MEASURED",
            "gpu_time": UNAVAILABLE,
            "throughput": "PARTIALLY_MEASURED",
            "peak_vram": "PARTIALLY_MEASURED_PYTORCH_PROCESS_COUNTERS",
            "checkpoint_payload_bytes": MEASURED,
            "isolated_checkpoint_save_time": UNAVAILABLE,
            "gpu_utilization": UNAVAILABLE,
            "temperature_time_series": UNAVAILABLE,
            "power_time_series": UNAVAILABLE,
            "quality_per_gpu_hour": (
                "MEASURED_DEVELOPMENT_GAIN_PER_EXCLUSIVE_GPU_WALL_HOUR"
                if quality_rows
                else UNAVAILABLE
            ),
            "quality_per_device_active_gpu_hour": UNAVAILABLE,
        },
        "metric_definitions": {
            "wall_seconds": "Trainer train_runtime, else complete telemetry train_end wall_seconds",
            "tokens_per_wall_second": "final cumulative num_tokens / measured wall_seconds",
            "peak_vram": "maximum process-local torch CUDA peak allocated/reserved bytes",
            "same_step_log_to_save_interval": "observed callback interval; not isolated save I/O",
            "checkpoint_payload_bytes": "CHECKPOINT_COMPLETE checkpoint_byte_size",
            "development_score_gain_per_exclusive_gpu_wall_hour": (
                "(candidate DEVELOPMENT mean_score - E10 DEVELOPMENT mean_score) / "
                "exclusive one-GPU training wall hours; not device-active GPU time"
            ),
        },
    }
    manifest_core["analysis_identity"] = identity_hash(manifest_core)
    manifest_path = output_directory / "SYSTEMS_ANALYSIS.json"
    atomic_write_json(manifest_path, manifest_core)
    atomic_write_bytes(
        report_path, _render_report(manifest_core, run_rows, checkpoint_rows, quality_rows)
    )

    output_bindings = {
        path.relative_to(root).as_posix(): {
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in [*outputs.values(), report_path]
    }
    completion = {
        **manifest_core,
        "outputs": output_bindings,
    }
    atomic_write_json(manifest_path, completion)
    return completion
