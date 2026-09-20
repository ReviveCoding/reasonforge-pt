"""Build the canonical ReasonForge evidence warehouse from repository artifacts.

The warehouse never synthesizes placeholder observations. Every scientific
table exists even when its source artifacts do not; ``table_status`` makes that
absence explicit and distinguishes it from a failed ingestion.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq
import yaml

from reasonforge.checkpoints import ProjectCheckpointStore, RuntimeCheckpointStore

CANONICAL_TABLES = (
    "sources",
    "data_roles",
    "models",
    "training",
    "rollouts",
    "rewards",
    "evaluations",
    "gpu_telemetry",
    "failures",
    "checkpoints",
    "resume_events",
    "statistics",
    "decisions",
)
ALLOWED_ROLES = frozenset(
    {"TRAIN", "DEVELOPMENT", "POLICY", "QUALIFICATION", "LOCKED_FINAL", "EXTERNAL_PROTECTED"}
)


class EvidenceValidationError(ValueError):
    """An artifact failed the canonical evidence contract."""


@dataclass(frozen=True)
class WarehouseBuildResult:
    database_path: Path
    manifest_path: Path
    row_counts: dict[str, int]
    table_states: dict[str, str]
    input_sha256: str


_DDL = {
    "sources": """
        CREATE TABLE sources (
            source_id VARCHAR PRIMARY KEY, name VARCHAR NOT NULL, kind VARCHAR,
            uri VARCHAR, revision VARCHAR, declared_role VARCHAR, protection VARCHAR,
            access_status VARCHAR, license_spdx VARCHAR, governance_status VARCHAR,
            training_allowed BOOLEAN, evidence_path VARCHAR NOT NULL, details_json JSON NOT NULL
        )
    """,
    "data_roles": """
        CREATE TABLE data_roles (
            dataset_id VARCHAR NOT NULL, artifact_path VARCHAR NOT NULL,
            example_id VARCHAR NOT NULL, source VARCHAR NOT NULL, source_revision VARCHAR,
            source_license VARCHAR, task_family VARCHAR NOT NULL, split_role VARCHAR NOT NULL,
            difficulty DOUBLE, reliability DOUBLE, verification_status VARCHAR,
            token_count BIGINT, content_hash VARCHAR NOT NULL, metadata_json JSON,
            PRIMARY KEY (dataset_id, example_id)
        )
    """,
    "models": """
        CREATE TABLE models (
            model_key VARCHAR PRIMARY KEY, model_id VARCHAR NOT NULL, revision VARCHAR NOT NULL,
            weight_bytes BIGINT, purpose VARCHAR, status VARCHAR NOT NULL,
            evidence_path VARCHAR NOT NULL, details_json JSON NOT NULL
        )
    """,
    "training": """
        CREATE TABLE training (
            record_id VARCHAR PRIMARY KEY, experiment_id VARCHAR, run_id VARCHAR,
            protocol_id VARCHAR, model_id VARCHAR, objective VARCHAR, optimizer_step BIGINT,
            loss DOUBLE, tokens BIGINT, examples BIGINT, wall_seconds DOUBLE, status VARCHAR,
            occurred_at TIMESTAMP, artifact_path VARCHAR NOT NULL, details_json JSON NOT NULL
        )
    """,
    "rollouts": """
        CREATE TABLE rollouts (
            record_id VARCHAR PRIMARY KEY, experiment_id VARCHAR, run_id VARCHAR,
            group_id VARCHAR, trajectory_id VARCHAR, example_id VARCHAR, task_family VARCHAR,
            reward DOUBLE, success BOOLEAN, included_in_gradient BOOLEAN,
            occurred_at TIMESTAMP, artifact_path VARCHAR NOT NULL, details_json JSON NOT NULL
        )
    """,
    "rewards": """
        CREATE TABLE rewards (
            record_id VARCHAR PRIMARY KEY, experiment_id VARCHAR, run_id VARCHAR,
            record_type VARCHAR NOT NULL, reward_name VARCHAR, example_id VARCHAR,
            component_value DOUBLE, aggregate_value DOUBLE, split_role VARCHAR,
            sample_size BIGINT, accuracy DOUBLE, sensitivity DOUBLE, specificity DOUBLE,
            perturbation_stability DOUBLE, judge_agreement DOUBLE,
            artifact_path VARCHAR NOT NULL, details_json JSON NOT NULL
        )
    """,
    "evaluations": """
        CREATE TABLE evaluations (
            record_id VARCHAR PRIMARY KEY, experiment_id VARCHAR, model_id VARCHAR,
            checkpoint_id VARCHAR, example_id VARCHAR, benchmark VARCHAR, split_role VARCHAR,
            metric_name VARCHAR, metric_value DOUBLE, passed BOOLEAN, seed BIGINT,
            artifact_path VARCHAR NOT NULL, details_json JSON NOT NULL
        )
    """,
    "gpu_telemetry": """
        CREATE TABLE gpu_telemetry (
            record_id VARCHAR PRIMARY KEY, experiment_id VARCHAR, run_id VARCHAR,
            phase VARCHAR, observed_at TIMESTAMP, gpu_index BIGINT, gpu_name VARCHAR,
            gpu_uuid VARCHAR, driver_version VARCHAR, memory_total_mib DOUBLE,
            memory_free_mib DOUBLE, utilization_percent DOUBLE, temperature_c DOUBLE,
            power_w DOUBLE, peak_allocated_bytes BIGINT, artifact_path VARCHAR NOT NULL,
            details_json JSON NOT NULL
        )
    """,
    "failures": """
        CREATE TABLE failures (
            record_id VARCHAR PRIMARY KEY, experiment_id VARCHAR, run_id VARCHAR,
            event_type VARCHAR NOT NULL, component VARCHAR, status VARCHAR NOT NULL,
            reason VARCHAR, occurred_at TIMESTAMP, artifact_path VARCHAR NOT NULL,
            details_json JSON NOT NULL
        )
    """,
    "checkpoints": """
        CREATE TABLE checkpoints (
            record_id VARCHAR PRIMARY KEY, checkpoint_class VARCHAR NOT NULL,
            checkpoint_id VARCHAR NOT NULL, experiment_id VARCHAR, run_id VARCHAR,
            protocol_id VARCHAR, checkpoint_step BIGINT, project_status VARCHAR,
            current_phase VARCHAR, created_at TIMESTAMP, byte_size BIGINT,
            verified BOOLEAN NOT NULL, artifact_path VARCHAR NOT NULL,
            manifest_sha256 VARCHAR NOT NULL, details_json JSON NOT NULL
        )
    """,
    "resume_events": """
        CREATE TABLE resume_events (
            record_id VARCHAR PRIMARY KEY, experiment_id VARCHAR, run_id VARCHAR,
            event_type VARCHAR NOT NULL, interrupted_group_id VARCHAR,
            resume_from_completed_step BIGINT, excluded_from_gradient BOOLEAN,
            occurred_at TIMESTAMP, artifact_path VARCHAR NOT NULL, details_json JSON NOT NULL
        )
    """,
    "statistics": """
        CREATE TABLE statistics (
            record_id VARCHAR PRIMARY KEY, experiment_id VARCHAR, comparison_id VARCHAR,
            metric_name VARCHAR NOT NULL, estimate DOUBLE, lower_ci DOUBLE, upper_ci DOUBLE,
            sample_size BIGINT, seed_count BIGINT, correction VARCHAR, qualifier VARCHAR,
            evidence_class VARCHAR, artifact_path VARCHAR NOT NULL, details_json JSON NOT NULL
        )
    """,
    "decisions": """
        CREATE TABLE decisions (
            record_id VARCHAR PRIMARY KEY, decision_type VARCHAR NOT NULL,
            decision VARCHAR NOT NULL, protocol_id VARCHAR, checkpoint_id VARCHAR,
            rationale VARCHAR, decided_at TIMESTAMP, artifact_path VARCHAR NOT NULL,
            details_json JSON NOT NULL
        )
    """,
}

_EVENT_COLUMNS: dict[str, tuple[str, ...]] = {
    "training": (
        "record_id",
        "experiment_id",
        "run_id",
        "protocol_id",
        "model_id",
        "objective",
        "optimizer_step",
        "loss",
        "tokens",
        "examples",
        "wall_seconds",
        "status",
        "occurred_at",
    ),
    "rollouts": (
        "record_id",
        "experiment_id",
        "run_id",
        "group_id",
        "trajectory_id",
        "example_id",
        "task_family",
        "reward",
        "success",
        "included_in_gradient",
        "occurred_at",
    ),
    "evaluations": (
        "record_id",
        "experiment_id",
        "model_id",
        "checkpoint_id",
        "example_id",
        "benchmark",
        "split_role",
        "metric_name",
        "metric_value",
        "passed",
        "seed",
    ),
    "rewards": (
        "record_id",
        "experiment_id",
        "run_id",
        "record_type",
        "reward_name",
        "example_id",
        "component_value",
        "aggregate_value",
        "split_role",
        "sample_size",
        "accuracy",
        "sensitivity",
        "specificity",
        "perturbation_stability",
        "judge_agreement",
    ),
    "gpu_telemetry": (
        "record_id",
        "experiment_id",
        "run_id",
        "phase",
        "observed_at",
        "gpu_index",
        "gpu_name",
        "gpu_uuid",
        "driver_version",
        "memory_total_mib",
        "memory_free_mib",
        "utilization_percent",
        "temperature_c",
        "power_w",
        "peak_allocated_bytes",
    ),
    "failures": (
        "record_id",
        "experiment_id",
        "run_id",
        "event_type",
        "component",
        "status",
        "reason",
        "occurred_at",
    ),
    "resume_events": (
        "record_id",
        "experiment_id",
        "run_id",
        "event_type",
        "interrupted_group_id",
        "resume_from_completed_step",
        "excluded_from_gradient",
        "occurred_at",
    ),
    "statistics": (
        "record_id",
        "experiment_id",
        "comparison_id",
        "metric_name",
        "estimate",
        "lower_ci",
        "upper_ci",
        "sample_size",
        "seed_count",
        "correction",
        "qualifier",
        "evidence_class",
    ),
    "decisions": (
        "record_id",
        "decision_type",
        "decision",
        "protocol_id",
        "checkpoint_id",
        "rationale",
        "decided_at",
    ),
}

_INTEGER_FIELDS = frozenset(
    {
        "optimizer_step",
        "tokens",
        "examples",
        "seed",
        "sample_size",
        "seed_count",
        "gpu_index",
        "peak_allocated_bytes",
        "resume_from_completed_step",
    }
)
_FLOAT_FIELDS = frozenset(
    {
        "loss",
        "wall_seconds",
        "reward",
        "metric_value",
        "estimate",
        "lower_ci",
        "upper_ci",
        "component_value",
        "aggregate_value",
        "accuracy",
        "sensitivity",
        "specificity",
        "perturbation_stability",
        "judge_agreement",
        "memory_total_mib",
        "memory_free_mib",
        "utilization_percent",
        "temperature_c",
        "power_w",
    }
)
_BOOLEAN_FIELDS = frozenset({"success", "included_in_gradient", "passed", "excluded_from_gradient"})


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceValidationError(f"invalid JSON artifact {path}: {error}") from error


def _read_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text("utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise EvidenceValidationError(f"invalid YAML artifact {path}: {error}") from error


def _relative(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError as error:
        raise EvidenceValidationError(f"artifact is outside workspace: {path}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceValidationError(f"{context} must be a mapping")
    return value


def _required_string(value: Mapping[str, Any], field: str, context: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item:
        raise EvidenceValidationError(f"{context}.{field} must be a non-empty string")
    return item


def _create_schema(connection: duckdb.DuckDBPyConnection) -> None:
    for table in CANONICAL_TABLES:
        connection.execute(_DDL[table])
    connection.execute(
        """
        CREATE TABLE table_status (
            table_name VARCHAR PRIMARY KEY,
            state VARCHAR NOT NULL CHECK (
                state IN ('POPULATED', 'EMPTY_NO_ARTIFACTS', 'EMPTY_ARTIFACTS_NO_ROWS')
            ),
            row_count BIGINT NOT NULL CHECK (row_count >= 0),
            input_artifacts_json JSON NOT NULL,
            built_at TIMESTAMP NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE warehouse_builds (
            build_id VARCHAR PRIMARY KEY, built_at TIMESTAMP NOT NULL,
            workspace VARCHAR NOT NULL, input_sha256 VARCHAR NOT NULL,
            schema_version INTEGER NOT NULL, manifest_path VARCHAR NOT NULL
        )
        """
    )


def _insert_sources(connection: duckdb.DuckDBPyConnection, workspace: Path) -> list[Path]:
    path = workspace / "data" / "source_registry.yaml"
    if not path.is_file():
        return []
    root = _mapping(_read_yaml(path), str(path))
    values = root.get("sources")
    if not isinstance(values, list):
        raise EvidenceValidationError("source registry sources must be a list")
    rows: list[tuple[object, ...]] = []
    for index, raw in enumerate(values):
        source = _mapping(raw, f"sources[{index}]")
        source_id = _required_string(source, "source_id", f"sources[{index}]")
        name = _required_string(source, "name", f"sources[{index}]")
        access = _mapping(source.get("access", {}), f"sources[{index}].access")
        license_value = _mapping(source.get("license", {}), f"sources[{index}].license")
        governance = _mapping(source.get("governance", {}), f"sources[{index}].governance")
        rows.append(
            (
                source_id,
                name,
                source.get("kind"),
                source.get("uri"),
                source.get("revision"),
                source.get("role"),
                source.get("protection"),
                access.get("status"),
                license_value.get("spdx"),
                governance.get("status"),
                governance.get("training_allowed"),
                _relative(path, workspace),
                _canonical_json(source),
            )
        )
    if rows:
        connection.executemany(
            "INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
        )
    return [path]


_CANONICAL_DATA_COLUMNS = frozenset(
    {
        "example_id",
        "source",
        "source_revision",
        "source_license",
        "task_family",
        "split_role",
        "prompt",
        "reference",
        "chosen",
        "rejected",
        "tool_schema",
        "difficulty",
        "reliability",
        "verification_status",
        "token_count",
        "content_hash",
        "metadata",
    }
)


def _insert_data_roles(connection: duckdb.DuckDBPyConnection, workspace: Path) -> list[Path]:
    curated = workspace / "data" / "curated"
    paths = sorted(curated.glob("*/*.parquet")) if curated.is_dir() else []
    for path in paths:
        schema_names = set(pq.read_schema(path).names)
        missing = sorted(_CANONICAL_DATA_COLUMNS - schema_names)
        if missing:
            raise EvidenceValidationError(f"canonical parquet {path} lacks columns: {missing}")
        dataset_id = path.parent.name
        expected_role = path.stem.upper()
        artifact_path = _relative(path, workspace)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
            rows: list[tuple[object, ...]] = []
            for raw in batch.to_pylist():
                row = _mapping(raw, artifact_path)
                role = _required_string(row, "split_role", artifact_path)
                if role not in ALLOWED_ROLES:
                    raise EvidenceValidationError(f"invalid split_role {role!r} in {path}")
                if expected_role != role:
                    raise EvidenceValidationError(
                        f"role/file mismatch in {path}: filename={expected_role}, row={role}"
                    )
                metadata = row.get("metadata")
                if metadata is not None:
                    try:
                        json.loads(metadata) if isinstance(metadata, str) else None
                    except json.JSONDecodeError as error:
                        raise EvidenceValidationError(
                            f"invalid metadata JSON in {path}: {error}"
                        ) from error
                rows.append(
                    (
                        dataset_id,
                        artifact_path,
                        _required_string(row, "example_id", artifact_path),
                        _required_string(row, "source", artifact_path),
                        row.get("source_revision"),
                        row.get("source_license"),
                        _required_string(row, "task_family", artifact_path),
                        role,
                        row.get("difficulty"),
                        row.get("reliability"),
                        row.get("verification_status"),
                        row.get("token_count"),
                        _required_string(row, "content_hash", artifact_path),
                        metadata,
                    )
                )
            if rows:
                connection.executemany(
                    "INSERT INTO data_roles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
                )
    return paths


def _insert_models(connection: duckdb.DuckDBPyConnection, workspace: Path) -> list[Path]:
    path = workspace / "configs" / "models.yaml"
    if not path.is_file():
        return []
    root = _mapping(_read_yaml(path), str(path))
    rows: list[tuple[object, ...]] = []
    for model_key, raw in root.items():
        model = _mapping(raw, f"models.{model_key}")
        rows.append(
            (
                str(model_key),
                _required_string(model, "model_id", str(model_key)),
                _required_string(model, "revision", str(model_key)),
                model.get("weight_bytes"),
                model.get("purpose"),
                "REGISTERED_NOT_EXECUTED",
                _relative(path, workspace),
                _canonical_json(model),
            )
        )
    if rows:
        connection.executemany("INSERT INTO models VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    return [path]


def _validate_event_value(field: str, value: object, context: str) -> None:
    if value is None:
        return
    if field in _INTEGER_FIELDS and (isinstance(value, bool) or not isinstance(value, int)):
        raise EvidenceValidationError(f"{context}.{field} must be an integer or null")
    if field in _FLOAT_FIELDS and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise EvidenceValidationError(f"{context}.{field} must be numeric or null")
    if field in _BOOLEAN_FIELDS and not isinstance(value, bool):
        raise EvidenceValidationError(f"{context}.{field} must be boolean or null")


def _insert_jsonl_events(
    connection: duckdb.DuckDBPyConnection, workspace: Path, table: str
) -> list[Path]:
    path = workspace / "evidence" / "warehouse_inputs" / f"{table}.jsonl"
    if not path.is_file():
        return []
    columns = _EVENT_COLUMNS[table]
    rows: list[tuple[object, ...]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = _mapping(json.loads(line), f"{path}:{line_number}")
            except json.JSONDecodeError as error:
                raise EvidenceValidationError(
                    f"invalid JSONL at {path}:{line_number}: {error}"
                ) from error
            record_id = _required_string(record, "record_id", f"{path}:{line_number}")
            if record_id in seen:
                raise EvidenceValidationError(f"duplicate record_id {record_id!r} in {path}")
            seen.add(record_id)
            for field in columns:
                _validate_event_value(field, record.get(field), f"{path}:{line_number}")
            if table == "evaluations" and record.get("split_role") not in ALLOWED_ROLES:
                raise EvidenceValidationError(
                    f"evaluation at {path}:{line_number} has invalid split_role"
                )
            if table == "rewards":
                _required_string(record, "record_type", f"{path}:{line_number}")
            if table in {"failures", "resume_events"}:
                _required_string(record, "event_type", f"{path}:{line_number}")
            if table == "failures":
                _required_string(record, "status", f"{path}:{line_number}")
            if table == "statistics":
                _required_string(record, "metric_name", f"{path}:{line_number}")
            if table == "decisions":
                _required_string(record, "decision_type", f"{path}:{line_number}")
                _required_string(record, "decision", f"{path}:{line_number}")
            details = {key: value for key, value in record.items() if key not in columns}
            rows.append(
                (
                    *(record.get(field) for field in columns),
                    _relative(path, workspace),
                    _canonical_json(details),
                )
            )
    if rows:
        placeholders = ", ".join("?" for _ in range(len(columns) + 2))
        connection.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)
    return [path]


def _insert_reward_qualification(
    connection: duckdb.DuckDBPyConnection, workspace: Path
) -> list[Path]:
    path = workspace / "reports" / "tables" / "reward_qualification.csv"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "verifier",
            "sample_size",
            "accuracy",
            "sensitivity",
            "specificity",
            "perturbation_stability",
            "judge_agreement",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise EvidenceValidationError(
                f"reward qualification CSV lacks required columns: {path}"
            )
        reward_rows: list[tuple[object, ...]] = []
        statistic_rows: list[tuple[object, ...]] = []
        for line_number, row in enumerate(reader, start=2):
            verifier = row.get("verifier", "")
            if not verifier:
                raise EvidenceValidationError(f"empty verifier at {path}:{line_number}")
            try:
                sample_size = int(row["sample_size"])
                metrics = {
                    field: float(row[field]) if row.get(field) else None
                    for field in (
                        "accuracy",
                        "sensitivity",
                        "specificity",
                        "perturbation_stability",
                        "judge_agreement",
                    )
                }
            except (TypeError, ValueError) as error:
                raise EvidenceValidationError(f"invalid metric at {path}:{line_number}") from error
            record_id = f"reward-qualification:{verifier}"
            reward_rows.append(
                (
                    record_id,
                    "E07",
                    None,
                    "VERIFIER_QUALIFICATION",
                    verifier,
                    None,
                    None,
                    None,
                    "QUALIFICATION",
                    sample_size,
                    metrics["accuracy"],
                    metrics["sensitivity"],
                    metrics["specificity"],
                    metrics["perturbation_stability"],
                    metrics["judge_agreement"],
                    _relative(path, workspace),
                    _canonical_json(row),
                )
            )
            for metric_name, estimate in metrics.items():
                if estimate is None:
                    continue
                statistic_rows.append(
                    (
                        f"{record_id}:{metric_name}",
                        "E07",
                        verifier,
                        metric_name,
                        estimate,
                        None,
                        None,
                        sample_size,
                        0,
                        None,
                        "deterministic verifier qualification",
                        "MEASURED",
                        _relative(path, workspace),
                        _canonical_json({"line": line_number}),
                    )
                )
        if reward_rows:
            connection.executemany(
                "INSERT INTO rewards VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                reward_rows,
            )
        if statistic_rows:
            connection.executemany(
                "INSERT INTO statistics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                statistic_rows,
            )
    return [path]


def _insert_gpu_qualification(connection: duckdb.DuckDBPyConnection, workspace: Path) -> list[Path]:
    path = workspace / "evidence" / "environment" / "CP00_environment.json"
    if not path.is_file():
        return []
    root = _mapping(_read_json(path), str(path))
    nvidia = _mapping(root.get("nvidia_smi", {}), "nvidia_smi")
    rows = nvidia.get("rows", [])
    if not isinstance(rows, list):
        raise EvidenceValidationError("nvidia_smi.rows must be a list")
    inserts: list[tuple[object, ...]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, str):
            raise EvidenceValidationError("nvidia_smi row must be a string")
        fields = [item.strip() for item in raw.split(",")]
        if len(fields) < 9:
            raise EvidenceValidationError(f"nvidia_smi row has too few fields: {raw!r}")

        def optional_number(value: str) -> float | None:
            try:
                return float(value)
            except ValueError:
                return None

        inserts.append(
            (
                f"E00:gpu:{index}",
                "E00",
                None,
                "QUALIFICATION",
                root.get("created_at"),
                int(fields[0]),
                fields[1],
                fields[2],
                fields[3],
                optional_number(fields[4]),
                optional_number(fields[5]),
                None,
                optional_number(fields[8]),
                None,
                _mapping(root.get("cuda", {}), "cuda").get("peak_allocated_bytes"),
                _relative(path, workspace),
                _canonical_json({"raw": raw, "query": nvidia.get("query")}),
            )
        )
    if inserts:
        connection.executemany(
            "INSERT INTO gpu_telemetry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            inserts,
        )
    return [path]


def _insert_checkpoints_and_failures(
    connection: duckdb.DuckDBPyConnection, workspace: Path
) -> tuple[list[Path], list[Path]]:
    checkpoint_inputs: list[Path] = []
    failure_inputs: list[Path] = []
    project_root = workspace / "evidence" / "checkpoints"
    project_store = ProjectCheckpointStore(project_root)
    if project_root.is_dir():
        for directory in sorted(project_root.glob("CP*")):
            if not directory.is_dir():
                continue
            project_manifest = project_store.verify(directory)
            manifest_path = directory / "CHECKPOINT_MANIFEST.json"
            connection.execute(
                "INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"project:{project_manifest.checkpoint_id}",
                    "PROJECT",
                    project_manifest.checkpoint_id,
                    None,
                    None,
                    project_manifest.protocol_id,
                    None,
                    project_manifest.project_status,
                    project_manifest.current_phase,
                    project_manifest.created_at,
                    sum(item.stat().st_size for item in directory.rglob("*") if item.is_file()),
                    True,
                    _relative(directory, workspace),
                    _sha256(manifest_path),
                    _canonical_json(project_manifest.to_dict()),
                ),
            )
            checkpoint_inputs.extend(item for item in directory.rglob("*") if item.is_file())
            blockers_path = directory / "BLOCKERS.json"
            blockers_root = _mapping(_read_json(blockers_path), str(blockers_path))
            blockers = blockers_root.get("blockers", [])
            if not isinstance(blockers, list):
                raise EvidenceValidationError(f"blockers must be a list: {blockers_path}")
            for index, raw in enumerate(blockers):
                blocker = _mapping(raw, f"{blockers_path}:{index}")
                status = _required_string(blocker, "status", str(blockers_path))
                component = _required_string(blocker, "component", str(blockers_path))
                connection.execute(
                    "INSERT INTO failures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"blocker:{project_manifest.checkpoint_id}:{component}",
                        None,
                        None,
                        "BLOCKER",
                        component,
                        status,
                        blocker.get("reason"),
                        project_manifest.created_at,
                        _relative(blockers_path, workspace),
                        _canonical_json(blocker),
                    ),
                )
            failure_inputs.append(blockers_path)

    runtime_root = workspace / "artifacts" / "checkpoints"
    runtime_store = RuntimeCheckpointStore(runtime_root)
    if runtime_root.is_dir():
        for directory in sorted(runtime_root.glob("*/*/COMPLETE/checkpoint_*")):
            if not directory.is_dir():
                continue
            runtime_manifest = runtime_store.verify(directory)
            manifest_path = directory / "RUNTIME_MANIFEST.json"
            connection.execute(
                "INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"runtime:{runtime_manifest.experiment_id}:{runtime_manifest.run_id}:"
                    f"{runtime_manifest.checkpoint_step}",
                    "RUNTIME",
                    directory.name,
                    runtime_manifest.experiment_id,
                    runtime_manifest.run_id,
                    runtime_manifest.protocol_id,
                    runtime_manifest.checkpoint_step,
                    "COMPLETE",
                    runtime_manifest.kind.value,
                    runtime_manifest.completed_at,
                    runtime_manifest.checkpoint_byte_size,
                    True,
                    _relative(directory, workspace),
                    _sha256(manifest_path),
                    _canonical_json(runtime_manifest.to_dict()),
                ),
            )
            checkpoint_inputs.extend(item for item in directory.rglob("*") if item.is_file())
        for invalid in sorted(runtime_root.glob("*/*/INVALID/*/CHECKPOINT_INVALID.json")):
            record = _mapping(_read_json(invalid), str(invalid))
            parts = invalid.relative_to(runtime_root).parts
            connection.execute(
                "INSERT INTO failures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"checkpoint-invalid:{_sha256(invalid)[:16]}",
                    parts[0],
                    parts[1],
                    "CHECKPOINT_INVALID",
                    "runtime_checkpoint",
                    "CHECKPOINT_INVALID",
                    record.get("reason"),
                    record.get("recorded_at"),
                    _relative(invalid, workspace),
                    _canonical_json(record),
                ),
            )
            failure_inputs.append(invalid)
    return checkpoint_inputs, failure_inputs


def _insert_resume_events(connection: duckdb.DuckDBPyConnection, workspace: Path) -> list[Path]:
    runtime_root = workspace / "artifacts" / "checkpoints"
    paths = sorted(runtime_root.glob("*/*/events/*.json")) if runtime_root.is_dir() else []
    for path in paths:
        record = _mapping(_read_json(path), str(path))
        event = _required_string(record, "event", str(path))
        parts = path.relative_to(runtime_root).parts
        connection.execute(
            "INSERT INTO resume_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"resume:{_sha256(path)[:16]}",
                parts[0],
                parts[1],
                event,
                record.get("interrupted_group_id"),
                record.get("resume_from_completed_step"),
                record.get("excluded_from_gradient_accounting"),
                record.get("recorded_at"),
                _relative(path, workspace),
                _canonical_json(record),
            ),
        )
    return paths


def _fingerprint_inputs(paths: Iterable[Path], workspace: Path) -> tuple[str, dict[str, str]]:
    inventory = {_relative(path, workspace): _sha256(path) for path in sorted(set(paths))}
    digest = hashlib.sha256(_canonical_json(inventory).encode("utf-8")).hexdigest()
    return digest, inventory


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_evidence_warehouse(
    workspace: Path | str,
    output_path: Path | str | None = None,
    *,
    manifest_path: Path | str | None = None,
) -> WarehouseBuildResult:
    """Validate artifacts and atomically rebuild the canonical DuckDB warehouse."""

    root = Path(workspace).resolve()
    output = (
        Path(output_path)
        if output_path is not None
        else root / "evidence" / "warehouse" / "reasonforge.duckdb"
    )
    if not output.is_absolute():
        output = root / output
    manifest_output = (
        Path(manifest_path)
        if manifest_path is not None
        else output.with_name("warehouse_manifest.json")
    )
    if not manifest_output.is_absolute():
        manifest_output = root / manifest_output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    all_inputs: dict[str, list[Path]] = {table: [] for table in CANONICAL_TABLES}
    connection: duckdb.DuckDBPyConnection | None = None
    try:
        connection = duckdb.connect(str(temporary))
        _create_schema(connection)
        all_inputs["sources"] = _insert_sources(connection, root)
        all_inputs["data_roles"] = _insert_data_roles(connection, root)
        all_inputs["models"] = _insert_models(connection, root)
        for table in ("training", "rollouts", "evaluations", "decisions"):
            all_inputs[table] = _insert_jsonl_events(connection, root, table)
        reward_inputs = _insert_reward_qualification(connection, root)
        all_inputs["rewards"].extend(reward_inputs)
        all_inputs["rewards"].extend(_insert_jsonl_events(connection, root, "rewards"))
        all_inputs["statistics"].extend(reward_inputs)
        all_inputs["statistics"].extend(_insert_jsonl_events(connection, root, "statistics"))
        all_inputs["gpu_telemetry"] = _insert_gpu_qualification(connection, root)
        all_inputs["gpu_telemetry"].extend(_insert_jsonl_events(connection, root, "gpu_telemetry"))
        checkpoint_inputs, failure_inputs = _insert_checkpoints_and_failures(connection, root)
        all_inputs["checkpoints"] = checkpoint_inputs
        all_inputs["failures"] = failure_inputs
        all_inputs["failures"].extend(_insert_jsonl_events(connection, root, "failures"))
        all_inputs["resume_events"] = _insert_resume_events(connection, root)
        all_inputs["resume_events"].extend(_insert_jsonl_events(connection, root, "resume_events"))

        row_counts: dict[str, int] = {}
        for table in CANONICAL_TABLES:
            result_row = connection.execute(f"SELECT count(*) FROM {table}").fetchone()
            if result_row is None:
                raise EvidenceValidationError(f"failed to count warehouse table {table}")
            row_counts[table] = int(result_row[0])
        table_states = {
            table: (
                "POPULATED"
                if count
                else "EMPTY_ARTIFACTS_NO_ROWS"
                if all_inputs[table]
                else "EMPTY_NO_ARTIFACTS"
            )
            for table, count in row_counts.items()
        }
        built_at = _utc_now()
        for table in CANONICAL_TABLES:
            connection.execute(
                "INSERT INTO table_status VALUES (?, ?, ?, ?, ?)",
                (
                    table,
                    table_states[table],
                    row_counts[table],
                    _canonical_json(
                        sorted(_relative(path, root) for path in set(all_inputs[table]))
                    ),
                    built_at,
                ),
            )
        input_digest, input_inventory = _fingerprint_inputs(
            (path for paths in all_inputs.values() for path in paths), root
        )
        build_id = f"warehouse-{input_digest[:16]}"
        connection.execute(
            "INSERT INTO warehouse_builds VALUES (?, ?, ?, ?, ?, ?)",
            (build_id, built_at, str(root), input_digest, 1, str(manifest_output)),
        )
        connection.execute("CHECKPOINT")
        connection.close()
        connection = None
        os.replace(temporary, output)
        manifest = {
            "schema_version": 1,
            "build_id": build_id,
            "built_at": built_at,
            "database_path": _relative(output, root),
            "database_sha256": _sha256(output),
            "input_sha256": input_digest,
            "input_artifacts": input_inventory,
            "row_counts": row_counts,
            "table_states": table_states,
            "empty_tables_are_not_measured_zeroes": True,
        }
        _atomic_write_json(manifest_output, manifest)
        return WarehouseBuildResult(output, manifest_output, row_counts, table_states, input_digest)
    except BaseException:
        if connection is not None:
            connection.close()
        temporary.unlink(missing_ok=True)
        raise
