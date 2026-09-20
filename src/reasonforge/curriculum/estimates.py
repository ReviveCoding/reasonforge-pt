"""Derive adaptive-sampling estimates from durable E12 rollout evidence.

The estimator deliberately distinguishes direct task observations from
family-pooled estimates for tasks that uniform sampling did not draw.  It never
turns a pooled observation into a claimed task-level observation.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from reasonforge.checkpoints.hashing import atomic_write_bytes, atomic_write_json
from reasonforge.config import file_sha256, identity_hash
from reasonforge.orchestration.rl import RLSuite

_METHOD_VERSION = "e12-rollout-estimates-v1"
_REQUIRED_REWARD_LOGS = {
    "rf_reward_scalar",
    "rf_reward_pre_veto_scalar",
    "rf_reward_hard_vetoed",
    "rf_rollout_group_id",
    "rf_source_example_id",
}
_CONTRIBUTION_PREFIX = "rf_reward_contribution_"


class AdaptiveEstimateError(ValueError):
    """Raised when source evidence cannot support qualified estimates."""


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AdaptiveEstimateError(f"unreadable JSON evidence {path}: {error}") from error
    if not isinstance(value, dict):
        raise AdaptiveEstimateError(f"JSON evidence is not an object: {path}")
    return value


def _bounded_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdaptiveEstimateError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise AdaptiveEstimateError(f"{label} is not finite and bounded in [0, 1]")
    return result


def _read_train_pool(path: Path, *, expected_fingerprint: str) -> list[dict[str, Any]]:
    columns = [
        "example_id",
        "task_family",
        "split_role",
        "content_hash",
        "reliability",
        "verification_status",
    ]
    try:
        rows = [dict(row) for row in pq.read_table(path, columns=columns).to_pylist()]
    except (OSError, ValueError, KeyError) as error:
        raise AdaptiveEstimateError(f"TRAIN pool is unreadable: {error}") from error
    if not rows:
        raise AdaptiveEstimateError("TRAIN pool is empty")
    task_ids = [str(row["example_id"]) for row in rows]
    if len(task_ids) != len(set(task_ids)):
        raise AdaptiveEstimateError("TRAIN pool contains duplicate example IDs")
    if {str(row["split_role"]) for row in rows} != {"TRAIN"}:
        raise AdaptiveEstimateError("adaptive candidate pool contains a non-TRAIN row")
    digest = hashlib.sha256()
    for content_hash in sorted(str(row["content_hash"]) for row in rows):
        digest.update(content_hash.encode())
        digest.update(b"\n")
    if digest.hexdigest() != expected_fingerprint:
        raise AdaptiveEstimateError("TRAIN pool semantic fingerprint mismatch")
    for row in rows:
        _bounded_number(row["reliability"], label=f"{row['example_id']} reliability")
        if row["verification_status"] != "VERIFIED_DETERMINISTIC":
            raise AdaptiveEstimateError(
                f"{row['example_id']} does not have deterministic verification"
            )
    return rows


def _same(values: Sequence[Any]) -> bool:
    return len({json.dumps(value, sort_keys=True, default=str) for value in values}) == 1


def _validate_group(
    event: Mapping[str, Any],
    *,
    ordinal: int,
    expected_group_id: str,
    expected_task_id: str,
    expected_generations: int,
) -> tuple[str, list[float]]:
    prefix = f"rollout group {ordinal}"
    if event.get("event") != "GROUP_REWARD_SCORED":
        raise AdaptiveEstimateError(f"{prefix} has an unexpected event type")
    if event.get("group_ordinal") != ordinal or event.get("group_id") != expected_group_id:
        raise AdaptiveEstimateError(f"{prefix} diverges from the frozen schedule")
    if event.get("example_id") != expected_task_id:
        raise AdaptiveEstimateError(f"{prefix} has the wrong source example")
    if event.get("attempt") != 1:
        raise AdaptiveEstimateError(f"{prefix} is not the sole completed attempt")
    if event.get("completion_count") != expected_generations:
        raise AdaptiveEstimateError(f"{prefix} has the wrong completion count")
    sequence_fields = ("completions", "rewards", "task_family", "reference", "metadata")
    for name in sequence_fields:
        values = event.get(name)
        if not isinstance(values, list) or len(values) != expected_generations:
            raise AdaptiveEstimateError(f"{prefix} has malformed {name}")
    if not _same(event["task_family"]) or not _same(event["reference"]):
        raise AdaptiveEstimateError(f"{prefix} mixes task identities")
    reward_logs = event.get("reward_logs")
    if not isinstance(reward_logs, Mapping):
        raise AdaptiveEstimateError(f"{prefix} has no reward log mapping")
    missing = _REQUIRED_REWARD_LOGS - set(reward_logs)
    if missing:
        raise AdaptiveEstimateError(f"{prefix} lacks reward logs: {sorted(missing)}")
    for name, values in reward_logs.items():
        if not isinstance(values, list) or len(values) != expected_generations:
            raise AdaptiveEstimateError(f"{prefix} has malformed reward log {name}")
    if reward_logs["rf_rollout_group_id"] != [expected_group_id] * expected_generations:
        raise AdaptiveEstimateError(f"{prefix} reward log has inconsistent group identity")
    if reward_logs["rf_source_example_id"] != [expected_task_id] * expected_generations:
        raise AdaptiveEstimateError(f"{prefix} reward log has inconsistent task identity")

    rewards = [
        _bounded_number(value, label=f"{prefix} reward {index}")
        for index, value in enumerate(event["rewards"])
    ]
    scalar = [
        _bounded_number(value, label=f"{prefix} logged scalar {index}")
        for index, value in enumerate(reward_logs["rf_reward_scalar"])
    ]
    pre_veto = [
        _bounded_number(value, label=f"{prefix} pre-veto scalar {index}")
        for index, value in enumerate(reward_logs["rf_reward_pre_veto_scalar"])
    ]
    contributions = [
        values for name, values in reward_logs.items() if str(name).startswith(_CONTRIBUTION_PREFIX)
    ]
    if not contributions:
        raise AdaptiveEstimateError(f"{prefix} has no component contributions")
    vetoes = reward_logs["rf_reward_hard_vetoed"]
    if any(not isinstance(value, bool) for value in vetoes):
        raise AdaptiveEstimateError(f"{prefix} has a non-boolean hard-veto log")
    for index, (observed, logged, before, vetoed) in enumerate(
        zip(rewards, scalar, pre_veto, vetoes, strict=True)
    ):
        contribution_sum = sum(float(values[index]) for values in contributions)
        if not math.isclose(contribution_sum, before, abs_tol=1e-9):
            raise AdaptiveEstimateError(f"{prefix} contribution sum mismatch at rollout {index}")
        expected = 0.0 if vetoed else before
        if not math.isclose(logged, expected, abs_tol=1e-9) or not math.isclose(
            observed, logged, abs_tol=1e-9
        ):
            raise AdaptiveEstimateError(f"{prefix} scalar mismatch at rollout {index}")
    return str(event["task_family"][0]), rewards


def _bootstrap_interval(
    groups: Sequence[Sequence[float]],
    statistic: str,
    *,
    seed_key: str,
    replicates: int,
) -> list[float] | None:
    if len(groups) < 2 or replicates < 1:
        return None
    rng_seed = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], "big")
    rng = random.Random(rng_seed)
    values: list[float] = []
    for _ in range(replicates):
        sampled = [groups[rng.randrange(len(groups))] for _ in groups]
        flat = [value for group in sampled for value in group]
        if statistic == "p":
            estimate = sum(math.isclose(value, 1.0, abs_tol=1e-9) for value in flat) / len(flat)
        else:
            estimate = min(1.0, 4.0 * statistics.pvariance(flat))
        values.append(estimate)
    values.sort()
    lower = values[max(0, math.floor(0.025 * replicates))]
    upper = values[min(replicates - 1, math.ceil(0.975 * replicates) - 1)]
    return [lower, upper]


def _summarize_groups(
    groups: Sequence[Sequence[float]], *, seed_key: str, bootstrap_replicates: int
) -> dict[str, Any]:
    flat = [value for group in groups for value in group]
    if not flat:
        raise AdaptiveEstimateError("cannot summarize an empty rollout collection")
    successes = sum(math.isclose(value, 1.0, abs_tol=1e-9) for value in flat)
    return {
        "group_count": len(groups),
        "rollout_count": len(flat),
        "complete_successes": successes,
        "p": successes / len(flat),
        "u": min(1.0, 4.0 * statistics.pvariance(flat)),
        "p_cluster_bootstrap_95": _bootstrap_interval(
            groups, "p", seed_key=f"{seed_key}:p", replicates=bootstrap_replicates
        ),
        "u_cluster_bootstrap_95": _bootstrap_interval(
            groups, "u", seed_key=f"{seed_key}:u", replicates=bootstrap_replicates
        ),
    }


def derive_e12_adaptive_estimates(
    workspace: Path,
    *,
    suite_path: Path = Path("configs/rl/e12_e20.yaml"),
    run_id: str = "E12-uniform-grpo-seed1701-v1",
    bootstrap_replicates: int = 2_000,
) -> dict[str, Any]:
    """Return a validated, deterministic adaptive-estimate evidence object."""

    root = workspace.resolve()
    suite_file = suite_path if suite_path.is_absolute() else root / suite_path
    suite = RLSuite.load(suite_file)
    run_dir = root / "artifacts" / "training" / "E12" / run_id
    paths = {
        "dataset": root / suite.dataset_path,
        "rollouts": run_dir / "ROLLOUT_REWARD_EVIDENCE.jsonl",
        "schedule": run_dir / "ROLLOUT_SCHEDULE.json",
        "run_identity": run_dir / "RUN_IDENTITY.json",
        "run_complete": run_dir / "RUN_COMPLETE.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise AdaptiveEstimateError(f"missing E12 evidence: {missing}")
    run_identity = _load_object(paths["run_identity"])
    run_complete = _load_object(paths["run_complete"])
    schedule = _load_object(paths["schedule"])
    if run_complete.get("status") != "COMPLETE" or run_complete.get("run_id") != run_id:
        raise AdaptiveEstimateError("E12 run is not durably COMPLETE")
    if run_identity.get("run_id") != run_id or run_identity.get("experiment_id") != "E12":
        raise AdaptiveEstimateError("E12 run identity mismatch")
    if run_complete.get("starting_adapter_identity") != run_identity.get(
        "starting_adapter_identity"
    ):
        raise AdaptiveEstimateError("starting adapter identity changed within E12")
    rollout_hash = file_sha256(paths["rollouts"])
    if run_complete.get("rollout_reward_evidence_sha256") != rollout_hash:
        raise AdaptiveEstimateError("rollout evidence hash does not match RUN_COMPLETE")
    if schedule.get("schedule_hash") != run_identity.get("schedule_hash") or schedule.get(
        "schedule_hash"
    ) != run_complete.get("schedule_hash"):
        raise AdaptiveEstimateError("frozen rollout schedule identity mismatch")

    pool = _read_train_pool(paths["dataset"], expected_fingerprint=suite.dataset_fingerprint)
    pool_by_id = {str(row["example_id"]): row for row in pool}
    scheduled_tasks = schedule.get("task_ids")
    scheduled_groups = schedule.get("group_ids")
    generations = schedule.get("num_generations")
    if (
        not isinstance(scheduled_tasks, list)
        or not isinstance(scheduled_groups, list)
        or len(scheduled_tasks) != len(scheduled_groups)
        or not isinstance(generations, int)
        or generations < 2
    ):
        raise AdaptiveEstimateError("frozen schedule is malformed")
    if any(str(task_id) not in pool_by_id for task_id in scheduled_tasks):
        raise AdaptiveEstimateError("frozen schedule references a task outside the TRAIN pool")

    raw_lines = paths["rollouts"].read_text(encoding="utf-8").splitlines()
    if len(raw_lines) != len(scheduled_groups):
        raise AdaptiveEstimateError("rollout evidence does not contain exactly one event per group")
    task_groups: dict[str, list[list[float]]] = defaultdict(list)
    family_groups: dict[str, list[list[float]]] = defaultdict(list)
    for ordinal, line in enumerate(raw_lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise AdaptiveEstimateError(f"partial JSONL record at group {ordinal}") from error
        if not isinstance(event, Mapping) or event.get("run_id") != run_id:
            raise AdaptiveEstimateError(f"rollout group {ordinal} has the wrong run identity")
        task_id = str(scheduled_tasks[ordinal])
        family, rewards = _validate_group(
            event,
            ordinal=ordinal,
            expected_group_id=str(scheduled_groups[ordinal]),
            expected_task_id=task_id,
            expected_generations=generations,
        )
        if family != str(pool_by_id[task_id]["task_family"]):
            raise AdaptiveEstimateError(f"rollout group {ordinal} has the wrong task family")
        task_groups[task_id].append(rewards)
        family_groups[family].append(rewards)

    expected_groups = int(run_complete.get("completed_rollout_groups", -1))
    expected_rollouts = int(run_complete.get("completed_rollouts", -1))
    if expected_groups != len(raw_lines) or expected_rollouts != len(raw_lines) * generations:
        raise AdaptiveEstimateError("RUN_COMPLETE rollout accounting mismatch")
    family_summary = {
        family: _summarize_groups(
            groups,
            seed_key=f"{run_id}:family:{family}",
            bootstrap_replicates=bootstrap_replicates,
        )
        for family, groups in sorted(family_groups.items())
    }
    estimates: list[dict[str, Any]] = []
    direct_count = 0
    for row in sorted(pool, key=lambda item: str(item["example_id"])):
        task_id = str(row["example_id"])
        family = str(row["task_family"])
        direct = task_groups.get(task_id)
        if direct:
            direct_count += 1
            summary = _summarize_groups(
                direct,
                seed_key=f"{run_id}:task:{task_id}",
                bootstrap_replicates=bootstrap_replicates,
            )
            estimate_kind = "DIRECT_TASK_ROLLOUTS"
            provenance = f"{run_id}:direct:{summary['group_count']}groups"
        else:
            summary = family_summary[family]
            estimate_kind = "FAMILY_POOLED_NO_DIRECT_TASK_OBSERVATION"
            provenance = f"{run_id}:family_pool:{family}:direct_task_groups=0"
        reliability = _bounded_number(row["reliability"], label=f"{task_id} reliability")
        estimates.append(
            {
                "task_id": task_id,
                "family": family,
                "p": summary["p"],
                "q": reliability,
                "u": summary["u"],
                "provenance": provenance,
                "estimate_kind": estimate_kind,
                "direct_group_count": len(direct or []),
                "direct_rollout_count": sum(len(group) for group in (direct or [])),
                "estimation_group_count": summary["group_count"],
                "estimation_rollout_count": summary["rollout_count"],
                "complete_successes_in_estimation_pool": summary["complete_successes"],
                "p_cluster_bootstrap_95": summary["p_cluster_bootstrap_95"],
                "u_cluster_bootstrap_95": summary["u_cluster_bootstrap_95"],
                "q_source": "canonical TRAIN reliability; VERIFIED_DETERMINISTIC",
            }
        )

    source_hashes = {name: file_sha256(path) for name, path in paths.items()}
    payload: dict[str, Any] = {
        "schema_version": 1,
        "method_version": _METHOD_VERSION,
        "status": "QUALIFIED",
        # This is a DEVELOPMENT decision artifact. Its rollout source rows remain
        # explicitly TRAIN and are never misrepresented as held-out observations.
        "data_role": "DEVELOPMENT",
        "source_data_roles": ["TRAIN"],
        "role_semantics": (
            "DEVELOPMENT denotes authorized downstream use; source rollouts came only "
            "from the frozen TRAIN candidate pool and are not held-out evaluation evidence."
        ),
        "experiment_id": "E12",
        "source_run_id": run_id,
        "dataset_fingerprint": suite.dataset_fingerprint,
        "dataset_file_sha256": source_hashes["dataset"],
        "candidate_pool_identity": identity_hash(sorted(pool_by_id)),
        "starting_adapter_identity": run_identity["starting_adapter_identity"],
        "reward_config_identity": run_identity["reward_config_hash"],
        "schedule_hash": schedule["schedule_hash"],
        "source_artifacts": {
            name: {"path": path.relative_to(root).as_posix(), "sha256": source_hashes[name]}
            for name, path in paths.items()
        },
        "estimator": {
            "p": "fraction of rollouts with canonical scalar reward exactly 1.0",
            "q": (
                "canonical per-task supervision reliability after deterministic-verifier status "
                "and complete reward-trace reconstruction were validated"
            ),
            "u": "4 * population variance of bounded canonical scalar rewards, clipped to 1",
            "unobserved_task_policy": (
                "family-pooled p/u; direct counts remain zero and provenance says pooled"
            ),
            "bootstrap": {
                "unit": "complete GRPO group",
                "replicates": bootstrap_replicates,
                "confidence": 0.95,
                "interval": "deterministic percentile cluster bootstrap",
                "single_group_interval": None,
            },
        },
        "coverage": {
            "candidate_tasks": len(pool),
            "directly_observed_tasks": direct_count,
            "family_pooled_unobserved_tasks": len(pool) - direct_count,
            "task_coverage_fraction": direct_count / len(pool),
            "observed_families": len(family_groups),
            "candidate_families": len({str(row["task_family"]) for row in pool}),
            "rollout_groups": len(raw_lines),
            "rollouts": len(raw_lines) * generations,
            "generations_per_group": generations,
        },
        "family_estimates": family_summary,
        "limitations": [
            "Uniform sampling left some tasks without direct rollouts; their p/u are "
            "family-pooled.",
            "The policy changed during E12, so pooled outcomes summarize an online training "
            "trace, not a stationary checkpoint.",
            "Generations within a GRPO group are correlated; intervals resample complete groups.",
            "q validates deterministic supervision identity and trace integrity, not independent "
            "semantic judge agreement.",
        ],
        "task_estimates": estimates,
    }
    payload["derivation_identity_sha256"] = identity_hash(payload)
    return payload


def publish_e12_adaptive_estimates(
    workspace: Path,
    *,
    output: Path = Path("evidence/training/E12/ADAPTIVE_ESTIMATES.json"),
    bootstrap_replicates: int = 2_000,
) -> tuple[Path, str]:
    """Atomically publish estimates and a hash sidecar."""

    root = workspace.resolve()
    destination = output if output.is_absolute() else root / output
    payload = derive_e12_adaptive_estimates(root, bootstrap_replicates=bootstrap_replicates)
    atomic_write_json(destination, payload)
    digest = file_sha256(destination)
    atomic_write_bytes(destination.with_suffix(".sha256"), f"{digest}\n".encode())
    return destination, digest
