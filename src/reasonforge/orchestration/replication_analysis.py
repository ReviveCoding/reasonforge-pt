"""Fail-closed E25 training and DEVELOPMENT replication aggregation."""

from __future__ import annotations

import csv
import io
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from reasonforge.checkpoints.hashing import atomic_write_bytes, atomic_write_json, sha256_file
from reasonforge.config import identity_hash
from reasonforge.statistics.paired import seed_aware_summary

_PROTOCOL_ID = "RFPT-E25-REPLICATION-v1"
_CONFIG_SHA256 = "50ed5c6e389dfc6985aeaf5dc97677bb477ab7ff24faecf1f84486ccbec78d1c"
_DESIGN_IDENTITY = "e54002d4333418574bfbcadad047fc7090628ffb6f93eee55b6b61221853c83f"
_SEEDS = (1701, 1702, 1703)
_CHECKPOINT_STEPS = (16, 32)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _load_json(path: Path, name: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"{name} is missing: {path}")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), name)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is invalid JSON: {path}") from error


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _verify_checkpoint(root: Path, run_dir: Path, run_hash: str, step: int) -> dict[str, Any]:
    directory = run_dir / f"checkpoint-{step}"
    completion_path = directory / "CHECKPOINT_COMPLETE.json"
    completion = _load_json(completion_path, f"checkpoint-{step} completion")
    if (
        completion.get("status") != "COMPLETE"
        or completion.get("checkpoint_step") != step
        or completion.get("run_identity_hash") != run_hash
    ):
        raise ValueError(f"checkpoint-{step} completion identity mismatch: {run_dir.name}")
    inventory = _mapping(completion.get("file_inventory"), "checkpoint file inventory")
    if not inventory:
        raise ValueError(f"checkpoint-{step} has an empty file inventory: {run_dir.name}")
    total_bytes = 0
    for name, raw_record in inventory.items():
        record = _mapping(raw_record, f"checkpoint file {name}")
        path = directory / str(name)
        expected_bytes = record.get("bytes")
        if not path.is_file() or path.stat().st_size != expected_bytes or path.stat().st_size == 0:
            raise ValueError(f"checkpoint payload size mismatch: {path}")
        if sha256_file(path) != record.get("sha256"):
            raise ValueError(f"checkpoint payload hash mismatch: {path}")
        total_bytes += path.stat().st_size
    if total_bytes != completion.get("checkpoint_byte_size"):
        raise ValueError(f"checkpoint byte accounting mismatch: {directory}")
    runtime = _load_json(directory / "REASONFORGE_CHECKPOINT.json", "runtime checkpoint")
    if (
        runtime.get("status") != "COMPLETE"
        or runtime.get("checkpoint_step") != step
        or runtime.get("run_identity_hash") != run_hash
    ):
        raise ValueError(f"runtime checkpoint identity mismatch: {directory}")
    if any(path.name.startswith(("checkpoint_tmp_", "INCOMPLETE")) for path in directory.iterdir()):
        raise ValueError(f"incomplete checkpoint sentinel found: {directory}")
    return {
        "step": step,
        "path": _relative(root, directory),
        "completion_sha256": sha256_file(completion_path),
        "checkpoint_byte_size": total_bytes,
        "payload_count": len(inventory),
    }


def _adapter_identity(adapter: Path) -> str:
    inventory = {
        path.relative_to(adapter).as_posix(): sha256_file(path)
        for path in sorted(adapter.rglob("*"))
        if path.is_file()
    }
    if not inventory:
        raise ValueError(f"final adapter is empty: {adapter}")
    return identity_hash(inventory)


def _verify_new_run(
    root: Path, frozen: Mapping[str, Any], *, arm: str, seed: int
) -> dict[str, Any]:
    run_id = str(frozen["run_id"])
    run_dir = root / str(frozen["output_dir"])
    complete_path = run_dir / "RUN_COMPLETE.json"
    complete = _load_json(complete_path, "E25 run completion")
    expected_source = "E16" if arm == "conventional" else "E18"
    expected = {
        "status": "COMPLETE",
        "experiment_id": "E25",
        "protocol_id": _PROTOCOL_ID,
        "execution_mode": "LOCKED_FINAL_REPLICATION",
        "run_id": run_id,
        "source_experiment_id": expected_source,
        "seed": seed,
        "optimizer_steps": 32,
        "completed_rollout_groups": 64,
        "completed_rollouts": 512,
        "reward_architecture": "terminal",
        "schedule_hash": frozen["schedule_hash"],
        "replication_config_sha256": _CONFIG_SHA256,
        "replication_design_identity": _DESIGN_IDENTITY,
    }
    mismatched = {
        key: (complete.get(key), value)
        for key, value in expected.items()
        if complete.get(key) != value
    }
    if mismatched:
        raise ValueError(f"E25 run identity mismatch for {run_id}: {mismatched}")
    run_identity = _load_json(run_dir / "RUN_IDENTITY.json", "E25 run identity")
    run_hash = identity_hash(dict(run_identity))
    if run_hash != complete.get("execution_identity_hash"):
        raise ValueError(f"E25 execution identity hash mismatch: {run_id}")
    evidence = root / str(complete["rollout_reward_evidence"])
    if sha256_file(evidence) != complete.get("rollout_reward_evidence_sha256"):
        raise ValueError(f"E25 rollout evidence hash mismatch: {run_id}")
    adapter = root / str(complete["final_adapter_path"])
    if _adapter_identity(adapter) != complete.get("final_adapter_identity"):
        raise ValueError(f"E25 final adapter identity mismatch: {run_id}")
    checkpoints = [_verify_checkpoint(root, run_dir, run_hash, step) for step in _CHECKPOINT_STEPS]
    present_steps = sorted(
        int(path.name.split("-", 1)[1])
        for path in run_dir.glob("checkpoint-*")
        if path.is_dir() and path.name.split("-", 1)[1].isdigit()
    )
    if present_steps != list(_CHECKPOINT_STEPS):
        raise ValueError(f"E25 checkpoint retention mismatch: {run_id}: {present_steps}")
    return {
        "arm": arm,
        "seed": seed,
        "run_id": run_id,
        "run_complete_path": _relative(root, complete_path),
        "run_complete_sha256": sha256_file(complete_path),
        "execution_identity_hash": run_hash,
        "final_adapter_path": str(complete["final_adapter_path"]),
        "final_adapter_identity": str(complete["final_adapter_identity"]),
        "optimizer_steps": 32,
        "rollout_groups": 64,
        "rollouts": 512,
        "train_runtime_seconds": float(
            _mapping(complete["train_result"], "train result")["train_runtime"]
        ),
        "checkpoints": checkpoints,
    }


def _read_jsonl(path: Path, name: str) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise ValueError(f"{name} is missing: {path}")
    rows: list[Mapping[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            rows.append(_mapping(json.loads(line), f"{name} row {index}"))
        except json.JSONDecodeError as error:
            raise ValueError(f"{name} row {index} is invalid JSON") from error
    return rows


def _verify_evaluation(
    root: Path,
    *,
    arm: str,
    seed: int,
    run_id: str,
    expected_model_identity: str,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    directory = root / "outputs" / run_id
    summary_path = directory / "summary.json"
    predictions_path = directory / "predictions.jsonl"
    diagnostics_path = directory / "model_diagnostics.json"
    telemetry_path = directory / "memory_telemetry.jsonl"
    summary = _load_json(summary_path, "DEVELOPMENT evaluation summary")
    if (
        summary.get("status") != "COMPLETE"
        or summary.get("run_id") != run_id
        or summary.get("scientific_state") != "DEVELOPMENT"
        or summary.get("generation_config_id") != "qwen3-dev-greedy-v3-bs1-bounded"
        or summary.get("dataset_rows") != 36
        or summary.get("model_identity") != expected_model_identity
    ):
        raise ValueError(f"DEVELOPMENT evaluation identity mismatch: {run_id}")
    generation = _mapping(summary.get("generation"), "generation configuration")
    if generation.get("do_sample") is not False or generation.get("batch_size") != 1:
        raise ValueError(f"DEVELOPMENT evaluation generation mismatch: {run_id}")
    rows = _read_jsonl(predictions_path, "DEVELOPMENT predictions")
    if len(rows) != 36:
        raise ValueError(f"DEVELOPMENT prediction count mismatch: {run_id}")
    by_id: dict[str, Mapping[str, Any]] = {}
    family_values: dict[str, list[float]] = defaultdict(list)
    scores: list[float] = []
    passes: list[bool] = []
    for row in rows:
        example_id = str(row.get("example_id", ""))
        if not example_id or example_id in by_id:
            raise ValueError(f"DEVELOPMENT example IDs are invalid or duplicated: {run_id}")
        if row.get("split_role") != "DEVELOPMENT" or row.get("run_id") != run_id:
            raise ValueError(f"DEVELOPMENT prediction role/identity mismatch: {run_id}")
        scoring = _mapping(row.get("scoring"), "prediction scoring")
        score = float(scoring["score"])
        passed = scoring["passed"]
        if not math.isfinite(score) or not 0.0 <= score <= 1.0 or type(passed) is not bool:
            raise ValueError(f"DEVELOPMENT prediction score is invalid: {run_id}")
        by_id[example_id] = row
        scores.append(score)
        passes.append(bool(passed))
        family_values[str(row["task_family"])].append(score)
    observed_mean = sum(scores) / len(scores)
    observed_pass = sum(passes) / len(passes)
    if not math.isclose(observed_mean, float(summary["mean_score"]), abs_tol=1e-12):
        raise ValueError(f"DEVELOPMENT mean-score mismatch: {run_id}")
    if not math.isclose(observed_pass, float(summary["pass_rate"]), abs_tol=1e-12):
        raise ValueError(f"DEVELOPMENT pass-rate mismatch: {run_id}")
    diagnostics = _load_json(diagnostics_path, "DEVELOPMENT model diagnostics")
    if (
        diagnostics.get("model_construction_count") != 1
        or diagnostics.get("adapter_wrapper_count") != 1
        or diagnostics.get("eval_mode") is not True
        or diagnostics.get("optimizer_constructed") is not False
        or diagnostics.get("trainable_parameter_count") != 0
        or diagnostics.get("loaded_in_4bit") is not True
    ):
        raise ValueError(f"DEVELOPMENT model diagnostics are not bounded: {run_id}")
    telemetry = _read_jsonl(telemetry_path, "DEVELOPMENT memory telemetry")
    if len(telemetry) != 36 or any(row.get("bounded_memory") is not True for row in telemetry):
        raise ValueError(f"DEVELOPMENT memory telemetry mismatch: {run_id}")
    return (
        {
            "arm": arm,
            "seed": seed,
            "run_id": run_id,
            "model_identity": expected_model_identity,
            "summary_path": _relative(root, summary_path),
            "summary_sha256": sha256_file(summary_path),
            "predictions_path": _relative(root, predictions_path),
            "predictions_sha256": sha256_file(predictions_path),
            "diagnostics_sha256": sha256_file(diagnostics_path),
            "memory_telemetry_sha256": sha256_file(telemetry_path),
            "example_count": 36,
            "mean_score": observed_mean,
            "pass_rate": observed_pass,
            "family_scores": {
                family: sum(values) / len(values)
                for family, values in sorted(family_values.items())
            },
        },
        by_id,
    )


def build_e25_completion_audit(workspace: Path) -> dict[str, Any]:
    """Verify all measured E25 work and aggregate protected-free DEVELOPMENT outcomes."""

    root = workspace.resolve()
    config = root / "configs/replication/e25_finalist_multiseed.yaml"
    design_path = root / "evidence/training/E25/EXECUTION_DESIGN.json"
    if sha256_file(config) != _CONFIG_SHA256:
        raise ValueError("E25 frozen replication configuration hash mismatch")
    design = _load_json(design_path, "E25 execution design")
    if (
        design.get("design_identity") != _DESIGN_IDENTITY
        or design.get("protocol_id") != _PROTOCOL_ID
    ):
        raise ValueError("E25 frozen execution design identity mismatch")
    frozen_runs = {
        (str(item["arm"]), int(item["seed"])): item
        for item in cast(Sequence[Mapping[str, Any]], design["new_runs"])
    }
    expected_keys = {(arm, seed) for arm in ("conventional", "rcac") for seed in (1702, 1703)}
    if set(frozen_runs) != expected_keys:
        raise ValueError("E25 frozen new-run matrix mismatch")

    training_runs = [
        _verify_new_run(root, frozen_runs[(arm, seed)], arm=arm, seed=seed)
        for seed in (1702, 1703)
        for arm in ("conventional", "rcac")
    ]
    adapter_identities = {
        (str(item["arm"]), int(item["seed"])): str(item["final_adapter_identity"])
        for item in training_runs
    }

    eval_specs = {
        ("conventional", 1701): (
            "E16-terminal-uniform-grpo-dev-eval-v1",
            "Qwen3-1.7B+E16-terminal-uniform-GRPO@1b378c73",
        ),
        ("rcac", 1701): (
            "E18-rcac-grpo-dev-eval-v1",
            "Qwen3-1.7B+E18-RCAC-GRPO@b895ef3f",
        ),
        **{
            (arm, seed): (
                str(frozen_runs[(arm, seed)]["development_eval_run_id"]),
                adapter_identities[(arm, seed)],
            )
            for arm, seed in expected_keys
        },
    }
    evaluations: list[dict[str, Any]] = []
    prediction_sets: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    for seed in _SEEDS:
        for arm in ("conventional", "rcac"):
            run_id, model_identity = eval_specs[(arm, seed)]
            audit, predictions = _verify_evaluation(
                root,
                arm=arm,
                seed=seed,
                run_id=run_id,
                expected_model_identity=model_identity,
            )
            evaluations.append(audit)
            prediction_sets[(arm, seed)] = predictions

    reference_ids = set(prediction_sets[("conventional", 1701)])
    if any(set(rows) != reference_ids for rows in prediction_sets.values()):
        raise ValueError("DEVELOPMENT example identities do not match across arms and seeds")
    ordered_ids = sorted(reference_ids)
    score_differences: dict[str, list[float]] = {}
    pass_differences: dict[str, list[float]] = {}
    per_seed: list[dict[str, Any]] = []
    for seed in _SEEDS:
        conventional = prediction_sets[("conventional", seed)]
        rcac = prediction_sets[("rcac", seed)]
        score_diffs = [
            float(_mapping(rcac[item]["scoring"], "RCAC scoring")["score"])
            - float(_mapping(conventional[item]["scoring"], "conventional scoring")["score"])
            for item in ordered_ids
        ]
        pass_diffs = [
            float(bool(_mapping(rcac[item]["scoring"], "RCAC scoring")["passed"]))
            - float(bool(_mapping(conventional[item]["scoring"], "conventional scoring")["passed"]))
            for item in ordered_ids
        ]
        key = str(seed)
        score_differences[key] = score_diffs
        pass_differences[key] = pass_diffs
        conventional_eval = next(
            item for item in evaluations if item["arm"] == "conventional" and item["seed"] == seed
        )
        rcac_eval = next(
            item for item in evaluations if item["arm"] == "rcac" and item["seed"] == seed
        )
        per_seed.append(
            {
                "seed": seed,
                "example_count": 36,
                "conventional_mean_score": conventional_eval["mean_score"],
                "rcac_mean_score": rcac_eval["mean_score"],
                "mean_score_difference_rcac_minus_conventional": sum(score_diffs) / 36,
                "conventional_pass_rate": conventional_eval["pass_rate"],
                "rcac_pass_rate": rcac_eval["pass_rate"],
                "pass_rate_difference_rcac_minus_conventional": sum(pass_diffs) / 36,
            }
        )
    unit_ids = {str(seed): ordered_ids for seed in _SEEDS}
    score_summary = seed_aware_summary(
        score_differences,
        unit_ids_by_seed=unit_ids,
        bootstrap_replicates=20_000,
        seed=2525,
    ).to_dict()
    pass_summary = seed_aware_summary(
        pass_differences,
        unit_ids_by_seed=unit_ids,
        bootstrap_replicates=20_000,
        seed=2526,
    ).to_dict()
    for result in (score_summary, pass_summary):
        result["qualifier"] = (
            "Hierarchical bootstrap over the three observed training seeds and 36 unique "
            "DEVELOPMENT task units. Every seed is reported explicitly; three seeds do not "
            "support high-powered population inference. Exact DEVELOPMENT unit identities "
            "matched across arms and seeds."
        )
    signs = [float(item["mean_score_difference_rcac_minus_conventional"]) for item in per_seed]
    conclusion = (
        "MIXED_DEVELOPMENT_RESULT_NO_STABILITY_OR_SUPERIORITY_CLAIM"
        if any(value < 0 for value in signs) and any(value > 0 for value in signs)
        else "DIRECTIONALLY_CONSISTENT_DEVELOPMENT_RESULT"
    )
    return {
        "schema_version": 1,
        "experiment_id": "E25",
        "protocol_id": _PROTOCOL_ID,
        "scientific_state": "LOCKED_FINAL",
        "scope": "TRAINING_AND_DEVELOPMENT_ONLY_NO_PROTECTED_OUTPUTS_ACCESSED",
        "status": "TRAINING_AND_DEVELOPMENT_COMPLETE_PROTECTED_REPLICATION_PENDING",
        "configuration": {"path": _relative(root, config), "sha256": sha256_file(config)},
        "execution_design": {
            "path": _relative(root, design_path),
            "sha256": sha256_file(design_path),
            "identity": _DESIGN_IDENTITY,
        },
        "training_runs": training_runs,
        "development_evaluations": evaluations,
        "development_analysis": {
            "analysis_units": (
                "36 unique DEVELOPMENT example IDs matched within each of 3 training seeds"
            ),
            "seed_count": 3,
            "per_seed": per_seed,
            "score_difference": score_summary,
            "pass_rate_difference": pass_summary,
            "conclusion": conclusion,
            "qualifier": (
                "DEVELOPMENT evidence describes local training variability only; it cannot "
                "tune frozen finalists or substitute for protected multi-seed evaluation."
            ),
        },
        "protected_accessed": False,
        "next_authorized_requirements": [
            (
                "Publish and verify an additive versioned E25 pre-protected binding for the four "
                "new adapter identities without changing CP60/CP70."
            ),
            (
                "Bind identical RFPT-PROTECTED-v1 dataset, prompt, generation, scorer, "
                "missing-output, and statistical semantics before new-seed inference."
            ),
            (
                "Generate and score new-seed protected outputs incrementally under new immutable "
                "identities; reuse seed-1701 outputs byte-for-byte."
            ),
            (
                "Aggregate every protected seed explicitly, then refresh E26 systems evidence "
                "before CP80 qualification."
            ),
        ],
        "prohibitions": [
            "NO_LOCKED_FINAL_TUNING",
            "NO_SEED1701_PROTECTED_REGENERATION",
            "NO_CP60_OR_CP70_MUTATION",
            "NO_E25_COMPLETE_CLAIM_BEFORE_PROTECTED_REPLICATION",
        ],
    }


def _csv_bytes(audit: Mapping[str, Any]) -> bytes:
    stream = io.StringIO(newline="")
    fields = [
        "seed",
        "example_count",
        "conventional_mean_score",
        "rcac_mean_score",
        "mean_score_difference_rcac_minus_conventional",
        "conventional_pass_rate",
        "rcac_pass_rate",
        "pass_rate_difference_rcac_minus_conventional",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    analysis = _mapping(audit["development_analysis"], "development analysis")
    for row in cast(Sequence[Mapping[str, Any]], analysis["per_seed"]):
        writer.writerow({field: row[field] for field in fields})
    return stream.getvalue().encode("utf-8")


def _report_bytes(audit: Mapping[str, Any]) -> bytes:
    analysis = _mapping(audit["development_analysis"], "development analysis")
    score = _mapping(analysis["score_difference"], "score analysis")
    passed = _mapping(analysis["pass_rate_difference"], "pass-rate analysis")
    lines = [
        "# E25 multi-seed replication: training and DEVELOPMENT audit",
        "",
        f"Status: **{audit['status']}**",
        "",
        "All four new matched-seed training runs reached 32 optimizer steps, 64 rollout groups,",
        "and 512 rollouts. Their step-16 and step-32 checkpoint manifests and every inventoried",
        "payload hash were revalidated. All four 36-row bounded DEVELOPMENT evaluations were also",
        "revalidated and combined with the frozen design's imported E16/E18 seed-1701 evaluations.",
        "No protected output was read by this audit.",
        "",
        "## DEVELOPMENT results",
        "",
        (
            "| Seed | Conventional mean | RCAC mean | RCAC - conventional | "
            "Conventional pass | RCAC pass | Pass difference |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cast(Sequence[Mapping[str, Any]], analysis["per_seed"]):
        lines.append(
            f"| {row['seed']} | {float(row['conventional_mean_score']):.4f} | "
            f"{float(row['rcac_mean_score']):.4f} | "
            f"{float(row['mean_score_difference_rcac_minus_conventional']):+.4f} | "
            f"{float(row['conventional_pass_rate']):.4f} | {float(row['rcac_pass_rate']):.4f} | "
            f"{float(row['pass_rate_difference_rcac_minus_conventional']):+.4f} |"
        )
    lines.extend(
        [
            "",
            f"The equal-seed mean score difference was {float(score['mean_difference']):+.4f} "
            f"(hierarchical bootstrap 95% interval {float(score['ci_low']):+.4f} to "
            f"{float(score['ci_high']):+.4f}; 3 seeds, 108 seed-example observations). "
            "The equal-seed "
            f"pass-rate difference was {float(passed['mean_difference']):+.4f} "
            f"(95% interval {float(passed['ci_low']):+.4f} to {float(passed['ci_high']):+.4f}).",
            "",
            "RCAC trailed at seeds 1701 and 1702 and led at seed 1703. This is a mixed DEVELOPMENT",
            "result and does not establish stability or superiority. Three training seeds do not",
            "support high-powered population inference, and DEVELOPMENT evidence may not alter the",
            "frozen finalists or protected protocol.",
            "",
            "## Remaining E25 gate",
            "",
            "E25 is not yet complete. Before new-seed protected inference, publish and verify an",
            "additive E25 pre-protected binding enumerating the four new adapter identities while",
            "preserving CP60/CP70 and every RFPT-PROTECTED-v1 semantic. Reuse seed-1701 protected",
            "outputs byte-for-byte, generate and score only the four new seed models under new",
            "immutable identities, report every seed, then refresh E26 systems evidence. Only then",
            "may CP80 replication qualification be attempted.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def publish_e25_completion_audit(workspace: Path) -> dict[str, Any]:
    """Atomically publish deterministic protected-free E25 audit artifacts."""

    root = workspace.resolve()
    audit = build_e25_completion_audit(root)
    evidence = root / "evidence/training/E25/TRAINING_DEVELOPMENT_AUDIT.json"
    completion_path = root / "evidence/training/E25/TRAINING_DEVELOPMENT_COMPLETE.json"
    table = root / "reports/tables/e25_development_multiseed.csv"
    report = root / "reports/E25_MULTI_SEED_REPLICATION.md"
    atomic_write_json(evidence, audit)
    atomic_write_bytes(table, _csv_bytes(audit))
    atomic_write_bytes(report, _report_bytes(audit))
    completion: dict[str, Any] = {
        **audit,
        "published_artifacts": {
            "audit": {"path": _relative(root, evidence), "sha256": sha256_file(evidence)},
            "table": {"path": _relative(root, table), "sha256": sha256_file(table)},
            "report": {"path": _relative(root, report), "sha256": sha256_file(report)},
        },
    }
    atomic_write_json(completion_path, completion)
    return completion
