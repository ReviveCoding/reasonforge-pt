"""E29: apply the frozen decision, without recomputing protected statistics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from reasonforge.checkpoints.hashing import atomic_write_bytes, atomic_write_json
from reasonforge.checkpoints.project import ProjectCheckpointStore
from reasonforge.config import file_sha256

CP80 = "evidence/checkpoints/CP80_REPLICATION_COMPLETE"
E25 = "evidence/protected_replication/RFPT-E25-PROTECTED-REPLICATION-v1"
POLICY = "docs/FROZEN_DECISION_POLICY.md"
OUTPUT = "evidence/decisions/E29/FINAL_DECISION_v2.json"
REPORT = "reports/final/E29_FINAL_DECISION_v2.md"


class DecisionEvidenceError(ValueError):
    """A required decision input is missing, changed, or inconsistent."""


def _read(root: Path, relative: str) -> dict[str, Any]:
    value = json.loads((root / relative).read_text("utf-8"))
    if not isinstance(value, dict):
        raise DecisionEvidenceError(f"Expected JSON object: {relative}")
    return value


def _binding(root: Path, relative: str) -> dict[str, Any]:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise DecisionEvidenceError(f"Source escapes repository: {relative}")
    return {"path": relative, "sha256": file_sha256(path), "bytes": path.stat().st_size}


def verify_binding(root: Path, binding: dict[str, Any]) -> None:
    actual = _binding(root, binding["path"])
    if actual["sha256"] != binding["sha256"] or (
        "bytes" in binding and actual["bytes"] != binding["bytes"]
    ):
        raise DecisionEvidenceError(f"Bound evidence changed: {binding['path']}")


def apply_frozen_decision(
    decision: dict[str, Any], *, integrity_passed: bool, recovery_qualified: bool
) -> str:
    """Accept the already-frozen E25 outcome; never retest or tune it."""
    if not integrity_passed or not recovery_qualified:
        return "BLOCK"
    if decision.get("status") != "COMPLETE":
        raise DecisionEvidenceError("E25 decision is incomplete")
    primary = decision.get("primary_pass", {})
    guardrails = decision.get("guardrail_pass", {})
    if set(primary) != {"reasonforge_reasoning", "reasonforge_tool"} or set(guardrails) != {
        "ifeval",
        "mmlu_pro_test_256",
        "reasonforge_authorization",
        "truthfulqa_mc",
    }:
        raise DecisionEvidenceError("Frozen endpoint inventory mismatch")
    if any(type(value) is not bool for value in [*primary.values(), *guardrails.values()]):
        raise DecisionEvidenceError("Decision flags must be booleans")
    outcome = decision.get("outcome")
    if not all(primary.values()):
        if outcome not in {"RETAIN_BASELINE", "ITERATE"}:
            raise DecisionEvidenceError("Outcome contradicts frozen primary gate")
    elif outcome != ("PROMOTE" if all(guardrails.values()) else "BLOCK"):
        raise DecisionEvidenceError("Outcome contradicts frozen guardrail gate")
    return str(outcome)


def _render(result: dict[str, Any]) -> str:
    lines = [
        "# E29 final scientific decision",
        "",
        f"Decision: **{result['decision']}**.",
        "",
        "E29 applies the completed E25 decision under the unchanged frozen policy. "
        "It neither regenerates outputs nor recomputes scores, confidence intervals, or tests. "
        "The retained comparator is E16 terminal-reward uniform GRPO; RCAC is E18.",
        "",
        "| Endpoint | Units per seed | RCAC minus conventional | 95% CI | Holm p |",
        "|---|---:|---:|---|---:|",
    ]
    for name, item in result["endpoints"].items():
        estimate = item["seed_aware"]
        holm = f"{item['holm_adjusted_p']:.6f}" if "holm_adjusted_p" in item else "not applicable"
        lines.append(
            f"| {name} | {item['analysis_unit_count_per_seed']} | "
            f"{estimate['mean_difference']:.6f} | "
            f"[{estimate['ci_low']:.6f}, {estimate['ci_high']:.6f}] | "
            f"{holm} |"
        )
    lines += [
        "",
        "Holm p values refer to the primary improvement family or the separate "
        "one-sided noninferiority guardrail family. The local instruction endpoint is "
        "exploratory and has no Holm-adjusted p value. These 95% intervals are recorded "
        "seed-aware descriptive intervals; the JSON separately retains the simultaneous "
        "lower bounds used for guardrail decisions.",
        "",
        "Both primary endpoints fail the frozen +0.141451 materiality/Holm rule. "
        "The completed E25 analysis reports all four relative guardrails passing. "
        "These relative comparisons do not establish adequate absolute safety or capability.",
        "",
        "All three training seeds (1701, 1702, 1703) remain individually reported below. "
        "The same 16 primary task units recur across seeds; 48 outcomes are not 48 independent "
        "tasks. The recorded hierarchical intervals describe observed seeds and units. "
        "Three seeds do not support high-powered population inference.",
        "",
        "| Endpoint | Seed | Conventional | RCAC | Difference |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in result["endpoints"].items():
        for seed in item["per_seed"]:
            values = seed["analysis"]
            baseline = values.get("baseline_rate", values.get("baseline_mean"))
            candidate = values.get("candidate_rate", values.get("candidate_mean"))
            lines.append(
                f"| {name} | {seed['seed']} | {baseline:.6f} | {candidate:.6f} | "
                f"{values['absolute_difference']:.6f} |"
            )
    lines += ["", "## Claim boundaries", ""]
    lines += [f"- {item}" for item in result["claim_boundaries"]]
    lines += [
        "",
        "E27 native SFT new-process recovery and E28 completed-step GRPO recovery "
        "are QUALIFIED. E28 preserves and excludes partial rollout groups. This qualification "
        "does not cover arbitrary mid-group restoration or untested stacks.",
        "",
        "E26 supplies measured one-GPU operational evidence. Its efficiency denominator is "
        "exclusive training wall time; device-active GPU time, power and isolated checkpoint "
        "save cost were not measured. Systems results do not cancel the primary failures.",
        "",
        "E36 bounded 4B CUDA inference passed, but C0 scientific evaluation and C1/C2 training "
        "confirmation did not execute. The frozen 1.7B starting adapter is incompatible with "
        "the 4B model and no 4B initialization rule was frozen. Its P1 disposition is "
        "BLOCKED_PROTOCOL_UNDERSPECIFIED. This is neither evidence of 4B transfer nor a "
        "measurement of compute infeasibility.",
        "",
        "This scientific decision is not a CP90/local-completion assertion. Final release "
        "artifacts, adversarial review and checkpoint gates remain separate requirements.",
        "",
        "## Bound evidence",
        "",
    ]
    lines += [f"- `{item['path']}` — SHA-256 `{item['sha256']}`" for item in result["inputs"]]
    return "\n".join(lines) + "\n"


def build_final_decision(root: Path, *, publish: bool = False) -> dict[str, Any]:
    root = root.resolve()
    store = ProjectCheckpointStore(root / "evidence/checkpoints")
    checkpoint = store.verify(root / CP80, expected_checkpoint_id="CP80_REPLICATION_COMPLETE")
    cp80 = _read(root, f"{CP80}/REPLICATION_COMPLETE_EVIDENCE_BINDINGS.json")
    for key in ("completion", "three_seed_analysis"):
        verify_binding(root, cp80["e25_replication"][key])
    verify_binding(root, cp80["e36_four_b_disposition"]["disposition"])
    completion = _read(root, f"{E25}/REPLICATION_COMPLETE.json")
    analysis = _read(root, f"{E25}/THREE_SEED_ANALYSIS.json")
    if completion["status"] != "COMPLETE" or completion["active_run_count"] != 0:
        raise DecisionEvidenceError("Replication is not durably complete")
    verify_binding(root, completion["three_seed_analysis"])
    if file_sha256(root / POLICY) != analysis["frozen_policy_sha256"]:
        raise DecisionEvidenceError("Frozen policy changed")
    if checkpoint.decision_policy_identity != analysis["frozen_policy_sha256"]:
        raise DecisionEvidenceError("Checkpoint policy identity mismatch")
    if analysis["status"] != "COMPLETE" or analysis["seeds"] != [1701, 1702, 1703]:
        raise DecisionEvidenceError("Three-seed analysis incomplete")
    qualification = _read(root, "evidence/checkpoint_qualification/QUALIFICATION_COMPLETE.json")
    qualified = qualification["status"] == "QUALIFIED"
    for value in qualification["experiments"].values():
        verify_binding(root, value)
        trial = _read(root, value["path"])
        qualified = qualified and trial["status"] == "QUALIFIED" and trial["new_process_verified"]
    systems = _read(root, "evidence/systems/E26/SYSTEMS_ANALYSIS.json")
    if systems["status"] != "ANALYSIS_ARTIFACT_GENERATED":
        raise DecisionEvidenceError("E26 analysis missing")
    for value in systems["source_bindings"]:
        verify_binding(root, value)
    for path, value in systems["outputs"].items():
        verify_binding(root, {"path": path, **value})
    disposition = _read(root, "evidence/training/E36/FOUR_B_DISPOSITION.json")
    registry = yaml.safe_load((root / "configs/EXPERIMENT_REGISTRY.yaml").read_text("utf-8"))
    statuses = {row["experiment_id"]: row.get("status") for row in registry["experiments"]}
    if any(
        statuses.get(key) != "COMPLETE" for key in ("E04", "E07", "E24", "E25", "E26", "E27", "E28")
    ):
        raise DecisionEvidenceError("E29 prerequisite registry status incomplete")
    outcome = apply_frozen_decision(
        analysis["decision"], integrity_passed=True, recovery_qualified=qualified
    )
    paths = [
        f"{CP80}/CHECKPOINT_MANIFEST.json",
        f"{CP80}/REPLICATION_COMPLETE_EVIDENCE_BINDINGS.json",
        f"{E25}/REPLICATION_COMPLETE.json",
        f"{E25}/THREE_SEED_ANALYSIS.json",
        POLICY,
        "evidence/systems/E26/SYSTEMS_ANALYSIS.json",
        "evidence/checkpoint_qualification/QUALIFICATION_COMPLETE.json",
        "evidence/checkpoint_qualification/E27_RESULT_v6.json",
        "evidence/checkpoint_qualification/E28_RESULT_v7.json",
        "evidence/training/E36/FOUR_B_DISPOSITION.json",
        "evidence/checkpoints/CP70_PROTECTED_COMPLETE/PROTECTED_COMPLETE_EVIDENCE_BINDINGS.json",
    ]
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "E29",
        "decision_id": "RFPT-E29-FINAL-DECISION-v2",
        "protocol_id": "RFPT-P0-v1",
        "status": "COMPLETE",
        "decision": outcome,
        "evidence_class": "DERIVED",
        "checkpoint_id": "CP80_REPLICATION_COMPLETE",
        "decision_application": "UNCHANGED_COMPLETED_E25_FROZEN_POLICY_OUTCOME",
        "frozen_e25_decision": analysis["decision"],
        "recovery_qualified": qualified,
        "protected_regeneration": False,
        "protected_rescoring": False,
        "statistical_recomputation": False,
        "seeds": analysis["seeds"],
        "endpoints": analysis["paired_analyses"],
        "safety": analysis["safety"],
        "strongest_conventional_baseline": "E16 terminal-reward uniform GRPO",
        "rcac_finalist": "E18 RCAC terminal-reward GRPO",
        "four_b_status": disposition["status"],
        "four_b_confirmation_available": disposition["four_b_confirmation_available"],
        "systems_availability": systems["availability"],
        "inputs": [_binding(root, path) for path in paths],
        "claim_boundaries": [
            "RCAC superiority and improved sample efficiency are not established by this study.",
            "The primary sets contain only 16 unique tasks each and are underpowered "
            "within the frozen budget.",
            "Tool and authorization success are zero in both arms at every seed; "
            "relative noninferiority is not absolute competence.",
            "MMLU-Pro is a registered 256-row slice, with 3/256 correct per arm/seed; "
            "it is not a full benchmark score or broad capability preservation claim.",
            "Degenerate empirical zero-difference intervals do not prove population equivalence.",
            "Truthfulness refers to the frozen MC1/MC2 aggregate, "
            "not general open-ended truthfulness.",
            "No human evaluation, production impact, multi-GPU scaling, "
            "or 4B scientific confirmation is claimed.",
        ],
    }
    if publish:
        for relative, content in ((REPORT, _render(result).encode()),):
            path = root / relative
            if path.exists() and path.read_bytes() != content:
                raise DecisionEvidenceError(
                    f"Refusing to overwrite published E29 report: {relative}"
                )
            atomic_write_bytes(path, content)
        result["report"] = _binding(root, REPORT)
        target = root / OUTPUT
        if target.exists() and _read(root, OUTPUT) != result:
            raise DecisionEvidenceError("Refusing to overwrite published E29 decision")
        atomic_write_json(target, result)
    return result
