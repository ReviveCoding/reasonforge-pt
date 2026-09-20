"""Validate portable release claims and exported aggregate artifacts without GPU work."""

from __future__ import annotations

import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "evidence/release/PUBLIC_RELEASE_SUMMARY.json"
LEDGER = ROOT / "evidence/claim_ledger.json"
MAX_BYTES = 25 * 1024 * 1024
REQUIRED_TABLES = (
    "primary_seed_effects.csv",
    "pairwise_transitions_reconciled.csv",
    "robustness_results.csv",
    "uncertainty_proxy_coverage.csv",
    "uncertainty_metrics.csv",
    "routing_results_reconciled.csv",
    "systems_tail_metrics_instrumented.csv",
    "statistical_sensitivity_reconciled.csv",
    "distribution_shift_semantic_audit.csv",
)


def load_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / "reports/tables" / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate() -> list[str]:
    errors: list[str] = []
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    if (
        summary["artifact_role"] != "DERIVED_PUBLIC_PACKAGING_SUMMARY"
        or summary["original_cp110_manifest"] is not False
    ):
        errors.append("public summary identity")
    if summary["scientific_decision"] != "RETAIN_CONVENTIONAL_4B":
        errors.append("scientific decision")
    design = summary["design"]
    if (
        design["seeds"] != [1701, 1702, 1703]
        or design["locked_unique_examples"] != 150
        or design["locked_seed_example_cells"] != 450
    ):
        errors.append("locked design")
    primary = summary["primary_result"]
    if not math.isclose(primary["c2_minus_c1"], 0.008518518518518517, abs_tol=1e-12):
        errors.append("primary estimate")
    if primary["seed_stratified_95_percent_interval"] != [
        -0.006296296296296297,
        0.025555555555555557,
    ]:
        errors.append("primary interval")
    if (
        primary["frozen_promotion_threshold"] != 0.03
        or primary["promotion_gate_met"] is not False
        or primary["rcac_superiority_established"] is not False
    ):
        errors.append("promotion boundary")
    checks = summary["internal_release_validation"]
    if (
        checks["bound_artifacts"],
        checks["adversarial_checks_passed"],
        checks["adversarial_checks_total"],
        checks["tests_passed"],
        checks["expected_xfails"],
    ) != (167, 37, 37, 16, 1):
        errors.append("internal release counts")
    seeds = load_csv("primary_seed_effects.csv")
    if len(seeds) != 3 or not math.isclose(
        sum(float(row["difference"]) for row in seeds) / 3, primary["c2_minus_c1"], abs_tol=1e-12
    ):
        errors.append("seed table disagrees with primary estimate")
    for name in REQUIRED_TABLES:
        rows = load_csv(name)
        if not rows:
            errors.append(f"empty table: {name}")
        if any(
            value.strip().lower() in {"nan", "inf", "-inf", "infinity", "-infinity"}
            for row in rows
            for value in row.values()
        ):
            errors.append(f"non-finite literal: {name}")
    routing = load_csv("routing_results_reconciled.csv")
    if {row["routing_decision"] for row in routing} != {"NO_ROBUST_ROUTING_BENEFIT"}:
        errors.append("routing result")
    uncertainty = load_csv("uncertainty_metrics.csv")
    if {row["semantic_status"] for row in uncertainty} != {"UNCERTAINTY_QUALITY_ASSOCIATION"}:
        errors.append("uncertainty claim")
    if (
        ledger["scientific_decision"] != "RETAIN_CONVENTIONAL_4B"
        or ledger["rcac_superiority"] != "NOT_ESTABLISHED"
        or ledger["human_review"] != "PENDING"
    ):
        errors.append("claim ledger boundary")
    fields = {
        "claim",
        "source_artifact",
        "evidence_type",
        "statistical_status",
        "scientific_role",
        "promotion_eligible",
        "limitations",
    }
    for index, claim in enumerate(ledger["claims"]):
        if not fields <= claim.keys() or claim["promotion_eligible"] is not False:
            errors.append(f"claim fields: {index}")
        if not (ROOT / claim["source_artifact"]).is_file():
            errors.append(f"claim source missing: {index}")
    figures = list((ROOT / "reports/figures").glob("*.svg"))
    if len(figures) < 5:
        errors.append("figures missing")
    for path in figures:
        ET.parse(path)
    excluded_parts = {
        ".git",
        ".venv",
        ".venv-ci",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "dist",
        "build",
    }
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or excluded_parts.intersection(path.parts)
            or any(part.endswith(".egg-info") for part in path.parts)
        ):
            continue
        if path.stat().st_size > MAX_BYTES:
            errors.append(f"oversized file: {path.relative_to(ROOT)}")
        if path.suffix.lower() in {
            ".safetensors",
            ".pt",
            ".pth",
            ".ckpt",
            ".bin",
            ".parquet",
            ".jsonl",
        }:
            errors.append(f"runtime artifact included: {path.relative_to(ROOT)}")
        if path.suffix.lower() in {
            ".md",
            ".py",
            ".json",
            ".yaml",
            ".yml",
            ".csv",
            ".toml",
            ".cff",
            ".svg",
        }:
            content = path.read_text(encoding="utf-8")
            if (
                re.search(r"[A-Za-z]:\\(?:Users|Documents and Settings)\\", content)
                or "".join(("bjw", "-0")) in content
            ):
                errors.append(f"personal path: {path.relative_to(ROOT)}")
            if path.suffix.lower() == ".md":
                for link in re.findall(r"\]\(([^)]+)\)", content):
                    if (
                        not re.match(r"[a-z]+://", link)
                        and not (path.parent / link.split("#", 1)[0]).exists()
                    ):
                        errors.append(f"broken link: {path.relative_to(ROOT)} -> {link}")
    return errors


if __name__ == "__main__":
    found = validate()
    if found:
        raise SystemExit("\n".join(found))
    print("Public release validation passed")
