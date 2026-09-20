from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_public_release import validate  # noqa: E402


def test_exported_release_contract() -> None:
    assert validate() == []


def test_frozen_primary_is_distinct_from_diagnostics() -> None:
    summary = json.loads(
        (ROOT / "evidence/release/PUBLIC_RELEASE_SUMMARY.json").read_text(encoding="utf-8")
    )
    assert summary["primary_result"]["promotion_gate_met"] is False
    assert summary["diagnostic_findings"]["promotion_eligible"] is False
    assert summary["diagnostic_findings"]["scientific_role"] == "POST_CP100_DIAGNOSTIC_ONLY"
    assert math.isclose(summary["primary_result"]["c2_minus_c1"], 0.008518518518518517)
    assert summary["artifact_role"] == "DERIVED_PUBLIC_PACKAGING_SUMMARY"
