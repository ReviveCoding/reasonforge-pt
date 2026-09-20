"""Deterministic verification for synthetic instruction constraints."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reasonforge.synthetic.instruction import Constraint, ConstraintKind


@dataclass(frozen=True)
class InstructionVerification:
    passed: bool
    score: float
    checks: Mapping[str, bool]
    details: Mapping[str, Any]


def _sections(text: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"(?m)^#{1,6}\s+(.+?)\s*$", text))
    return [
        (
            match.group(1).strip(),
            text[
                match.end() : matches[index + 1].start() if index + 1 < len(matches) else len(text)
            ],
        )
        for index, match in enumerate(matches)
    ]


def _json_schema_ok(text: str, spec: Mapping[str, Any]) -> bool:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return False
    if spec.get("type") == "object" and not isinstance(value, dict):
        return False
    if spec.get("type") == "array" and not isinstance(value, list):
        return False
    if isinstance(value, dict):
        required = spec.get("required", [])
        if not all(key in value for key in required):
            return False
        properties = spec.get("properties", {})
        type_map: dict[str, type[Any] | tuple[type[Any], ...]] = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        for key, rules in properties.items():
            if key in value and "type" in rules:
                expected_type = type_map.get(rules["type"])
                if expected_type is None or not isinstance(value[key], expected_type):
                    return False
                if rules["type"] in {"number", "integer"} and isinstance(value[key], bool):
                    return False
    return True


def verify_instruction(text: str, constraints: Sequence[Constraint]) -> InstructionVerification:
    """Verify constraints exactly and expose every decision independently."""

    sections = _sections(text)
    headings = [heading for heading, _ in sections]
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {"headings": headings}
    for index, constraint in enumerate(constraints):
        params = constraint.parameters
        key = f"{index}:{constraint.kind.value}"
        if constraint.kind is ConstraintKind.EXACT_SECTIONS:
            checks[key] = headings == list(params["headings"])
        elif constraint.kind is ConstraintKind.EXACT_BULLETS:
            bullet_count = len(re.findall(r"(?m)^\s*[-*+]\s+\S", text))
            details["bullet_count"] = bullet_count
            checks[key] = bullet_count == int(params["count"])
        elif constraint.kind in (ConstraintKind.REQUIRED_PHRASE, ConstraintKind.FORBIDDEN_PHRASE):
            phrase = str(params["phrase"])
            haystack = text if params.get("case_sensitive", False) else text.casefold()
            needle = phrase if params.get("case_sensitive", False) else phrase.casefold()
            present = needle in haystack
            checks[key] = (
                present if constraint.kind is ConstraintKind.REQUIRED_PHRASE else not present
            )
        elif constraint.kind is ConstraintKind.WORD_RANGE:
            count = len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))
            details["word_count"] = count
            checks[key] = int(params["minimum"]) <= count <= int(params["maximum"])
        elif constraint.kind is ConstraintKind.SENTENCE_COUNT:
            count = len(re.findall(r"[^.!?\s][^.!?]*[.!?](?:\s|$)", text))
            details["sentence_count"] = count
            checks[key] = count == int(params["count"])
        elif constraint.kind is ConstraintKind.JSON_SCHEMA:
            checks[key] = _json_schema_ok(text, params["schema"])
        elif constraint.kind is ConstraintKind.ORDERING:
            items = [str(item) for item in params["items"]]
            positions = [text.find(item) for item in items]
            checks[key] = all(pos >= 0 for pos in positions) and positions == sorted(positions)
        elif constraint.kind is ConstraintKind.CONDITIONAL:
            trigger = str(params["if_phrase"])
            consequence = str(params["then_required"])
            checks[key] = trigger not in text or consequence in text
        elif constraint.kind is ConstraintKind.CROSS_REFERENCE:
            target = str(params["target_section"])
            marker = str(params["marker"])
            bodies = {heading: body for heading, body in sections}
            checks[key] = target in bodies and marker.casefold() in bodies[target].casefold()
        else:  # defensive for future enum extensions
            raise ValueError(f"unsupported constraint kind: {constraint.kind}")
    score = sum(checks.values()) / len(checks) if checks else 1.0
    return InstructionVerification(all(checks.values()), score, checks, details)
