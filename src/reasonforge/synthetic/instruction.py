"""Independent, deterministic instruction-following task generation.

The generator emits both the natural-language task and a machine-readable
constraint specification.  Downstream code should score the specification,
not attempt to recover constraints from the prompt text.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ConstraintKind(StrEnum):
    EXACT_SECTIONS = "exact_sections"
    EXACT_BULLETS = "exact_bullets"
    REQUIRED_PHRASE = "required_phrase"
    FORBIDDEN_PHRASE = "forbidden_phrase"
    WORD_RANGE = "word_range"
    SENTENCE_COUNT = "sentence_count"
    JSON_SCHEMA = "json_schema"
    ORDERING = "ordering"
    CONDITIONAL = "conditional"
    CROSS_REFERENCE = "cross_reference"


@dataclass(frozen=True)
class Constraint:
    kind: ConstraintKind
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class InstructionExample:
    example_id: str
    prompt: str
    constraints: tuple[Constraint, ...]
    task_family: str = "synthetic_instruction"
    generator_version: str = "rf-instruction-v1"

    @property
    def difficulty(self) -> int:
        """Structural proxy only; model-outcome difficulty belongs downstream."""

        return len(self.constraints)


class InstructionGenerator:
    """Generate reproducible multi-constraint writing tasks.

    Calls are index-addressed: ``generate(5)`` is identical regardless of which
    other indices were generated before it.  This makes interrupted generation
    safe to resume without depending on a mutable PRNG cursor.
    """

    _TOPICS = (
        "a neighborhood seed library",
        "a museum's quiet-hour program",
        "a bicycle repair workshop",
        "a community rain-garden project",
        "a school astronomy evening",
        "a local food-waste audit",
    )
    _REQUIRED = ("next step", "evidence", "trade-off", "verification")
    _FORBIDDEN = ("obviously", "guaranteed", "revolutionary", "perfect")

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def _rng(self, index: int) -> random.Random:
        digest = hashlib.sha256(f"{self.seed}:{index}".encode()).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def generate(self, index: int, *, constraint_count: int = 4) -> InstructionExample:
        if index < 0:
            raise ValueError("index must be non-negative")
        if not 2 <= constraint_count <= 9:
            raise ValueError("constraint_count must be in [2, 9]")
        rng = self._rng(index)
        topic = rng.choice(self._TOPICS)
        required = rng.choice(self._REQUIRED)
        forbidden = rng.choice(self._FORBIDDEN)
        bullet_count = rng.randint(3, 5)
        minimum_words = rng.choice((45, 55, 65))
        maximum_words = minimum_words + rng.choice((25, 35))

        candidates = [
            Constraint(ConstraintKind.EXACT_SECTIONS, {"headings": ["Context", "Actions"]}),
            Constraint(ConstraintKind.EXACT_BULLETS, {"count": bullet_count}),
            Constraint(
                ConstraintKind.REQUIRED_PHRASE, {"phrase": required, "case_sensitive": False}
            ),
            Constraint(
                ConstraintKind.FORBIDDEN_PHRASE, {"phrase": forbidden, "case_sensitive": False}
            ),
            Constraint(
                ConstraintKind.WORD_RANGE, {"minimum": minimum_words, "maximum": maximum_words}
            ),
            Constraint(ConstraintKind.SENTENCE_COUNT, {"count": bullet_count}),
            Constraint(
                ConstraintKind.CONDITIONAL,
                {"if_phrase": required, "then_required": "verification"},
            ),
            Constraint(ConstraintKind.ORDERING, {"items": ["Context", "Actions"]}),
            Constraint(
                ConstraintKind.CROSS_REFERENCE,
                {"source_section": "Context", "target_section": "Actions", "marker": "Context"},
            ),
        ]
        constraints = tuple(candidates[:constraint_count])
        constraint_clauses = [
            "Use exactly two Markdown sections headed 'Context' and 'Actions'.",
            f"Use exactly {bullet_count} bullet points in total.",
            f"Include the phrase '{required}' and do not use '{forbidden}'.",
            f"Do not use the phrase '{forbidden}'.",
            f"Keep the response between {minimum_words} and {maximum_words}, inclusive.",
            f"Use exactly {bullet_count} complete sentences.",
            f"If '{required}' appears, also include 'verification'.",
            "Put Context before Actions.",
            "In Actions, explicitly refer back to Context.",
        ]
        # Required and forbidden phrases are separate constraints and therefore
        # receive separate clauses even though they are often presented together.
        constraint_clauses[2] = f"Include the phrase '{required}'."
        prompt = " ".join(
            [f"Write a practical note about {topic}.", *constraint_clauses[:constraint_count]]
        )
        identity = hashlib.sha256(
            f"rf-instruction-v1\0{self.seed}\0{index}\0{prompt}".encode()
        ).hexdigest()[:20]
        return InstructionExample(f"syn-inst-{identity}", prompt, constraints)

    def generate_json(self, index: int) -> InstructionExample:
        """Generate a JSON-only task with a deterministic, verifiable schema."""

        if index < 0:
            raise ValueError("index must be non-negative")
        topic = self._rng(index).choice(self._TOPICS)
        schema = {
            "type": "object",
            "required": ["topic", "actions", "ready"],
            "properties": {
                "topic": {"type": "string"},
                "actions": {"type": "array"},
                "ready": {"type": "boolean"},
            },
        }
        prompt = (
            f"Return only JSON for {topic}. Include string 'topic', array 'actions', "
            "and boolean 'ready' fields; all three fields are required."
        )
        constraint = Constraint(ConstraintKind.JSON_SCHEMA, {"schema": schema})
        identity = hashlib.sha256(
            f"rf-instruction-json-v1\0{self.seed}\0{index}\0{prompt}".encode()
        ).hexdigest()[:20]
        return InstructionExample(f"syn-inst-json-{identity}", prompt, (constraint,))

    def generate_many(
        self, count: int, *, start: int = 0, constraint_count: int = 4
    ) -> list[InstructionExample]:
        if count < 0:
            raise ValueError("count must be non-negative")
        return [
            self.generate(index, constraint_count=constraint_count)
            for index in range(start, start + count)
        ]

    def replay(self, ids: Iterable[int], *, constraint_count: int = 4) -> list[InstructionExample]:
        return [self.generate(index, constraint_count=constraint_count) for index in ids]
