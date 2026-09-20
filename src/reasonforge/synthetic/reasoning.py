"""Programmatic reasoning tasks independent of protected benchmark templates."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class ReasoningExample:
    example_id: str
    prompt: str
    reference: str
    answer: Decimal
    task_family: str
    parameters: Mapping[str, int | str]
    generator_version: str = "rf-reasoning-v1"


class ReasoningGenerator:
    """Generate deterministic compositional arithmetic and unit tasks."""

    _NAMES = ("Ari", "Bo", "Cleo", "Devi", "Emi", "Fara")
    _OBJECTS = ("tiles", "seed packets", "cables", "sample jars")

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def _rng(self, index: int) -> random.Random:
        digest = hashlib.sha256(f"reason:{self.seed}:{index}".encode()).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def generate(self, index: int) -> ReasoningExample:
        if index < 0:
            raise ValueError("index must be non-negative")
        rng = self._rng(index)
        family = ("latent_inventory", "unit_conversion", "conditional_plan")[index % 3]
        if family == "latent_inventory":
            return self._inventory(index, rng)
        if family == "unit_conversion":
            return self._units(index, rng)
        return self._conditional(index, rng)

    def _identity(self, index: int, prompt: str) -> str:
        digest = hashlib.sha256(
            f"rf-reasoning-v1\0{self.seed}\0{index}\0{prompt}".encode()
        ).hexdigest()[:20]
        return f"syn-reason-{digest}"

    def _inventory(self, index: int, rng: random.Random) -> ReasoningExample:
        name = rng.choice(self._NAMES)
        obj = rng.choice(self._OBJECTS)
        boxes = rng.randint(3, 9)
        per_box = rng.randint(4, 12)
        donated = rng.randint(2, per_box + 3)
        damaged = rng.randint(1, 4)
        irrelevant = rng.randint(10, 40)
        answer = boxes * per_box + donated - damaged
        facts = [
            f"{name} inventories {boxes} sealed boxes with {per_box} {obj} in each box.",
            f"A volunteer donates {donated} more {obj}.",
            f"Quality control removes {damaged} damaged {obj}.",
            f"The storeroom is {irrelevant} square metres; this does not affect inventory.",
        ]
        rng.shuffle(facts)
        prompt = " ".join(facts) + f" How many usable {obj} remain?"
        reference = f"{boxes}*{per_box}+{donated}-{damaged}={answer}."
        return ReasoningExample(
            self._identity(index, prompt),
            prompt,
            reference,
            Decimal(answer),
            task_family="latent_inventory",
            parameters={"boxes": boxes, "per_box": per_box, "donated": donated, "damaged": damaged},
        )

    def _units(self, index: int, rng: random.Random) -> ReasoningExample:
        length_m = rng.randint(4, 18)
        pieces = rng.randint(2, 5)
        used_cm = rng.randint(20, 90)
        total_cm = length_m * 100 * pieces
        answer = total_cm - used_cm
        prompt = (
            f"A workshop has {pieces} equal cable reels, each {length_m} metres long. "
            f"It uses {used_cm} centimetres from the combined cable. "
            "Convert units and report the remaining combined length in centimetres."
        )
        reference = f"{pieces}*{length_m}*100-{used_cm}={answer} cm."
        return ReasoningExample(
            self._identity(index, prompt),
            prompt,
            reference,
            Decimal(answer),
            task_family="unit_conversion",
            parameters={"metres_per_reel": length_m, "reels": pieces, "used_cm": used_cm},
        )

    def _conditional(self, index: int, rng: random.Random) -> ReasoningExample:
        attendees = rng.randint(17, 57)
        capacity = rng.randint(4, 9)
        reserve = rng.randint(1, 3)
        required = (attendees + capacity - 1) // capacity
        applied_reserve = reserve if attendees > capacity * 3 else 0
        answer = required + applied_reserve
        prompt = (
            f"An event expects {attendees} attendees. Each table seats at most {capacity}. "
            f"If attendance exceeds {capacity * 3}, add {reserve} spare table(s); "
            "otherwise add none. "
            "What is the minimum total number of tables to prepare?"
        )
        reference = f"ceil({attendees}/{capacity})+{applied_reserve}={answer}."
        return ReasoningExample(
            self._identity(index, prompt),
            prompt,
            reference,
            Decimal(answer),
            task_family="conditional_plan",
            parameters={
                "attendees": attendees,
                "capacity": capacity,
                "reserve": reserve,
                "applied_reserve": applied_reserve,
            },
        )

    def generate_many(self, count: int, *, start: int = 0) -> list[ReasoningExample]:
        if count < 0:
            raise ValueError("count must be non-negative")
        return [self.generate(index) for index in range(start, start + count)]
