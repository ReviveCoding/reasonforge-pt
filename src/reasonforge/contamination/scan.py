"""Exact, near, semantic-candidate, and structural overlap screening."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from reasonforge.data.schema import normalize_text


@dataclass(frozen=True, slots=True)
class ContaminationFinding:
    train_id: str
    protected_id: str
    method: str
    score: float


def _ngrams(text: str, n: int = 5) -> set[str]:
    tokens = normalize_text(text).split()
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _char_vector(text: str) -> Counter[str]:
    normalized = normalize_text(text)
    return Counter(normalized[index : index + 3] for index in range(max(0, len(normalized) - 2)))


def cosine(left: Counter[str], right: Counter[str]) -> float:
    dot = sum(value * right[key] for key, value in left.items())
    lnorm = math.sqrt(sum(value * value for value in left.values()))
    rnorm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (lnorm * rnorm) if lnorm and rnorm else float(left == right)


def structural_signature(text: str) -> str:
    normalized = normalize_text(text)
    abstracted = re.sub(r"\b\d+(?:\.\d+)?\b", "<NUM>", normalized)
    abstracted = re.sub(r"\b[a-f0-9]{8,}\b", "<ID>", abstracted)
    return hashlib.sha256(abstracted.encode("utf-8")).hexdigest()


def scan_overlap(
    train: Iterable[tuple[str, str]],
    protected: Iterable[tuple[str, str]],
    *,
    near_threshold: float = 0.80,
    semantic_candidate_threshold: float = 0.92,
) -> list[ContaminationFinding]:
    train_rows = list(train)
    protected_rows = [
        (
            protected_id,
            normalize_text(protected_text),
            _ngrams(protected_text),
            structural_signature(protected_text),
        )
        for protected_id, protected_text in protected
    ]
    semantic_scores: dict[tuple[int, int], float] = {}
    if train_rows and protected_rows:
        # This is the same normalized character-trigram cosine signal used by
        # ``cosine`` above, computed sparsely so a registered benchmark corpus
        # does not turn the audit into a quadratic Python hot loop.
        from sklearn.feature_extraction.text import HashingVectorizer

        vectorizer = HashingVectorizer(
            analyzer="char",
            ngram_range=(3, 3),
            n_features=2**18,
            alternate_sign=False,
            norm="l2",
        )
        train_matrix = vectorizer.transform(normalize_text(text) for _, text in train_rows)
        protected_matrix = vectorizer.transform(row[1] for row in protected_rows)
        similarities = (train_matrix @ protected_matrix.T).tocoo()
        semantic_scores = {
            (int(row), int(column)): float(value)
            for row, column, value in zip(
                similarities.row, similarities.col, similarities.data, strict=True
            )
            if value >= semantic_candidate_threshold
        }
    findings: list[ContaminationFinding] = []
    for train_index, (train_id, train_text) in enumerate(train_rows):
        train_norm = normalize_text(train_text)
        train_grams = _ngrams(train_text)
        train_structure = structural_signature(train_text)
        for protected_index, (
            protected_id,
            protected_norm,
            protected_grams,
            protected_structure,
        ) in enumerate(protected_rows):
            if train_norm == protected_norm:
                findings.append(ContaminationFinding(train_id, protected_id, "exact", 1.0))
                continue
            size_ratio = min(len(train_grams), len(protected_grams)) / max(
                len(train_grams), len(protected_grams), 1
            )
            near_score = (
                jaccard(train_grams, protected_grams) if size_ratio >= near_threshold else 0.0
            )
            if near_score >= near_threshold:
                findings.append(
                    ContaminationFinding(train_id, protected_id, "near_ngram", near_score)
                )
            semantic_score = semantic_scores.get((train_index, protected_index))
            if semantic_score is not None:
                findings.append(
                    ContaminationFinding(
                        train_id, protected_id, "semantic_candidate", semantic_score
                    )
                )
            if train_structure == protected_structure:
                findings.append(ContaminationFinding(train_id, protected_id, "structural", 1.0))
    return findings
