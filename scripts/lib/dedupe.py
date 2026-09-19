"""Deterministic exact and near-duplicate detection without embedding dependencies."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable

from scripts.lib.io_utils import sha256_text, text_from_message, utc_now

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[:.=(){}\[\],]+")


@dataclass(frozen=True)
class Similarity:
    exact: bool
    jaccard: float
    left_tokens: int
    right_tokens: int


def normalized_tokens(text: str) -> list[str]:
    """Keep identifiers/punctuation because code similarity matters in this corpus."""
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def shingles(text: str, *, size: int = 3) -> set[tuple[str, ...]]:
    tokens = normalized_tokens(text)
    if not tokens:
        return set()
    if len(tokens) < size:
        return {tuple(tokens)}
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def similarity(left: str, right: str, *, shingle_size: int = 3) -> Similarity:
    left_normalized = " ".join(normalized_tokens(left))
    right_normalized = " ".join(normalized_tokens(right))
    if left_normalized == right_normalized:
        return Similarity(True, 1.0, len(normalized_tokens(left)), len(normalized_tokens(right)))
    left_shingles = shingles(left, size=shingle_size)
    right_shingles = shingles(right, size=shingle_size)
    if not left_shingles or not right_shingles:
        return Similarity(False, 0.0, len(normalized_tokens(left)), len(normalized_tokens(right)))
    score = len(left_shingles & right_shingles) / len(left_shingles | right_shingles)
    return Similarity(False, score, len(normalized_tokens(left)), len(normalized_tokens(right)))


def record_text(record: dict[str, Any]) -> str:
    return f"USER:\n{text_from_message(record, 'user')}\nASSISTANT:\n{text_from_message(record, 'assistant')}"


def prompt_text(record: dict[str, Any]) -> str:
    return text_from_message(record, "user")


def mark_deduplicated(
    records: Iterable[dict[str, Any]],
    *,
    threshold: float = 0.82,
    shingle_size: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Annotate candidates and return (records, duplicate audit rows).

    The corpus is expected to be in the hundreds or low thousands during early data work.
    Pairwise comparison is chosen intentionally: it is transparent, deterministic, and does
    not hide semantically important code overlap behind an opaque embedding model.
    """
    output: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    exact_index: dict[str, dict[str, Any]] = {}

    for source in records:
        record = deepcopy(source)
        candidate_text = record_text(record)
        candidate_hash = sha256_text(" ".join(normalized_tokens(candidate_text)))
        duplicate_of: str | None = None
        metric: Similarity | None = None
        reason: str | None = None
        if candidate_hash in exact_index:
            duplicate_of = str(exact_index[candidate_hash].get("record_id"))
            metric = Similarity(True, 1.0, 0, 0)
            reason = "exact_normalized_match"
        else:
            for prior in kept:
                compared = similarity(candidate_text, record_text(prior), shingle_size=shingle_size)
                if compared.jaccard >= threshold:
                    duplicate_of = str(prior.get("record_id"))
                    metric = compared
                    reason = "near_duplicate_jaccard"
                    break
        quality = record.setdefault("quality", {})
        dedupe = quality.setdefault("deduplication", {})
        dedupe.update(
            {
                "status": "duplicate" if duplicate_of else "unique",
                "duplicate_of": duplicate_of,
                "similarity": round(metric.jaccard, 6) if metric else None,
                "method": f"normalized_{shingle_size}gram_jaccard",
                "threshold": threshold,
                "checked_at": utc_now(),
            }
        )
        if duplicate_of:
            audit.append(
                {
                    "record_id": record.get("record_id"),
                    "duplicate_of": duplicate_of,
                    "reason": reason,
                    "similarity": round(metric.jaccard, 6) if metric else None,
                }
            )
        else:
            exact_index[candidate_hash] = record
            kept.append(record)
        output.append(record)
    return output, audit


def cross_split_prompt_collisions(
    train_records: Iterable[dict[str, Any]],
    evaluation_tasks: Iterable[dict[str, Any]],
    *,
    threshold: float = 0.93,
    shingle_size: int = 3,
) -> list[dict[str, Any]]:
    """Find wording-level leakage, not ordinary conceptual overlap, across train/eval."""
    collisions: list[dict[str, Any]] = []
    tasks = list(evaluation_tasks)
    for record in train_records:
        prompt = prompt_text(record)
        for task in tasks:
            evaluation_prompt = str(task.get("prompt", ""))
            compared = similarity(prompt, evaluation_prompt, shingle_size=shingle_size)
            if compared.jaccard >= threshold:
                collisions.append(
                    {
                        "record_id": record.get("record_id"),
                        "source_seed_id": record.get("source_seed_id"),
                        "evaluation_task_id": task.get("id"),
                        "similarity": round(compared.jaccard, 6),
                        "threshold": threshold,
                    }
                )
    return collisions
