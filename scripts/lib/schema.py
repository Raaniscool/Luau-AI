"""Versioned schemas and record constructors for the Luau-AI data pipeline.

The project uses plain dictionaries and explicit validators rather than a heavyweight
runtime dependency so the data pipeline can run on the same Windows machine as Ollama.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from scripts.lib.io_utils import canonical_json, sha256_text, utc_now

SCHEMA_VERSION = "1.0"

VALID_DIFFICULTIES = {"beginner", "intermediate", "advanced"}
VALID_TASK_TYPES = {
    "code_generation",
    "code_explanation",
    "bug_fix",
    "code_review",
    "architecture_design",
    "security_review",
    "optimization",
    "api_usage",
    "natural_language_to_luau",
}
VALID_REVIEW_DECISIONS = {"accept", "revise", "reject"}


class SchemaError(ValueError):
    pass


def issue(code: str, message: str, severity: str = "error") -> dict[str, str]:
    return {"code": code, "message": message, "severity": severity}


def validate_seed(seed: dict[str, Any]) -> list[dict[str, str]]:
    """Validate a curated task specification before asking a model to expand it."""
    problems: list[dict[str, str]] = []
    required_strings = ("id", "title", "task_type", "difficulty", "user_request")
    for key in required_strings:
        if not isinstance(seed.get(key), str) or not seed[key].strip():
            problems.append(issue("seed.required", f"Seed requires a non-empty string {key!r}"))
    if seed.get("task_type") not in VALID_TASK_TYPES:
        problems.append(issue("seed.task_type", f"Unknown task type: {seed.get('task_type')!r}"))
    if seed.get("difficulty") not in VALID_DIFFICULTIES:
        problems.append(issue("seed.difficulty", f"Unknown difficulty: {seed.get('difficulty')!r}"))
    for key in ("requirements", "concepts", "expected_evidence"):
        value = seed.get(key)
        if not isinstance(value, list) or not value or not all(isinstance(item, str) and item.strip() for item in value):
            problems.append(issue("seed.list", f"Seed {key!r} must be a non-empty list of strings"))
    if seed.get("split", "train") != "train":
        problems.append(issue("seed.split", "Generation seed split must be 'train'"))
    source = seed.get("source")
    if not isinstance(source, dict) or source.get("kind") != "project_authored":
        problems.append(issue("seed.provenance", "Seed must declare project_authored source provenance"))
    return problems


def validate_example_structure(record: dict[str, Any]) -> list[dict[str, str]]:
    """Check the canonical chat-record contract independently of content quality."""
    problems: list[dict[str, str]] = []
    if record.get("schema_version") != SCHEMA_VERSION:
        problems.append(issue("record.schema_version", f"Expected schema_version {SCHEMA_VERSION!r}"))
    for key in ("record_id", "source_seed_id"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            problems.append(issue("record.required", f"Record requires a non-empty string {key!r}"))

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        problems.append(issue("record.messages", "Record requires at least system, user, and assistant messages"))
    else:
        roles = [message.get("role") if isinstance(message, dict) else None for message in messages]
        if roles[:3] != ["system", "user", "assistant"]:
            problems.append(issue("record.message_roles", "First messages must be system, user, assistant in that order"))
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                problems.append(issue("record.message_object", f"Message {index} is not an object"))
                continue
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                problems.append(issue("record.message_content", f"Message {index} has no non-empty string content"))
            elif len(content) > 30000:
                problems.append(issue("record.message_length", f"Message {index} exceeds 30,000 characters"))
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        problems.append(issue("record.metadata", "Record requires metadata object"))
    else:
        if metadata.get("split") != "train":
            problems.append(issue("record.split", "Training candidate metadata.split must be 'train'"))
        if metadata.get("task_type") not in VALID_TASK_TYPES:
            problems.append(issue("record.task_type", "Record metadata.task_type is missing or unknown"))
        if metadata.get("difficulty") not in VALID_DIFFICULTIES:
            problems.append(issue("record.difficulty", "Record metadata.difficulty is missing or unknown"))
        topics = metadata.get("topics")
        if not isinstance(topics, list) or not topics:
            problems.append(issue("record.topics", "Record needs at least one topic"))
    quality = record.get("quality")
    if not isinstance(quality, dict):
        problems.append(issue("record.quality", "Record requires a quality object"))
    return problems


def make_generated_record(
    seed: dict[str, Any],
    assistant_response: str,
    *,
    generator: dict[str, Any],
    variant: int,
    coverage: list[str] | None = None,
    generation_notes: list[str] | None = None,
) -> dict[str, Any]:
    """Construct a canonical training candidate with full, immutable provenance."""
    response_hash = sha256_text(assistant_response)[:12]
    source_seed_id = seed["id"]
    record_id = f"gen-{source_seed_id}-v{variant}-{response_hash}"
    topics = list(dict.fromkeys([*seed.get("concepts", []), *seed.get("tags", [])]))
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": record_id,
        "source_seed_id": source_seed_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful Roblox and Luau engineering assistant. Give correct, "
                    "practical, security-aware guidance. Clearly state assumptions and never "
                    "treat a client as authoritative for server-owned game state."
                ),
            },
            {"role": "user", "content": seed["user_request"].strip()},
            {"role": "assistant", "content": assistant_response.strip()},
        ],
        "metadata": {
            "split": "train",
            "task_type": seed["task_type"],
            "difficulty": seed["difficulty"],
            "title": seed["title"],
            "topics": topics,
            "requirements": list(seed.get("requirements", [])),
            "expected_evidence": list(seed.get("expected_evidence", [])),
            "avoid": list(seed.get("avoid", [])),
            "source": deepcopy(seed.get("source", {})),
            "created_at": utc_now(),
            "stage": "generated",
            "generator": deepcopy(generator),
            "generation_variant": variant,
            "coverage_claimed": coverage or [],
            "generation_notes": generation_notes or [],
        },
        "quality": initial_quality(),
    }


def initial_quality() -> dict[str, Any]:
    return {
        "static": {"status": "not_run", "issues": [], "checked_at": None},
        "llm_review": {"status": "not_run", "decision": None, "review": None, "checked_at": None},
        "human_review": {"status": "not_requested", "decision": None, "reviewer": None, "checked_at": None},
        "deduplication": {"status": "not_run", "duplicate_of": None, "similarity": None, "checked_at": None},
    }


def clone_for_correction(record: dict[str, Any], corrected_response: str, *, correction: dict[str, Any]) -> dict[str, Any]:
    """Create a new candidate, preserving the unmodified parent record identifier."""
    clone = deepcopy(record)
    parent_id = clone["record_id"]
    correction_round = int(clone.get("metadata", {}).get("correction_round", 0)) + 1
    response_hash = sha256_text(corrected_response)[:12]
    clone["record_id"] = f"corr-{parent_id}-r{correction_round}-{response_hash}"
    clone["messages"][2]["content"] = corrected_response.strip()
    clone["metadata"]["stage"] = "corrected"
    clone["metadata"]["parent_record_id"] = parent_id
    clone["metadata"]["correction_round"] = correction_round
    clone["metadata"]["correction"] = deepcopy(correction)
    clone["metadata"]["created_at"] = utc_now()
    clone["quality"] = initial_quality()
    return clone


def record_fingerprint(record: dict[str, Any]) -> str:
    """Stable hash used in manifests; excludes mutable validation timestamps."""
    stable = {
        "source_seed_id": record.get("source_seed_id"),
        "messages": record.get("messages"),
        "task_type": record.get("metadata", {}).get("task_type"),
        "difficulty": record.get("metadata", {}).get("difficulty"),
    }
    return sha256_text(canonical_json(stable))


def reviewer_approved(record: dict[str, Any]) -> bool:
    """A human approval can substitute only for a recorded LLM review, never static checks."""
    quality = record.get("quality", {})
    llm = quality.get("llm_review", {})
    human = quality.get("human_review", {})
    return (
        (llm.get("status") == "complete" and llm.get("decision") == "accept")
        or (human.get("status") == "complete" and human.get("decision") == "accept")
    )


def quality_gate_status(record: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return final eligibility and all reasons it cannot enter final training data."""
    reasons: list[str] = []
    structure = validate_example_structure(record)
    if structure:
        reasons.extend(f"schema:{item['code']}" for item in structure)
    quality = record.get("quality", {})
    if quality.get("static", {}).get("status") != "pass":
        reasons.append("static_validation_not_passed")
    if not reviewer_approved(record):
        reasons.append("no_recorded_accepting_reviewer")
    dedupe = quality.get("deduplication", {})
    if dedupe.get("status") not in {"unique", "not_applicable"}:
        reasons.append("deduplication_not_unique")
    return not reasons, reasons
