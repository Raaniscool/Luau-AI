"""Shared held-out evaluation task and answer-record helpers."""

from __future__ import annotations

import re
from typing import Any

from scripts.lib.io_utils import canonical_json, sha256_text, utc_now


def validate_evaluation_tasks(tasks: list[dict[str, Any]]) -> None:
    ids: set[str] = set()
    errors: list[str] = []
    for task in tasks:
        task_id = task.get("id")
        if not isinstance(task_id, str) or not task_id:
            errors.append("evaluation task has no id")
            continue
        if task_id in ids:
            errors.append(f"duplicate evaluation id {task_id}")
        ids.add(task_id)
        if task.get("schema_version") != "1.0":
            errors.append(f"{task_id}: unexpected schema version")
        if task.get("split") != "evaluation":
            errors.append(f"{task_id}: split must be evaluation")
        if not isinstance(task.get("prompt"), str) or not task["prompt"].strip():
            errors.append(f"{task_id}: prompt missing")
        rubric = task.get("rubric")
        if not isinstance(rubric, list) or not rubric:
            errors.append(f"{task_id}: rubric missing")
        elif sum(item.get("max_points", 0) for item in rubric if isinstance(item, dict)) != 100:
            errors.append(f"{task_id}: rubric must total 100 points")
    if errors:
        raise ValueError("Evaluation suite invalid:\n- " + "\n- ".join(errors))


def task_selection(tasks: list[dict[str, Any]], wanted_ids: list[str], limit: int) -> list[dict[str, Any]]:
    if limit < 0:
        raise ValueError("--limit must be zero or positive")
    known = {str(task.get("id")) for task in tasks}
    unknown = sorted(set(wanted_ids) - known)
    if unknown:
        raise ValueError(f"Unknown --task-id values: {', '.join(unknown)}")
    if wanted_ids:
        selected = [task for task in tasks if task.get("id") in set(wanted_ids)]
    else:
        selected = tasks
    return selected[:limit] if limit else selected


def model_answer_record(
    *,
    run_id: str,
    run_kind: str,
    task: dict[str, Any],
    model: str,
    model_fingerprint: str | None,
    options: dict[str, Any],
    answer: str | None,
    elapsed_seconds: float | None,
    ollama_raw: dict[str, Any] | None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "run_kind": run_kind,
        "created_at": utc_now(),
        "task_id": task["id"],
        "task_category": task.get("category"),
        "task_difficulty": task.get("difficulty"),
        "prompt": task["prompt"],
        "evaluation_prompt_fingerprint": sha256_text(task["prompt"]),
        "model": model,
        "model_fingerprint": model_fingerprint,
        "generation_options": options,
        "status": "error" if error else "complete",
        "answer": answer,
        "elapsed_seconds": round(elapsed_seconds, 3) if elapsed_seconds is not None else None,
        "ollama_response_metadata": _response_metadata(ollama_raw),
        "error": error,
    }


def _response_metadata(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    """Persist reproducibility-relevant API fields without duplicating the answer body."""
    if raw is None:
        return None
    allowed = (
        "model",
        "created_at",
        "done_reason",
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    )
    return {key: raw[key] for key in allowed if key in raw}


CLIENT_CANNOT_FIRESERVER_FLAG = "Claims clients cannot initiate a client-to-server RemoteEvent request with FireServer."
REMOTE_EVENT_INHERENTLY_SECURE_FLAG = "Claims RemoteEvents are inherently or automatically secure."


def remoteevent_regression_flags(answer: str) -> list[str]:
    """Detect the two known RemoteEvent factual errors in arbitrary instructional text."""
    checks = (
        (
            (
                r"\bclients?\s+(?:cannot|can't|can\s+not|are\s+unable\s+to)\s+(?:call|use|fire|invoke)\s+(?:a\s+)?(?:remoteevent\s*:\s*)?fireserver\b",
                r"\bclients?\s+(?:cannot|can't|can\s+not|are\s+unable\s+to)\s+(?:initiate|send|make)\s+(?:a\s+)?(?:client[\s-]+to[\s-]+server\s+)?(?:remoteevent\s+)?(?:request|call)(?:\s+(?:with|using|via|by\s+calling)\s+(?:remoteevent\s*:\s*)?fireserver)?\b",
                r"\b(?:remoteevent\s*:\s*)?fireserver\s+(?:can|may|is)\s+only\s+(?:be\s+)?(?:called|used|invoked)\s+(?:by|from)\s+(?:the\s+)?server\b",
                r"\bonly\s+(?:the\s+)?server\s+(?:can|may)\s+(?:call|use|invoke)\s+(?:a\s+)?(?:remoteevent\s*:\s*)?fireserver\b",
            ),
            CLIENT_CANNOT_FIRESERVER_FLAG,
        ),
        (
            (
                r"\bremoteevents?\s+(?:are|is)\s+(?:(?:inherently|automatically|by\s+default)\s+)?(?:secure|safe)\b",
                r"\bremoteevents?\s+(?:automatically|inherently|by\s+default)\s+(?:validate|sanitize|authorize|secure|protect)\b",
                r"\bremoteevents?\s+(?:itself\s+)?(?:automatically\s+)?(?:validates?|sanitizes?|authorizes?)\s+(?:all\s+)?(?:client\s+)?(?:input|requests?)\b",
                r"\bremoteevents?\s+(?:make|keep|ensure|guarantee)\s+(?:client\s+)?(?:input|requests?)\s+(?:secure|safe|trusted|validated)\b",
            ),
            REMOTE_EVENT_INHERENTLY_SECURE_FLAG,
        ),
    )
    corrective_prefix = r"(?:incorrect|wrong|false|not\s+true|myth|misconception|mistake|avoid|do\s+not\s+say|should\s+not\s+say).{0,24}$"
    corrective_suffix = r"^\s*(?:[\"'”.)!?]\s*)*(?:is\s+|was\s+)?(?:incorrect|wrong|false|not\s+true|a\s+myth|a\s+misconception)\b"
    flags: list[str] = []
    for patterns, message in checks:
        for pattern in patterns:
            for match in re.finditer(pattern, answer, re.IGNORECASE):
                prefix = answer[max(0, match.start() - 64) : match.start()].lower()
                suffix = answer[match.end() : match.end() + 64].lower()
                if re.search(corrective_prefix, prefix) or re.search(corrective_suffix, suffix):
                    continue
                flags.append(message)
                break
            else:
                continue
            break
    return flags


def deterministic_regression_flags(task: dict[str, Any], answer: str) -> list[str]:
    """Apply known factual regression checks only to a task explicitly marked for them."""
    if not isinstance(task.get("baseline_regression_focus"), dict):
        return []
    return remoteevent_regression_flags(answer)


def score_record_fingerprint(answer_record: dict[str, Any]) -> str:
    stable = {
        "run_id": answer_record.get("run_id"),
        "task_id": answer_record.get("task_id"),
        "model": answer_record.get("model"),
        "answer": answer_record.get("answer"),
        "options": answer_record.get("generation_options"),
    }
    return sha256_text(canonical_json(stable))
