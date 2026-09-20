"""Top-level runtime request routing for DukeOTR.

The existing adaptive Builder → Reviewer/Tester → Fixer implementation remains the fallback for
all requests that are not a narrow, source-checked fast answer. This module only decides whether
to return an unmodified verified entry or hand off to that existing adaptive effort router; it
does not run a model or generate a candidate itself.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from scripts.lib.effort_routing import classify_task
from scripts.lib.verified_knowledge import load_entries, load_source_catalog, match_fast_answer, source_index


_CODE_OR_IMPLEMENTATION_RE = re.compile(
    r"\b(?:write|build|implement|create|make|debug|fix|refactor|design|architect|optimi[sz]e|code)\b", re.IGNORECASE
)


def _fallback_seed(prompt: str) -> dict[str, Any]:
    """Build a conservative ephemeral brief for the already-implemented effort classifier.

    A runtime prompt has no project-authored seed metadata. The fallback intentionally defaults
    to intermediate rather than the simple fast path; only the verified knowledge layer may
    return an answer without Builder/Reviewer/Fixer work.
    """
    task_type = "code_generation" if _CODE_OR_IMPLEMENTATION_RE.search(prompt) else "question_answer"
    return {
        "id": "runtime-unverified-request",
        "title": "Runtime request requiring adaptive quality routing",
        "user_request": prompt,
        "task_type": task_type,
        "difficulty": "intermediate",
        "concepts": [],
        "tags": ["runtime_request"],
        "requirements": [],
        "expected_evidence": [],
    }


def route_user_request(
    prompt: str,
    *,
    effort_config: dict[str, Any],
    knowledge_config: dict[str, Any],
    entries: list[dict[str, Any]] | None = None,
    sources: dict[str, dict[str, Any]] | None = None,
    as_of: date | None = None,
) -> dict[str, Any]:
    """Choose verified knowledge or the existing adaptive route without invoking either role.

    Callers that receive ``adaptive_pipeline`` are responsible for handing the request to the
    existing Builder → Reviewer/Tester → Fixer runtime. This separation prevents the verified
    library from becoming an unreviewed code-generation shortcut.
    """
    if not isinstance(knowledge_config, dict):
        raise ValueError("Verified knowledge configuration must be an object")
    matcher_config = knowledge_config.get("matcher")
    if not isinstance(matcher_config, dict):
        raise ValueError("Verified knowledge configuration requires matcher object")
    if entries is None:
        path = knowledge_config.get("entries_file")
        if not isinstance(path, str) or not path:
            raise ValueError("Verified knowledge configuration requires entries_file")
        entries = load_entries(path)
    if sources is None:
        path = knowledge_config.get("sources_file")
        if not isinstance(path, str) or not path:
            raise ValueError("Verified knowledge configuration requires sources_file")
        sources = source_index(load_source_catalog(path))

    fast_answer = match_fast_answer(
        prompt,
        entries=entries,
        sources=sources,
        matcher_config=matcher_config,
        as_of=as_of,
    )
    if fast_answer.get("status") == "matched":
        return {
            "route": "verified_knowledge_fast_answer",
            "fast_answer": fast_answer,
            "adaptive_route": None,
            "notice": "A source-checked entry answered this narrow request without Builder/Reviewer/Fixer inference. The entry remains separate from training data.",
        }

    adaptive_route = classify_task(_fallback_seed(prompt), effort_config)
    return {
        "route": "adaptive_pipeline",
        "fast_answer": fast_answer,
        "adaptive_route": adaptive_route.as_dict(),
        "notice": "No verified fast answer was returned. Continue through the existing adaptive Builder → Reviewer/Tester → Fixer path; this router did not invoke a model or create training data.",
    }
