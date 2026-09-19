"""Deterministic and LLM-assisted quality gates for candidate training examples."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from scripts.lib.io_utils import extract_json_object, text_from_message, utc_now
from scripts.lib.ollama import OllamaClient, OllamaError
from scripts.lib.prompts import REVIEW_SYSTEM, review_prompt
from scripts.lib.schema import VALID_REVIEW_DECISIONS, issue, validate_example_structure

_CODE_FENCE_RE = re.compile(r"```(?:luau|lua)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def code_blocks(text: str) -> list[str]:
    return _CODE_FENCE_RE.findall(text)


def _finding(code: str, message: str, severity: str = "warning", *, evidence: str | None = None) -> dict[str, str]:
    value = issue(code, message, severity)
    if evidence:
        value["evidence"] = evidence[:240]
    return value


def _is_review_or_fix_task(record: dict[str, Any]) -> bool:
    return record.get("metadata", {}).get("task_type") in {"bug_fix", "code_review", "security_review"}


def static_validate(record: dict[str, Any], *, minimum_assistant_characters: int = 160) -> dict[str, Any]:
    """Run explainable heuristics; this is intentionally not a substitute for an expert review.

    A rule is marked `block` only for a clear structural/security anti-pattern. Ambiguous
    issues remain warnings so a reviewer can distinguish a deliberately bad code excerpt
    from the answer's actual recommendation.
    """
    findings: list[dict[str, str]] = []
    for schema_finding in validate_example_structure(record):
        findings.append({**schema_finding, "severity": "block"})

    answer = text_from_message(record, "assistant")
    user = text_from_message(record, "user")
    task_type = record.get("metadata", {}).get("task_type")
    if len(answer.strip()) < minimum_assistant_characters:
        findings.append(_finding("answer.too_short", "Assistant response is too short to be a useful reviewed example", "block"))
    if "<think" in answer.lower() or "</think>" in answer.lower():
        findings.append(_finding("answer.reasoning_markup", "Answer contains hidden-reasoning markup", "block"))
    if answer.count("```") % 2:
        findings.append(_finding("answer.unbalanced_fence", "Answer has an unmatched Markdown code fence", "block"))
    if re.search(r"\b(?:TODO|TBD|INSERT\s+CODE\s+HERE)\b", answer, re.IGNORECASE):
        findings.append(_finding("answer.placeholder", "Answer includes an unresolved placeholder", "warning"))

    should_contain_code = task_type in {"code_generation", "bug_fix", "natural_language_to_luau", "optimization"}
    blocks = code_blocks(answer)
    code = "\n".join(blocks)
    if should_contain_code and not blocks:
        findings.append(_finding("answer.missing_luau_code", "This task type normally needs a fenced Luau implementation", "warning"))
    if blocks and not re.search(r"\b(?:local|function|if|for|while|return|type|export)\b", code):
        findings.append(_finding("code.not_luau_like", "Fenced code has no recognizable Luau structure", "warning"))

    # Patterns that should never be taught as a production solution.
    for pattern, code_name, message in (
        (r"\bloadstring\s*\(", "security.loadstring", "Do not teach dynamic loadstring execution in Roblox game code"),
        (r"\b(?:getfenv|setfenv|getgenv|hookfunction|hookmetamethod)\s*\(", "security.exploit_api", "Answer uses exploit-oriented or unsafe environment manipulation"),
        (r"\btrust(?:ing)?\s+the\s+client\b", "security.trust_client", "Answer explicitly suggests trusting the client"),
    ):
        for match in re.finditer(pattern, answer, re.IGNORECASE):
            # A correct security explanation commonly says “never trust the client.”
            # Do not punish that negated guidance merely because it contains the phrase.
            prefix = answer[max(0, match.start() - 40) : match.start()].lower()
            if code_name == "security.trust_client" and re.search(r"(?:never|do not|don't|should not|must not)\s*$", prefix):
                continue
            severity = "warning" if _is_review_or_fix_task(record) else "block"
            findings.append(_finding(code_name, message, severity, evidence=match.group(0)))

    # Networking signatures are inspected primarily in code, so prose explaining an
    # anti-pattern does not itself trigger a false block.
    if code:
        # Parameter names vary (player, plr, sender), so only flag the unambiguous
        # no-parameter form here. The LLM reviewer assesses the first-argument semantics.
        missing_player_remote = re.search(
            r"\.OnServerEvent\s*:\s*Connect\s*\(\s*function\s*\(\s*\)", code, re.IGNORECASE
        )
        if missing_player_remote:
            severity = "warning" if _is_review_or_fix_task(record) else "block"
            findings.append(
                _finding(
                    "network.remote_missing_player",
                    "OnServerEvent handler has no Player first argument",
                    severity,
                    evidence=missing_player_remote.group(0),
                )
            )
        missing_player_invoke = re.search(r"\.OnServerInvoke\s*=\s*function\s*\(\s*\)", code, re.IGNORECASE)
        if missing_player_invoke:
            severity = "warning" if _is_review_or_fix_task(record) else "block"
            findings.append(
                _finding(
                    "network.remote_function_missing_player",
                    "OnServerInvoke handler has no Player first argument",
                    severity,
                    evidence=missing_player_invoke.group(0),
                )
            )
        for pattern, code_name, message in (
            (r"\.OnServerEvent\s*\(", "api.remoteevent_subscription", "OnServerEvent is an event; use :Connect(...)"),
            (r"\.OnClientEvent\s*\(", "api.remoteevent_client_subscription", "OnClientEvent is an event; use :Connect(...)"),
            (r"\.OnServerInvoke\s*:\s*Connect", "api.remotefunction_subscription", "OnServerInvoke is assigned a callback, not connected"),
            (r"game:GetService\(\s*[\"'](?:PlayerService|ReplicatedStorageService|WorkspaceService)[\"']\s*\)", "api.unknown_service", "Code appears to use a non-existent Roblox service name"),
        ):
            match = re.search(pattern, code, re.IGNORECASE)
            if match:
                findings.append(_finding(code_name, message, "block", evidence=match.group(0)))
        if re.search(r"while\s+true\s+do", code, re.IGNORECASE) and not re.search(r"task\.wait\s*\(", code, re.IGNORECASE):
            findings.append(_finding("performance.busy_loop", "Infinite loop has no visible task.wait yield", "block"))
        if re.search(r"\bwait\s*\(", code):
            findings.append(_finding("style.legacy_wait", "Prefer task.wait over legacy wait", "warning"))
        if re.search(r"(?:GetAsync|SetAsync|UpdateAsync|RemoveAsync)\s*\(", code) and not re.search(r"\bpcall\s*\(", code):
            findings.append(_finding("datastore.no_pcall", "DataStore operation has no visible pcall error handling", "warning"))
        client_datastore_marker = re.search(
            r"^\s*--\s*(?:LocalScript|StarterPlayerScripts|StarterGui)\b", code, re.IGNORECASE | re.MULTILINE
        )
        if re.search(r"(?:GetDataStore|OrderedDataStore)", code) and client_datastore_marker:
            findings.append(
                _finding(
                    "datastore.client_access",
                    "DataStore access appears in a code block explicitly marked client-side; it must remain server-side",
                    "block",
                    evidence=client_datastore_marker.group(0),
                )
            )
        if re.search(r"\.FireClient\s*\(\s*\)", code):
            findings.append(_finding("network.fireclient_missing_player", "FireClient needs a target Player argument", "block"))
        if re.search(r"\.InvokeServer\s*\(", code) and re.search(r"RenderStepped\s*:\s*Connect", code):
            findings.append(_finding("performance.sync_remote_per_frame", "InvokeServer may be called from a per-frame callback", "warning"))

    # Evidence checking is deliberately advisory: reviewers can recognize correct
    # implementation expressed with terminology different from the seed wording.
    lower_answer = answer.lower()
    for expected in record.get("metadata", {}).get("expected_evidence", []):
        keywords = [word.lower() for word in _WORD_RE.findall(expected) if len(word) >= 4]
        if keywords and not any(keyword in lower_answer for keyword in keywords):
            findings.append(
                _finding(
                    "coverage.unverified_expected_evidence",
                    f"Could not visibly confirm expected evidence: {expected}",
                    "warning",
                )
            )

    blockers = [entry for entry in findings if entry.get("severity") in {"block", "error"}]
    return {
        "status": "fail" if blockers else "pass",
        "issues": findings,
        "checked_at": utc_now(),
        "checker": "static-v1",
        "assistant_characters": len(answer),
        "code_block_count": len(blocks),
        "user_characters": len(user),
    }


def parse_review(raw_text: str, *, minimums: dict[str, int]) -> dict[str, Any]:
    """Parse and policy-check the independent reviewer response."""
    value = extract_json_object(raw_text)
    decision = value.get("decision")
    if decision not in VALID_REVIEW_DECISIONS:
        raise ValueError(f"Review decision must be one of {sorted(VALID_REVIEW_DECISIONS)}, got {decision!r}")
    scores = value.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("Review must contain scores object")
    normalized_scores: dict[str, int] = {}
    for key in ("accuracy", "security", "requirement_coverage", "pedagogy"):
        score = scores.get(key)
        if not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"Review score {key!r} must be integer 1..5")
        normalized_scores[key] = score
    for list_key in ("blocking_issues", "required_fixes", "strengths", "api_claims_to_verify"):
        if not isinstance(value.get(list_key, []), list) or not all(isinstance(item, str) for item in value.get(list_key, [])):
            raise ValueError(f"Review field {list_key!r} must be a list of strings")
    summary = value.get("summary", "")
    if not isinstance(summary, str):
        raise ValueError("Review summary must be a string")

    effective_decision = decision
    if decision == "accept" and value.get("blocking_issues"):
        effective_decision = "revise"
        value.setdefault("required_fixes", []).append("Resolve every reviewer-reported blocking issue before acceptance")
    if effective_decision == "accept":
        for key, minimum in minimums.items():
            if normalized_scores.get(key, 0) < minimum:
                effective_decision = "revise"
                value.setdefault("required_fixes", []).append(
                    f"Policy requires {key} >= {minimum}; reviewer gave {normalized_scores[key]}"
                )
                break
    value["reported_decision"] = decision
    value["decision"] = effective_decision
    value["scores"] = normalized_scores
    return value


def review_record(
    record: dict[str, Any],
    *,
    client: OllamaClient,
    model: str,
    options: dict[str, Any],
    minimums: dict[str, int],
) -> dict[str, Any]:
    """Ask a separate reviewer pass and retain errors as audit data instead of guessing."""
    static_issues = record.get("quality", {}).get("static", {}).get("issues", [])
    try:
        response = client.generate(
            model=model,
            system=REVIEW_SYSTEM,
            prompt=review_prompt(record, static_issues),
            options=options,
            think=False,
        )
        review = parse_review(response.content, minimums=minimums)
        return {
            "status": "complete",
            "decision": review["decision"],
            "review": review,
            "checked_at": utc_now(),
            "reviewer": {"kind": "ollama", "model": model, "elapsed_seconds": round(response.elapsed_seconds, 3)},
        }
    except (OllamaError, ValueError) as exc:
        return {
            "status": "error",
            "decision": None,
            "review": None,
            "checked_at": utc_now(),
            "reviewer": {"kind": "ollama", "model": model},
            "error": str(exc),
        }


def validate_record(
    record: dict[str, Any],
    *,
    client: OllamaClient | None,
    model: str | None,
    review_options: dict[str, Any],
    minimums: dict[str, int],
    minimum_assistant_characters: int,
) -> dict[str, Any]:
    """Return a copy enriched with static and (when available) review results."""
    candidate = deepcopy(record)
    quality = candidate.setdefault("quality", {})
    quality["static"] = static_validate(candidate, minimum_assistant_characters=minimum_assistant_characters)
    if client is None or model is None:
        quality["llm_review"] = {
            "status": "skipped",
            "decision": None,
            "review": None,
            "checked_at": utc_now(),
            "reason": "LLM review explicitly skipped; record is not final-dataset eligible",
        }
    else:
        quality["llm_review"] = review_record(
            candidate, client=client, model=model, options=review_options, minimums=minimums
        )
    return candidate


def needs_correction(record: dict[str, Any], *, include_rejected: bool = False) -> bool:
    static_status = record.get("quality", {}).get("static", {}).get("status")
    decision = record.get("quality", {}).get("llm_review", {}).get("decision")
    if static_status == "fail" or decision == "revise":
        return True
    return include_rejected and decision == "reject"
