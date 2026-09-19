"""Schema checks and deterministic retrieval for the source-attributed Code Book."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any
from urllib.parse import urlparse

from scripts.lib.io_utils import read_jsonl

CODE_BOOK_SCHEMA_VERSION = "1.0"
VALID_CARD_STATUSES = {"draft", "source_checked", "human_verified", "deprecated"}
VALID_CLAIM_KINDS = {"official_api", "official_security_guidance", "engineering_pattern"}
VALID_SOURCE_TYPES = {"official_docs", "official_api_reference", "official_luau_docs"}
DEFAULT_RETRIEVABLE_STATUSES = {"source_checked", "human_verified"}
_CARD_ID_RE = re.compile(r"^cb-[a-z0-9-]+$")
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def issue(code: str, message: str, severity: str = "error") -> dict[str, str]:
    return {"code": code, "message": message, "severity": severity}


def text_tokens(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(value.lower()))


def validate_card(card: dict[str, Any], *, allowed_source_host_suffixes: tuple[str, ...] = ("create.roblox.com",)) -> list[dict[str, str]]:
    """Return structural/provenance findings without pretending to prove API truth."""
    problems: list[dict[str, str]] = []
    if card.get("schema_version") != CODE_BOOK_SCHEMA_VERSION:
        problems.append(issue("card.schema", f"Expected schema_version {CODE_BOOK_SCHEMA_VERSION!r}"))
    card_id = card.get("id")
    if not isinstance(card_id, str) or not _CARD_ID_RE.fullmatch(card_id):
        problems.append(issue("card.id", "Card id must match cb-lowercase-hyphenated"))
    if not isinstance(card.get("revision"), int) or card.get("revision", 0) < 1:
        problems.append(issue("card.revision", "Card revision must be an integer >= 1"))
    status = card.get("status")
    if status not in VALID_CARD_STATUSES:
        problems.append(issue("card.status", f"Unknown card status: {status!r}"))
    for field, minimum in (("title", 8), ("summary", 40), ("created_at", 1), ("last_source_check", 1)):
        value = card.get(field)
        if not isinstance(value, str) or len(value.strip()) < minimum:
            problems.append(issue("card.required_text", f"Card requires {field!r} with at least {minimum} characters"))
    for field, minimum in (("domains", 1), ("keywords", 2), ("pitfalls", 1), ("reviewer_checks", 1)):
        value = card.get(field)
        if not isinstance(value, list) or len(value) < minimum or not all(isinstance(item, str) and item.strip() for item in value):
            problems.append(issue("card.list", f"Card {field!r} must contain at least {minimum} non-empty strings"))
    if card.get("training_policy") != "manual_curation_only":
        problems.append(issue("card.training_policy", "Cards must be manual_curation_only; do not auto-inject cards into SFT"))

    patterns = card.get("patterns")
    pattern_ids: set[str] = set()
    if not isinstance(patterns, list) or not patterns:
        problems.append(issue("card.patterns", "Card requires at least one explicit implementation pattern"))
    else:
        for index, pattern in enumerate(patterns):
            if not isinstance(pattern, dict):
                problems.append(issue("pattern.object", f"Pattern {index} is not an object"))
                continue
            pattern_id = pattern.get("id")
            if not isinstance(pattern_id, str) or not pattern_id.strip():
                problems.append(issue("pattern.id", f"Pattern {index} needs a non-empty id"))
            elif pattern_id in pattern_ids:
                problems.append(issue("pattern.duplicate_id", f"Duplicate pattern id {pattern_id!r}"))
            else:
                pattern_ids.add(pattern_id)
            if not isinstance(pattern.get("name"), str) or len(pattern["name"].strip()) < 5:
                problems.append(issue("pattern.name", f"Pattern {pattern_id!r} needs a meaningful name"))
            steps = pattern.get("steps")
            if not isinstance(steps, list) or not steps or not all(isinstance(step, str) and len(step.strip()) >= 5 for step in steps):
                problems.append(issue("pattern.steps", f"Pattern {pattern_id!r} needs one or more meaningful steps"))

    sources = card.get("sources")
    source_ids: set[str] = set()
    if not isinstance(sources, list) or not sources:
        problems.append(issue("card.sources", "Card requires at least one source"))
    else:
        for index, source in enumerate(sources):
            if not isinstance(source, dict):
                problems.append(issue("source.object", f"Source {index} is not an object"))
                continue
            source_id = source.get("id")
            if not isinstance(source_id, str) or not source_id:
                problems.append(issue("source.id", f"Source {index} needs a non-empty id"))
            elif source_id in source_ids:
                problems.append(issue("source.duplicate_id", f"Duplicate source id {source_id!r}"))
            else:
                source_ids.add(source_id)
            if not isinstance(source.get("publisher"), str) or not source["publisher"].strip():
                problems.append(issue("source.publisher", f"Source {source_id!r} needs publisher"))
            url = source.get("url")
            parsed = urlparse(url) if isinstance(url, str) else None
            if not parsed or parsed.scheme != "https" or not parsed.hostname:
                problems.append(issue("source.url", f"Source {source_id!r} requires an HTTPS URL"))
            elif allowed_source_host_suffixes and not any(
                parsed.hostname == suffix or parsed.hostname.endswith(f".{suffix}") for suffix in allowed_source_host_suffixes
            ):
                problems.append(issue("source.host", f"Source {source_id!r} host {parsed.hostname!r} violates source policy"))
            if source.get("source_type") not in VALID_SOURCE_TYPES:
                problems.append(issue("source.type", f"Source {source_id!r} has unsupported source_type"))
            if not isinstance(source.get("checked_at"), str) or not source["checked_at"].strip():
                problems.append(issue("source.checked_at", f"Source {source_id!r} needs checked_at"))

    claims = card.get("claims")
    claim_ids: set[str] = set()
    if not isinstance(claims, list) or not claims:
        problems.append(issue("card.claims", "Card requires at least one claim"))
    else:
        for index, claim in enumerate(claims):
            if not isinstance(claim, dict):
                problems.append(issue("claim.object", f"Claim {index} is not an object"))
                continue
            claim_id = claim.get("id")
            if not isinstance(claim_id, str) or not claim_id:
                problems.append(issue("claim.id", f"Claim {index} needs id"))
            elif claim_id in claim_ids:
                problems.append(issue("claim.duplicate_id", f"Duplicate claim id {claim_id!r}"))
            else:
                claim_ids.add(claim_id)
            statement = claim.get("statement")
            if not isinstance(statement, str) or len(statement.strip()) < 15:
                problems.append(issue("claim.statement", f"Claim {claim_id!r} needs a meaningful statement"))
            if claim.get("claim_kind") not in VALID_CLAIM_KINDS:
                problems.append(issue("claim.kind", f"Claim {claim_id!r} has unsupported claim_kind"))
            refs = claim.get("source_refs")
            if not isinstance(refs, list) or not refs or not all(isinstance(reference, str) for reference in refs):
                problems.append(issue("claim.source_refs", f"Claim {claim_id!r} needs source_refs"))
            elif unknown := sorted(set(refs) - source_ids):
                problems.append(issue("claim.unknown_source", f"Claim {claim_id!r} references absent sources: {', '.join(unknown)}"))
            caveats = claim.get("caveats")
            if not isinstance(caveats, list) or not all(isinstance(caveat, str) for caveat in caveats):
                problems.append(issue("claim.caveats", f"Claim {claim_id!r} caveats must be a list of strings"))

    if status == "human_verified":
        verification = card.get("human_verification")
        if not isinstance(verification, dict) or not isinstance(verification.get("reviewer"), str) or not isinstance(verification.get("verified_at"), str):
            problems.append(issue("card.human_verification", "human_verified card needs reviewer and verified_at"))
    elif card.get("human_verification") is not None:
        problems.append(issue("card.human_verification_status", "Only human_verified cards may contain human_verification", "warning"))
    return problems


def load_cards(path: str) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def catalog_summary(cards: list[dict[str, Any]]) -> dict[str, Any]:
    domains: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    claim_kinds: Counter[str] = Counter()
    for card in cards:
        statuses[str(card.get("status"))] += 1
        domains.update(str(value) for value in card.get("domains", []))
        claim_kinds.update(str(claim.get("claim_kind")) for claim in card.get("claims", []) if isinstance(claim, dict))
    return {
        "card_count": len(cards),
        "status_counts": dict(sorted(statuses.items())),
        "domain_counts": dict(sorted(domains.items())),
        "claim_kind_counts": dict(sorted(claim_kinds.items())),
    }


def search_cards(
    cards: list[dict[str, Any]], query: str, *, allowed_statuses: set[str] | None = None, limit: int = 5
) -> list[dict[str, Any]]:
    """Return transparent lexical matches, preserving cards and source attribution."""
    if limit < 1:
        raise ValueError("limit must be at least one")
    allowed = allowed_statuses or DEFAULT_RETRIEVABLE_STATUSES
    query_tokens = text_tokens(query)
    if not query_tokens:
        raise ValueError("query must include searchable letters or numbers")
    matches: list[dict[str, Any]] = []
    for card in cards:
        if card.get("status") not in allowed:
            continue
        title_tokens = text_tokens(str(card.get("title", "")))
        keyword_tokens = text_tokens(" ".join(str(value) for value in card.get("keywords", [])))
        domain_tokens = text_tokens(" ".join(str(value) for value in card.get("domains", [])))
        summary_tokens = text_tokens(str(card.get("summary", "")))
        claim_tokens = text_tokens(" ".join(str(claim.get("statement", "")) for claim in card.get("claims", []) if isinstance(claim, dict)))
        matched = query_tokens & (title_tokens | keyword_tokens | domain_tokens | summary_tokens | claim_tokens)
        if not matched:
            continue
        score = (
            4 * len(query_tokens & keyword_tokens)
            + 3 * len(query_tokens & title_tokens)
            + 2 * len(query_tokens & domain_tokens)
            + len(query_tokens & summary_tokens)
            + len(query_tokens & claim_tokens)
        )
        matches.append({"score": score, "matched_terms": sorted(matched), "card": card})
    return sorted(matches, key=lambda item: (-item["score"], item["card"].get("id", "")))[:limit]


def context_view(match: dict[str, Any]) -> dict[str, Any]:
    """Return a deliberately bounded, source-preserving card view for future roles."""
    card = match["card"]
    return {
        "id": card["id"],
        "revision": card["revision"],
        "status": card["status"],
        "title": card["title"],
        "summary": card["summary"],
        "matched_terms": match["matched_terms"],
        "claims": card["claims"],
        "patterns": card["patterns"],
        "pitfalls": card["pitfalls"],
        "reviewer_checks": card["reviewer_checks"],
        "sources": card["sources"],
        "notice": "Reference context only. Verify current APIs and game-specific assumptions before implementation.",
    }
