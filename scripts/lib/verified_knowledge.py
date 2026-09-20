"""Structured, conservative retrieval for DukeOTR verified Roblox/Luau fast answers.

This module is deliberately data-driven: entry text, aliases, examples, source references, and
verification dates live in ``verified_knowledge/`` rather than in a growing set of prompt
conditionals. A match is only a fast-answer eligibility decision. It never writes training data,
reads hidden evaluation material into an answer, invokes a model, or replaces the existing
Builder → Reviewer/Tester → Fixer path for requests that need reasoning.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from scripts.lib.io_utils import read_json, read_jsonl


KNOWLEDGE_SCHEMA_VERSION = "1.0"
SOURCE_CATALOG_SCHEMA_VERSION = "1.0"
VALID_ENTRY_STATUSES = {"source_checked", "human_verified", "deprecated"}
VALID_DIFFICULTIES = {"beginner", "intermediate", "advanced"}
VALID_DEPTHS = ("quick", "normal", "deep")
VALID_VERSION_SENSITIVITY = {"stable", "review_before_fast_answer", "deprecated"}
VALID_SOURCE_TYPES = {"official_docs", "official_api_reference", "official_luau_docs"}
_ENTRY_ID_RE = re.compile(r"^vk-[a-z0-9-]+$")
_SOURCE_ID_RE = re.compile(r"^vk-src-[a-z0-9-]+$")
_TOKEN_RE = re.compile(r"[a-z0-9_]+")

_DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "about",
    "answer",
    "brief",
    "briefly",
    "can",
    "cases",
    "do",
    "does",
    "deep",
    "deeply",
    "detail",
    "detailed",
    "explain",
    "for",
    "hood",
    "in",
    "is",
    "it",
    "luau",
    "me",
    "mean",
    "misconceptions",
    "of",
    "one",
    "please",
    "quick",
    "quickly",
    "roblox",
    "short",
    "sentence",
    "tell",
    "the",
    "to",
    "under",
    "use",
    "what",
    "why",
    "with",
    "you",
}


def finding(code: str, message: str, severity: str = "error") -> dict[str, str]:
    """Return a stable audit finding payload."""
    return {"code": code, "message": message, "severity": severity}


def load_entries(path: str | Path) -> list[dict[str, Any]]:
    """Load data-only verified knowledge entries; no model or network call occurs."""
    return list(read_jsonl(path))


def load_source_catalog(path: str | Path) -> dict[str, Any]:
    """Load the normalized official-source catalog shared by all entries."""
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("Verified knowledge source catalog must be a JSON object")
    return value


def source_index(source_catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = source_catalog.get("sources", [])
    if not isinstance(sources, list):
        return {}
    return {str(source.get("id")): source for source in sources if isinstance(source, dict) and isinstance(source.get("id"), str)}


def _date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _host_is_allowed(url: str, allowed_source_host_suffixes: tuple[str, ...]) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    return any(parsed.hostname == suffix or parsed.hostname.endswith(f".{suffix}") for suffix in allowed_source_host_suffixes)


def validate_source_catalog(
    source_catalog: dict[str, Any], *, allowed_source_host_suffixes: tuple[str, ...]
) -> list[dict[str, str]]:
    """Validate official source metadata without pretending to fetch or re-verify it online."""
    problems: list[dict[str, str]] = []
    if source_catalog.get("schema_version") != SOURCE_CATALOG_SCHEMA_VERSION:
        problems.append(finding("source_catalog.schema", f"Expected schema_version {SOURCE_CATALOG_SCHEMA_VERSION!r}"))
    if source_catalog.get("project_name") != "DukeOTR":
        problems.append(finding("source_catalog.project", "Source catalog must identify DukeOTR"))
    if source_catalog.get("training_policy") != "not_training_data_or_auto_promotion":
        problems.append(finding("source_catalog.training_policy", "Source catalog must remain outside automatic training-data promotion"))
    if source_catalog.get("evaluation_policy") != "not_sourced_from_held_out_evaluation":
        problems.append(finding("source_catalog.evaluation_policy", "Source catalog must explicitly preserve held-out evaluation isolation"))
    rights_policy = source_catalog.get("rights_and_usage_policy")
    if not isinstance(rights_policy, str) or len(rights_policy.strip()) < 30:
        problems.append(finding("source_catalog.rights_policy", "Source catalog needs a clear rights_and_usage_policy statement"))
    sources = source_catalog.get("sources")
    if not isinstance(sources, list) or not sources:
        return [*problems, finding("source_catalog.sources", "Source catalog needs a non-empty sources list")]
    seen: set[str] = set()
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            problems.append(finding("source.object", f"Source {index} is not an object"))
            continue
        source_id = source.get("id")
        if not isinstance(source_id, str) or not _SOURCE_ID_RE.fullmatch(source_id):
            problems.append(finding("source.id", f"Source {index} id must match vk-src-lowercase-hyphenated"))
        elif source_id in seen:
            problems.append(finding("source.duplicate_id", f"Duplicate source id {source_id!r}"))
        else:
            seen.add(source_id)
        for field, minimum in (("title", 4), ("publisher", 3), ("source_version", 3), ("checked_at", 10)):
            value = source.get(field)
            if not isinstance(value, str) or len(value.strip()) < minimum:
                problems.append(finding("source.required_text", f"Source {source_id!r} needs {field!r}"))
        if _date(source.get("checked_at")) is None:
            problems.append(finding("source.checked_at", f"Source {source_id!r} checked_at must be an ISO date"))
        url = source.get("url")
        if not isinstance(url, str) or not _host_is_allowed(url, allowed_source_host_suffixes):
            problems.append(finding("source.url", f"Source {source_id!r} needs an allowed official HTTPS URL"))
        if source.get("source_type") not in VALID_SOURCE_TYPES:
            problems.append(finding("source.type", f"Source {source_id!r} has unsupported source_type"))
    return problems


def _normalise_text(value: str) -> str:
    """Normalize casing, simple contractions, and CamelCase for transparent lexical intent matching."""
    expanded = value.strip()
    expanded = re.sub(r"\bwhat['’]s\b", "what is", expanded, flags=re.IGNORECASE)
    expanded = re.sub(r"\bcan't\b", "cannot", expanded, flags=re.IGNORECASE)
    expanded = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", expanded)
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", expanded)
    return " ".join(_TOKEN_RE.findall(expanded.casefold()))


def _content_tokens(value: str, stopwords: set[str]) -> set[str]:
    return {token for token in _TOKEN_RE.findall(_normalise_text(value)) if token not in stopwords}


def _entry_source_refs(entry: dict[str, Any]) -> list[str]:
    refs = entry.get("source_refs", [])
    return [str(value) for value in refs] if isinstance(refs, list) else []


def _api_terminal(api_name: str) -> str:
    return re.split(r"[.:]", api_name)[-1]


def validate_entry(
    entry: dict[str, Any],
    *,
    sources: dict[str, dict[str, Any]],
    allowed_api_names: set[str],
    rejected_api_names: set[str],
    known_code_book_ids: set[str] | None = None,
) -> list[dict[str, str]]:
    """Validate one audited fast-answer entry with its resolved source catalog.

    The checks are intentionally structural and conservative. They can reject unknown API names
    listed by an entry, but they do not claim to execute code or prove all prose semantics.
    """
    problems: list[dict[str, str]] = []
    if entry.get("schema_version") != KNOWLEDGE_SCHEMA_VERSION:
        problems.append(finding("entry.schema", f"Expected schema_version {KNOWLEDGE_SCHEMA_VERSION!r}"))
    entry_id = entry.get("id")
    if not isinstance(entry_id, str) or not _ENTRY_ID_RE.fullmatch(entry_id):
        problems.append(finding("entry.id", "Entry id must match vk-lowercase-hyphenated"))
    if entry.get("status") not in VALID_ENTRY_STATUSES:
        problems.append(finding("entry.status", f"Entry {entry_id!r} has an invalid status"))
    for field, minimum in (("category", 3), ("canonical_name", 3)):
        value = entry.get(field)
        if not isinstance(value, str) or len(value.strip()) < minimum:
            problems.append(finding("entry.required_text", f"Entry {entry_id!r} needs {field!r}"))
    if entry.get("difficulty") not in VALID_DIFFICULTIES:
        problems.append(finding("entry.difficulty", f"Entry {entry_id!r} has invalid difficulty"))
    if entry.get("version_sensitivity") not in VALID_VERSION_SENSITIVITY:
        problems.append(finding("entry.version_sensitivity", f"Entry {entry_id!r} has invalid version_sensitivity"))
    aliases = entry.get("aliases")
    if not isinstance(aliases, list) or len(aliases) < 2 or not all(isinstance(alias, str) and len(alias.strip()) >= 3 for alias in aliases):
        problems.append(finding("entry.aliases", f"Entry {entry_id!r} needs at least two meaningful aliases"))
    elif len({_normalise_text(alias) for alias in aliases}) != len(aliases):
        problems.append(finding("entry.duplicate_alias", f"Entry {entry_id!r} has duplicate normalized aliases"))
    match_terms = entry.get("match_terms")
    if not isinstance(match_terms, list) or not match_terms or not all(isinstance(term, str) and term.strip() for term in match_terms):
        problems.append(finding("entry.match_terms", f"Entry {entry_id!r} needs non-empty match_terms"))

    depths = entry.get("fast_answer_depths")
    if not isinstance(depths, list) or not depths or not all(depth in VALID_DEPTHS for depth in depths):
        problems.append(finding("entry.fast_answer_depths", f"Entry {entry_id!r} has invalid fast_answer_depths"))
    elif len(set(depths)) != len(depths):
        problems.append(finding("entry.fast_answer_depths", f"Entry {entry_id!r} repeats a fast-answer depth"))
    explanations = entry.get("explanations")
    if not isinstance(explanations, dict):
        problems.append(finding("entry.explanations", f"Entry {entry_id!r} needs explanations object"))
    else:
        # Every curated concept records Quick/Normal/Deep explanations. ``fast_answer_depths``
        # separately controls which depths are safe to serve without adaptive reasoning.
        for depth in VALID_DEPTHS:
            text = explanations.get(depth)
            minimum = 25 if depth == "quick" else 55
            if not isinstance(text, str) or len(text.strip()) < minimum:
                problems.append(finding("entry.explanation_depth", f"Entry {entry_id!r} needs a meaningful {depth} explanation"))

    misconceptions = entry.get("common_misconceptions")
    if not isinstance(misconceptions, list) or not misconceptions or not all(isinstance(item, str) and len(item.strip()) >= 12 for item in misconceptions):
        problems.append(finding("entry.misconceptions", f"Entry {entry_id!r} needs one or more meaningful misconceptions"))

    api_names = entry.get("api_names")
    if not isinstance(api_names, list) or not api_names or not all(isinstance(name, str) and name for name in api_names):
        problems.append(finding("entry.api_names", f"Entry {entry_id!r} api_names must be a non-empty list of strings"))
        api_names = []
    elif len(set(api_names)) != len(api_names):
        problems.append(finding("entry.duplicate_api", f"Entry {entry_id!r} repeats an API name"))
    for api_name in api_names:
        if api_name in rejected_api_names:
            problems.append(finding("entry.rejected_api", f"Entry {entry_id!r} names rejected/hallucinated API {api_name!r}"))
        elif api_name not in allowed_api_names:
            problems.append(finding("entry.unsupported_api", f"Entry {entry_id!r} names API absent from the approved API allowlist: {api_name!r}"))

    refs = _entry_source_refs(entry)
    if not refs:
        problems.append(finding("entry.source_refs", f"Entry {entry_id!r} needs one or more official source_refs"))
    else:
        if len(set(refs)) != len(refs):
            problems.append(finding("entry.duplicate_source_ref", f"Entry {entry_id!r} repeats an official source reference"))
        unknown = sorted(set(refs) - set(sources))
        if unknown:
            problems.append(finding("entry.unknown_source", f"Entry {entry_id!r} references absent sources: {', '.join(unknown)}"))
    if entry.get("primary_source_ref") not in refs:
        problems.append(finding("entry.primary_source", f"Entry {entry_id!r} primary_source_ref must be in source_refs"))

    verification = entry.get("last_verification")
    if not isinstance(verification, dict):
        problems.append(finding("entry.last_verification", f"Entry {entry_id!r} needs last_verification object"))
    else:
        verified_at = _date(verification.get("verified_at"))
        review_by = _date(verification.get("review_by"))
        if verified_at is None:
            problems.append(finding("entry.verified_at", f"Entry {entry_id!r} verified_at must be an ISO date"))
        if review_by is None:
            problems.append(finding("entry.review_by", f"Entry {entry_id!r} review_by must be an ISO date"))
        elif verified_at is not None and review_by < verified_at:
            problems.append(finding("entry.review_before_verification", f"Entry {entry_id!r} review_by cannot precede verified_at"))
        if not isinstance(verification.get("verifier"), str) or len(verification["verifier"].strip()) < 3:
            problems.append(finding("entry.verifier", f"Entry {entry_id!r} needs a named verification role"))
        if verification.get("verification_status") not in {"source_checked", "human_verified"}:
            problems.append(finding("entry.verification_status", f"Entry {entry_id!r} has invalid verification status"))

    if entry.get("training_policy") != "not_training_data_or_auto_promotion":
        problems.append(finding("entry.training_policy", f"Entry {entry_id!r} must remain separate from training data"))
    if entry.get("evaluation_policy") != "not_sourced_from_held_out_evaluation":
        problems.append(finding("entry.evaluation_policy", f"Entry {entry_id!r} must explicitly preserve held-out isolation"))

    code_book_refs = entry.get("code_book_refs", [])
    if not isinstance(code_book_refs, list) or not all(isinstance(card_id, str) for card_id in code_book_refs):
        problems.append(finding("entry.code_book_refs", f"Entry {entry_id!r} code_book_refs must be a list of strings"))
    elif known_code_book_ids is not None:
        unknown_cards = sorted(set(code_book_refs) - known_code_book_ids)
        if unknown_cards:
            problems.append(finding("entry.unknown_code_book", f"Entry {entry_id!r} references absent Code Book cards: {', '.join(unknown_cards)}"))

    examples = entry.get("examples", [])
    if not isinstance(examples, list):
        problems.append(finding("entry.examples", f"Entry {entry_id!r} examples must be a list"))
    else:
        example_ids: set[str] = set()
        for index, example in enumerate(examples, start=1):
            if not isinstance(example, dict):
                problems.append(finding("example.object", f"Entry {entry_id!r} example {index} is not an object"))
                continue
            example_id = example.get("id")
            if not isinstance(example_id, str) or not example_id:
                problems.append(finding("example.id", f"Entry {entry_id!r} example {index} needs id"))
            elif example_id in example_ids:
                problems.append(finding("example.duplicate_id", f"Entry {entry_id!r} repeats example id {example_id!r}"))
            else:
                example_ids.add(example_id)
            if example.get("language") != "luau":
                problems.append(finding("example.language", f"Entry {entry_id!r} example {example_id!r} must be Luau"))
            for field, minimum in (("purpose", 12), ("code", 12)):
                value = example.get(field)
                if not isinstance(value, str) or len(value.strip()) < minimum:
                    problems.append(finding("example.required_text", f"Entry {entry_id!r} example {example_id!r} needs {field!r}"))
            example_depths = example.get("depths")
            if not isinstance(example_depths, list) or not example_depths or not all(depth in VALID_DEPTHS for depth in example_depths):
                problems.append(finding("example.depths", f"Entry {entry_id!r} example {example_id!r} needs valid depths"))
            elif isinstance(depths, list) and not set(example_depths).issubset(depths):
                problems.append(finding("example.depth_scope", f"Entry {entry_id!r} example {example_id!r} uses a depth absent from the entry"))
            example_apis = example.get("api_names")
            if not isinstance(example_apis, list) or not example_apis or not all(isinstance(name, str) for name in example_apis):
                problems.append(finding("example.api_names", f"Entry {entry_id!r} example {example_id!r} needs API names"))
                example_apis = []
            elif not set(example_apis).issubset(set(api_names)):
                problems.append(finding("example.api_scope", f"Entry {entry_id!r} example {example_id!r} names API outside entry api_names"))
            code = str(example.get("code", ""))
            for api_name in example_apis:
                terminal = _api_terminal(api_name)
                if terminal.casefold() not in code.casefold():
                    problems.append(finding("example.api_not_present", f"Entry {entry_id!r} example {example_id!r} does not visibly use {api_name!r}"))
            example_refs = example.get("source_refs")
            if not isinstance(example_refs, list) or not example_refs or not set(example_refs).issubset(set(refs)):
                problems.append(finding("example.source_refs", f"Entry {entry_id!r} example {example_id!r} needs source refs owned by its entry"))
    return problems


def entry_is_stale(entry: dict[str, Any], *, as_of: date, stale_after_days: int) -> bool:
    """Return whether a source-checked entry needs re-verification before fast use."""
    verification = entry.get("last_verification", {})
    verified_at = _date(verification.get("verified_at")) if isinstance(verification, dict) else None
    review_by = _date(verification.get("review_by")) if isinstance(verification, dict) else None
    if verified_at is None or review_by is None:
        return True
    return as_of > review_by or as_of > verified_at + timedelta(days=stale_after_days)


def _matcher_stopwords(matcher_config: dict[str, Any]) -> set[str]:
    configured = matcher_config.get("stopwords")
    if isinstance(configured, list) and all(isinstance(item, str) for item in configured):
        return {item.casefold() for item in configured}
    return set(_DEFAULT_STOPWORDS)


def _phrase_matches(prompt: str, patterns: Any) -> list[str]:
    if not isinstance(patterns, list):
        return []
    matches: list[str] = []
    for pattern in patterns:
        if not isinstance(pattern, str) or not pattern:
            continue
        try:
            if re.search(pattern, prompt, re.IGNORECASE):
                matches.append(pattern)
        except re.error:
            # Configuration validation reports malformed patterns; matching stays fail-closed.
            continue
    return matches


def _candidate_score(
    prompt_normalized: str, prompt_content: set[str], entry: dict[str, Any], stopwords: set[str]
) -> tuple[float, str | None]:
    best_score = 0.0
    best_phrase: str | None = None
    phrases = [str(entry.get("canonical_name", "")), *[str(value) for value in entry.get("aliases", [])]]
    match_terms = entry.get("match_terms", [])
    if isinstance(match_terms, list) and match_terms:
        phrases.append(" ".join(str(value) for value in match_terms))
    for phrase in phrases:
        phrase_normalized = _normalise_text(phrase)
        profile = _content_tokens(phrase, stopwords)
        if not profile:
            continue
        overlap = prompt_content & profile
        if not overlap:
            continue
        coverage = len(overlap) / len(profile)
        extras = prompt_content - profile
        extra_penalty = 0.55 * (len(extras) / max(1, len(prompt_content)))
        specificity_bonus = min(len(profile), 4) * 0.02
        exact_bonus = 0.18 if prompt_normalized == phrase_normalized else 0.0
        score = coverage - extra_penalty + specificity_bonus + exact_bonus
        if score > best_score:
            best_score = score
            best_phrase = phrase
    return min(1.0, max(0.0, best_score)), best_phrase


def _escalation_reasons(prompt_normalized: str, matcher_config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    escalation = matcher_config.get("escalation_patterns", {})
    if not isinstance(escalation, dict):
        return reasons
    for category, patterns in escalation.items():
        for pattern in _phrase_matches(prompt_normalized, patterns):
            reasons.append(f"{category}:{pattern}")
    return reasons


def choose_depth(prompt: str, entry: dict[str, Any], matcher_config: dict[str, Any]) -> tuple[str | None, str | None]:
    """Select a verified depth from request wording, refusing unsupported requested depth."""
    normalized = _normalise_text(prompt)
    if _phrase_matches(normalized, matcher_config.get("quick_depth_patterns", [])):
        requested = "quick"
    elif _phrase_matches(normalized, matcher_config.get("deep_depth_patterns", [])):
        requested = "deep"
    else:
        requested = "normal"
    available = entry.get("fast_answer_depths", [])
    if not isinstance(available, list) or requested not in available:
        return None, f"requested_depth_not_verified:{requested}"
    return requested, None


def resolve_entry_sources(entry: dict[str, Any], sources: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Return source provenance in the stable entry-declared order."""
    resolved: list[dict[str, Any]] = []
    for reference in _entry_source_refs(entry):
        source = sources.get(reference)
        if source:
            resolved.append(
                {
                    "id": source["id"],
                    "title": source["title"],
                    "publisher": source["publisher"],
                    "url": source["url"],
                    "source_type": source["source_type"],
                    "checked_at": source["checked_at"],
                    "source_version": source["source_version"],
                }
            )
    return resolved


def render_answer(entry: dict[str, Any], depth: str, sources: dict[str, dict[str, Any]]) -> str:
    """Render only an entry-approved curated depth; this performs no model rewriting."""
    approved_depths = entry.get("fast_answer_depths", [])
    if depth not in approved_depths:
        raise ValueError(f"Depth {depth!r} is not approved for a verified fast answer from {entry.get('id')!r}")
    explanations = entry["explanations"]
    lines = [str(explanations[depth]).strip()]
    if depth != "quick":
        selected_examples = [
            example
            for example in entry.get("examples", [])
            if isinstance(example, dict) and depth in example.get("depths", [])
        ]
        for example in selected_examples:
            lines.extend(["", f"**Example — {example['purpose']}**", "", "```luau", example["code"].strip(), "```"])
    if depth == "deep":
        misconceptions = entry.get("common_misconceptions", [])
        if misconceptions:
            lines.extend(["", "**Common misconception(s)**"])
            lines.extend(f"- {item}" for item in misconceptions)
    resolved = resolve_entry_sources(entry, sources)
    if resolved:
        lines.extend(["", "**Verified source(s)**"])
        lines.extend(f"- [{source['title']}]({source['url']})" for source in resolved)
    return "\n".join(lines).strip()


def match_fast_answer(
    prompt: str,
    *,
    entries: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    matcher_config: dict[str, Any],
    as_of: date | None = None,
) -> dict[str, Any]:
    """Return a verified fast answer or a reason to fall back to adaptive quality routing.

    This matcher is intentionally transparent lexical/intent matching, not an embedding or model
    classifier. It handles phrasing variants through normalized aliases and intent templates, and
    fails closed for ambiguity, multiple concepts, custom implementation, deep high-risk detail,
    version-sensitive requests, stale entries, or insufficient match confidence.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        return {"status": "fallback", "reason": "empty_or_non_text_prompt", "candidate_concept_ids": []}
    normalized = _normalise_text(prompt)
    max_characters = matcher_config.get("maximum_prompt_characters", 500)
    if not isinstance(max_characters, int) or max_characters < 1:
        raise ValueError("matcher.maximum_prompt_characters must be a positive integer")
    if len(prompt) > max_characters:
        return {"status": "fallback", "reason": "prompt_too_long_for_fast_answer", "candidate_concept_ids": []}
    escalation_reasons = _escalation_reasons(normalized, matcher_config)
    if escalation_reasons:
        return {"status": "fallback", "reason": "requires_adaptive_pipeline", "escalation_reasons": escalation_reasons, "candidate_concept_ids": []}

    intent_patterns = matcher_config.get("definition_intent_patterns", [])
    if not _phrase_matches(normalized, intent_patterns):
        return {"status": "fallback", "reason": "not_a_narrow_definition_or_explanation_request", "candidate_concept_ids": []}

    stopwords = _matcher_stopwords(matcher_config)
    prompt_content = _content_tokens(prompt, stopwords)
    max_content_tokens = matcher_config.get("maximum_content_tokens", 12)
    if not isinstance(max_content_tokens, int) or max_content_tokens < 1:
        raise ValueError("matcher.maximum_content_tokens must be a positive integer")
    if not prompt_content:
        return {"status": "fallback", "reason": "ambiguous_no_concept_terms", "candidate_concept_ids": []}
    if len(prompt_content) > max_content_tokens:
        return {"status": "fallback", "reason": "too_many_concept_terms", "candidate_concept_ids": []}

    as_of = as_of or date.today()
    stale_after_days = matcher_config.get("stale_after_days", 180)
    if not isinstance(stale_after_days, int) or stale_after_days < 1:
        raise ValueError("matcher.stale_after_days must be a positive integer")
    candidates: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("status") not in {"source_checked", "human_verified"}:
            continue
        if entry.get("version_sensitivity") != "stable":
            continue
        if entry_is_stale(entry, as_of=as_of, stale_after_days=stale_after_days):
            continue
        score, phrase = _candidate_score(normalized, prompt_content, entry, stopwords)
        if score:
            candidates.append({"entry": entry, "confidence": score, "matched_phrase": phrase})
    candidates.sort(key=lambda item: (-item["confidence"], str(item["entry"].get("id", ""))))
    minimum_confidence = matcher_config.get("minimum_confidence", 0.84)
    ambiguity_margin = matcher_config.get("ambiguity_margin", 0.08)
    if not isinstance(minimum_confidence, (float, int)) or not 0 < float(minimum_confidence) <= 1:
        raise ValueError("matcher.minimum_confidence must be a number in (0, 1]")
    if not isinstance(ambiguity_margin, (float, int)) or not 0 <= float(ambiguity_margin) < 1:
        raise ValueError("matcher.ambiguity_margin must be a number in [0, 1)")
    candidate_ids = [str(item["entry"].get("id")) for item in candidates[:5]]
    if not candidates or candidates[0]["confidence"] < float(minimum_confidence):
        return {"status": "fallback", "reason": "no_confident_verified_concept_match", "candidate_concept_ids": candidate_ids}
    if len(candidates) > 1 and candidates[1]["confidence"] >= float(minimum_confidence) and (
        candidates[0]["confidence"] - candidates[1]["confidence"] <= float(ambiguity_margin)
    ):
        return {"status": "fallback", "reason": "multiple_concepts_or_ambiguous_match", "candidate_concept_ids": candidate_ids}

    selected = candidates[0]
    entry = selected["entry"]
    depth, depth_problem = choose_depth(prompt, entry, matcher_config)
    if depth_problem:
        return {
            "status": "fallback",
            "reason": depth_problem,
            "candidate_concept_ids": candidate_ids,
            "matched_concept_id": entry.get("id"),
        }
    assert depth is not None
    return {
        "status": "matched",
        "answer_kind": "verified_knowledge_fast_answer",
        "concept_id": entry["id"],
        "canonical_name": entry["canonical_name"],
        "category": entry["category"],
        "confidence": round(float(selected["confidence"]), 4),
        "matched_phrase": selected["matched_phrase"],
        "depth": depth,
        "answer": render_answer(entry, depth, sources),
        "sources": resolve_entry_sources(entry, sources),
        "last_verification": entry["last_verification"],
        "training_policy": entry["training_policy"],
        "evaluation_policy": entry["evaluation_policy"],
        "notice": "Returned verbatim from a source-checked knowledge entry. It is not model inference, training data, or a substitute for the adaptive quality pipeline when the request needs reasoning.",
    }
