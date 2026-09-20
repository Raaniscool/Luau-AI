"""Read-only audits for DukeOTR's verified Roblox knowledge library.

The audit can inspect the held-out suite only for local wording-collision detection and a hash. It
never exposes held-out prompts in its report, sends them to a model, or turns them into knowledge
entries. Writes are guarded so the audit cannot overwrite that held-out JSONL file.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path
from typing import Any

from scripts.lib.dedupe import cross_split_prompt_collisions, similarity
from scripts.lib.io_utils import read_json, read_jsonl, sha256_file, write_json_atomic
from scripts.lib.verified_knowledge import (
    _date,
    _normalise_text,
    entry_is_stale,
    finding,
    load_entries,
    load_source_catalog,
    source_index,
    validate_entry,
    validate_source_catalog,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_EVALUATION_FILE = _REPOSITORY_ROOT / "evaluation_data" / "roblox_luau_eval.jsonl"


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _file_hash_or_none(path: str | Path) -> str | None:
    try:
        return sha256_file(path)
    except OSError:
        return None


def assert_safe_output_path(output_path: str | Path, evaluation_path: str | Path) -> None:
    """Refuse writes to the configured or canonical held-out suite (or its protected directory).

    The canonical guard remains in force even if a caller supplies a temporary ``--evaluation``
    override. That prevents an output flag from using the override to bypass permanent project
    evaluation isolation.
    """
    destination = _resolved(output_path)
    configured_evaluation = _resolved(evaluation_path)
    canonical_evaluation = _CANONICAL_EVALUATION_FILE.resolve(strict=False)
    canonical_directory = canonical_evaluation.parent
    if destination in {configured_evaluation, canonical_evaluation} or canonical_directory in destination.parents:
        raise ValueError(
            "Refusing to write to the permanently held-out evaluation file or evaluation_data directory. "
            "Choose a report/output path outside evaluation_data/."
        )


def write_audit_report(output_path: str | Path, report: dict[str, Any], *, evaluation_path: str | Path) -> None:
    assert_safe_output_path(output_path, evaluation_path)
    write_json_atomic(output_path, report)


def _config_problems(config: dict[str, Any]) -> list[dict[str, str]]:
    problems: list[dict[str, str]] = []
    if config.get("schema_version") != "1.0":
        problems.append(finding("config.schema", "Verified knowledge config must use schema_version 1.0"))
    if config.get("project_name") != "DukeOTR":
        problems.append(finding("config.project", "Verified knowledge config must identify DukeOTR"))
    if config.get("training_policy") != "not_training_data_or_auto_promotion":
        problems.append(finding("config.training_policy", "Config must prohibit automatic training-data promotion"))
    if config.get("evaluation_policy") != "not_sourced_from_held_out_evaluation":
        problems.append(finding("config.evaluation_policy", "Config must preserve held-out evaluation isolation"))
    for field in ("entries_file", "sources_file", "schema_file", "evaluation_file", "code_book_file"):
        value = config.get(field)
        if not isinstance(value, str) or not value:
            problems.append(finding("config.path", f"Config requires non-empty {field!r}"))
    hosts = config.get("allowed_source_host_suffixes")
    if not isinstance(hosts, list) or not hosts or not all(isinstance(item, str) and item for item in hosts):
        problems.append(finding("config.source_hosts", "Config needs one or more allowed official source host suffixes"))
    adaptive_fallback = config.get("adaptive_fallback")
    if not isinstance(adaptive_fallback, dict):
        problems.append(finding("config.adaptive_fallback", "Config needs an adaptive_fallback object"))
    else:
        if adaptive_fallback.get("classifier") != "scripts.lib.effort_routing.classify_task":
            problems.append(finding("config.adaptive_classifier", "adaptive_fallback must preserve the existing effort_routing.classify_task classifier"))
        if not isinstance(adaptive_fallback.get("effort_config"), str) or not adaptive_fallback["effort_config"]:
            problems.append(finding("config.adaptive_effort_config", "adaptive_fallback needs an effort_config path"))
        if adaptive_fallback.get("downstream_orchestrator") != "python -m scripts.run_builder_reviewer_fixer":
            problems.append(finding("config.adaptive_orchestrator", "adaptive_fallback must name the existing Builder/Reviewer/Tester/Fixer orchestrator"))
    matcher = config.get("matcher")
    if not isinstance(matcher, dict):
        problems.append(finding("config.matcher", "Config needs a matcher object"))
    else:
        for field in ("minimum_confidence", "ambiguity_margin", "stale_after_days", "maximum_prompt_characters", "maximum_content_tokens"):
            if field not in matcher:
                problems.append(finding("config.matcher_field", f"Matcher needs {field!r}"))
        pattern_groups: dict[str, Any] = {
            field: matcher.get(field)
            for field in ("definition_intent_patterns", "quick_depth_patterns", "deep_depth_patterns")
        }
        escalation = matcher.get("escalation_patterns")
        if not isinstance(escalation, dict) or not escalation:
            problems.append(finding("config.escalation_patterns", "Matcher needs a non-empty escalation_patterns object"))
        else:
            pattern_groups.update({f"escalation_patterns.{name}": value for name, value in escalation.items()})
        for field, patterns in pattern_groups.items():
            if not isinstance(patterns, list) or not patterns or not all(isinstance(pattern, str) and pattern for pattern in patterns):
                problems.append(finding("config.matcher_patterns", f"Matcher {field!r} must be a non-empty list of regex strings"))
                continue
            for pattern in patterns:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    problems.append(finding("config.matcher_regex", f"Matcher {field!r} has invalid regex {pattern!r}: {exc}"))
    api_names = config.get("supported_api_names")
    if not isinstance(api_names, list) or not api_names or not all(isinstance(item, str) and item for item in api_names):
        problems.append(finding("config.supported_api_names", "Config needs a non-empty supported API allowlist"))
    return problems


def _load_code_book_ids(path: str | Path) -> tuple[set[str], list[dict[str, str]]]:
    try:
        cards = list(read_jsonl(path))
    except (FileNotFoundError, ValueError) as exc:
        return set(), [finding("code_book.load", str(exc))]
    return {str(card.get("id")) for card in cards if isinstance(card.get("id"), str)}, []


def _alias_records(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build ephemeral one-message records only for local held-out collision checking."""
    records: list[dict[str, Any]] = []
    for entry in entries:
        entry_id = str(entry.get("id", "unknown"))
        aliases = [entry.get("canonical_name", ""), *entry.get("aliases", [])]
        for index, alias in enumerate(aliases):
            if not isinstance(alias, str) or not alias.strip():
                continue
            records.append(
                {
                    "record_id": f"{entry_id}:alias:{index}",
                    "source_seed_id": entry_id,
                    "messages": [{"role": "user", "content": alias}],
                }
            )
    return records


def _duplicate_findings(entries: list[dict[str, Any]], *, similarity_threshold: float) -> tuple[list[dict[str, str]], dict[str, Any]]:
    problems: list[dict[str, str]] = []
    duplicate_ids: list[str] = []
    duplicate_canonicals: list[dict[str, str]] = []
    alias_collisions: list[dict[str, Any]] = []
    near_concepts: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    canonical_index: dict[str, str] = {}
    alias_index: dict[str, tuple[str, str]] = {}
    for entry in entries:
        entry_id = str(entry.get("id", ""))
        if entry_id in seen_ids:
            duplicate_ids.append(entry_id)
            problems.append(finding("library.duplicate_id", f"Duplicate verified knowledge id {entry_id!r}"))
        seen_ids.add(entry_id)
        canonical = _normalise_text(str(entry.get("canonical_name", "")))
        if canonical:
            prior = canonical_index.get(canonical)
            if prior and prior != entry_id:
                duplicate_canonicals.append({"canonical_name": canonical, "left_id": prior, "right_id": entry_id})
                problems.append(finding("library.duplicate_canonical", f"Entries {prior!r} and {entry_id!r} duplicate canonical concept {canonical!r}"))
            canonical_index[canonical] = entry_id
        for alias in [entry.get("canonical_name", ""), *entry.get("aliases", [])]:
            if not isinstance(alias, str):
                continue
            normalized = _normalise_text(alias)
            if not normalized:
                continue
            prior = alias_index.get(normalized)
            if prior and prior[0] != entry_id:
                row = {"phrase": normalized, "left_id": prior[0], "right_id": entry_id}
                alias_collisions.append(row)
                problems.append(finding("library.alias_collision", f"Entries {prior[0]!r} and {entry_id!r} share alias {normalized!r}"))
            alias_index[normalized] = (entry_id, alias)
    for left, right in combinations(entries, 2):
        left_id = str(left.get("id", ""))
        right_id = str(right.get("id", ""))
        left_text = " ".join([str(left.get("canonical_name", "")), *[str(item) for item in left.get("match_terms", [])]])
        right_text = " ".join([str(right.get("canonical_name", "")), *[str(item) for item in right.get("match_terms", [])]])
        compared = similarity(left_text, right_text)
        if compared.jaccard >= similarity_threshold:
            row = {"left_id": left_id, "right_id": right_id, "similarity": round(compared.jaccard, 6), "threshold": similarity_threshold}
            near_concepts.append(row)
            problems.append(finding("library.near_duplicate_concept", f"Entries {left_id!r} and {right_id!r} are too similar ({compared.jaccard:.3f})"))
    return problems, {
        "duplicate_ids": sorted(set(duplicate_ids)),
        "duplicate_canonicals": duplicate_canonicals,
        "alias_collisions": alias_collisions,
        "near_duplicate_concepts": near_concepts,
    }


def _entry_leakage_findings(entries: list[dict[str, Any]], sources_catalog: dict[str, Any]) -> list[dict[str, str]]:
    problems: list[dict[str, str]] = []
    forbidden_markers = ("evaluation_data/", "roblox_luau_eval.jsonl", "project_authored_held_out")
    for entry in entries:
        rendered = json.dumps(entry, ensure_ascii=False).casefold()
        for marker in forbidden_markers:
            if marker in rendered:
                problems.append(finding("isolation.entry_reference", f"Entry {entry.get('id')!r} contains prohibited held-out marker {marker!r}"))
    rendered_catalog = json.dumps(sources_catalog, ensure_ascii=False).casefold()
    for marker in forbidden_markers:
        if marker in rendered_catalog:
            problems.append(finding("isolation.source_reference", f"Source catalog contains prohibited held-out marker {marker!r}"))
    return problems


def audit_verified_knowledge(
    config: dict[str, Any],
    *,
    entries_path: str | Path | None = None,
    sources_path: str | Path | None = None,
    evaluation_path: str | Path | None = None,
    code_book_path: str | Path | None = None,
    as_of: date | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Audit provenance, structure, freshness, concept uniqueness, and held-out isolation.

    The returned report deliberately includes only held-out task IDs/similarity values for
    collision findings, never prompts, rubrics, expected answers, baselines, or model scores.
    """
    as_of = as_of or date.today()
    problems = _config_problems(config)
    warnings: list[dict[str, str]] = []
    entries_path = entries_path or str(config.get("entries_file", ""))
    sources_path = sources_path or str(config.get("sources_file", ""))
    evaluation_path = evaluation_path or str(config.get("evaluation_file", ""))
    code_book_path = code_book_path or str(config.get("code_book_file", ""))
    entries: list[dict[str, Any]] = []
    sources_catalog: dict[str, Any] = {}
    try:
        entries = load_entries(entries_path)
    except (FileNotFoundError, ValueError) as exc:
        problems.append(finding("library.load", str(exc)))
    try:
        sources_catalog = load_source_catalog(sources_path)
    except (FileNotFoundError, ValueError) as exc:
        problems.append(finding("source_catalog.load", str(exc)))
    schema_path = config.get("schema_file")
    if isinstance(schema_path, str) and schema_path:
        try:
            schema = read_json(schema_path)
            if not isinstance(schema, dict) or schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
                problems.append(finding("schema.definition", "Verified knowledge schema must be a Draft 2020-12 JSON Schema object"))
            elif "required" not in schema or "properties" not in schema:
                problems.append(finding("schema.definition", "Verified knowledge schema must declare required fields and properties"))
        except (FileNotFoundError, ValueError) as exc:
            problems.append(finding("schema.load", str(exc)))
    adaptive_fallback_status = "unavailable"
    fallback = config.get("adaptive_fallback", {})
    fallback_config_path = fallback.get("effort_config") if isinstance(fallback, dict) else None
    if isinstance(fallback_config_path, str) and fallback_config_path:
        try:
            fallback_config = read_json(fallback_config_path)
            adaptive_enabled = fallback_config.get("adaptive_routing", {}).get("enabled") if isinstance(fallback_config, dict) else None
            if not isinstance(fallback_config, dict) or fallback_config.get("project_name") != "DukeOTR" or adaptive_enabled is not True:
                problems.append(finding("adaptive_fallback.config", "Configured adaptive fallback is not the enabled DukeOTR effort-routing configuration"))
            else:
                adaptive_fallback_status = "checked_existing_adaptive_routing"
        except (FileNotFoundError, ValueError) as exc:
            problems.append(finding("adaptive_fallback.load", str(exc)))
    allowed_hosts_raw = config.get("allowed_source_host_suffixes", [])
    allowed_hosts = tuple(item for item in allowed_hosts_raw if isinstance(item, str))
    if sources_catalog:
        problems.extend(validate_source_catalog(sources_catalog, allowed_source_host_suffixes=allowed_hosts))
    sources = source_index(sources_catalog)
    code_book_ids, code_book_problems = _load_code_book_ids(code_book_path)
    problems.extend(code_book_problems)
    supported_api_names = {str(value) for value in config.get("supported_api_names", []) if isinstance(value, str)}
    rejected_api_names = {str(value) for value in config.get("rejected_api_names", []) if isinstance(value, str)}
    for entry in entries:
        problems.extend(
            validate_entry(
                entry,
                sources=sources,
                allowed_api_names=supported_api_names,
                rejected_api_names=rejected_api_names,
                known_code_book_ids=code_book_ids,
            )
        )
    minimum_entries = config.get("minimum_source_checked_entries", 0)
    source_checked_count = sum(1 for entry in entries if entry.get("status") in {"source_checked", "human_verified"})
    if not isinstance(minimum_entries, int) or source_checked_count < minimum_entries:
        problems.append(finding("library.minimum_entries", f"Need at least {minimum_entries} source-checked entries; found {source_checked_count}"))
    similarity_threshold = config.get("duplicate_similarity_threshold", 0.9)
    if not isinstance(similarity_threshold, (float, int)) or not 0 < float(similarity_threshold) <= 1:
        problems.append(finding("config.duplicate_similarity", "duplicate_similarity_threshold must be in (0, 1]"))
        similarity_threshold = 0.9
    duplicate_problems, duplicate_summary = _duplicate_findings(entries, similarity_threshold=float(similarity_threshold))
    problems.extend(duplicate_problems)
    problems.extend(_entry_leakage_findings(entries, sources_catalog))

    matcher = config.get("matcher", {}) if isinstance(config.get("matcher"), dict) else {}
    stale_after_days = matcher.get("stale_after_days", 180)
    if not isinstance(stale_after_days, int) or stale_after_days < 1:
        stale_after_days = 180
        problems.append(finding("config.stale_after_days", "matcher.stale_after_days must be a positive integer"))
    stale_entry_ids = [str(entry.get("id")) for entry in entries if entry_is_stale(entry, as_of=as_of, stale_after_days=stale_after_days)]
    future_verification_ids: list[str] = []
    for entry in entries:
        verification = entry.get("last_verification", {})
        verified_at = _date(verification.get("verified_at")) if isinstance(verification, dict) else None
        if verified_at and verified_at > as_of:
            future_verification_ids.append(str(entry.get("id")))
    if future_verification_ids:
        problems.append(finding("freshness.future_verification", f"Entries have verification dates after audit date: {', '.join(future_verification_ids)}"))
    if stale_entry_ids:
        stale_finding = finding("freshness.stale_entry", f"Entries need re-verification: {', '.join(stale_entry_ids)}", "error" if strict else "warning")
        (problems if strict else warnings).append(stale_finding)
    stale_source_ids: list[str] = []
    future_source_ids: list[str] = []
    for source_id, source in sources.items():
        checked_at = _date(source.get("checked_at"))
        if checked_at is None:
            continue  # Structural validation already emitted the malformed-date finding.
        if checked_at > as_of:
            future_source_ids.append(source_id)
        elif as_of > checked_at + timedelta(days=stale_after_days):
            stale_source_ids.append(source_id)
    if future_source_ids:
        problems.append(finding("freshness.future_source_check", f"Sources have check dates after audit date: {', '.join(future_source_ids)}"))
    if stale_source_ids:
        stale_source_finding = finding("freshness.stale_source", f"Official sources need re-checking: {', '.join(stale_source_ids)}", "error" if strict else "warning")
        (problems if strict else warnings).append(stale_source_finding)

    evaluation_summary: dict[str, Any]
    try:
        evaluation_tasks = list(read_jsonl(evaluation_path))
        collision_threshold = 0.93
        collisions = cross_split_prompt_collisions(_alias_records(entries), evaluation_tasks, threshold=collision_threshold)
        # No held-out prompt text is retained or emitted.
        if collisions:
            problems.append(finding("isolation.wording_collision", f"Found {len(collisions)} wording-level verified-alias / held-out prompt collision(s)"))
        evaluation_summary = {
            "status": "clear" if not collisions else "collision",
            "evaluation_file": str(evaluation_path),
            "evaluation_sha256": sha256_file(evaluation_path),
            "comparison": "local alias-to-held-out wording-level normalized 3-gram Jaccard; held-out content was not sent to a model or emitted",
            "threshold": collision_threshold,
            "collision_count": len(collisions),
            "collisions": collisions,
            "output_overwrite_protection": "enabled: audit and query CLIs refuse the resolved evaluation file as an output path",
        }
    except (FileNotFoundError, ValueError) as exc:
        problems.append(finding("isolation.evaluation_load", str(exc)))
        evaluation_summary = {
            "status": "unavailable",
            "evaluation_file": str(evaluation_path),
            "comparison": "not run because the protected evaluation file was unavailable",
            "collision_count": None,
            "collisions": [],
            "output_overwrite_protection": "enabled",
        }

    # Keep exact duplicate findings stable even if a malformed library has repeated rows.
    unique_problems: list[dict[str, str]] = []
    seen_problem_keys: set[tuple[str, str, str]] = set()
    for item in problems:
        key = (item["code"], item["message"], item["severity"])
        if key not in seen_problem_keys:
            unique_problems.append(item)
            seen_problem_keys.add(key)
    unique_warnings: list[dict[str, str]] = []
    seen_warning_keys: set[tuple[str, str, str]] = set()
    for item in warnings:
        key = (item["code"], item["message"], item["severity"])
        if key not in seen_warning_keys:
            unique_warnings.append(item)
            seen_warning_keys.add(key)
    return {
        "schema_version": "1.0",
        "project_name": "DukeOTR",
        "audit_kind": "verified_knowledge_strict_audit",
        "as_of": as_of.isoformat(),
        "status": "pass" if not unique_problems else "fail",
        "strict": strict,
        "library": {
            "entries_file": str(entries_path),
            "entries_sha256": _file_hash_or_none(entries_path),
            "entry_count": len(entries),
            "source_checked_entry_count": source_checked_count,
            "minimum_source_checked_entries": minimum_entries,
            "sources_file": str(sources_path),
            "sources_sha256": _file_hash_or_none(sources_path),
            "source_count": len(sources),
            "schema_file": str(config.get("schema_file", "")),
            "schema_sha256": _file_hash_or_none(str(config.get("schema_file", ""))),
            "training_policy": "not_training_data_or_auto_promotion",
            "automatic_promotion": "prohibited",
        },
        "provenance": {
            "allowed_source_host_suffixes": list(allowed_hosts),
            "source_catalog_status": "checked" if sources_catalog else "unavailable",
            "unsupported_or_rejected_api_policy": "approved allowlist plus explicit rejected-name checks",
            "code_book_reference_status": "checked against separate Code Book IDs; cards were not promoted or copied into this library",
            "adaptive_fallback_status": adaptive_fallback_status,
            "adaptive_fallback": fallback,
        },
        "freshness": {
            "stale_after_days": stale_after_days,
            "stale_entry_ids": stale_entry_ids,
            "future_verification_ids": future_verification_ids,
            "stale_source_ids": stale_source_ids,
            "future_source_ids": future_source_ids,
        },
        "duplicates": duplicate_summary,
        "evaluation_isolation": evaluation_summary,
        "errors": unique_problems,
        "warnings": unique_warnings,
    }


def audit_from_config_path(
    config_path: str | Path,
    *,
    as_of: date | None = None,
    strict: bool = False,
    **overrides: Any,
) -> dict[str, Any]:
    """Convenience loader for scripts and tests."""
    config = read_json(config_path)
    if not isinstance(config, dict):
        raise ValueError("Verified knowledge config must be an object")
    return audit_verified_knowledge(config, as_of=as_of, strict=strict, **overrides)
