"""Audit the source-attributed Roblox/Luau Code Book without calling a model."""

from __future__ import annotations

# Support both `python -m scripts.audit_code_book` and direct Windows invocation.
if __package__ in {None, ""}:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import sys
from typing import Any

from scripts.lib.code_book import catalog_summary, load_cards, validate_card
from scripts.lib.io_utils import read_json, utc_now, write_json_atomic

REQUIRED_DOMAINS: dict[str, tuple[str, ...]] = {
    "Roblox APIs": ("roblox_apis",),
    "Luau syntax": ("Luau",),
    "client/server architecture": ("client_server", "architecture"),
    "RemoteEvents": ("RemoteEvents",),
    "RemoteFunctions": ("RemoteFunctions",),
    "security": ("security",),
    "Instances": ("Instances",),
    "services": ("services",),
    "events": ("events",),
    "functions": ("functions",),
    "tables": ("tables",),
    "ModuleScripts": ("ModuleScripts",),
    "OOP/metatables": ("OOP", "metatables"),
    "UI": ("UI",),
    "DataStores": ("DataStores",),
    "Tools": ("Tools",),
    "NPCs": ("NPCs",),
    "inventories": ("inventories",),
    "shops": ("shops",),
    "combat": ("combat",),
    "currencies": ("currencies",),
    "round systems": ("round_systems",),
    "matchmaking": ("matchmaking",),
    "developer products": ("developer_products",),
    "gamepasses": ("gamepasses",),
    "debugging": ("debugging",),
    "optimization": ("optimization",),
    "common mistakes": ("common_mistakes",),
}

# This deliberately names the minimum granular concepts requested for DukeOTR readiness.
# A card can satisfy more than one row, but every row must be traceable to a source-checked
# card domain.  It is a coverage floor, not a claim that one card makes the model competent.
REQUIRED_CONCEPTS: dict[str, tuple[str, ...]] = {
    "Luau values": ("values",),
    "local scope": ("scope",),
    "nil": ("nil",),
    "operators": ("operators",),
    "control flow": ("control_flow",),
    "loops": ("loops",),
    "functions": ("functions",),
    "returns": ("returns",),
    "tables": ("tables",),
    "strings": ("strings",),
    "math": ("math",),
    "randomness": ("randomness",),
    "type annotations": ("type_annotations",),
    "typed tables": ("typed_tables",),
    "unions": ("unions",),
    "callbacks": ("callbacks",),
    "errors": ("error_handling",),
    "pcall": ("pcall",),
    "assert": ("assert",),
    "ModuleScripts": ("ModuleScripts",),
    "require": ("require",),
    "Instances": ("Instances",),
    "services": ("services",),
    "players": ("players",),
    "characters": ("characters",),
    "GUIs": ("GUIs",),
    "events": ("events",),
    "connections": ("connections",),
    "closures": ("closures",),
    "refactoring": ("refactoring",),
    "RemoteEvents": ("RemoteEvents",),
    "RemoteFunctions": ("RemoteFunctions",),
    "client/server": ("client_server",),
    "replication": ("replication",),
    "server authority": ("server_authority",),
    "security": ("security",),
    "DataStores": ("DataStores",),
    "raycasting": ("raycasting",),
    "TweenService": ("TweenService",),
    "RunService": ("RunService",),
    "physics": ("physics",),
    "performance": ("performance",),
    "common API hallucinations": ("common_api_hallucinations",),
}


def coverage_matrix(cards: list[dict[str, Any]], required: dict[str, tuple[str, ...]]) -> tuple[dict[str, list[str]], dict[str, tuple[str, ...]]]:
    """Return explicit card evidence and rows that have no tagged source-checked coverage."""
    eligible = [card for card in cards if card.get("status") in {"source_checked", "human_verified"}]
    matrix: dict[str, list[str]] = {}
    missing: dict[str, tuple[str, ...]] = {}
    for label, alternatives in required.items():
        matching_ids = sorted(
            str(card.get("id"))
            for card in eligible
            if any(value in set(str(item) for item in card.get("domains", [])) for value in alternatives)
        )
        matrix[label] = matching_ids
        if not matching_ids:
            missing[label] = alternatives
    return matrix, missing


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Audit Code Book provenance, schema, and coverage")
    value.add_argument("--cards", default=None, help="Code Book JSONL; defaults to configs/code_book.json")
    value.add_argument("--config", default="configs/code_book.json")
    value.add_argument("--output", default="reports/code_book_audit.json")
    value.add_argument("--strict", action="store_true", help="Exit nonzero for schema/source/coverage errors")
    return value


def run(arguments: argparse.Namespace) -> int:
    config = read_json(arguments.config)
    cards_path = arguments.cards or config.get("cards_file")
    if not isinstance(cards_path, str) or not cards_path:
        raise ValueError("Code Book config requires cards_file")
    cards = load_cards(cards_path)
    if not cards:
        raise ValueError("Code Book has no cards")
    suffixes = tuple(str(item) for item in config.get("required_source_host_suffixes", []))
    findings: list[dict[str, Any]] = []
    ids: set[str] = set()
    duplicate_ids: list[str] = []
    for card in cards:
        card_id = str(card.get("id"))
        if card_id in ids:
            duplicate_ids.append(card_id)
        ids.add(card_id)
        for finding in validate_card(card, allowed_source_host_suffixes=suffixes):
            findings.append({"card_id": card_id, **finding})
    domain_coverage, missing_domains = coverage_matrix(cards, REQUIRED_DOMAINS)
    concept_coverage, missing_concepts = coverage_matrix(cards, REQUIRED_CONCEPTS)
    minimum_card_count = config.get("minimum_source_checked_cards", 1)
    if not isinstance(minimum_card_count, int) or minimum_card_count < 1:
        raise ValueError("Code Book minimum_source_checked_cards must be an integer >= 1")
    source_checked_count = sum(card.get("status") in {"source_checked", "human_verified"} for card in cards)
    cards_without_patterns = [card["id"] for card in cards if not card.get("patterns")]
    errors = [finding for finding in findings if finding["severity"] == "error"]
    warnings = [finding for finding in findings if finding["severity"] != "error"]
    report = {
        "stage": "code_book_audit",
        "created_at": utc_now(),
        "config": str(arguments.config),
        "cards": str(cards_path),
        **catalog_summary(cards),
        "duplicate_ids": sorted(set(duplicate_ids)),
        "minimum_source_checked_cards": minimum_card_count,
        "source_checked_card_count": source_checked_count,
        "source_checked_card_count_status": "pass" if source_checked_count >= minimum_card_count else "fail",
        "domain_coverage": domain_coverage,
        "missing_required_domains": missing_domains,
        "granular_concept_coverage": concept_coverage,
        "missing_required_concepts": missing_concepts,
        "cards_without_explicit_pattern": cards_without_patterns,
        "errors": errors,
        "warnings": warnings,
        "status": "pass"
        if not (errors or duplicate_ids or missing_domains or missing_concepts or cards_without_patterns or source_checked_count < minimum_card_count)
        else "fail",
        "policy_note": "source_checked cards are source-attributed reference material, not human-verified facts or automatic training data.",
    }
    write_json_atomic(arguments.output, report)
    print(
        f"Code Book audit: {report['status']} — {report['card_count']} cards, "
        f"{source_checked_count}/{minimum_card_count} source-checked floor, {len(errors)} errors, "
        f"{len(warnings)} warnings, {len(missing_domains)} missing broad domains, "
        f"{len(missing_concepts)} missing granular concepts. Report: {arguments.output}"
    )
    if arguments.strict and report["status"] != "pass":
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Code Book audit error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
