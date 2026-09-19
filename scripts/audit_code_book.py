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
    domain_set = set().union(*(set(str(item) for item in card.get("domains", [])) for card in cards))
    missing_domains = {
        label: alternatives for label, alternatives in REQUIRED_DOMAINS.items() if not any(value in domain_set for value in alternatives)
    }
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
        "missing_required_domains": missing_domains,
        "cards_without_explicit_pattern": cards_without_patterns,
        "errors": errors,
        "warnings": warnings,
        "status": "pass" if not (errors or duplicate_ids or missing_domains or cards_without_patterns) else "fail",
        "policy_note": "source_checked cards are source-attributed reference material, not human-verified facts or automatic training data.",
    }
    write_json_atomic(arguments.output, report)
    print(
        f"Code Book audit: {report['status']} — {report['card_count']} cards, "
        f"{len(errors)} errors, {len(warnings)} warnings, {len(missing_domains)} missing domains. "
        f"Report: {arguments.output}"
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
