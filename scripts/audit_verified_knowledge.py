"""Audit DukeOTR's separate, source-checked verified Roblox knowledge library."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import json
import sys
from datetime import date
from typing import Any

from scripts.lib.io_utils import read_json
from scripts.lib.verified_knowledge_audit import audit_verified_knowledge, assert_safe_output_path, write_audit_report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Validate verified-knowledge provenance, API claims, freshness, duplicates, and held-out evaluation isolation."
    )
    value.add_argument("--config", default="configs/verified_knowledge.json")
    value.add_argument("--entries", default=None)
    value.add_argument("--sources", default=None)
    value.add_argument("--evaluation", default=None)
    value.add_argument("--code-book", default=None)
    value.add_argument("--as-of", default=None, help="ISO date used for deterministic stale-entry identification")
    value.add_argument("--strict", action="store_true", help="Treat stale entries as audit failures instead of warnings")
    value.add_argument("--output", default=None, help="Optional JSON report; cannot be the protected held-out evaluation file")
    return value


def _as_of(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("--as-of must use ISO YYYY-MM-DD format") from exc


def _summary(report: dict[str, Any]) -> str:
    library = report["library"]
    isolation = report["evaluation_isolation"]
    return (
        f"Verified knowledge audit: {report['status']} | "
        f"entries={library['entry_count']} source_checked={library['source_checked_entry_count']} "
        f"sources={library['source_count']} stale={len(report['freshness']['stale_entry_ids'])} "
        f"held_out_isolation={isolation['status']} errors={len(report['errors'])} warnings={len(report['warnings'])}"
    )


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        config = read_json(arguments.config)
        if not isinstance(config, dict):
            raise ValueError("Verified knowledge config must be a JSON object")
        evaluation_path = arguments.evaluation or config.get("evaluation_file")
        if not isinstance(evaluation_path, str) or not evaluation_path:
            raise ValueError("Verified knowledge config requires evaluation_file for isolation and output protection")
        output = arguments.output or config.get("audit_output")
        if output:
            assert_safe_output_path(output, evaluation_path)
        report = audit_verified_knowledge(
            config,
            entries_path=arguments.entries,
            sources_path=arguments.sources,
            evaluation_path=evaluation_path,
            code_book_path=arguments.code_book,
            as_of=_as_of(arguments.as_of),
            strict=arguments.strict,
        )
        if output:
            write_audit_report(output, report, evaluation_path=evaluation_path)
        print(_summary(report))
        if report["errors"]:
            for item in report["errors"]:
                print(f"ERROR {item['code']}: {item['message']}", file=sys.stderr)
        if report["warnings"]:
            for item in report["warnings"]:
                print(f"WARNING {item['code']}: {item['message']}", file=sys.stderr)
        return 0 if report["status"] == "pass" else 2
    except (FileNotFoundError, ValueError) as exc:
        print(f"Verified knowledge audit failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
