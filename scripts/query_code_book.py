"""Retrieve source-attributed Code Book context without invoking Ollama."""

from __future__ import annotations

# Support both `python -m scripts.query_code_book` and direct Windows invocation.
if __package__ in {None, ""}:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scripts.lib.code_book import context_view, load_cards, search_cards
from scripts.lib.io_utils import read_json


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Search the source-attributed Roblox/Luau Code Book")
    value.add_argument("--query", required=True, help="Topic or review question, such as 'secure RemoteEvent shop purchase'")
    value.add_argument("--cards", default=None)
    value.add_argument("--config", default="configs/code_book.json")
    value.add_argument("--max-results", type=int, default=None)
    value.add_argument("--include-drafts", action="store_true", help="Include draft/deprecated cards for editorial work")
    value.add_argument("--format", choices=["json", "markdown"], default="json")
    value.add_argument("--output", default=None, help="Optional output file; otherwise prints to stdout")
    return value


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Code Book context: {payload['query']}",
        "",
        "> Reference context only. Verify current APIs and game-specific assumptions before implementation.",
        "",
    ]
    if not payload["results"]:
        lines.append("No eligible Code Book cards matched this query.")
    for result in payload["results"]:
        lines.extend(
            [
                f"## {result['title']} (`{result['id']}`, revision {result['revision']}, {result['status']})",
                "",
                result["summary"],
                "",
                "### Claims",
            ]
        )
        for claim in result["claims"]:
            lines.append(f"- **{claim['claim_kind']}** — {claim['statement']}")
            if claim["caveats"]:
                lines.append(f"  - Caveat: {' '.join(claim['caveats'])}")
        lines.extend(["", "### Implementation patterns"])
        for pattern in result["patterns"]:
            lines.append(f"- **{pattern['name']}** (`{pattern['id']}`)")
            lines.extend(f"  - {step}" for step in pattern["steps"])
        lines.extend(["", "### Common pitfalls"])
        lines.extend(f"- {pitfall}" for pitfall in result["pitfalls"])
        lines.extend(["", "### Reviewer checks"])
        lines.extend(f"- {check}" for check in result["reviewer_checks"])
        lines.extend(["", "### Sources"])
        lines.extend(f"- [{source['id']}]({source['url']}) — {source['publisher']} ({source['source_type']})" for source in result["sources"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run(arguments: argparse.Namespace) -> int:
    config = read_json(arguments.config)
    cards_path = arguments.cards or config.get("cards_file")
    if not isinstance(cards_path, str) or not cards_path:
        raise ValueError("Code Book config requires cards_file")
    max_results = arguments.max_results if arguments.max_results is not None else int(config.get("default_max_query_results", 5))
    statuses = {"draft", "source_checked", "human_verified", "deprecated"} if arguments.include_drafts else set(
        config.get("allowed_statuses_for_default_retrieval", ["source_checked", "human_verified"])
    )
    matches = search_cards(load_cards(cards_path), arguments.query, allowed_statuses=statuses, limit=max_results)
    payload = {
        "schema_version": "1.0",
        "query": arguments.query,
        "cards_file": cards_path,
        "allowed_statuses": sorted(statuses),
        "result_count": len(matches),
        "results": [context_view(match) | {"retrieval_score": match["score"]} for match in matches],
        "notice": "This is cited reference context only. It is not a model answer, does not execute code, and does not make generated output safe.",
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n" if arguments.format == "json" else markdown(payload)
    if arguments.output:
        destination = Path(arguments.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
        print(f"Wrote {len(matches)} Code Book matches to {destination}")
    else:
        print(rendered, end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Code Book query error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
