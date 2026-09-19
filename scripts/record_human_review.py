"""Record auditable human review decisions without editing JSONL by hand.

Decision JSONL rows must contain `record_id`, `decision` (accept/revise/reject), `reviewer`,
and `notes`. A human acceptance can supplement an LLM review, but static failures and
duplicates still cannot enter final data.
"""

from __future__ import annotations

# Support both `python -m scripts.name` and `python scripts/name.py` from the repo.
if __package__ in {None, ""}:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import sys
from copy import deepcopy
from typing import Any

from scripts.lib.io_utils import read_jsonl, utc_now, write_jsonl_atomic
from scripts.lib.schema import VALID_REVIEW_DECISIONS


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Apply human review decisions to candidate JSONL")
    value.add_argument("--input", required=True)
    value.add_argument("--decisions", required=True)
    value.add_argument("--output", required=True)
    return value


def run(arguments: argparse.Namespace) -> int:
    decisions: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(arguments.decisions):
        record_id = row.get("record_id")
        decision = row.get("decision")
        reviewer = row.get("reviewer")
        notes = row.get("notes")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("Every human decision needs record_id")
        if decision not in VALID_REVIEW_DECISIONS:
            raise ValueError(f"Decision for {record_id} must be accept, revise, or reject")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError(f"Decision for {record_id} needs reviewer")
        if not isinstance(notes, str) or not notes.strip():
            raise ValueError(f"Decision for {record_id} needs non-empty notes")
        if record_id in decisions:
            raise ValueError(f"Duplicate human decision for {record_id}")
        decisions[record_id] = row
    output: list[dict[str, Any]] = []
    found: set[str] = set()
    for record in read_jsonl(arguments.input):
        clone = deepcopy(record)
        record_id = clone.get("record_id")
        if record_id in decisions:
            row = decisions[str(record_id)]
            clone.setdefault("quality", {})["human_review"] = {
                "status": "complete",
                "decision": row["decision"],
                "reviewer": row["reviewer"],
                "notes": row["notes"],
                "checked_at": utc_now(),
            }
            found.add(str(record_id))
        output.append(clone)
    unknown = sorted(set(decisions) - found)
    if unknown:
        raise ValueError(f"Human decision IDs not found in input: {', '.join(unknown[:20])}")
    write_jsonl_atomic(arguments.output, output)
    print(f"Applied {len(found)} human review decisions to {arguments.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"human review error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
