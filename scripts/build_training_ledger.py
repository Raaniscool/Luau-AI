"""Build an audit-only sidecar ledger for Builder → Reviewer/Tester → Fixer lineage.

The ledger complements canonical candidates; it never replaces them, feeds the dataset builder,
or promotes an example. Each row keeps original Builder output, review evidence, correction
relationship/explanation, and verification state together for human audit.
"""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.lib.io_utils import read_json, read_jsonl, utc_now, write_json_atomic, write_jsonl_atomic
from scripts.lib.training_factory import (
    TrainingFactoryError,
    assert_safe_factory_output_path,
    candidate_is_evaluation_like,
    make_lineage_ledger_record,
    select_corrections,
    validate_failure_taxonomy,
    validate_ledger_record,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Build DukeOTR Builder/Reviewer/Fixer audit ledger sidecar")
    value.add_argument("--input", action="append", required=True, help="Original candidate JSONL; repeat as needed")
    value.add_argument("--corrections", action="append", default=[], help="Corrected/revalidated JSONL; repeat as needed")
    value.add_argument("--output", required=True)
    value.add_argument("--report", default=None)
    value.add_argument("--taxonomy", default="training_factory/failure_taxonomy.json")
    value.add_argument("--created-at", default=None)
    return value


def _read_unique(paths: list[str], *, role: str) -> list[tuple[str, dict[str, Any]]]:
    output: list[tuple[str, dict[str, Any]]] = []
    ids: set[str] = set()
    for source in paths:
        for row in read_jsonl(source):
            record_id = row.get("record_id")
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError(f"{role} artifact {source} contains a row without record_id")
            if record_id in ids:
                raise ValueError(f"Duplicate {role} record_id {record_id!r}")
            if candidate_is_evaluation_like(row):
                raise TrainingFactoryError(f"Refusing evaluation-like record {record_id!r} as ledger input")
            ids.add(record_id)
            output.append((source, row))
    return output


def run(arguments: argparse.Namespace) -> int:
    assert_safe_factory_output_path(arguments.output)
    if arguments.report:
        assert_safe_factory_output_path(arguments.report)
    taxonomy = read_json(arguments.taxonomy)
    if not isinstance(taxonomy, dict):
        raise ValueError("Failure taxonomy must be a JSON object")
    taxonomy_problems = validate_failure_taxonomy(taxonomy)
    if taxonomy_problems:
        raise ValueError(f"Invalid failure taxonomy: {taxonomy_problems}")
    original_rows = _read_unique(arguments.input, role="candidate")
    correction_rows = _read_unique(arguments.corrections, role="correction")
    corrections = select_corrections(row for _source, row in correction_rows)
    created_at = arguments.created_at or utc_now()
    ledgers: list[dict[str, Any]] = []
    for source, record in original_rows:
        ledger = make_lineage_ledger_record(
            record,
            taxonomy=taxonomy,
            candidate_artifact=source,
            correction=corrections.get(str(record.get("record_id"))),
            created_at=created_at,
        )
        problems = validate_ledger_record(ledger, taxonomy=taxonomy)
        if problems:
            raise ValueError(f"Invalid ledger row for {record.get('record_id')!r}: {problems}")
        ledgers.append(ledger)
    ledger_ids = [row["ledger_id"] for row in ledgers]
    if len(ledger_ids) != len(set(ledger_ids)):
        raise ValueError("Ledger construction produced duplicate ledger IDs")
    write_jsonl_atomic(arguments.output, ledgers)
    report_path = Path(arguments.report) if arguments.report else Path(arguments.output).with_suffix(".report.json")
    verification_counts = Counter(str(row["verification"]["status"]) for row in ledgers)
    failure_count = sum(bool(row["failure_analysis"]["observed_failure"]) for row in ledgers)
    report = {
        "schema_version": "1.0",
        "stage": "training_factory_lineage_ledger",
        "created_at": created_at,
        "candidate_inputs": list(arguments.input),
        "correction_inputs": list(arguments.corrections),
        "output": arguments.output,
        "ledger_record_count": len(ledgers),
        "records_with_observed_failure": failure_count,
        "verification_counts": dict(sorted(verification_counts.items())),
        "non_claim": "The ledger preserves audit evidence only. It does not make a candidate eligible, promote a candidate, or establish factual correctness.",
    }
    write_json_atomic(report_path, report)
    print(f"Wrote {len(ledgers)} Builder/Reviewer/Fixer lineage rows to {arguments.output}; report: {report_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError, TrainingFactoryError) as exc:
        print(f"training ledger error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
