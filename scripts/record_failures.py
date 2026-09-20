"""Record observed candidate failures in a separate DukeOTR failure database.

This script never promotes records, calls a model, or reads held-out evaluation content for
prompt construction. It serializes only actual static/reviewer/deduplication failure evidence
already present in canonical candidate traces.
"""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import sys
from pathlib import Path
from typing import Any

from scripts.lib.io_utils import read_json, read_jsonl, text_from_message, utc_now, write_json_atomic, write_jsonl_atomic
from scripts.lib.training_factory import (
    TrainingFactoryError,
    assert_safe_factory_output_path,
    candidate_is_evaluation_like,
    make_failure_record,
    observed_failure,
    select_corrections,
    summarize_failures,
    validate_failure_record,
    validate_failure_taxonomy,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Record observed Builder/Reviewer/Fixer candidate failures separately")
    value.add_argument("--input", action="append", required=True, help="Candidate JSONL artifact; repeat as needed")
    value.add_argument("--corrections", action="append", default=[], help="Corrected/revalidated candidate JSONL; repeat as needed")
    value.add_argument("--output", default="failure_data/observed_failures.jsonl")
    value.add_argument("--report", default=None)
    value.add_argument("--taxonomy", default="training_factory/failure_taxonomy.json")
    value.add_argument("--created-at", default=None, help="UTC timestamp override for reproducible artifact generation")
    return value


def _read_unique(paths: list[str], *, role: str) -> list[tuple[str, dict[str, Any]]]:
    """Read candidate stages, letting a later same-answer stage enrich quality metadata.

    A validated candidate can legitimately reappear in the deduplicated artifact with the same
    immutable ID and answer but richer `deduplication`/`split_isolation` evidence. This is not a
    second failure. We reject any duplicate ID whose prompt or answer differs, then retain the
    later stage so isolation failures are recordable alongside earlier review failures.
    """
    rows: list[tuple[str, dict[str, Any]]] = []
    indexes: dict[str, int] = {}
    for source in paths:
        for row in read_jsonl(source):
            record_id = row.get("record_id")
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError(f"{role} artifact {source} contains a record without record_id")
            if candidate_is_evaluation_like(row):
                raise TrainingFactoryError(f"Refusing evaluation-like record {record_id!r} as failure database input")
            if record_id in indexes:
                prior_source, prior = rows[indexes[record_id]]
                if text_from_message(prior, "user") != text_from_message(row, "user") or text_from_message(prior, "assistant") != text_from_message(row, "assistant"):
                    raise ValueError(
                        f"Duplicate record_id {record_id!r} has conflicting prompt/answer content across {prior_source} and {source}"
                    )
                rows[indexes[record_id]] = (source, row)
                continue
            indexes[record_id] = len(rows)
            rows.append((source, row))
    return rows


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
    source_rows = _read_unique(arguments.input, role="candidate")
    correction_rows = _read_unique(arguments.corrections, role="correction")
    corrections = select_corrections(row for _source, row in correction_rows)
    created_at = arguments.created_at or utc_now()
    failures: list[dict[str, Any]] = []
    skipped = 0
    for source, record in source_rows:
        if not observed_failure(record):
            skipped += 1
            continue
        correction = corrections.get(str(record.get("record_id")))
        failure = make_failure_record(
            record,
            taxonomy=taxonomy,
            candidate_artifact=source,
            correction=correction,
            created_at=created_at,
        )
        problems = validate_failure_record(failure, taxonomy=taxonomy)
        if problems:
            raise ValueError(f"Factory failure record for {record.get('record_id')!r} is invalid: {problems}")
        failures.append(failure)
    failure_ids = [row["failure_id"] for row in failures]
    if len(failure_ids) != len(set(failure_ids)):
        raise ValueError("Failure construction produced duplicate failure IDs")
    write_jsonl_atomic(arguments.output, failures)
    report_path = Path(arguments.report) if arguments.report else Path(arguments.output).with_suffix(".report.json")
    summary = summarize_failures(failures, taxonomy=taxonomy)
    report = {
        "stage": "failure_recording",
        "created_at": created_at,
        "candidate_inputs": list(arguments.input),
        "correction_inputs": list(arguments.corrections),
        "output": str(arguments.output),
        "candidate_record_count": len(source_rows),
        "recorded_failure_count": len(failures),
        "skipped_without_stored_failure_evidence": skipped,
        "summary": summary,
        "non_claim": "The database contains observed pipeline failures only. A category count is not an estimate of total model weakness or a factual-correctness oracle.",
    }
    write_json_atomic(report_path, report)
    print(
        f"Recorded {len(failures)} observed failures from {len(source_rows)} candidate records to {arguments.output}; "
        f"report: {report_path}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError, TrainingFactoryError) as exc:
        print(f"failure recording error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
