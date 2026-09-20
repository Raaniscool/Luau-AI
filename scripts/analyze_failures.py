"""Derive auditable weakness statistics and targeting recommendations from stored failures only."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import sys
from typing import Any

from scripts.lib.io_utils import read_json, read_jsonl, utc_now, write_json_atomic
from scripts.lib.training_factory import (
    TrainingFactoryError,
    assert_safe_factory_output_path,
    recommend_targeting,
    registry_file_fingerprint,
    summarize_failures,
    validate_failure_record,
    validate_failure_taxonomy,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Analyze observed DukeOTR failure records without inventing counts")
    value.add_argument("--input", action="append", required=True, help="Failure JSONL database artifact; repeat as needed")
    value.add_argument("--output", default="reports/training_factory/failure_analysis.json")
    value.add_argument("--taxonomy", default="training_factory/failure_taxonomy.json")
    value.add_argument("--config", default="configs/training_factory.json")
    value.add_argument("--minimum-failure-count", type=int, default=None)
    value.add_argument("--created-at", default=None, help="UTC timestamp override for reproducible reports")
    return value


def run(arguments: argparse.Namespace) -> int:
    assert_safe_factory_output_path(arguments.output)
    taxonomy = read_json(arguments.taxonomy)
    config = read_json(arguments.config)
    if not isinstance(taxonomy, dict) or not isinstance(config, dict):
        raise ValueError("Factory taxonomy and config must be JSON objects")
    taxonomy_problems = validate_failure_taxonomy(taxonomy)
    if taxonomy_problems:
        raise ValueError(f"Invalid failure taxonomy: {taxonomy_problems}")
    policy = config.get("targeted_generation", {})
    minimum = arguments.minimum_failure_count
    if minimum is None:
        minimum = policy.get("minimum_failure_count_for_recommendation", 1)
    if not isinstance(minimum, int) or minimum < 1:
        raise ValueError("minimum-failure-count must be a positive integer")
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for source in arguments.input:
        for row in read_jsonl(source):
            problems = validate_failure_record(row, taxonomy=taxonomy)
            if problems:
                raise ValueError(f"Invalid failure record in {source}: {problems}")
            failure_id = row["failure_id"]
            if failure_id in seen_ids:
                raise ValueError(f"Duplicate failure_id across analysis inputs: {failure_id}")
            seen_ids.add(failure_id)
            rows.append(row)
    summary = summarize_failures(rows, taxonomy=taxonomy)
    recommendations = recommend_targeting(summary, taxonomy=taxonomy, minimum_failure_count=minimum)
    report = {
        "schema_version": "1.0",
        "stage": "failure_analysis",
        "created_at": arguments.created_at or utc_now(),
        "failure_inputs": list(arguments.input),
        "failure_input_fingerprint": registry_file_fingerprint(arguments.input),
        "minimum_failure_count_for_recommendation": minimum,
        "summary": summary,
        "targeting_recommendations": recommendations,
        "non_claim": "Recommendations exist only for categories represented in the stored input rows. They are source-brief planning priorities, not generated examples, training data, or a claim of factual correctness.",
    }
    write_json_atomic(arguments.output, report)
    print(
        f"Analyzed {len(rows)} stored failure records: {len(recommendations)} evidence-backed targeting recommendations. "
        f"Report: {arguments.output}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError, TrainingFactoryError) as exc:
        print(f"failure analysis error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
