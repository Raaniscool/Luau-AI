"""Stage 2: static and independent LLM review of generated/corrected candidates.

Examples with a skipped or failed LLM review are intentionally ineligible for final training
data. Use --skip-llm-review only to inspect deterministic checks while offline.
"""

from __future__ import annotations

# Support both `python -m scripts.name` and `python scripts/name.py` from the repo.
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
from scripts.lib.ollama import OllamaClient, OllamaError
from scripts.lib.quality import validate_record


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Validate Roblox/Luau candidate JSONL")
    value.add_argument("--input", default="generated_data/generated_examples.jsonl")
    value.add_argument("--output", default="validated_data/validated_examples.jsonl")
    value.add_argument("--report", default=None)
    value.add_argument("--config", default="configs/pipeline.json")
    value.add_argument("--model", default=None, help="Independent reviewer model; default qwen3:4b")
    value.add_argument("--host", default=None)
    value.add_argument("--limit", type=int, default=0)
    value.add_argument("--skip-llm-review", action="store_true", help="Static checks only; results cannot be finalized")
    value.add_argument("--allow-review-errors", action="store_true", help="Exit zero despite reviewer transport/format errors")
    return value


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    static = Counter(record.get("quality", {}).get("static", {}).get("status") for record in records)
    review = Counter(record.get("quality", {}).get("llm_review", {}).get("status") for record in records)
    decisions = Counter(record.get("quality", {}).get("llm_review", {}).get("decision") for record in records)
    return {
        "records": len(records),
        "static_statuses": dict(sorted(static.items(), key=lambda pair: str(pair[0]))),
        "review_statuses": dict(sorted(review.items(), key=lambda pair: str(pair[0]))),
        "review_decisions": dict(sorted(decisions.items(), key=lambda pair: str(pair[0]))),
        "static_pass_and_review_accept": sum(
            record.get("quality", {}).get("static", {}).get("status") == "pass"
            and record.get("quality", {}).get("llm_review", {}).get("decision") == "accept"
            for record in records
        ),
    }


def run(arguments: argparse.Namespace) -> int:
    if arguments.limit < 0:
        raise ValueError("--limit must be zero or positive")
    config = read_json(arguments.config)
    reviewer_config = dict(config.get("reviewer", {}))
    build_config = dict(config.get("dataset_build", {}))
    model = arguments.model or config.get("base_ollama_model", "qwen3:4b")
    source_records = list(read_jsonl(arguments.input))
    if arguments.limit:
        source_records = source_records[: arguments.limit]
    if not source_records:
        raise ValueError("Input has no candidate records")

    client: OllamaClient | None = None
    if not arguments.skip_llm_review:
        client = OllamaClient(arguments.host)
        client.assert_model_present(model)
    options = {
        "temperature": reviewer_config.get("temperature", 0.0),
        "num_predict": reviewer_config.get("num_predict", 1200),
        "num_ctx": config.get("generator", {}).get("num_ctx", 8192),
    }
    minimums = {
        "accuracy": int(reviewer_config.get("minimum_accuracy", 4)),
        "security": int(reviewer_config.get("minimum_security", 4)),
        "requirement_coverage": int(reviewer_config.get("minimum_requirement_coverage", 4)),
        "pedagogy": int(reviewer_config.get("minimum_pedagogy", 3)),
    }
    validated: list[dict[str, Any]] = []
    for index, record in enumerate(source_records, start=1):
        print(f"[{index}/{len(source_records)}] validating {record.get('record_id', '(missing id)')}", flush=True)
        validated.append(
            validate_record(
                record,
                client=client,
                model=None if arguments.skip_llm_review else model,
                review_options=options,
                minimums=minimums,
                minimum_assistant_characters=int(build_config.get("minimum_assistant_characters", 160)),
            )
        )
    write_jsonl_atomic(arguments.output, validated)
    report_path = Path(arguments.report) if arguments.report else Path(arguments.output).with_suffix(".validation_report.json")
    summary = summarize(validated)
    report = {
        "stage": "validation",
        "created_at": utc_now(),
        "input": str(arguments.input),
        "output": str(arguments.output),
        "reviewer_model": None if arguments.skip_llm_review else model,
        "review_options": options,
        "review_minimums": minimums,
        "llm_review_skipped": bool(arguments.skip_llm_review),
        **summary,
    }
    write_json_atomic(report_path, report)
    print(f"Wrote {len(validated)} reviewed records to {arguments.output}; report: {report_path}")
    review_errors = summary["review_statuses"].get("error", 0)
    if review_errors and not arguments.allow_review_errors:
        print("Validation completed with reviewer errors; affected records are not eligible for final data.", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError, OllamaError) as exc:
        print(f"validation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
