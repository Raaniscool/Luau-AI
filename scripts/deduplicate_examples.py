"""Stage 4: deterministic duplicate detection and held-out prompt leakage checks."""

from __future__ import annotations

# Support both `python -m scripts.name` and `python scripts/name.py` from the repo.
if __package__ in {None, ""}:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from scripts.lib.dedupe import cross_split_prompt_collisions, mark_deduplicated
from scripts.lib.io_utils import read_json, read_jsonl, utc_now, write_json_atomic, write_jsonl_atomic
from scripts.lib.schema import STATIC_CHECKER_VERSION, reviewer_approved, validate_example_structure


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Deduplicate reviewed Roblox/Luau training candidates")
    value.add_argument("--input", action="append", default=[], help="Reviewed JSONL input; repeat for corrected records")
    value.add_argument("--output", default="validated_data/dukeotr_phase1_candidates.deduplicated.jsonl")
    value.add_argument("--report", default=None)
    value.add_argument("--evaluation", default="evaluation_data/roblox_luau_eval.jsonl")
    value.add_argument("--config", default="configs/pipeline.json")
    value.add_argument("--threshold", type=float, default=None, help="Near-duplicate Jaccard threshold")
    value.add_argument("--cross-split-threshold", type=float, default=None)
    value.add_argument("--include-ineligible", action="store_true", help="Also deduplicate unreviewed/rejected records (never final eligible)")
    return value


def pre_dedupe_eligibility(record: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if validate_example_structure(record):
        reasons.append("schema_invalid")
    quality = record.get("quality", {})
    static = quality.get("static", {})
    if static.get("status") != "pass":
        reasons.append("static_not_pass")
    elif static.get("checker") != STATIC_CHECKER_VERSION:
        reasons.append("static_checker_version_not_current")
    if not reviewer_approved(record):
        reasons.append("no_accepting_reviewer")
    return not reasons, reasons


def run(arguments: argparse.Namespace) -> int:
    config = read_json(arguments.config)
    dedupe_config = dict(config.get("deduplication", {}))
    threshold = arguments.threshold if arguments.threshold is not None else float(dedupe_config.get("near_duplicate_threshold", 0.82))
    cross_threshold = (
        arguments.cross_split_threshold
        if arguments.cross_split_threshold is not None
        else float(dedupe_config.get("cross_split_prompt_threshold", 0.93))
    )
    if not 0 < threshold <= 1 or not 0 < cross_threshold <= 1:
        raise ValueError("Deduplication thresholds must be in (0, 1]")
    inputs = arguments.input or ["validated_data/dukeotr_phase1_candidates.validated.jsonl"]
    source_records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    for source in inputs:
        for record in read_jsonl(source):
            record_id = str(record.get("record_id"))
            if record_id in seen_ids:
                duplicate_ids.append(record_id)
                continue
            seen_ids.add(record_id)
            source_records.append(record)
    if duplicate_ids:
        raise ValueError(f"Input artifacts contain duplicate record IDs: {', '.join(duplicate_ids[:10])}")
    if not source_records:
        raise ValueError("No records provided for deduplication")

    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for record in source_records:
        is_eligible, reasons = pre_dedupe_eligibility(record)
        if is_eligible or arguments.include_ineligible:
            eligible.append(record)
        else:
            excluded.append({"record_id": record.get("record_id"), "reasons": reasons})
    marked, duplicate_audit = mark_deduplicated(
        eligible,
        threshold=threshold,
        shingle_size=int(dedupe_config.get("shingle_size", 3)),
    )
    evaluation_tasks = list(read_jsonl(arguments.evaluation))
    collisions = cross_split_prompt_collisions(
        marked,
        evaluation_tasks,
        threshold=cross_threshold,
        shingle_size=int(dedupe_config.get("shingle_size", 3)),
    )
    collision_by_id: dict[str, list[dict[str, Any]]] = {}
    for collision in collisions:
        collision_by_id.setdefault(str(collision["record_id"]), []).append(collision)
    output: list[dict[str, Any]] = []
    for record in marked:
        clone = deepcopy(record)
        clone.setdefault("quality", {})["split_isolation"] = {
            "status": "collision" if clone.get("record_id") in collision_by_id else "clear",
            "collisions": collision_by_id.get(str(clone.get("record_id")), []),
            "threshold": cross_threshold,
            "checked_at": utc_now(),
        }
        output.append(clone)
    write_jsonl_atomic(arguments.output, output)
    report_path = Path(arguments.report) if arguments.report else Path(arguments.output).with_suffix(".deduplication_report.json")
    report = {
        "stage": "deduplication",
        "created_at": utc_now(),
        "inputs": inputs,
        "output": str(arguments.output),
        "input_records": len(source_records),
        "considered_records": len(eligible),
        "excluded_before_deduplication": excluded,
        "unique_records": sum(record.get("quality", {}).get("deduplication", {}).get("status") == "unique" for record in output),
        "duplicate_records": len(duplicate_audit),
        "duplicates": duplicate_audit,
        "near_duplicate_threshold": threshold,
        "cross_split_prompt_threshold": cross_threshold,
        "cross_split_collisions": collisions,
        "evaluation_file": str(arguments.evaluation),
    }
    write_json_atomic(report_path, report)
    print(
        f"Wrote {len(output)} deduplicated candidates ({report['unique_records']} unique, "
        f"{report['duplicate_records']} duplicates) to {arguments.output}; report: {report_path}"
    )
    if collisions:
        print(f"Warning: found {len(collisions)} potential held-out prompt collisions; builder will exclude them.", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"deduplication error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
