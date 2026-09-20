"""Audit DukeOTR's combined project-authored source-brief portfolio.

This deterministic audit verifies diversity of *brief specifications*, not model output quality.
It is intentionally offline and only uses held-out evaluation prompts locally to detect
wording-level leakage; no held-out text is sent to a generation or review model.
"""

from __future__ import annotations

# Support both `python -m scripts.audit_source_portfolio` and direct Windows invocation.
if __package__ in {None, ""}:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import sys
from collections import Counter
from typing import Any

from scripts.lib.dedupe import cross_split_prompt_collisions, normalized_tokens
from scripts.lib.io_utils import read_json, read_jsonl, utc_now, write_json_atomic
from scripts.lib.schema import validate_seed


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Audit DukeOTR source-brief diversity and held-out wording isolation")
    value.add_argument("--config", default="configs/source_portfolio.json")
    value.add_argument("--catalog", action="append", default=[], help="Override/repeat curated source JSONL catalog paths")
    value.add_argument("--evaluation", default=None)
    value.add_argument("--output", default="reports/source_portfolio_audit.json")
    value.add_argument("--strict", action="store_true", help="Fail when a variety, schema, duplication, or isolation gate fails")
    return value


def _normal_prompt(value: str) -> str:
    return " ".join(normalized_tokens(value))


def _validate_config(config: dict[str, Any]) -> None:
    errors: list[str] = []
    if config.get("schema_version") != "1.0":
        errors.append("schema_version must be '1.0'")
    if config.get("project_name") != "DukeOTR":
        errors.append("project_name must be DukeOTR")
    catalogs = config.get("source_catalogs")
    if not isinstance(catalogs, list) or len(catalogs) < 2 or not all(isinstance(item, str) and item for item in catalogs):
        errors.append("source_catalogs must list at least two non-empty JSONL paths")
    if not isinstance(config.get("minimum_project_authored_briefs"), int) or config["minimum_project_authored_briefs"] < 1:
        errors.append("minimum_project_authored_briefs must be a positive integer")
    maximum_share = config.get("maximum_single_task_type_share")
    if not isinstance(maximum_share, (int, float)) or not 0 < float(maximum_share) <= 1:
        errors.append("maximum_single_task_type_share must be in (0, 1]")
    threshold = config.get("cross_split_prompt_threshold")
    if not isinstance(threshold, (int, float)) or not 0 < float(threshold) <= 1:
        errors.append("cross_split_prompt_threshold must be in (0, 1]")
    forms = config.get("required_task_forms")
    if not isinstance(forms, dict) or len(forms) < 10:
        errors.append("required_task_forms must contain at least ten named forms")
    elif any(not isinstance(name, str) or not isinstance(types, list) or not types or not all(isinstance(item, str) for item in types) for name, types in forms.items()):
        errors.append("required_task_forms must map names to non-empty task-type lists")
    if errors:
        raise ValueError("Source portfolio config is invalid:\n- " + "\n- ".join(errors))


def audit(config: dict[str, Any], catalogs: list[str], evaluation_path: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    source_by_id: dict[str, str] = {}
    invalid: dict[str, list[dict[str, str]]] = {}
    duplicate_ids: list[str] = []
    for catalog in catalogs:
        for row_number, seed in enumerate(read_jsonl(catalog), start=1):
            seed_id = str(seed.get("id", f"{catalog}:{row_number}"))
            if seed_id in source_by_id:
                duplicate_ids.append(seed_id)
            source_by_id[seed_id] = catalog
            problems = validate_seed(seed)
            if problems:
                invalid[seed_id] = problems
            record = dict(seed)
            record["_catalog"] = catalog
            records.append(record)

    prompt_index: dict[str, list[str]] = {}
    for seed in records:
        prompt_index.setdefault(_normal_prompt(str(seed.get("user_request", ""))), []).append(str(seed.get("id")))
    duplicate_prompts = sorted(ids for prompt, ids in prompt_index.items() if prompt and len(ids) > 1)
    task_types = Counter(str(seed.get("task_type")) for seed in records)
    difficulties = Counter(str(seed.get("difficulty")) for seed in records)
    forms: dict[str, list[str]] = {}
    for name, accepted_types in config["required_task_forms"].items():
        forms[name] = sorted(str(seed.get("id")) for seed in records if seed.get("task_type") in accepted_types)
    missing_forms = sorted(name for name, ids in forms.items() if not ids)
    catalog_counts = Counter(str(seed["_catalog"]) for seed in records)
    max_type_count = max(task_types.values(), default=0)
    max_type_share = (max_type_count / len(records)) if records else 1.0

    evaluation_tasks = list(read_jsonl(evaluation_path))
    candidate_records = [
        {
            "record_id": f"portfolio-{seed['id']}",
            "source_seed_id": seed["id"],
            "messages": [{"role": "user", "content": seed["user_request"]}],
        }
        for seed in records
    ]
    collisions = cross_split_prompt_collisions(
        candidate_records,
        evaluation_tasks,
        threshold=float(config["cross_split_prompt_threshold"]),
    )
    isolation_collisions = [
        {
            "source_seed_id": item["source_seed_id"],
            "evaluation_task_id": item["evaluation_task_id"],
            "similarity": item["similarity"],
            "threshold": item["threshold"],
        }
        for item in collisions
    ]
    count_below_floor = len(records) < int(config["minimum_project_authored_briefs"])
    concentration_failure = max_type_share > float(config["maximum_single_task_type_share"])
    failed = bool(invalid or duplicate_ids or duplicate_prompts or missing_forms or isolation_collisions or count_below_floor or concentration_failure)
    return {
        "stage": "source_portfolio_audit",
        "created_at": utc_now(),
        "project_name": "DukeOTR",
        "catalogs": catalogs,
        "evaluation_file": evaluation_path,
        "source_brief_count": len(records),
        "minimum_project_authored_briefs": config["minimum_project_authored_briefs"],
        "source_brief_count_status": "fail" if count_below_floor else "pass",
        "catalog_counts": dict(sorted(catalog_counts.items())),
        "task_type_counts": dict(sorted(task_types.items())),
        "difficulty_counts": dict(sorted(difficulties.items())),
        "maximum_single_task_type_share": float(config["maximum_single_task_type_share"]),
        "observed_maximum_task_type_share": round(max_type_share, 6),
        "task_type_concentration_status": "fail" if concentration_failure else "pass",
        "required_task_form_evidence": forms,
        "missing_required_task_forms": missing_forms,
        "invalid_seed_records": invalid,
        "duplicate_seed_ids": sorted(set(duplicate_ids)),
        "exact_duplicate_source_prompts": duplicate_prompts,
        "evaluation_wording_collisions": isolation_collisions,
        "status": "fail" if failed else "pass",
        "non_claim": "This report measures curated source-brief variety and isolation only; it does not certify generated candidates, review quality, final data, training, or model behavior.",
    }


def run(arguments: argparse.Namespace) -> int:
    config = read_json(arguments.config)
    if not isinstance(config, dict):
        raise ValueError("Source portfolio config must be a JSON object")
    _validate_config(config)
    catalogs = arguments.catalog or list(config["source_catalogs"])
    evaluation_path = arguments.evaluation or str(config["evaluation_file"])
    report = audit(config, catalogs, evaluation_path)
    write_json_atomic(arguments.output, report)
    print(
        f"Source portfolio audit: {report['status']} — {report['source_brief_count']} briefs across "
        f"{len(catalogs)} catalogs, {len(report['missing_required_task_forms'])} missing task forms, "
        f"{len(report['evaluation_wording_collisions'])} held-out wording collisions. Report: {arguments.output}"
    )
    if arguments.strict and report["status"] != "pass":
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"source portfolio audit error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
