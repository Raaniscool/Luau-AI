"""Audit DukeOTR's permanently held-out evaluation suite and expansion plan.

The audit reports coverage and authoring gaps without reading evaluation content into any
training/generation role. It never generates answers, scores a model, or treats the planned
100–300 task target as already achieved.
"""

from __future__ import annotations

# Support both `python -m scripts.audit_evaluation_suite` and direct Windows invocation.
if __package__ in {None, ""}:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import sys
from collections import Counter
from typing import Any

from scripts.lib.evaluation import validate_evaluation_tasks
from scripts.lib.io_utils import read_json, read_jsonl, sha256_file, utc_now, write_json_atomic


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Audit held-out DukeOTR evaluation coverage and readiness plan")
    value.add_argument("--plan", default="evaluation_data/coverage_plan.json")
    value.add_argument("--suite", default=None, help="Override the suite file declared by the plan")
    value.add_argument("--output", default="reports/evaluation_suite_audit.json")
    value.add_argument("--strict", action="store_true", help="Fail for an invalid suite/plan or a regression below the current suite floor")
    value.add_argument(
        "--require-mature-target",
        action="store_true",
        help="Also fail unless the planned minimum 100-task mature suite and every mature track floor are reached",
    )
    return value


def _error(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _load_plan(path: str) -> dict[str, Any]:
    plan = read_json(path)
    if not isinstance(plan, dict):
        raise ValueError("Evaluation coverage plan must be a JSON object")
    errors: list[str] = []
    _error(errors, plan.get("schema_version") == "1.0", "schema_version must be '1.0'")
    _error(errors, plan.get("project_name") == "DukeOTR", "project_name must be DukeOTR")
    _error(errors, plan.get("suite_policy") == "permanently_held_out_not_generation_or_training_input", "suite_policy must preserve permanent isolation")
    _error(errors, isinstance(plan.get("suite_file"), str) and bool(plan["suite_file"]), "suite_file must be a non-empty string")
    floor = plan.get("current_suite_floor")
    _error(errors, isinstance(floor, int) and floor > 0, "current_suite_floor must be a positive integer")
    mature = plan.get("mature_suite_target")
    _error(errors, isinstance(mature, dict), "mature_suite_target must be an object")
    if isinstance(mature, dict):
        minimum = mature.get("minimum_tasks")
        recommended = mature.get("recommended_tasks")
        maximum = mature.get("maximum_tasks")
        _error(
            errors,
            all(isinstance(item, int) for item in (minimum, recommended, maximum)) and minimum <= recommended <= maximum,
            "mature_suite_target must have ordered integer minimum/recommended/maximum counts",
        )
    tracks = plan.get("coverage_tracks")
    _error(errors, isinstance(tracks, list) and bool(tracks), "coverage_tracks must be a non-empty list")
    ids: set[str] = set()
    if isinstance(tracks, list):
        for index, track in enumerate(tracks, start=1):
            if not isinstance(track, dict):
                errors.append(f"coverage track {index} must be an object")
                continue
            track_id = track.get("id")
            if not isinstance(track_id, str) or not track_id or track_id in ids:
                errors.append(f"coverage track {index} needs a unique non-empty id")
            else:
                ids.add(track_id)
            tags = track.get("match_tags")
            if not isinstance(tags, list) or not tags or not all(isinstance(tag, str) and tag for tag in tags):
                errors.append(f"coverage track {track_id!r} needs non-empty match_tags")
            minimum = track.get("mature_minimum_tasks")
            if not isinstance(minimum, int) or minimum < 1:
                errors.append(f"coverage track {track_id!r} needs mature_minimum_tasks >= 1")
    forms = plan.get("authoring_form_targets")
    _error(errors, isinstance(forms, list) and len(forms) >= 8 and all(isinstance(item, str) and item for item in forms), "authoring_form_targets must have at least eight named forms")
    if errors:
        raise ValueError("Evaluation coverage plan is invalid:\n- " + "\n- ".join(errors))
    return plan


def audit(plan: dict[str, Any], tasks: list[dict[str, Any]], *, suite_path: str) -> dict[str, Any]:
    validate_evaluation_tasks(tasks)
    task_ids = [str(task["id"]) for task in tasks]
    categories = Counter(str(task.get("category", "unclassified")) for task in tasks)
    difficulties = Counter(str(task.get("difficulty", "unclassified")) for task in tasks)
    tags = Counter(str(tag) for task in tasks for tag in task.get("tags", []) if isinstance(tag, str))
    tracks: list[dict[str, Any]] = []
    for track in plan["coverage_tracks"]:
        matching_ids = sorted(
            str(task["id"])
            for task in tasks
            if set(str(tag) for tag in task.get("tags", []) if isinstance(tag, str)).intersection(track["match_tags"])
        )
        target = int(track["mature_minimum_tasks"])
        tracks.append(
            {
                "id": track["id"],
                "description": track.get("description", ""),
                "match_tags": track["match_tags"],
                "current_task_count": len(matching_ids),
                "mature_minimum_tasks": target,
                "remaining_to_mature_minimum": max(0, target - len(matching_ids)),
                "mature_target_reached": len(matching_ids) >= target,
                "task_ids": matching_ids,
            }
        )
    current_floor = int(plan["current_suite_floor"])
    mature = plan["mature_suite_target"]
    mature_minimum = int(mature["minimum_tasks"])
    plan_status = "mature_target_reached" if len(tasks) >= mature_minimum and all(item["mature_target_reached"] for item in tracks) else "expansion_in_progress"
    return {
        "stage": "evaluation_suite_audit",
        "created_at": utc_now(),
        "project_name": "DukeOTR",
        "suite": suite_path,
        "suite_sha256": sha256_file(suite_path),
        "suite_policy": plan["suite_policy"],
        "task_count": len(tasks),
        "current_suite_floor": current_floor,
        "current_floor_status": "pass" if len(tasks) >= current_floor else "fail",
        "mature_suite_target": mature,
        "mature_suite_status": plan_status,
        "remaining_to_mature_minimum": max(0, mature_minimum - len(tasks)),
        "category_counts": dict(sorted(categories.items())),
        "difficulty_counts": dict(sorted(difficulties.items())),
        "tag_counts": dict(sorted(tags.items())),
        "coverage_tracks": tracks,
        "missing_mature_tracks": [item["id"] for item in tracks if not item["mature_target_reached"]],
        "authoring_form_targets": plan["authoring_form_targets"],
        "planned_authoring_batches": plan.get("planned_authoring_batches", []),
        "isolation_note": "This audit reads held-out data only for evaluation stewardship. Its output must not be supplied to data-generation or SFT roles.",
        "task_ids": task_ids,
    }


def run(arguments: argparse.Namespace) -> int:
    plan = _load_plan(arguments.plan)
    suite_path = arguments.suite or plan["suite_file"]
    tasks = list(read_jsonl(suite_path))
    report = audit(plan, tasks, suite_path=suite_path)
    strict_failure = report["current_floor_status"] != "pass"
    mature_failure = arguments.require_mature_target and report["mature_suite_status"] != "mature_target_reached"
    report["status"] = "fail" if strict_failure or mature_failure else "pass"
    report["strict_mode"] = bool(arguments.strict)
    report["mature_target_required"] = bool(arguments.require_mature_target)
    write_json_atomic(arguments.output, report)
    print(
        f"Evaluation suite audit: {report['status']} — {report['task_count']} tasks, "
        f"current floor {report['current_floor_status']}, mature status {report['mature_suite_status']}. "
        f"Report: {arguments.output}"
    )
    if (arguments.strict and strict_failure) or mature_failure:
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"evaluation suite audit error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
