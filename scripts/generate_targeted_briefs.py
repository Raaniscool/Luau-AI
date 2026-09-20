"""Create deterministic, review-required targeted source briefs from observed failures.

No LLM is called. The output contains task specifications only, never answers or final training
candidates. The held-out suite is read only after construction for local wording-collision
rejection and is never supplied to the construction templates.
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

from scripts.lib.io_utils import read_json, read_jsonl, utc_now, write_json_atomic, write_jsonl_atomic
from scripts.lib.training_factory import (
    TrainingFactoryError,
    assert_safe_factory_output_path,
    build_targeted_drafts,
    factory_draft_eval_collisions,
    registry_file_fingerprint,
    summarize_failures,
    validate_failure_record,
    validate_failure_taxonomy,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Make human-review-required targeted task briefs from stored failure evidence")
    value.add_argument("--failure-input", action="append", required=True, help="Observed failure database JSONL; repeat as needed")
    value.add_argument("--failure-category", required=True, help="Controlled failure category with observed rows")
    value.add_argument("--output", required=True, help="JSONL draft output; never an active source catalog")
    value.add_argument("--report", default=None)
    value.add_argument("--templates", default="training_factory/targeted_brief_templates.json")
    value.add_argument("--taxonomy", default="training_factory/failure_taxonomy.json")
    value.add_argument("--config", default="configs/training_factory.json")
    value.add_argument("--evaluation", default=None, help="Held-out suite for local collision rejection only")
    value.add_argument("--beginner", type=int, default=0)
    value.add_argument("--intermediate", type=int, default=0)
    value.add_argument("--advanced", type=int, default=0)
    value.add_argument("--seed", type=int, default=None)
    value.add_argument("--created-at", default=None, help="UTC timestamp override for byte-reproducible draft content")
    return value


def _profile_for_category(templates: dict[str, Any], taxonomy: dict[str, Any], category: str) -> dict[str, Any]:
    category_info = {entry.get("id"): entry for entry in taxonomy.get("categories", []) if isinstance(entry, dict)}.get(category)
    if not isinstance(category_info, dict):
        raise ValueError(f"Unknown failure category {category!r}")
    target_profile = category_info.get("target_profile")
    profiles = templates.get("profiles")
    if not isinstance(profiles, list):
        raise ValueError("Target templates require profiles list")
    candidates = [
        profile
        for profile in profiles
        if isinstance(profile, dict)
        and profile.get("id") == target_profile
        and category in profile.get("failure_categories", [])
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one targeting profile for {category!r}; found {len(candidates)}")
    return candidates[0]


def run(arguments: argparse.Namespace) -> int:
    assert_safe_factory_output_path(arguments.output)
    if arguments.report:
        assert_safe_factory_output_path(arguments.report)
    if any(value < 0 for value in (arguments.beginner, arguments.intermediate, arguments.advanced)):
        raise ValueError("Requested draft counts cannot be negative")
    requested = arguments.beginner + arguments.intermediate + arguments.advanced
    if requested < 1:
        raise ValueError("Request at least one targeted draft")
    taxonomy = read_json(arguments.taxonomy)
    templates = read_json(arguments.templates)
    config = read_json(arguments.config)
    if not isinstance(taxonomy, dict) or not isinstance(templates, dict) or not isinstance(config, dict):
        raise ValueError("Factory taxonomy, templates, and config must be JSON objects")
    taxonomy_problems = validate_failure_taxonomy(taxonomy)
    if taxonomy_problems:
        raise ValueError(f"Invalid failure taxonomy: {taxonomy_problems}")
    target_policy = config.get("targeted_generation", {})
    allowed_categories = config.get("task_categories")
    if not isinstance(allowed_categories, list) or not allowed_categories or not all(isinstance(item, str) and item for item in allowed_categories):
        raise ValueError("Training Factory config requires a non-empty controlled task_categories list")
    if target_policy.get("requires_observed_failure_evidence") is not True or target_policy.get("requires_human_review_before_promotion") is not True:
        raise ValueError("Training Factory config must require observed evidence and human review")
    maximum = target_policy.get("maximum_drafts_per_request", 250)
    if not isinstance(maximum, int) or requested > maximum:
        raise ValueError(f"Requested {requested} drafts exceeds configured maximum {maximum}")
    failures: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for source in arguments.failure_input:
        for row in read_jsonl(source):
            problems = validate_failure_record(row, taxonomy=taxonomy)
            if problems:
                raise ValueError(f"Invalid failure record in {source}: {problems}")
            if row["failure_id"] in seen_ids:
                raise ValueError(f"Duplicate failure_id across inputs: {row['failure_id']}")
            seen_ids.add(row["failure_id"])
            failures.append(row)
    summary = summarize_failures(failures, taxonomy=taxonomy)
    observed_count = summary["category_counts"].get(arguments.failure_category, 0)
    if not isinstance(observed_count, int) or observed_count < 1:
        raise TrainingFactoryError(
            f"Refusing targeted generation for {arguments.failure_category!r}: no stored failure record supports it"
        )
    profile = _profile_for_category(templates, taxonomy, arguments.failure_category)
    if profile.get("category") not in set(allowed_categories):
        raise ValueError(f"Target profile category {profile.get('category')!r} is absent from controlled task_categories")
    seed = arguments.seed if arguments.seed is not None else int(target_policy.get("seed", 3407))
    created_at = arguments.created_at or utc_now()
    drafts = build_targeted_drafts(
        profile=profile,
        failure_category=arguments.failure_category,
        observed_failure_count=observed_count,
        counts_by_difficulty={
            "beginner": arguments.beginner,
            "intermediate": arguments.intermediate,
            "advanced": arguments.advanced,
        },
        failure_input_sha256=registry_file_fingerprint(arguments.failure_input),
        seed=seed,
        created_at=created_at,
    )
    evaluation_path = arguments.evaluation or config.get("held_out_evaluation")
    if not isinstance(evaluation_path, str) or not evaluation_path:
        raise ValueError("Factory config needs held_out_evaluation")
    collisions = factory_draft_eval_collisions(
        drafts,
        read_jsonl(evaluation_path),
        threshold=float(target_policy.get("cross_split_prompt_threshold", 0.93)),
    )
    if collisions:
        raise TrainingFactoryError(
            f"Refusing to write {len(collisions)} targeted drafts with held-out wording collisions; no draft artifact was written"
        )
    write_jsonl_atomic(arguments.output, drafts)
    report_path = Path(arguments.report) if arguments.report else Path(arguments.output).with_suffix(".report.json")
    report = {
        "schema_version": "1.0",
        "stage": "targeted_brief_planning",
        "created_at": created_at,
        "output": str(arguments.output),
        "failure_category": arguments.failure_category,
        "observed_failure_count": observed_count,
        "failure_inputs": list(arguments.failure_input),
        "failure_input_fingerprint": registry_file_fingerprint(arguments.failure_input),
        "profile": profile["id"],
        "seed": seed,
        "draft_count": len(drafts),
        "draft_counts_by_difficulty": {
            "beginner": arguments.beginner,
            "intermediate": arguments.intermediate,
            "advanced": arguments.advanced,
        },
        "draft_counts_by_depth": {
            depth: sum(draft.get("task_depth") == depth for draft in drafts)
            for depth in ("short", "normal", "deep")
        },
        "evaluation_collision_check": {
            "evaluation_file": evaluation_path,
            "collisions": [],
            "status": "clear",
        },
        "promotion": "not_automatic; every row requires an explicit human decision and a separate promotion artifact",
        "non_claim": "Drafts are source-task plans only. They are not answers, generated candidates, reviewed training data, final data, or evidence that any model was trained.",
    }
    write_json_atomic(report_path, report)
    print(
        f"Wrote {len(drafts)} review-required targeted briefs for observed category {arguments.failure_category!r} to "
        f"{arguments.output}; report: {report_path}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError, TrainingFactoryError) as exc:
        print(f"targeted brief generation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
