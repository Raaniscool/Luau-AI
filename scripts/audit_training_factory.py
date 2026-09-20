"""Offline strict audit for DukeOTR Training Factory boundaries and sidecar contracts."""

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

from scripts.lib.io_utils import read_json, read_jsonl, utc_now, write_json_atomic
from scripts.lib.schema import validate_seed
from scripts.lib.training_factory import (
    TrainingFactoryError,
    assert_safe_factory_output_path,
    candidate_is_evaluation_like,
    factory_draft_eval_collisions,
    validate_factory_draft,
    validate_failure_record,
    validate_failure_taxonomy,
    validate_ledger_record,
    validate_model_version_registry,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Audit DukeOTR Training Factory schemas, separation, and evidence boundaries")
    value.add_argument("--config", default="configs/training_factory.json")
    value.add_argument("--taxonomy", default=None)
    value.add_argument("--templates", default=None)
    value.add_argument("--versions", default=None)
    value.add_argument("--catalog", action="append", default=[], help="Override/repeat source catalogs")
    value.add_argument("--failure-input", action="append", default=[])
    value.add_argument("--draft-input", action="append", default=[])
    value.add_argument("--ledger-input", action="append", default=[])
    value.add_argument("--evaluation", default=None)
    value.add_argument("--output", default="reports/training_factory/factory_audit.json")
    value.add_argument("--strict", action="store_true")
    return value


def _load_records(paths: list[str]) -> list[tuple[str, dict[str, Any]]]:
    result: list[tuple[str, dict[str, Any]]] = []
    for source in paths:
        result.extend((source, row) for row in read_jsonl(source))
    return result


def _template_problems(
    templates: dict[str, Any], taxonomy: dict[str, Any], *, allowed_task_categories: set[str]
) -> list[dict[str, str]]:
    problems: list[dict[str, str]] = []
    if templates.get("schema_version") != "1.0":
        problems.append({"code": "templates.schema_version", "message": "Expected schema_version '1.0'", "severity": "error"})
    profiles = templates.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        return [*problems, {"code": "templates.profiles", "message": "Templates need non-empty profiles", "severity": "error"}]
    known_categories = {entry.get("id") for entry in taxonomy.get("categories", []) if isinstance(entry, dict)}
    profile_ids: set[str] = set()
    categories_covered: set[str] = set()
    for index, profile in enumerate(profiles, start=1):
        if not isinstance(profile, dict) or not isinstance(profile.get("id"), str) or not profile["id"]:
            problems.append({"code": "templates.profile", "message": f"Profile {index} needs id", "severity": "error"})
            continue
        if profile["id"] in profile_ids:
            problems.append({"code": "templates.profile", "message": f"Duplicate profile id {profile['id']}", "severity": "error"})
        profile_ids.add(profile["id"])
        if profile.get("category") not in allowed_task_categories:
            problems.append({"code": "templates.task_category", "message": f"Profile {profile['id']} has uncontrolled task category", "severity": "error"})
        categories = profile.get("failure_categories")
        if not isinstance(categories, list) or not categories or any(category not in known_categories for category in categories):
            problems.append({"code": "templates.category", "message": f"Profile {profile['id']} has unknown/no failure categories", "severity": "error"})
        else:
            categories_covered.update(categories)
        for field in ("task_forms", "scenarios", "failure_modes", "concepts", "base_requirements", "expected_evidence", "avoid"):
            if not isinstance(profile.get(field), list) or not profile[field] or not all(isinstance(item, str) and item.strip() for item in profile[field]):
                problems.append({"code": "templates.field", "message": f"Profile {profile['id']} needs non-empty {field}", "severity": "error"})
        for difficulty in ("beginner", "intermediate", "advanced"):
            values = profile.get("difficulty_requirements", {}).get(difficulty) if isinstance(profile.get("difficulty_requirements"), dict) else None
            if not isinstance(values, list) or not values:
                problems.append({"code": "templates.difficulty", "message": f"Profile {profile['id']} misses {difficulty} construction", "severity": "error"})
    expected_categories = {entry.get("id") for entry in taxonomy.get("categories", []) if isinstance(entry, dict) and entry.get("target_profile")}
    missing = sorted(expected_categories - categories_covered)
    if missing:
        problems.append({"code": "templates.coverage", "message": f"No targeted template covers: {', '.join(missing)}", "severity": "error"})
    return problems


def audit(arguments: argparse.Namespace) -> dict[str, Any]:
    config = read_json(arguments.config)
    if not isinstance(config, dict):
        raise ValueError("Training Factory config must be a JSON object")
    taxonomy_path = arguments.taxonomy or config.get("failure_taxonomy")
    templates_path = arguments.templates or config.get("targeted_brief_templates")
    versions_path = arguments.versions or config.get("model_version_registry")
    evaluation_path = arguments.evaluation or config.get("held_out_evaluation")
    if not all(isinstance(value, str) and value for value in (taxonomy_path, templates_path, versions_path, evaluation_path)):
        raise ValueError("Training Factory config has incomplete path configuration")
    taxonomy = read_json(taxonomy_path)
    templates = read_json(templates_path)
    versions = read_json(versions_path)
    if not isinstance(taxonomy, dict) or not isinstance(templates, dict) or not isinstance(versions, dict):
        raise ValueError("Taxonomy, templates, and version registry must be JSON objects")
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    if config.get("schema_version") != "1.0" or config.get("project_name") != "DukeOTR":
        errors.append({"code": "config.identity", "message": "Config must be schema 1.0 for DukeOTR", "severity": "error"})
    if config.get("targeted_generation", {}).get("requires_observed_failure_evidence") is not True:
        errors.append({"code": "config.failure_evidence", "message": "Targeting must require observed failure evidence", "severity": "error"})
    if config.get("targeted_generation", {}).get("requires_human_review_before_promotion") is not True:
        errors.append({"code": "config.human_review", "message": "Targeted briefs must require human review", "severity": "error"})
    allowed_categories_raw = config.get("task_categories")
    if not isinstance(allowed_categories_raw, list) or not allowed_categories_raw or not all(
        isinstance(item, str) and item for item in allowed_categories_raw
    ) or len(set(allowed_categories_raw)) != len(allowed_categories_raw):
        errors.append({"code": "config.task_categories", "message": "Config requires unique controlled task_categories", "severity": "error"})
        allowed_categories: set[str] = set()
    else:
        allowed_categories = set(allowed_categories_raw)
    errors.extend(validate_failure_taxonomy(taxonomy))
    errors.extend(_template_problems(templates, taxonomy, allowed_task_categories=allowed_categories))
    errors.extend(validate_model_version_registry(versions))
    catalogs = arguments.catalog or config.get("source_catalogs", [])
    if not isinstance(catalogs, list) or not catalogs:
        errors.append({"code": "catalogs.missing", "message": "At least one source catalog is required", "severity": "error"})
        catalogs = []
    source_rows = _load_records(catalogs)
    invalid_seeds: dict[str, list[dict[str, str]]] = {}
    source_ids: set[str] = set()
    duplicate_source_ids: list[str] = []
    category_counts: Counter[str] = Counter()
    depth_counts: Counter[str] = Counter()
    for source, seed in source_rows:
        identifier = str(seed.get("id", f"{source}:unknown"))
        if identifier in source_ids:
            duplicate_source_ids.append(identifier)
        source_ids.add(identifier)
        issues = validate_seed(seed)
        if issues:
            invalid_seeds[identifier] = issues
        if candidate_is_evaluation_like({"record_id": identifier, "source_seed_id": identifier, "metadata": {"source": seed.get("source", {})}}):
            errors.append({"code": "source.evaluation_isolation", "message": f"Source seed {identifier} has evaluation-like provenance", "severity": "error"})
        seed_category = str(seed.get("category", "unspecified"))
        category_counts[seed_category] += 1
        depth_counts[str(seed.get("task_depth", "unspecified"))] += 1
        if seed_category != "unspecified" and seed_category not in allowed_categories:
            errors.append({"code": "source.task_category", "message": f"Source seed {identifier} has uncontrolled category {seed_category!r}", "severity": "error"})
    if invalid_seeds:
        errors.append({"code": "source.schema", "message": f"{len(invalid_seeds)} source seeds fail canonical validation", "severity": "error"})
    if duplicate_source_ids:
        errors.append({"code": "source.duplicate_id", "message": f"Duplicate source IDs: {', '.join(sorted(set(duplicate_source_ids))[:10])}", "severity": "error"})
    if category_counts.get("unspecified", 0) or depth_counts.get("unspecified", 0):
        warnings.append(
            {
                "code": "source.factory_metadata_optional",
                "message": "Legacy source briefs may omit category/task_depth; new factory-targeted briefs must provide them. This is not a claim that legacy task labels have been inferred.",
                "severity": "warning",
            }
        )
    failure_rows = _load_records(arguments.failure_input)
    failure_ids: set[str] = set()
    invalid_failure_count = 0
    for source, row in failure_rows:
        problems = validate_failure_record(row, taxonomy=taxonomy)
        if problems:
            invalid_failure_count += 1
        identifier = row.get("failure_id")
        if isinstance(identifier, str):
            if identifier in failure_ids:
                errors.append({"code": "failure.duplicate_id", "message": f"Duplicate failure ID {identifier}", "severity": "error"})
            failure_ids.add(identifier)
    if invalid_failure_count:
        errors.append({"code": "failure.schema", "message": f"{invalid_failure_count} failure rows are invalid", "severity": "error"})
    drafts = _load_records(arguments.draft_input)
    invalid_draft_count = 0
    draft_rows = [row for _source, row in drafts]
    draft_depth_counts: Counter[str] = Counter()
    draft_category_counts: Counter[str] = Counter()
    for _source, draft in drafts:
        if validate_factory_draft(draft, allowed_categories=allowed_categories):
            invalid_draft_count += 1
        draft_depth_counts[str(draft.get("task_depth", "unspecified"))] += 1
        draft_category_counts[str(draft.get("category", "unspecified"))] += 1
    if invalid_draft_count:
        errors.append({"code": "draft.schema", "message": f"{invalid_draft_count} targeted drafts are invalid", "severity": "error"})
    draft_collisions = factory_draft_eval_collisions(
        draft_rows,
        read_jsonl(evaluation_path),
        threshold=float(config.get("targeted_generation", {}).get("cross_split_prompt_threshold", 0.93)),
    ) if draft_rows else []
    if draft_collisions:
        errors.append({"code": "draft.evaluation_isolation", "message": f"{len(draft_collisions)} targeted draft/evaluation wording collisions", "severity": "error"})
    ledgers = _load_records(arguments.ledger_input)
    invalid_ledger_count = sum(bool(validate_ledger_record(row, taxonomy=taxonomy)) for _source, row in ledgers)
    if invalid_ledger_count:
        errors.append({"code": "ledger.schema", "message": f"{invalid_ledger_count} lineage ledger rows are invalid", "severity": "error"})
    return {
        "schema_version": "1.0",
        "stage": "training_factory_audit",
        "created_at": utc_now(),
        "config": arguments.config,
        "catalogs": catalogs,
        "held_out_evaluation": evaluation_path,
        "source_brief_count": len(source_rows),
        "source_category_counts": dict(sorted(category_counts.items())),
        "source_task_depth_counts": dict(sorted(depth_counts.items())),
        "failure_record_count": len(failure_rows),
        "targeted_draft_count": len(drafts),
        "targeted_draft_category_counts": dict(sorted(draft_category_counts.items())),
        "targeted_draft_depth_counts": dict(sorted(draft_depth_counts.items())),
        "lineage_ledger_count": len(ledgers),
        "targeted_draft_evaluation_collisions": [
            {
                "draft_id": collision.get("record_id"),
                "evaluation_task_id": collision.get("evaluation_task_id"),
                "similarity": collision.get("similarity"),
            }
            for collision in draft_collisions
        ],
        "errors": errors,
        "warnings": warnings,
        "status": "fail" if errors else "pass",
        "non_claim": "This audit checks contracts, provenance boundaries, and visible evidence only. Passing does not independently establish every Roblox/Luau fact, model quality, a built final corpus, trained adapter, or released DukeOTR model.",
    }


def run(arguments: argparse.Namespace) -> int:
    assert_safe_factory_output_path(arguments.output)
    report = audit(arguments)
    write_json_atomic(arguments.output, report)
    print(
        f"Training Factory audit: {report['status']} — {report['source_brief_count']} source briefs, "
        f"{report['failure_record_count']} failure records, {report['targeted_draft_count']} targeted drafts. "
        f"Report: {arguments.output}"
    )
    if arguments.strict and report["status"] != "pass":
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError, TrainingFactoryError) as exc:
        print(f"training factory audit error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
