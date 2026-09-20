"""Human-gated promotion of selected targeted briefs into a separate curated seed catalog.

Promotion never appends to an active catalog and never makes an answer/training record. A
promoted row still travels through the normal Builder → Reviewer/Tester → Fixer pipeline.
"""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from scripts.lib.dedupe import similarity
from scripts.lib.io_utils import read_json, read_jsonl, utc_now, write_json_atomic, write_jsonl_atomic
from scripts.lib.schema import validate_seed
from scripts.lib.training_factory import (
    TrainingFactoryError,
    assert_safe_factory_output_path,
    factory_draft_eval_collisions,
    validate_factory_draft,
    validate_failure_taxonomy,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Apply explicit human decisions to targeted DukeOTR task briefs")
    value.add_argument("--drafts", required=True)
    value.add_argument("--decisions", required=True, help="JSONL rows: draft_id, decision accept/reject, reviewer, notes")
    value.add_argument("--output", required=True, help="New reviewed seed catalog; never an existing active catalog")
    value.add_argument("--report", default=None)
    value.add_argument("--taxonomy", default="training_factory/failure_taxonomy.json")
    value.add_argument("--config", default="configs/training_factory.json")
    value.add_argument("--evaluation", default=None)
    value.add_argument("--against-catalog", action="append", default=[], help="Existing source catalog to compare locally for duplicate prompts")
    value.add_argument("--threshold", type=float, default=None)
    value.add_argument("--created-at", default=None)
    return value


def _read_decisions(path: str) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        draft_id = row.get("draft_id")
        if not isinstance(draft_id, str) or not draft_id.strip():
            raise ValueError("Every targeted-brief decision requires non-empty draft_id")
        if row.get("decision") not in {"accept", "reject"}:
            raise ValueError(f"Decision for {draft_id!r} must be accept or reject")
        for key in ("reviewer", "notes"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Decision for {draft_id!r} needs non-empty {key}")
        if draft_id in decisions:
            raise ValueError(f"Duplicate targeted-brief decision for {draft_id!r}")
        decisions[draft_id] = row
    return decisions


def _seed_from_accepted_draft(draft: dict[str, Any], decision: dict[str, Any], *, created_at: str) -> dict[str, Any]:
    source = deepcopy(draft["source"])
    source.update(
        {
            "kind": "project_authored",
            "origin": "training_factory_targeted_draft",
            "draft_id": draft["draft_id"],
            "human_reviewer": decision["reviewer"],
            "human_review_notes": decision["notes"],
            "promoted_at": created_at,
        }
    )
    return {
        "id": f"targeted-{draft['draft_id']}",
        "title": draft["title"],
        "task_type": draft["task_type"],
        "difficulty": draft["difficulty"],
        "task_depth": draft["task_depth"],
        "category": draft["category"],
        "difficulty_rationale": draft["difficulty_rationale"],
        "task_depth_rationale": draft["task_depth_rationale"],
        "user_request": draft["user_request"],
        "requirements": deepcopy(draft["requirements"]),
        "concepts": deepcopy(draft["concepts"]),
        "expected_evidence": deepcopy(draft["expected_evidence"]),
        "avoid": deepcopy(draft["avoid"]),
        "tags": [draft["category"], draft["targeting"]["failure_category"], "training_factory_targeted"],
        "split": "train",
        "source": source,
        "targeting": deepcopy(draft["targeting"]),
    }


def _existing_prompts(paths: list[str]) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    for source in paths:
        for seed in read_jsonl(source):
            prompt = seed.get("user_request")
            identifier = seed.get("id")
            if isinstance(prompt, str) and isinstance(identifier, str):
                output.append((identifier, prompt))
    return output


def run(arguments: argparse.Namespace) -> int:
    assert_safe_factory_output_path(arguments.output)
    if arguments.report:
        assert_safe_factory_output_path(arguments.report)
    taxonomy = read_json(arguments.taxonomy)
    config = read_json(arguments.config)
    if not isinstance(taxonomy, dict) or not isinstance(config, dict):
        raise ValueError("Factory taxonomy and config must be JSON objects")
    taxonomy_problems = validate_failure_taxonomy(taxonomy)
    if taxonomy_problems:
        raise ValueError(f"Invalid failure taxonomy: {taxonomy_problems}")
    target_policy = config.get("targeted_generation", {})
    allowed_categories_raw = config.get("task_categories")
    if not isinstance(allowed_categories_raw, list) or not allowed_categories_raw or not all(
        isinstance(item, str) and item for item in allowed_categories_raw
    ):
        raise ValueError("Training Factory config requires a non-empty controlled task_categories list")
    allowed_categories = set(allowed_categories_raw)
    threshold = arguments.threshold if arguments.threshold is not None else float(target_policy.get("deduplication_threshold", 0.82))
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    decisions = _read_decisions(arguments.decisions)
    drafts: list[dict[str, Any]] = []
    seen_draft_ids: set[str] = set()
    for draft in read_jsonl(arguments.drafts):
        problems = validate_factory_draft(draft, allowed_categories=allowed_categories)
        if problems:
            raise ValueError(f"Invalid targeted draft: {problems}")
        draft_id = draft["draft_id"]
        if draft_id in seen_draft_ids:
            raise ValueError(f"Duplicate draft_id {draft_id!r}")
        seen_draft_ids.add(draft_id)
        drafts.append(draft)
    if set(decisions) != seen_draft_ids:
        unknown = sorted(set(decisions) - seen_draft_ids)
        missing = sorted(seen_draft_ids - set(decisions))
        messages: list[str] = []
        if unknown:
            messages.append(f"unknown decision IDs: {', '.join(unknown[:10])}")
        if missing:
            messages.append(f"drafts without an explicit decision: {', '.join(missing[:10])}")
        raise ValueError("; ".join(messages))
    created_at = arguments.created_at or utc_now()
    accepted = [draft for draft in drafts if decisions[draft["draft_id"]]["decision"] == "accept"]
    promoted = [_seed_from_accepted_draft(draft, decisions[draft["draft_id"]], created_at=created_at) for draft in accepted]
    invalid = {seed["id"]: validate_seed(seed) for seed in promoted if validate_seed(seed)}
    if invalid:
        raise ValueError(f"Human-accepted targeted drafts did not produce valid source seeds: {invalid}")
    existing = _existing_prompts(arguments.against_catalog)
    duplicate_pairs: list[dict[str, Any]] = []
    for left_index, seed in enumerate(promoted):
        prompt = seed["user_request"]
        for existing_id, existing_prompt in existing:
            metric = similarity(prompt, existing_prompt)
            if metric.jaccard >= threshold:
                duplicate_pairs.append({"seed_id": seed["id"], "other_id": existing_id, "similarity": round(metric.jaccard, 6)})
        for other in promoted[left_index + 1 :]:
            metric = similarity(prompt, other["user_request"])
            if metric.jaccard >= threshold:
                duplicate_pairs.append({"seed_id": seed["id"], "other_id": other["id"], "similarity": round(metric.jaccard, 6)})
    if duplicate_pairs:
        raise TrainingFactoryError(f"Refusing promotion with {len(duplicate_pairs)} near-duplicate source prompts")
    evaluation_path = arguments.evaluation or config.get("held_out_evaluation")
    if not isinstance(evaluation_path, str) or not evaluation_path:
        raise ValueError("Factory config needs held_out_evaluation")
    # Adapt promoted seeds to draft-shaped prompts only for the existing local-only collision checker.
    promoted_as_drafts = [
        {"draft_id": seed["id"], "user_request": seed["user_request"]}
        for seed in promoted
    ]
    collisions = factory_draft_eval_collisions(
        promoted_as_drafts,
        read_jsonl(evaluation_path),
        threshold=float(target_policy.get("cross_split_prompt_threshold", 0.93)),
    )
    if collisions:
        raise TrainingFactoryError(f"Refusing promotion with {len(collisions)} held-out wording collisions")
    write_jsonl_atomic(arguments.output, promoted)
    report_path = Path(arguments.report) if arguments.report else Path(arguments.output).with_suffix(".promotion_report.json")
    report = {
        "schema_version": "1.0",
        "stage": "targeted_brief_human_promotion",
        "created_at": created_at,
        "draft_input": arguments.drafts,
        "decision_input": arguments.decisions,
        "output": arguments.output,
        "draft_count": len(drafts),
        "accepted_count": len(accepted),
        "rejected_count": len(drafts) - len(accepted),
        "existing_catalogs_compared": list(arguments.against_catalog),
        "near_duplicate_threshold": threshold,
        "held_out_collision_status": "clear",
        "held_out_evaluation_file": evaluation_path,
        "automatic_catalog_append": False,
        "automatic_candidate_promotion": False,
        "non_claim": "Accepted source briefs still require normal Builder, reviewer/tester, correction, deduplication, and final-dataset gates. They are not answers or final training data.",
    }
    write_json_atomic(report_path, report)
    print(
        f"Promoted {len(promoted)} explicitly human-accepted targeted briefs into separate seed catalog {arguments.output}; "
        f"report: {report_path}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError, TrainingFactoryError) as exc:
        print(f"targeted brief promotion error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
