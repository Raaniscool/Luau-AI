"""Audit the curated DukeOTR Phase-1 Luau-fundamentals source catalog.

This is a source-brief coverage gate. A passing result does not claim that examples were
model-generated, reviewed, deduplicated, or fit for training.
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

from scripts.lib.dedupe import normalized_tokens
from scripts.lib.io_utils import read_jsonl, utc_now, write_json_atomic
from scripts.lib.schema import validate_seed


# These are concept labels in the project-authored seed catalog, rather than fuzzy keywords in
# prose. A new curriculum item should deliberately declare what it teaches.
REQUIRED_CONCEPTS: tuple[str, ...] = (
    "variables",
    "basic types",
    "local scope",
    "operators",
    "conditionals",
    "numeric for loop",
    "generic for loop",
    "while loop",
    "repeat until loop",
    "functions",
    "tables",
    "strings",
    "math",
    "randomness",
    "type checks",
    "type annotations",
    "ModuleScripts",
    "error handling",
    "events",
    "connections",
    "closures",
    "callbacks",
)

REQUIRED_INSTRUCTIONAL_MODES: tuple[str, ...] = (
    "question_answer",
    "natural_language_to_luau",
    "code_explanation",
    "bug_fix",
    "diagnosis_correction",
    "completion",
    "bad_answer_critique",
    "subtle_bug_analysis",
    "runtime_reasoning",
    "requirements_implementation",
    "code_review",
    "output_prediction",
    "refactoring",
    "tradeoff_analysis",
)

REQUIRED_DIFFICULTIES: tuple[str, ...] = ("beginner", "intermediate", "advanced")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Audit DukeOTR Phase-1 Luau coverage without running a model"
    )
    value.add_argument(
        "--seeds",
        default="raw_data/dukeotr_phase1_luau_seed_tasks.jsonl",
        help="Project-authored Phase-1 source briefs",
    )
    value.add_argument(
        "--evaluation",
        default="evaluation_data/roblox_luau_eval.jsonl",
        help="Held-out task catalog used only for collision detection",
    )
    value.add_argument(
        "--project-config",
        default="configs/dukeotr_project.json",
        help="DukeOTR identity/version configuration",
    )
    value.add_argument(
        "--output",
        default="reports/dukeotr_phase1_curriculum_audit.json",
        help="Ignored local JSON report path",
    )
    value.add_argument("--strict", action="store_true", help="Return non-zero for any failed gate")
    return value


def _normalised_metadata(record: dict[str, Any]) -> set[str]:
    return {
        str(term).strip().casefold()
        for term in [*record.get("concepts", []), *record.get("tags", [])]
        if str(term).strip()
    }


def _normalised_prompt(record: dict[str, Any], key: str) -> str:
    return " ".join(normalized_tokens(str(record.get(key, ""))))


def _read_project_config(path: str) -> dict[str, Any]:
    import json

    content = Path(path).read_text(encoding="utf-8")
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError(f"Project config must be a JSON object: {path}")
    return value


def audit(
    seeds: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    project_config: dict[str, Any],
    *,
    seed_path: str,
    evaluation_path: str,
    project_config_path: str,
) -> dict[str, Any]:
    invalid_records = {
        str(seed.get("id", f"row-{index + 1}")): problems
        for index, seed in enumerate(seeds)
        if (problems := validate_seed(seed))
    }
    ids = [str(seed.get("id", "")) for seed in seeds]
    duplicate_ids = sorted({item for item in ids if ids.count(item) > 1})

    prompt_to_ids: dict[str, list[str]] = {}
    for seed in seeds:
        prompt_to_ids.setdefault(_normalised_prompt(seed, "user_request"), []).append(str(seed.get("id", "")))
    duplicate_prompts = sorted(
        ids_for_prompt for prompt, ids_for_prompt in prompt_to_ids.items() if prompt and len(ids_for_prompt) > 1
    )

    concepts = set().union(*(_normalised_metadata(seed) for seed in seeds)) if seeds else set()
    missing_concepts = [
        concept for concept in REQUIRED_CONCEPTS if concept.casefold() not in concepts
    ]
    task_types = Counter(str(seed.get("task_type", "")) for seed in seeds)
    missing_instructional_modes = [
        mode for mode in REQUIRED_INSTRUCTIONAL_MODES if mode not in task_types
    ]
    difficulties = Counter(str(seed.get("difficulty", "")) for seed in seeds)
    missing_difficulties = [
        difficulty for difficulty in REQUIRED_DIFFICULTIES if difficulty not in difficulties
    ]

    evaluation_ids = {str(task.get("id", "")) for task in evaluation}
    evaluation_prompts = {
        _normalised_prompt(task, "prompt") for task in evaluation if _normalised_prompt(task, "prompt")
    }
    id_collisions = sorted(set(ids) & evaluation_ids)
    prompt_collisions = sorted(
        str(seed.get("id", ""))
        for seed in seeds
        if _normalised_prompt(seed, "user_request") in evaluation_prompts
    )

    project_name = project_config.get("project_name")
    base_model = project_config.get("starting_ollama_model")
    planned_dataset = project_config.get("dataset_identity", {}).get("planned_first_dataset_version")
    model_identity = project_config.get("model_identity", {})
    planned_model = model_identity.get("planned_first_adapter_version")
    planned_candidate_tag = model_identity.get("planned_versioned_ollama_tag")
    planned_release_tag = model_identity.get("planned_release_ollama_tag")
    identity_errors: list[str] = []
    if project_name != "DukeOTR":
        identity_errors.append("project_name must be DukeOTR")
    if base_model != "qwen3:4b":
        identity_errors.append("starting_ollama_model must be qwen3:4b")
    if planned_dataset != "dukeotr_dataset_v1":
        identity_errors.append("planned dataset version must be dukeotr_dataset_v1")
    if planned_model != "dukeotr_v1":
        identity_errors.append("planned adapter version must be dukeotr_v1")
    if planned_candidate_tag != "dukeotr-v1":
        identity_errors.append("planned versioned Ollama tag must be dukeotr-v1")
    if planned_release_tag != "dukeotr":
        identity_errors.append("planned release Ollama tag must be dukeotr")

    failed = bool(
        invalid_records
        or duplicate_ids
        or duplicate_prompts
        or missing_concepts
        or missing_instructional_modes
        or missing_difficulties
        or id_collisions
        or prompt_collisions
        or identity_errors
    )
    return {
        "stage": "dukeotr_phase1_curriculum_audit",
        "created_at": utc_now(),
        "project_name": project_name,
        "project_config": project_config_path,
        "seed_file": seed_path,
        "evaluation_file": evaluation_path,
        "source_brief_count": len(seeds),
        "task_type_counts": dict(sorted(task_types.items())),
        "difficulty_counts": dict(sorted(difficulties.items())),
        "required_concepts": list(REQUIRED_CONCEPTS),
        "missing_concepts": missing_concepts,
        "required_instructional_modes": list(REQUIRED_INSTRUCTIONAL_MODES),
        "missing_instructional_modes": missing_instructional_modes,
        "missing_difficulties": missing_difficulties,
        "invalid_seed_records": invalid_records,
        "duplicate_seed_ids": duplicate_ids,
        "exact_duplicate_seed_prompts": duplicate_prompts,
        "evaluation_id_collisions": id_collisions,
        "evaluation_prompt_collisions": prompt_collisions,
        "identity_errors": identity_errors,
        "status": "fail" if failed else "pass",
        "non_claim": "This report audits curated source-brief coverage only; it does not certify generated or final training examples.",
    }


def run(arguments: argparse.Namespace) -> int:
    seeds = list(read_jsonl(arguments.seeds))
    evaluation = list(read_jsonl(arguments.evaluation))
    project_config = _read_project_config(arguments.project_config)
    report = audit(
        seeds,
        evaluation,
        project_config,
        seed_path=arguments.seeds,
        evaluation_path=arguments.evaluation,
        project_config_path=arguments.project_config,
    )
    write_json_atomic(arguments.output, report)
    print(
        f"DukeOTR Phase-1 curriculum audit: {report['status']} — "
        f"{report['source_brief_count']} briefs, "
        f"{len(report['missing_concepts'])} missing concepts, "
        f"{len(report['missing_instructional_modes'])} missing instructional modes. "
        f"Report: {arguments.output}"
    )
    if arguments.strict and report["status"] != "pass":
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"DukeOTR curriculum audit error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
