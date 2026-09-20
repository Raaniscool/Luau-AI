"""Stage 5: create the final train/development corpus behind hard quality gates.

The held-out evaluation JSONL is read only to prevent leakage. It is never concatenated
into a training artifact.
"""

from __future__ import annotations

# Support both `python -m scripts.name` and `python scripts/name.py` from the repo.
if __package__ in {None, ""}:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from scripts.lib.dedupe import cross_split_prompt_collisions
from scripts.lib.io_utils import canonical_json, read_json, read_jsonl, sha256_text, text_from_message, utc_now, write_json_atomic, write_jsonl_atomic
from scripts.lib.schema import STATIC_CHECKER_VERSION, quality_gate_status, record_fingerprint


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Build final quality-gated Roblox/Luau training data")
    value.add_argument("--input", default="validated_data/dukeotr_phase1_candidates.deduplicated.jsonl")
    value.add_argument("--evaluation", default="evaluation_data/roblox_luau_eval.jsonl")
    value.add_argument("--output-dir", default="training_data/dukeotr_phase1_candidates")
    value.add_argument("--report", default=None)
    value.add_argument("--config", default="configs/pipeline.json")
    value.add_argument("--validation-ratio", type=float, default=None)
    value.add_argument("--strict", action="store_true", help="Return nonzero if any leakage collision is found")
    value.add_argument("--allow-empty", action="store_true", help="For pipeline plumbing checks only; never use for training")
    return value


def validate_evaluation_tasks(tasks: list[dict[str, Any]]) -> None:
    ids: set[str] = set()
    errors: list[str] = []
    for task in tasks:
        task_id = task.get("id")
        if not isinstance(task_id, str) or not task_id:
            errors.append("evaluation task has no id")
            continue
        if task_id in ids:
            errors.append(f"duplicate evaluation id {task_id}")
        ids.add(task_id)
        if task.get("split") != "evaluation":
            errors.append(f"{task_id}: split must be evaluation")
        if not isinstance(task.get("prompt"), str) or not task["prompt"].strip():
            errors.append(f"{task_id}: prompt missing")
        rubric = task.get("rubric")
        if not isinstance(rubric, list) or not rubric:
            errors.append(f"{task_id}: rubric missing")
        elif sum(item.get("max_points", 0) for item in rubric if isinstance(item, dict)) != 100:
            errors.append(f"{task_id}: rubric must total 100 points")
    if errors:
        raise ValueError("Evaluation suite invalid:\n- " + "\n- ".join(errors))


def deterministic_partitions(records: list[dict[str, Any]], ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create a reproducible stratified development split without touching held-out eval."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        metadata = record.get("metadata", {})
        groups[(str(metadata.get("task_type")), str(metadata.get("difficulty")))].append(record)
    train: list[dict[str, Any]] = []
    development: list[dict[str, Any]] = []
    for _key, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda entry: sha256_text(str(entry.get("record_id"))))
        desired = round(len(ordered) * ratio)
        # Small groups remain entirely in train so every category has examples. Larger
        # groups contribute at least one development item but never lose all train items.
        if len(ordered) >= 5:
            count = min(max(desired, 1), len(ordered) - 1)
        else:
            count = 0
        development.extend(ordered[:count])
        train.extend(ordered[count:])
    return train, development


def mark_partition(records: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in records:
        clone = deepcopy(record)
        clone.setdefault("metadata", {})["dataset_partition"] = name
        clone["metadata"]["finalized_at"] = utc_now()
        clone["metadata"]["final_record_fingerprint"] = record_fingerprint(clone)
        result.append(clone)
    return result


def coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    topics = Counter()
    task_types = Counter()
    difficulties = Counter()
    for record in records:
        metadata = record.get("metadata", {})
        task_types[str(metadata.get("task_type"))] += 1
        difficulties[str(metadata.get("difficulty"))] += 1
        for topic in metadata.get("topics", []):
            topics[str(topic)] += 1
    return {
        "task_types": dict(sorted(task_types.items())),
        "difficulties": dict(sorted(difficulties.items())),
        "topics": dict(sorted(topics.items())),
    }


def run(arguments: argparse.Namespace) -> int:
    config = read_json(arguments.config)
    dedupe_config = dict(config.get("deduplication", {}))
    build_config = dict(config.get("dataset_build", {}))
    ratio = arguments.validation_ratio if arguments.validation_ratio is not None else float(build_config.get("validation_ratio", 0.08))
    if not 0 <= ratio < 1:
        raise ValueError("--validation-ratio must be in [0, 1)")
    records = list(read_jsonl(arguments.input))
    tasks = list(read_jsonl(arguments.evaluation))
    validate_evaluation_tasks(tasks)
    if not records and not arguments.allow_empty:
        raise ValueError("No deduplicated candidates supplied")

    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    minimum_characters = int(build_config.get("minimum_assistant_characters", 160))
    maximum_characters = int(build_config.get("maximum_assistant_characters", 24000))
    for record in records:
        passes, reasons = quality_gate_status(record)
        assistant_length = len(text_from_message(record, "assistant"))
        if assistant_length < minimum_characters or assistant_length > maximum_characters:
            passes = False
            reasons.append(f"assistant_length_outside_{minimum_characters}_{maximum_characters}")
        if str(record.get("source_seed_id", "")).startswith("eval-"):
            passes = False
            reasons.append("evaluation_source_id_not_allowed")
        if passes:
            eligible.append(record)
        else:
            excluded.append({"record_id": record.get("record_id"), "reasons": reasons})

    collisions = cross_split_prompt_collisions(
        eligible,
        tasks,
        threshold=float(dedupe_config.get("cross_split_prompt_threshold", 0.93)),
        shingle_size=int(dedupe_config.get("shingle_size", 3)),
    )
    collided_ids = {str(collision["record_id"]) for collision in collisions}
    leakage_excluded = [record for record in eligible if str(record.get("record_id")) in collided_ids]
    if leakage_excluded:
        eligible = [record for record in eligible if str(record.get("record_id")) not in collided_ids]
        excluded.extend(
            {
                "record_id": record.get("record_id"),
                "reasons": ["held_out_prompt_collision"],
            }
            for record in leakage_excluded
        )

    if not eligible and not arguments.allow_empty:
        raise ValueError("No records passed schema, validation/review, deduplication, and split-isolation gates")
    train_raw, dev_raw = deterministic_partitions(eligible, ratio)
    train = mark_partition(train_raw, "train")
    development = mark_partition(dev_raw, "development")
    final = mark_partition(eligible, "final")
    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "validation.jsonl"
    final_path = output_dir / "final_dataset.jsonl"
    write_jsonl_atomic(train_path, train)
    write_jsonl_atomic(dev_path, development)
    write_jsonl_atomic(final_path, final)

    manifest = {
        "schema_version": "1.0",
        "stage": "final_dataset_creation",
        "created_at": utc_now(),
        "source_input": str(arguments.input),
        "held_out_evaluation": str(arguments.evaluation),
        "input_records": len(records),
        "accepted_records": len(final),
        "excluded_records": excluded,
        "held_out_prompt_collisions": collisions,
        "partition_counts": {"train": len(train), "development": len(development), "final": len(final)},
        "validation_ratio_requested": ratio,
        "coverage": coverage(final),
        "final_record_fingerprints_sha256": sha256_text(canonical_json(sorted(record_fingerprint(item) for item in final))),
        "files": {"train": str(train_path), "validation": str(dev_path), "final": str(final_path)},
        "quality_gate": (
            f"schema + {STATIC_CHECKER_VERSION} pass + recorded reviewer acceptance + "
            "unique deduplication + held-out collision exclusion"
        ),
    }
    manifest_path = output_dir / "dataset_manifest.json"
    write_json_atomic(manifest_path, manifest)
    report_path = Path(arguments.report) if arguments.report else output_dir / "dataset_build_report.json"
    write_json_atomic(report_path, manifest)
    print(
        f"Built {len(final)} final records ({len(train)} train, {len(development)} development). "
        f"Held-out evaluation remains separate at {arguments.evaluation}."
    )
    if collisions and arguments.strict:
        print("Strict mode: held-out prompt collisions were excluded.", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"dataset build error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
