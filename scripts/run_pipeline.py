"""Convenience orchestrator for the DukeOTR staged data pipeline.

It defaults to the Phase-1 Luau-fundamentals source catalog and performs generation →
validation → correction → re-validation → deduplication → final build. Fine-tuning is
deliberately not included: it requires a separately verified GPU/cloud environment and a
recorded baseline.
"""

from __future__ import annotations

# Support both `python -m scripts.name` and `python scripts/name.py` from the repo.
if __package__ in {None, ""}:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.lib.io_utils import read_jsonl, utc_now, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}")


def stage_paths(run_id: str) -> dict[str, str]:
    """Keep a pilot's ignored artifacts together without reusing generic filenames."""
    return {
        "generated": f"generated_data/{run_id}.generated.jsonl",
        "validated": f"validated_data/{run_id}.validated.jsonl",
        "corrected": f"validated_data/{run_id}.corrected.jsonl",
        "corrected_validated": f"validated_data/{run_id}.corrected_validated.jsonl",
        "deduplicated": f"validated_data/{run_id}.deduplicated.jsonl",
        "training_dir": f"training_data/{run_id}",
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Run the auditable Roblox/Luau data pipeline")
    value.add_argument("--model", default="qwen3:4b")
    value.add_argument("--host", default=None)
    value.add_argument(
        "--seeds",
        default="raw_data/dukeotr_phase1_luau_seed_tasks.jsonl",
        help="Curated train source briefs; defaults to DukeOTR Phase-1 Luau fundamentals",
    )
    value.add_argument("--evaluation", default="evaluation_data/roblox_luau_eval.jsonl")
    value.add_argument("--limit", type=int, default=0, help="Pilot with a small number of source briefs; 0 means all")
    value.add_argument("--variants", type=int, default=1)
    value.add_argument("--with-baseline", action="store_true", help="Capture base-model answers before data generation")
    value.add_argument("--baseline-model", default="qwen3:4b")
    value.add_argument("--baseline-limit", type=int, default=0)
    value.add_argument(
        "--run-id",
        default="dukeotr_phase1",
        help="Safe stage-output identifier; choose a new one for each pilot rather than overwriting artifacts",
    )
    value.add_argument("--overwrite", action="store_true", help="Allow replacing prior artifacts for this exact --run-id")
    value.add_argument("--dry-run", action="store_true", help="Write a pipeline plan only")
    value.add_argument("--report", default=None, help="Defaults to reports/<run-id>.pipeline_run.json")
    return value


def command(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


def run_command(name: str, args: list[str], results: list[dict[str, Any]]) -> bool:
    print(f"\n== {name} ==\n$ {' '.join(args)}", flush=True)
    completed = subprocess.run(args, cwd=ROOT, check=False)
    results.append({"stage": name, "command": args, "returncode": completed.returncode})
    return completed.returncode == 0


def run(arguments: argparse.Namespace) -> int:
    if arguments.limit < 0 or arguments.variants < 1 or arguments.baseline_limit < 0:
        raise ValueError("limits must be non-negative and --variants must be at least 1")
    if not _RUN_ID_RE.fullmatch(arguments.run_id):
        raise ValueError("--run-id must be 1-80 letters, numbers, underscores, or hyphens and start with a letter/number")
    paths = stage_paths(arguments.run_id)
    generated = paths["generated"]
    validated = paths["validated"]
    corrected = paths["corrected"]
    corrected_validated = paths["corrected_validated"]
    deduplicated = paths["deduplicated"]
    training_dir = paths["training_dir"]
    report_path = arguments.report or f"reports/{arguments.run_id}.pipeline_run.json"
    existing = [
        path
        for path in [generated, validated, corrected, corrected_validated, deduplicated, training_dir, report_path]
        if Path(path).exists()
    ]
    if existing and not arguments.overwrite:
        raise ValueError(
            "Refusing to overwrite prior pipeline artifacts: "
            + ", ".join(existing)
            + ". Choose a new --run-id, archive compatible material in legacy_data, or use --overwrite only after review."
        )
    shared = ["--host", arguments.host] if arguments.host else []
    plan = {
        "stage": "dukeotr_phase_1_pipeline",
        "created_at": utc_now(),
        "status": "planned" if arguments.dry_run else "running",
        "run_id": arguments.run_id,
        "model": arguments.model,
        "seeds": arguments.seeds,
        "evaluation": arguments.evaluation,
        "limit": arguments.limit,
        "variants": arguments.variants,
        "baseline_requested": arguments.with_baseline,
        "stage_paths": paths,
        "training_note": "This pipeline never invokes fine-tuning or assigns a final dataset version.",
    }
    if arguments.dry_run:
        plan["commands"] = [
            command("scripts.generate_examples", "--seeds", arguments.seeds, "--output", generated, "--model", arguments.model, "--limit", str(arguments.limit), "--variants", str(arguments.variants)),
            command("scripts.validate_examples", "--input", generated, "--output", validated, "--model", arguments.model),
            command("scripts.correct_examples", "--input", validated, "--output", corrected, "--model", arguments.model),
            command("scripts.validate_examples", "--input", corrected, "--output", corrected_validated, "--model", arguments.model),
            command("scripts.deduplicate_examples", "--input", validated, "--input", corrected_validated, "--output", deduplicated),
            command("scripts.build_datasets", "--input", deduplicated, "--evaluation", arguments.evaluation, "--output-dir", training_dir, "--strict"),
        ]
        write_json_atomic(report_path, plan)
        print(f"Pipeline plan written to {report_path}; no model or training was run.")
        return 0

    results: list[dict[str, Any]] = []
    if arguments.with_baseline:
        baseline_args = command("scripts.run_baseline", "--model", arguments.baseline_model, "--limit", str(arguments.baseline_limit), *shared)
        if not run_command("baseline", baseline_args, results):
            plan.update({"status": "failed", "results": results, "completed_at": utc_now()})
            write_json_atomic(report_path, plan)
            return 2
    generate_args = command(
        "scripts.generate_examples",
        "--seeds", arguments.seeds,
        "--output", generated,
        "--model", arguments.model,
        "--limit", str(arguments.limit),
        "--variants", str(arguments.variants),
        "--continue-on-error",
        *shared,
    )
    if not run_command("generation", generate_args, results):
        plan.update({"status": "failed", "results": results, "completed_at": utc_now()})
        write_json_atomic(report_path, plan)
        return 2
    validate_args = command(
        "scripts.validate_examples",
        "--input", generated,
        "--output", validated,
        "--model", arguments.model,
        "--allow-review-errors",
        *shared,
    )
    if not run_command("validation", validate_args, results):
        plan.update({"status": "failed", "results": results, "completed_at": utc_now()})
        write_json_atomic(report_path, plan)
        return 2
    correction_args = command(
        "scripts.correct_examples",
        "--input", validated,
        "--output", corrected,
        "--model", arguments.model,
        "--continue-on-error",
        *shared,
    )
    if not run_command("correction", correction_args, results):
        plan.update({"status": "failed", "results": results, "completed_at": utc_now()})
        write_json_atomic(report_path, plan)
        return 2
    dedupe_inputs = ["--input", validated]
    correction_count = sum(1 for _ in read_jsonl(corrected)) if Path(corrected).exists() else 0
    if correction_count:
        corrected_validate_args = command(
            "scripts.validate_examples",
            "--input", corrected,
            "--output", corrected_validated,
            "--model", arguments.model,
            "--allow-review-errors",
            *shared,
        )
        if not run_command("revalidation", corrected_validate_args, results):
            plan.update({"status": "failed", "results": results, "completed_at": utc_now()})
            write_json_atomic(report_path, plan)
            return 2
        dedupe_inputs.extend(["--input", corrected_validated])
    else:
        results.append({"stage": "revalidation", "status": "skipped", "reason": "No correction candidates"})
    dedupe_args = command(
        "scripts.deduplicate_examples",
        *dedupe_inputs,
        "--output", deduplicated,
        "--evaluation", arguments.evaluation,
    )
    if not run_command("deduplication", dedupe_args, results):
        plan.update({"status": "failed", "results": results, "completed_at": utc_now()})
        write_json_atomic(report_path, plan)
        return 2
    build_args = command(
        "scripts.build_datasets",
        "--input", deduplicated,
        "--evaluation", arguments.evaluation,
        "--output-dir", training_dir,
        "--strict",
    )
    success = run_command("final_dataset_creation", build_args, results)
    plan.update({"status": "completed" if success else "failed", "results": results, "completed_at": utc_now()})
    write_json_atomic(report_path, plan)
    return 0 if success else 2


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"pipeline error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
