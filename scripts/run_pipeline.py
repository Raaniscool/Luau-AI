"""Convenience orchestrator for the Phase 1 data pipeline.

It performs generation → validation → correction → re-validation → deduplication → final
build. Fine-tuning is deliberately not included: it requires a separately verified GPU/cloud
environment and a recorded baseline.
"""

from __future__ import annotations

# Support both `python -m scripts.name` and `python scripts/name.py` from the repo.
if __package__ in {None, ""}:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.lib.io_utils import read_jsonl, utc_now, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Run the auditable Roblox/Luau data pipeline")
    value.add_argument("--model", default="qwen3:4b")
    value.add_argument("--host", default=None)
    value.add_argument("--seeds", default="raw_data/roblox_luau_seed_tasks.jsonl")
    value.add_argument("--evaluation", default="evaluation_data/roblox_luau_eval.jsonl")
    value.add_argument("--limit", type=int, default=0, help="Pilot with a small number of source briefs; 0 means all")
    value.add_argument("--variants", type=int, default=1)
    value.add_argument("--with-baseline", action="store_true", help="Capture base-model answers before data generation")
    value.add_argument("--baseline-model", default="qwen3:4b")
    value.add_argument("--baseline-limit", type=int, default=0)
    value.add_argument("--dry-run", action="store_true", help="Write a pipeline plan only")
    value.add_argument("--report", default="reports/pipeline_run.json")
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
    generated = "generated_data/generated_examples.jsonl"
    validated = "validated_data/validated_examples.jsonl"
    corrected = "validated_data/corrected_candidates.jsonl"
    corrected_validated = "validated_data/corrected_validated_examples.jsonl"
    deduplicated = "validated_data/deduplicated_examples.jsonl"
    shared = ["--host", arguments.host] if arguments.host else []
    plan = {
        "stage": "phase_1_pipeline",
        "created_at": utc_now(),
        "status": "planned" if arguments.dry_run else "running",
        "model": arguments.model,
        "seeds": arguments.seeds,
        "evaluation": arguments.evaluation,
        "limit": arguments.limit,
        "variants": arguments.variants,
        "baseline_requested": arguments.with_baseline,
        "training_note": "This pipeline never invokes fine-tuning.",
    }
    if arguments.dry_run:
        plan["commands"] = [
            command("scripts.generate_examples", "--seeds", arguments.seeds, "--output", generated, "--model", arguments.model, "--limit", str(arguments.limit), "--variants", str(arguments.variants)),
            command("scripts.validate_examples", "--input", generated, "--output", validated, "--model", arguments.model),
            command("scripts.correct_examples", "--input", validated, "--output", corrected, "--model", arguments.model),
            command("scripts.validate_examples", "--input", corrected, "--output", corrected_validated, "--model", arguments.model),
            command("scripts.deduplicate_examples", "--input", validated, "--input", corrected_validated, "--output", deduplicated),
            command("scripts.build_datasets", "--input", deduplicated, "--evaluation", arguments.evaluation),
        ]
        write_json_atomic(arguments.report, plan)
        print(f"Pipeline plan written to {arguments.report}; no model or training was run.")
        return 0

    results: list[dict[str, Any]] = []
    if arguments.with_baseline:
        baseline_args = command("scripts.run_baseline", "--model", arguments.baseline_model, "--limit", str(arguments.baseline_limit), *shared)
        if not run_command("baseline", baseline_args, results):
            plan.update({"status": "failed", "results": results, "completed_at": utc_now()})
            write_json_atomic(arguments.report, plan)
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
        write_json_atomic(arguments.report, plan)
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
        write_json_atomic(arguments.report, plan)
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
        write_json_atomic(arguments.report, plan)
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
            write_json_atomic(arguments.report, plan)
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
        write_json_atomic(arguments.report, plan)
        return 2
    build_args = command(
        "scripts.build_datasets",
        "--input", deduplicated,
        "--evaluation", arguments.evaluation,
        "--strict",
    )
    success = run_command("final_dataset_creation", build_args, results)
    plan.update({"status": "completed" if success else "failed", "results": results, "completed_at": utc_now()})
    write_json_atomic(arguments.report, plan)
    return 0 if success else 2


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"pipeline error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
