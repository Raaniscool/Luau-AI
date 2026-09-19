"""Run a held-out Roblox/Luau evaluation suite against an Ollama model.

The model receives only task prompts and the evaluation system instruction. Rubrics remain
on disk for scoring and are never included in the generation request.
"""

from __future__ import annotations

# Support both `python -m scripts.name` and `python scripts/name.py` from the repo.
if __package__ in {None, ""}:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.lib.evaluation import model_answer_record, task_selection, validate_evaluation_tasks
from scripts.lib.io_utils import canonical_json, read_jsonl, sha256_text, slugify, utc_now, write_json_atomic, write_jsonl_atomic
from scripts.lib.ollama import OllamaClient, OllamaError
from scripts.lib.prompts import EVALUATION_SYSTEM


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Run held-out Roblox/Luau evaluation through Ollama")
    value.add_argument("--evaluation", default="evaluation_data/roblox_luau_eval.jsonl")
    value.add_argument("--model", default=None, help="Ollama model tag to evaluate (required except through run_baseline.py)")
    value.add_argument("--host", default=None)
    value.add_argument("--run-kind", choices=["baseline", "candidate"], default="candidate")
    value.add_argument("--run-id", default=None)
    value.add_argument("--output", default=None, help="Answer JSONL; default is timestamped under reports/evaluations")
    value.add_argument("--report", default=None)
    value.add_argument("--task-id", action="append", default=[])
    value.add_argument("--limit", type=int, default=0)
    value.add_argument("--temperature", type=float, default=0.0)
    value.add_argument("--top-p", type=float, default=0.9)
    value.add_argument("--num-predict", type=int, default=2200)
    value.add_argument("--num-ctx", type=int, default=8192)
    value.add_argument("--timeout-seconds", type=int, default=600, help="Per-response/chunk Ollama HTTP timeout")
    value.add_argument("--seed", type=int, default=3407)
    value.add_argument("--dry-run", action="store_true")
    return value


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run(arguments: argparse.Namespace) -> int:
    if not arguments.model:
        raise ValueError("--model is required (or use run_baseline.py for the qwen3:4b default)")
    if arguments.timeout_seconds < 1:
        raise ValueError("--timeout-seconds must be at least one second")
    tasks = list(read_jsonl(arguments.evaluation))
    validate_evaluation_tasks(tasks)
    selected = task_selection(tasks, arguments.task_id, arguments.limit)
    if not selected:
        raise ValueError("No evaluation tasks selected")
    run_id = arguments.run_id or f"{arguments.run_kind}-{slugify(arguments.model)}-{timestamp_slug()}"
    output_path = Path(arguments.output or f"reports/evaluations/{run_id}.jsonl")
    report_path = Path(arguments.report or output_path.with_suffix(".run_report.json"))
    options = {
        "temperature": arguments.temperature,
        "top_p": arguments.top_p,
        "num_predict": arguments.num_predict,
        "num_ctx": arguments.num_ctx,
        "seed": arguments.seed,
    }
    plan = {
        "stage": "held_out_evaluation",
        "created_at": utc_now(),
        "run_id": run_id,
        "run_kind": arguments.run_kind,
        "evaluation_file": str(arguments.evaluation),
        "model": arguments.model,
        "tasks_selected": [task["id"] for task in selected],
        "generation_options": options,
        "request_timeout_seconds": arguments.timeout_seconds,
        "output": str(output_path),
        "dry_run": bool(arguments.dry_run),
    }
    if arguments.dry_run:
        write_json_atomic(report_path, plan)
        print(f"Evaluation plan written to {report_path}; no model was called.")
        return 0

    client = OllamaClient(arguments.host, timeout_seconds=arguments.timeout_seconds)
    client.assert_model_present(arguments.model)
    try:
        fingerprint = sha256_text(canonical_json(client.show(arguments.model)))[:16]
    except OllamaError:
        fingerprint = None
    answers: list[dict[str, Any]] = []
    for index, task in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] evaluating {task['id']}", flush=True)
        try:
            response = client.generate(
                model=arguments.model,
                system=EVALUATION_SYSTEM,
                prompt=task["prompt"],
                options=options,
                think=False,
            )
            answers.append(
                model_answer_record(
                    run_id=run_id,
                    run_kind=arguments.run_kind,
                    task=task,
                    model=arguments.model,
                    model_fingerprint=fingerprint,
                    options=options,
                    answer=response.content,
                    elapsed_seconds=response.elapsed_seconds,
                    ollama_raw=response.raw,
                )
            )
        except OllamaError as exc:
            print(f"Evaluation task {task['id']} failed: {exc}", file=sys.stderr)
            answers.append(
                model_answer_record(
                    run_id=run_id,
                    run_kind=arguments.run_kind,
                    task=task,
                    model=arguments.model,
                    model_fingerprint=fingerprint,
                    options=options,
                    answer=None,
                    elapsed_seconds=None,
                    ollama_raw=None,
                    error=str(exc),
                )
            )
    write_jsonl_atomic(output_path, answers)
    failures = [record for record in answers if record["status"] != "complete"]
    report = {
        **plan,
        "completed_at": utc_now(),
        "model_fingerprint": fingerprint,
        "completed_tasks": len(answers) - len(failures),
        "failed_tasks": [{"task_id": item["task_id"], "error": item["error"]} for item in failures],
        "answer_file_sha256": sha256_text("\n".join(str(item.get("answer") or "") for item in answers)),
    }
    write_json_atomic(report_path, report)
    print(f"Wrote {len(answers)} held-out answers to {output_path}; report: {report_path}")
    return 2 if failures else 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except KeyboardInterrupt:
        print("evaluation interrupted; no result was fabricated.", file=sys.stderr)
        return 130
    except (FileNotFoundError, ValueError, OSError, OllamaError) as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
