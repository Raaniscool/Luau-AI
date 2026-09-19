"""Run the deliberately small baseline/Code Book proof of concept.

This command never generates training examples and never fine-tunes. On a real run it:
1) runs `ollama list` and verifies the existing qwen3:4b tag;
2) audits the source-attributed Code Book;
3) captures the one held-out RemoteEvent baseline answer; and
4) scores that answer, including deterministic regression flags.
"""

from __future__ import annotations

# Support both `python -m scripts.run_first_poc` and direct Windows invocation.
if __package__ in {None, ""}:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from scripts.audit_code_book import main as audit_code_book_main
from scripts.check_ollama import main as check_ollama_main
from scripts.lib.io_utils import utc_now, write_json_atomic
from scripts.run_baseline import main as run_baseline_main
from scripts.score_evaluation import main as score_evaluation_main

REMOTE_BASELINE_TASK = "eval-remoteevent-secure-001"
_LOCAL_HOSTNAMES = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def effective_local_host(host: str | None) -> str:
    """Resolve the POC endpoint and reject remote endpoints the CLI cannot attest."""
    value = host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
    parsed = urlparse(value if value.startswith(("http://", "https://")) else f"http://{value}")
    if parsed.hostname not in _LOCAL_HOSTNAMES:
        raise ValueError(
            "run_first_poc.py is intentionally local-only: `ollama list` cannot verify a remote endpoint's model registry. "
            "Run this POC against the existing local Ollama service."
        )
    return value


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Run a small, non-training Code Book + RemoteEvent baseline proof of concept")
    value.add_argument("--model", default="qwen3:4b")
    value.add_argument("--host", default=None, help="Local Ollama API host; remote hosts are intentionally rejected")
    value.add_argument("--output-dir", default="reports/poc")
    value.add_argument("--overwrite", action="store_true", help="Allow replacement of an existing raw POC baseline answer")
    value.add_argument("--dry-run", action="store_true", help="Write a plan only; do not call Ollama")
    return value


def run(arguments: argparse.Namespace) -> int:
    if arguments.model != "qwen3:4b":
        raise ValueError("The first proof of concept is intentionally pinned to the existing exact base tag qwen3:4b.")
    host = effective_local_host(arguments.host)
    output_dir = Path(arguments.output_dir)
    answer_file = output_dir / "qwen3_4b_remoteevent_baseline.jsonl"
    score_file = output_dir / "qwen3_4b_remoteevent_baseline.scored.jsonl"
    report_file = output_dir / "first_poc_report.json"
    report: dict[str, Any] = {
        "stage": "first_proof_of_concept",
        "created_at": utc_now(),
        "model": arguments.model,
        "host": host,
        "task_id": REMOTE_BASELINE_TASK,
        "answer_file": str(answer_file),
        "score_file": str(score_file),
        "guarantees": [
            "No model download is requested.",
            "No training data is generated.",
            "No fine-tuning is started.",
            "The model must pass `ollama list` registration before baseline inference.",
        ],
    }
    if arguments.dry_run:
        report["status"] = "planned_not_executed"
        report["commands"] = [
            ["ollama", "list"],
            ["audit_code_book", "--strict"],
            ["run_baseline", "--task-id", REMOTE_BASELINE_TASK],
            ["score_evaluation", "--answers", str(answer_file)],
        ]
        write_json_atomic(report_file, report)
        print(f"POC plan written to {report_file}; no Ollama request or training was performed.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    if answer_file.exists() and not arguments.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite the existing raw POC baseline at {answer_file}. "
            "Inspect/archive it first, or rerun intentionally with --overwrite."
        )
    steps: list[dict[str, Any]] = []

    def execute(name: str, function: Any, args: list[str]) -> bool:
        code = function(args)
        steps.append({"step": name, "returncode": code})
        return code == 0

    if not execute("ollama_cli_preflight", check_ollama_main, ["--model", arguments.model]):
        report.update({"status": "failed", "steps": steps, "completed_at": utc_now()})
        write_json_atomic(report_file, report)
        return 2
    if not execute("code_book_audit", audit_code_book_main, ["--strict", "--output", str(output_dir / "code_book_audit.json")]):
        report.update({"status": "failed", "steps": steps, "completed_at": utc_now()})
        write_json_atomic(report_file, report)
        return 2
    baseline_args = [
        "--model", arguments.model,
        "--task-id", REMOTE_BASELINE_TASK,
        "--output", str(answer_file),
    ]
    baseline_args.extend(["--host", host])
    if not execute("remoteevent_baseline", run_baseline_main, baseline_args):
        report.update({"status": "failed", "steps": steps, "completed_at": utc_now()})
        write_json_atomic(report_file, report)
        return 2
    score_args = ["--answers", str(answer_file), "--output", str(score_file), "--judge-model", arguments.model]
    score_args.extend(["--host", host])
    if not execute("remoteevent_score", score_evaluation_main, score_args):
        report.update({"status": "failed", "steps": steps, "completed_at": utc_now()})
        write_json_atomic(report_file, report)
        return 2
    report.update(
        {
            "status": "completed",
            "steps": steps,
            "completed_at": utc_now(),
            "next_step": "Inspect the raw answer and score record. Only then run a small reviewed data pilot; do not start fine-tuning on this hardware.",
        }
    )
    write_json_atomic(report_file, report)
    print(f"POC completed. Inspect {answer_file}, {score_file}, and {report_file} before scaling.")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"first POC error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
