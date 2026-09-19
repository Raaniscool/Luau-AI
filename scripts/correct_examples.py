"""Stage 3: repair candidates that received deterministic blockers or a revise decision.

Correction output is a new artifact with parent provenance. It is deliberately not marked
valid; run `validate_examples.py` on it before deduplication/building.
"""

from __future__ import annotations

# Support both `python -m scripts.name` and `python scripts/name.py` from the repo.
if __package__ in {None, ""}:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import sys
from pathlib import Path
from typing import Any

from scripts.lib.io_utils import extract_json_object, read_json, read_jsonl, utc_now, write_json_atomic, write_jsonl_atomic
from scripts.lib.ollama import OllamaClient, OllamaError
from scripts.lib.prompts import CORRECTION_SYSTEM, correction_prompt
from scripts.lib.quality import needs_correction
from scripts.lib.schema import clone_for_correction


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Correct failed/revise Roblox/Luau candidates")
    value.add_argument("--input", default="validated_data/dukeotr_phase1_candidates.validated.jsonl")
    value.add_argument("--output", default="validated_data/dukeotr_phase1_candidates.corrected.jsonl")
    value.add_argument("--report", default=None)
    value.add_argument("--config", default="configs/pipeline.json")
    value.add_argument("--model", default=None)
    value.add_argument("--host", default=None)
    value.add_argument("--limit", type=int, default=0)
    value.add_argument("--include-rejected", action="store_true", help="Also attempt correction of reviewer rejects")
    value.add_argument("--continue-on-error", action="store_true")
    return value


def run(arguments: argparse.Namespace) -> int:
    if arguments.limit < 0:
        raise ValueError("--limit must be zero or positive")
    config = read_json(arguments.config)
    generator_config = dict(config.get("generator", {}))
    model = arguments.model or config.get("base_ollama_model", "qwen3:4b")
    input_records = list(read_jsonl(arguments.input))
    selected = [
        record for record in input_records if needs_correction(record, include_rejected=arguments.include_rejected)
    ]
    if arguments.limit:
        selected = selected[: arguments.limit]
    report_path = Path(arguments.report) if arguments.report else Path(arguments.output).with_suffix(".correction_report.json")
    options = {
        "temperature": 0.15,
        "top_p": generator_config.get("top_p", 0.9),
        "seed": generator_config.get("seed", 3407),
        "num_predict": generator_config.get("num_predict", 2200),
        "num_ctx": generator_config.get("num_ctx", 8192),
    }
    if not selected:
        write_jsonl_atomic(arguments.output, [])
        write_json_atomic(
            report_path,
            {
                "stage": "correction",
                "created_at": utc_now(),
                "input": str(arguments.input),
                "selected_records": 0,
                "corrected_records": 0,
                "note": "No static failures or revise decisions were selected for correction.",
            },
        )
        print("No candidates needed correction; wrote empty correction artifact.")
        return 0

    client = OllamaClient(arguments.host)
    client.assert_model_present(model)
    corrected: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, record in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] correcting {record.get('record_id')}", flush=True)
        try:
            response = client.generate(
                model=model,
                system=CORRECTION_SYSTEM,
                prompt=correction_prompt(record),
                options=options,
                think=False,
            )
            payload = extract_json_object(response.content)
            answer = payload.get("assistant_response")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Correction JSON has no non-empty assistant_response")
            changes = payload.get("changes_made", [])
            assumptions = payload.get("remaining_assumptions", [])
            if not isinstance(changes, list) or not all(isinstance(item, str) for item in changes):
                raise ValueError("Correction changes_made must be a list of strings")
            if not isinstance(assumptions, list) or not all(isinstance(item, str) for item in assumptions):
                raise ValueError("Correction remaining_assumptions must be a list of strings")
            corrected.append(
                clone_for_correction(
                    record,
                    answer,
                    correction={
                        "model": model,
                        "options": options,
                        "elapsed_seconds": round(response.elapsed_seconds, 3),
                        "changes_made": changes,
                        "remaining_assumptions": assumptions,
                        "source_quality_snapshot": record.get("quality", {}),
                    },
                )
            )
        except (OllamaError, ValueError) as exc:
            failures.append({"record_id": record.get("record_id"), "error": str(exc)})
            print(f"Correction failed: {exc}", file=sys.stderr)
            if not arguments.continue_on_error:
                break
    write_jsonl_atomic(arguments.output, corrected)
    write_json_atomic(
        report_path,
        {
            "stage": "correction",
            "created_at": utc_now(),
            "input": str(arguments.input),
            "output": str(arguments.output),
            "model": model,
            "options": options,
            "selected_records": len(selected),
            "corrected_records": len(corrected),
            "failures": failures,
        },
    )
    print(f"Wrote {len(corrected)} corrected candidates to {arguments.output}; report: {report_path}")
    return 2 if failures else 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError, OllamaError) as exc:
        print(f"correction error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
