"""Stage 1: expand curated DukeOTR Roblox/Luau task briefs through local Ollama.

The default catalog is DukeOTR's Phase-1 Luau-fundamentals curriculum. Run from the
repository root, for example:
    python -m scripts.generate_examples --model qwen3:4b --limit 8

Output is only a candidate corpus. It must proceed through validation before training.
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

from scripts.lib.io_utils import canonical_json, extract_json_object, read_json, read_jsonl, sha256_text, utc_now, write_json_atomic, write_jsonl_atomic
from scripts.lib.ollama import OllamaClient, OllamaError
from scripts.lib.prompts import GENERATION_SYSTEM, generation_prompt
from scripts.lib.schema import make_generated_record, validate_seed


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Generate candidate Roblox/Luau examples with local Ollama")
    value.add_argument(
        "--seeds",
        default="raw_data/dukeotr_phase1_luau_seed_tasks.jsonl",
        help="Curated train seed JSONL; defaults to the DukeOTR Phase-1 fundamentals catalog",
    )
    value.add_argument("--config", default="configs/pipeline.json", help="Pipeline JSON config")
    value.add_argument(
        "--output",
        default="generated_data/dukeotr_phase1_candidates.jsonl",
        help="Candidate JSONL destination; choose a unique pilot path to preserve prior candidates",
    )
    value.add_argument("--report", default=None, help="Optional generation report path")
    value.add_argument("--model", default=None, help="Ollama model tag; defaults to pipeline config")
    value.add_argument("--host", default=None, help="Ollama API host, or set OLLAMA_HOST")
    value.add_argument("--limit", type=int, default=0, help="Generate from at most N seeds; 0 means all")
    value.add_argument("--seed-id", action="append", default=[], help="Generate only this seed id (repeatable)")
    value.add_argument("--variants", type=int, default=None, help="Variants per seed; defaults to config")
    value.add_argument("--temperature", type=float, default=None)
    value.add_argument("--num-predict", type=int, default=None)
    value.add_argument("--dry-run", action="store_true", help="Validate/plan only; do not contact Ollama")
    value.add_argument("--overwrite", action="store_true", help="Replace an existing candidate JSONL only after reviewing it")
    value.add_argument("--continue-on-error", action="store_true", help="Write successful candidates if one generation fails")
    return value


def select_seeds(seeds: list[dict[str, Any]], wanted_ids: list[str], limit: int) -> list[dict[str, Any]]:
    seen: set[str] = set()
    problems: list[str] = []
    for seed in seeds:
        seed_id = seed.get("id")
        if seed_id in seen:
            problems.append(f"duplicate seed id {seed_id!r}")
        seen.add(seed_id)
        errors = validate_seed(seed)
        if errors:
            rendered = "; ".join(f"{entry['code']}: {entry['message']}" for entry in errors)
            problems.append(f"{seed_id}: {rendered}")
    if problems:
        raise ValueError("Seed catalog is invalid:\n- " + "\n- ".join(problems))
    if wanted_ids:
        unknown = sorted(set(wanted_ids) - seen)
        if unknown:
            raise ValueError(f"Unknown --seed-id values: {', '.join(unknown)}")
        allowed = set(wanted_ids)
        seeds = [seed for seed in seeds if seed["id"] in allowed]
    if limit < 0:
        raise ValueError("--limit must be zero or positive")
    return seeds[:limit] if limit else seeds


def run(arguments: argparse.Namespace) -> int:
    config = read_json(arguments.config)
    generator_config = dict(config.get("generator", {}))
    model = arguments.model or config.get("base_ollama_model", "qwen3:4b")
    variants = arguments.variants if arguments.variants is not None else int(generator_config.get("variants_per_seed", 1))
    if variants < 1:
        raise ValueError("--variants must be at least 1")
    selected = select_seeds(list(read_jsonl(arguments.seeds)), arguments.seed_id, arguments.limit)
    output_path = Path(arguments.output)
    if output_path.exists() and not arguments.dry_run and not arguments.overwrite:
        raise ValueError(
            f"Refusing to overwrite existing candidate output: {output_path}. "
            "Choose a new --output path or use --overwrite only after preserving/reviewing it."
        )
    report_path = Path(arguments.report) if arguments.report else output_path.with_suffix(".generation_report.json")
    planned = {
        "stage": "generation",
        "created_at": utc_now(),
        "model": model,
        "seed_file": str(arguments.seeds),
        "selected_seed_count": len(selected),
        "variants_per_seed": variants,
        "planned_records": len(selected) * variants,
        "dry_run": bool(arguments.dry_run),
    }
    if arguments.dry_run:
        write_json_atomic(report_path, planned)
        print(f"Validated {len(selected)} training seeds. Plan written to {report_path}.")
        return 0

    options = {
        "temperature": arguments.temperature if arguments.temperature is not None else generator_config.get("temperature", 0.35),
        "top_p": generator_config.get("top_p", 0.9),
        "seed": generator_config.get("seed", 3407),
        "num_predict": arguments.num_predict if arguments.num_predict is not None else generator_config.get("num_predict", 2200),
        "num_ctx": generator_config.get("num_ctx", 8192),
    }
    client = OllamaClient(arguments.host)
    client.assert_model_present(model)
    try:
        model_info = client.show(model)
        model_fingerprint = sha256_text(canonical_json(model_info))[:16]
    except OllamaError:
        # Generation can still be auditable when an older Ollama server lacks /api/show.
        model_fingerprint = None

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(selected, start=1):
        for variant in range(1, variants + 1):
            print(f"[{seed_index}/{len(selected)}] {seed['id']} variant {variant}/{variants}", flush=True)
            try:
                response = client.generate(
                    model=model,
                    system=GENERATION_SYSTEM,
                    prompt=generation_prompt(seed, variant),
                    options=options,
                    think=False,
                )
                generated = extract_json_object(response.content)
                assistant_response = generated.get("assistant_response")
                if not isinstance(assistant_response, str) or not assistant_response.strip():
                    raise ValueError("Generator JSON has no non-empty assistant_response")
                coverage = generated.get("coverage", [])
                notes = generated.get("self_check", [])
                if not isinstance(coverage, list) or not all(isinstance(item, str) for item in coverage):
                    raise ValueError("Generator JSON coverage must be a list of strings")
                if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
                    raise ValueError("Generator JSON self_check must be a list of strings")
                records.append(
                    make_generated_record(
                        seed,
                        assistant_response,
                        generator={
                            "kind": "ollama",
                            "model": model,
                            "model_fingerprint": model_fingerprint,
                            "options": options,
                            "elapsed_seconds": round(response.elapsed_seconds, 3),
                        },
                        variant=variant,
                        coverage=coverage,
                        generation_notes=notes,
                    )
                )
            except (OllamaError, ValueError) as exc:
                failure = {"seed_id": seed["id"], "variant": variant, "error": str(exc)}
                failures.append(failure)
                print(f"Generation failed: {failure['error']}", file=sys.stderr)
                if not arguments.continue_on_error:
                    break
        if failures and not arguments.continue_on_error:
            break

    report = {
        **planned,
        "completed_at": utc_now(),
        "options": options,
        "generated_records": len(records),
        "failures": failures,
        "output": str(output_path),
        "output_sha256": sha256_text("\n".join(record["record_id"] for record in records)) if records else None,
    }
    if records:
        write_jsonl_atomic(output_path, records)
    write_json_atomic(report_path, report)
    print(f"Wrote {len(records)} candidate records to {output_path}; report: {report_path}")
    if failures:
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError, OllamaError) as exc:
        print(f"generation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
