"""Score held-out model answers against hidden project rubrics with an independent judge pass.

LLM-as-judge is a repeatable triage signal, not proof of correctness. Preserve raw answers,
inspect critical failures manually, and record the judge model alongside every score.
"""

from __future__ import annotations

# Support both `python -m scripts.name` and `python scripts/name.py` from the repo.
if __package__ in {None, ""}:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.lib.evaluation import score_record_fingerprint, validate_evaluation_tasks
from scripts.lib.io_utils import extract_json_object, read_jsonl, utc_now, write_json_atomic, write_jsonl_atomic
from scripts.lib.ollama import OllamaClient, OllamaError
from scripts.lib.prompts import SCORING_SYSTEM, scoring_prompt


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Score held-out Roblox/Luau answers")
    value.add_argument("--answers", required=True, help="JSONL emitted by run_baseline.py or run_evaluation.py")
    value.add_argument("--evaluation", default="evaluation_data/roblox_luau_eval.jsonl")
    value.add_argument("--judge-model", default="qwen3:4b")
    value.add_argument("--host", default=None)
    value.add_argument("--output", default=None)
    value.add_argument("--report", default=None)
    value.add_argument("--temperature", type=float, default=0.0)
    value.add_argument("--num-predict", type=int, default=1600)
    value.add_argument("--num-ctx", type=int, default=8192)
    value.add_argument("--limit", type=int, default=0)
    value.add_argument("--dry-run", action="store_true")
    return value


def parse_score(raw: str, task: dict[str, Any]) -> dict[str, Any]:
    value = extract_json_object(raw)
    criterion_map = {item["id"]: item for item in task["rubric"]}
    rows = value.get("criterion_scores")
    if not isinstance(rows, list):
        raise ValueError("Score response criterion_scores must be a list")
    seen: set[str] = set()
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each criterion score must be an object")
        criterion_id = row.get("id")
        if criterion_id not in criterion_map or criterion_id in seen:
            raise ValueError(f"Unknown or duplicate criterion id {criterion_id!r}")
        seen.add(criterion_id)
        points = row.get("points")
        if not isinstance(points, (int, float)) or isinstance(points, bool):
            raise ValueError(f"Criterion {criterion_id} has invalid points")
        max_points = criterion_map[criterion_id]["max_points"]
        if points < 0 or points > max_points:
            raise ValueError(f"Criterion {criterion_id} points must be within 0..{max_points}")
        evidence = row.get("evidence", "")
        if not isinstance(evidence, str):
            raise ValueError(f"Criterion {criterion_id} evidence must be a string")
        normalized_rows.append(
            {
                "id": criterion_id,
                "points": round(float(points), 2),
                "max_points": max_points,
                "evidence": evidence,
            }
        )
    missing = sorted(set(criterion_map) - seen)
    if missing:
        raise ValueError(f"Score response omitted criteria: {', '.join(missing)}")
    critical = value.get("critical_failures", [])
    missing_requirements = value.get("missing_requirements", [])
    strengths = value.get("strengths", [])
    for key, field in (("critical_failures", critical), ("missing_requirements", missing_requirements), ("strengths", strengths)):
        if not isinstance(field, list) or not all(isinstance(item, str) for item in field):
            raise ValueError(f"Score response {key} must be a list of strings")
    verdict = value.get("verdict")
    if verdict not in {"pass", "borderline", "fail"}:
        raise ValueError("Score response verdict must be pass, borderline, or fail")
    calculated = round(sum(row["points"] for row in normalized_rows), 2)
    reported = value.get("overall_score")
    if not isinstance(reported, (int, float)) or isinstance(reported, bool) or not 0 <= reported <= 100:
        raise ValueError("Score response overall_score must be 0..100")
    return {
        "overall_score": calculated,
        "reported_overall_score": float(reported),
        "criterion_scores": normalized_rows,
        "critical_failures": critical,
        "missing_requirements": missing_requirements,
        "strengths": strengths,
        "verdict": verdict,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record.get("status") == "complete"]
    scores = [float(record["score"]["overall_score"]) for record in completed]
    groups: dict[str, list[float]] = defaultdict(list)
    verdicts: dict[str, int] = defaultdict(int)
    for record in completed:
        groups[str(record.get("task_category"))].append(float(record["score"]["overall_score"]))
        verdicts[str(record["score"].get("verdict"))] += 1
    return {
        "answers": len(records),
        "scored": len(completed),
        "unscored": len(records) - len(completed),
        "mean_score": round(sum(scores) / len(scores), 2) if scores else None,
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "category_mean_scores": {key: round(sum(values) / len(values), 2) for key, values in sorted(groups.items())},
        "verdict_counts": dict(sorted(verdicts.items())),
        "critical_failure_count": sum(len(record["score"].get("critical_failures", [])) for record in completed),
    }


def run(arguments: argparse.Namespace) -> int:
    if arguments.limit < 0:
        raise ValueError("--limit must be zero or positive")
    tasks = list(read_jsonl(arguments.evaluation))
    validate_evaluation_tasks(tasks)
    task_by_id = {task["id"]: task for task in tasks}
    answers = list(read_jsonl(arguments.answers))
    if arguments.limit:
        answers = answers[: arguments.limit]
    if not answers:
        raise ValueError("No answer records supplied")
    unknown = [str(answer.get("task_id")) for answer in answers if answer.get("task_id") not in task_by_id]
    if unknown:
        raise ValueError(f"Answers refer to unknown evaluation task IDs: {', '.join(unknown[:10])}")
    answer_path = Path(arguments.answers)
    output_path = Path(arguments.output or answer_path.with_name(f"{answer_path.stem}.scored.jsonl"))
    report_path = Path(arguments.report or output_path.with_suffix(".score_report.json"))
    options = {"temperature": arguments.temperature, "num_predict": arguments.num_predict, "num_ctx": arguments.num_ctx}
    if arguments.dry_run:
        write_json_atomic(
            report_path,
            {
                "stage": "evaluation_scoring",
                "created_at": utc_now(),
                "answers": str(arguments.answers),
                "judge_model": arguments.judge_model,
                "planned_records": len(answers),
                "dry_run": True,
            },
        )
        print(f"Scoring plan written to {report_path}; no judge model was called.")
        return 0

    client = OllamaClient(arguments.host)
    client.assert_model_present(arguments.judge_model)
    output: list[dict[str, Any]] = []
    for index, answer_record in enumerate(answers, start=1):
        task = task_by_id[answer_record["task_id"]]
        print(f"[{index}/{len(answers)}] scoring {task['id']}", flush=True)
        base = {
            "schema_version": "1.0",
            "created_at": utc_now(),
            "answer_record_fingerprint": score_record_fingerprint(answer_record),
            "run_id": answer_record.get("run_id"),
            "run_kind": answer_record.get("run_kind"),
            "task_id": task["id"],
            "task_category": task.get("category"),
            "task_difficulty": task.get("difficulty"),
            "evaluated_model": answer_record.get("model"),
            "judge": {"kind": "ollama_llm_as_judge", "model": arguments.judge_model, "options": options},
        }
        if answer_record.get("status") != "complete" or not isinstance(answer_record.get("answer"), str):
            output.append(
                {
                    **base,
                    "status": "unavailable",
                    "score": None,
                    "error": f"Answer run did not complete: {answer_record.get('error')}",
                }
            )
            continue
        try:
            response = client.generate(
                model=arguments.judge_model,
                system=SCORING_SYSTEM,
                prompt=scoring_prompt(task, answer_record["answer"]),
                options=options,
                think=False,
            )
            score = parse_score(response.content, task)
            output.append(
                {
                    **base,
                    "status": "complete",
                    "score": score,
                    "judge_elapsed_seconds": round(response.elapsed_seconds, 3),
                    "error": None,
                }
            )
        except (OllamaError, ValueError) as exc:
            print(f"Scoring task {task['id']} failed: {exc}", file=sys.stderr)
            output.append({**base, "status": "error", "score": None, "error": str(exc)})
    write_jsonl_atomic(output_path, output)
    report = {
        "stage": "evaluation_scoring",
        "created_at": utc_now(),
        "answers": str(arguments.answers),
        "output": str(output_path),
        "evaluation": str(arguments.evaluation),
        "judge_model": arguments.judge_model,
        "judge_options": options,
        "method_note": "LLM-as-judge is a repeatable triage signal; inspect raw answers and critical failures manually.",
        **aggregate(output),
    }
    write_json_atomic(report_path, report)
    print(f"Wrote {len(output)} score records to {output_path}; report: {report_path}")
    return 2 if any(record.get("status") == "error" for record in output) else 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError, OllamaError) as exc:
        print(f"scoring error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
