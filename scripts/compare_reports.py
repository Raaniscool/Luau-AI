"""Compare scored baseline and candidate held-out evaluation runs."""

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

from scripts.lib.io_utils import read_jsonl, utc_now, write_json_atomic


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Compare baseline vs candidate evaluation score JSONL")
    value.add_argument("--baseline", required=True, help="Scored baseline JSONL")
    value.add_argument("--candidate", required=True, help="Scored candidate JSONL")
    value.add_argument("--output", default=None, help="Comparison JSON report")
    value.add_argument("--minimum-mean-improvement", type=float, default=0.0)
    return value


def usable(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        task_id = record.get("task_id")
        if record.get("status") == "complete" and isinstance(record.get("score"), dict) and isinstance(task_id, str):
            if task_id in indexed:
                raise ValueError(f"Duplicate completed score for task {task_id}")
            indexed[task_id] = record
    return indexed


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


def run(arguments: argparse.Namespace) -> int:
    baseline_records = list(read_jsonl(arguments.baseline))
    candidate_records = list(read_jsonl(arguments.candidate))
    baseline = usable(baseline_records)
    candidate = usable(candidate_records)
    shared = sorted(set(baseline) & set(candidate))
    if not shared:
        raise ValueError("No tasks have complete scores in both baseline and candidate runs")
    rows: list[dict[str, Any]] = []
    category_deltas: dict[str, list[float]] = defaultdict(list)
    for task_id in shared:
        baseline_score = float(baseline[task_id]["score"]["overall_score"])
        candidate_score = float(candidate[task_id]["score"]["overall_score"])
        delta = round(candidate_score - baseline_score, 2)
        category = str(candidate[task_id].get("task_category") or baseline[task_id].get("task_category"))
        category_deltas[category].append(delta)
        rows.append(
            {
                "task_id": task_id,
                "category": category,
                "baseline_score": baseline_score,
                "candidate_score": candidate_score,
                "delta": delta,
                "baseline_verdict": baseline[task_id]["score"].get("verdict"),
                "candidate_verdict": candidate[task_id]["score"].get("verdict"),
                "baseline_critical_failures": baseline[task_id]["score"].get("critical_failures", []),
                "candidate_critical_failures": candidate[task_id]["score"].get("critical_failures", []),
            }
        )
    baseline_values = [row["baseline_score"] for row in rows]
    candidate_values = [row["candidate_score"] for row in rows]
    deltas = [row["delta"] for row in rows]
    report = {
        "schema_version": "1.0",
        "stage": "evaluation_comparison",
        "created_at": utc_now(),
        "baseline_file": str(arguments.baseline),
        "candidate_file": str(arguments.candidate),
        "baseline_run_ids": sorted({str(record.get("run_id") or "unknown") for record in baseline.values()}),
        "candidate_run_ids": sorted({str(record.get("run_id") or "unknown") for record in candidate.values()}),
        "baseline_models": sorted({str(record.get("evaluated_model") or "unknown") for record in baseline.values()}),
        "candidate_models": sorted({str(record.get("evaluated_model") or "unknown") for record in candidate.values()}),
        "shared_tasks": len(rows),
        "missing_from_candidate": sorted(set(baseline) - set(candidate)),
        "missing_from_baseline": sorted(set(candidate) - set(baseline)),
        "aggregate": {
            "baseline_mean": mean(baseline_values),
            "candidate_mean": mean(candidate_values),
            "mean_delta": mean(deltas),
            "median_delta": sorted(deltas)[len(deltas) // 2] if len(deltas) % 2 else round((sorted(deltas)[len(deltas)//2 - 1] + sorted(deltas)[len(deltas)//2]) / 2, 2),
            "improved_tasks": sum(delta > 0 for delta in deltas),
            "unchanged_tasks": sum(delta == 0 for delta in deltas),
            "regressed_tasks": sum(delta < 0 for delta in deltas),
            "baseline_critical_failures": sum(len(row["baseline_critical_failures"]) for row in rows),
            "candidate_critical_failures": sum(len(row["candidate_critical_failures"]) for row in rows),
            "category_mean_delta": {key: mean(values) for key, values in sorted(category_deltas.items())},
        },
        "threshold": {"minimum_mean_improvement": arguments.minimum_mean_improvement},
        "meets_requested_mean_improvement": mean(deltas) is not None and mean(deltas) >= arguments.minimum_mean_improvement,
        "per_task": rows,
        "interpretation_note": "Scores reflect the recorded judge model and rubric. Read raw answers and critical failures before claiming a real capability improvement.",
    }
    output_path = Path(arguments.output or "reports/comparisons/baseline_vs_candidate.json")
    write_json_atomic(output_path, report)
    print(
        f"Compared {len(rows)} shared held-out tasks. Mean delta: {report['aggregate']['mean_delta']}. "
        f"Report: {output_path}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"comparison error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
