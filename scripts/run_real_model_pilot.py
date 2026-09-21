"""Run a small, sequential, non-promoting DukeOTR real-model discovery pilot.

The command is intentionally a thin coordinator around the existing local Ollama client,
Builder → Reviewer → Fixer runner, and Training Factory sidecars. It is safe to run on a
Windows machine that already has the exact local ``qwen3:4b`` tag. It never downloads,
replaces, trains, imports, or promotes a model/dataset artifact.
"""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import shutil
import sys
from typing import Any

from scripts import analyze_failures, build_training_ledger, deduplicate_examples, record_failures, run_builder_reviewer_fixer
from scripts.lib.io_utils import read_json, read_jsonl, sha256_file, utc_now, write_json_atomic, write_jsonl_atomic
from scripts.lib.ollama import OllamaClient, OllamaError
from scripts.lib.real_model_pilot import (
    PILOT_SCHEMA_VERSION,
    PilotMaterialization,
    PilotPaths,
    RealModelPilotError,
    correction_explanation_evidence,
    initial_pilot_state,
    load_pilot_state,
    load_pilot_tasks,
    materialize_trace_candidates,
    merge_materialized_records,
    pilot_fingerprint,
    quality_eligible_records,
    recover_interrupted_tasks,
    runnable_task_ids,
    summarize_state,
    task_state,
    validate_pilot_tasks,
    validate_run_id,
    write_pilot_state,
)
from scripts.lib.training_factory import TrainingFactoryError, assert_safe_factory_output_path


PILOT_MODEL = "qwen3:4b"
_DEFAULT_TASKS = "pilot_data/real_model_pilot_v1.jsonl"
_DEFAULT_CONFIG = "configs/real_model_pilot.json"
_DEFAULT_OUTPUT_ROOT = "reports/real_model_pilots"


class PilotExecutionError(RuntimeError):
    """Raised when a reusable sidecar cannot safely materialize pilot evidence."""


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Sequentially run the separate DukeOTR Real Model Pilot through the existing local "
            "Builder/Reviewer/Tester/Fixer and Training Factory sidecars. Never promotes training data."
        )
    )
    value.add_argument("--run-id", required=True, help="New/resumable pilot run ID, e.g. win-qwen3-4b-20260920")
    value.add_argument("--tasks", default=_DEFAULT_TASKS, help="Separate pilot-only JSONL source brief catalog")
    value.add_argument("--config", default=_DEFAULT_CONFIG, help="Real Model Pilot policy/config JSON")
    value.add_argument("--evaluation", default="evaluation_data/roblox_luau_eval.jsonl")
    value.add_argument("--brf-config", default="configs/builder_verifier_reviewer.json")
    value.add_argument("--pipeline-config", default="configs/pipeline.json")
    value.add_argument("--code-book", default="code_book/roblox_luau_cards.jsonl")
    value.add_argument("--factory-config", default="configs/training_factory.json")
    value.add_argument("--taxonomy", default="training_factory/failure_taxonomy.json")
    value.add_argument("--output-root", default=_DEFAULT_OUTPUT_ROOT)
    value.add_argument("--host", default=None, help="Local Ollama HTTP host (defaults to OLLAMA_HOST or 127.0.0.1:11434)")
    value.add_argument("--model", default=PILOT_MODEL, help=f"Pinned v1 pilot tag; must remain {PILOT_MODEL}")
    value.add_argument("--timeout-seconds", type=int, default=600, help="Per local Ollama request/chunk timeout (default: 600)")
    value.add_argument("--max-tasks", type=int, default=None, help="Run at most this many pending/interrupted tasks this invocation")
    value.add_argument(
        "--retry-failed",
        action="store_true",
        help="Explicitly retry tasks whose prior Builder/Reviewer/Fixer generation failed; creates a new trace attempt.",
    )
    value.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate task/isolation/run layout and write a no-model plan; does not call Ollama or B/R/F.",
    )
    value.add_argument(
        "--repair-sidecars",
        action="store_true",
        help=(
            "Explicitly rebuild only derived Training Factory sidecars from an existing pilot's "
            "persisted candidate artifacts. It never calls Ollama/B/R/F, snapshots existing "
            "derived artifacts first, and may process an older pipeline fingerprint."
        ),
    )
    return value


def _require_positive_timeout(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise RealModelPilotError("--timeout-seconds must be a positive integer")
    return value


def _read_config(path: str) -> dict[str, Any]:
    config = read_json(path)
    if not isinstance(config, dict):
        raise RealModelPilotError("Real Model Pilot config must be a JSON object")
    if config.get("schema_version") != PILOT_SCHEMA_VERSION:
        raise RealModelPilotError(f"Real Model Pilot config must use schema_version {PILOT_SCHEMA_VERSION!r}")
    if config.get("model") != PILOT_MODEL:
        raise RealModelPilotError(f"Real Model Pilot config must remain pinned to {PILOT_MODEL!r}")
    if config.get("execution", {}).get("parallelism") != 1:
        raise RealModelPilotError("Real Model Pilot v1 must set execution.parallelism to 1")
    return config


def _pipeline_version(arguments: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """Hash only local config/code artifacts; no held-out content is copied to inference context."""

    return {
        "pilot_runner": {
            "path": "scripts/run_real_model_pilot.py",
            "sha256": sha256_file(__file__),
            "schema_version": PILOT_SCHEMA_VERSION,
        },
        "pilot_helpers": {
            "path": "scripts/lib/real_model_pilot.py",
            "sha256": sha256_file(Path(__file__).with_name("lib") / "real_model_pilot.py"),
        },
        "pilot_config": {"path": arguments.config, "sha256": sha256_file(arguments.config)},
        "builder_reviewer_fixer": {
            "config_path": arguments.brf_config,
            "config_sha256": sha256_file(arguments.brf_config),
            "entrypoint_path": "scripts/run_builder_reviewer_fixer.py",
            "entrypoint_sha256": sha256_file(Path(__file__).with_name("run_builder_reviewer_fixer.py")),
        },
        "pipeline_config": {"path": arguments.pipeline_config, "sha256": sha256_file(arguments.pipeline_config)},
        "code_book": {"path": arguments.code_book, "sha256": sha256_file(arguments.code_book)},
        "failure_taxonomy": {"path": arguments.taxonomy, "sha256": sha256_file(arguments.taxonomy)},
        "training_factory_config": {"path": arguments.factory_config, "sha256": sha256_file(arguments.factory_config)},
        "training_factory_sidecars": {
            name: {
                "path": f"scripts/{name}.py",
                "sha256": sha256_file(Path(__file__).with_name(f"{name}.py")),
            }
            for name in ("deduplicate_examples", "build_training_ledger", "record_failures", "analyze_failures")
        },
        "dataset_version_target": config.get("dataset_version_target", "dukeotr_dataset_v1"),
        "dataset_version_status": "planned_not_built_not_promoted",
    }


def _assert_safe_paths(paths: PilotPaths) -> None:
    for path in (
        paths.root,
        paths.state,
        paths.manifest,
        paths.report,
        paths.raw_candidates,
        paths.corrected_candidates,
        paths.linkable_corrections,
        paths.correction_linkage_report,
        paths.deduplication_input_candidates,
        paths.final_candidates,
        paths.deduplicated_candidates,
        paths.training_eligible_candidates,
        paths.ledger,
        paths.failures,
        paths.failure_analysis,
    ):
        assert_safe_factory_output_path(path)


def _assert_pilot_output_boundary(paths: PilotPaths) -> None:
    """Keep discovery artifacts out of curated/source/final-data repository layers."""

    repository_root = Path(__file__).resolve().parents[1]
    root = paths.root.resolve()
    prohibited = (
        "evaluation_data",
        "training_data",
        "raw_data",
        "generated_data",
        "validated_data",
        "failure_data",
        "knowledge_base",
        "code_book",
        "verified_knowledge",
        "models",
    )
    for name in prohibited:
        protected = (repository_root / name).resolve()
        try:
            root.relative_to(protected)
        except ValueError:
            continue
        raise RealModelPilotError(
            f"Pilot output must remain a separate discovery artifact, not under {name}/: {root}"
        )


def _state_matches(
    state: dict[str, Any],
    *,
    run_id: str,
    task_path: str,
    tasks: list[dict[str, Any]],
    model: str,
    evaluation_isolation: dict[str, Any],
    pipeline_version: dict[str, Any],
    allow_pipeline_mismatch_for_sidecar_repair: bool = False,
) -> None:
    if state.get("run_id") != run_id:
        raise RealModelPilotError("Existing pilot state run_id does not match --run-id")
    task_set = state.get("task_set")
    if not isinstance(task_set, dict) or task_set.get("sha256") != sha256_file(task_path):
        raise RealModelPilotError("Pilot task catalog changed since this run started; use a new --run-id rather than mixing evidence")
    if task_set.get("task_ids") != [task["id"] for task in tasks]:
        raise RealModelPilotError("Pilot task IDs changed since this run started; use a new --run-id")
    state_model = state.get("model")
    if not isinstance(state_model, dict) or state_model.get("requested_tag") != model:
        raise RealModelPilotError("Existing pilot state used a different model tag; use a new --run-id")
    stored_isolation = state.get("evaluation_isolation")
    if (
        not isinstance(stored_isolation, dict)
        or stored_isolation.get("evaluation_sha256") != evaluation_isolation.get("evaluation_sha256")
        or stored_isolation.get("threshold") != evaluation_isolation.get("threshold")
    ):
        raise RealModelPilotError(
            "Held-out evaluation fingerprint/isolation threshold changed since this run started; use a new --run-id rather than reprocess against different held-out material"
        )
    if state.get("pipeline_version") != pipeline_version and not allow_pipeline_mismatch_for_sidecar_repair:
        raise RealModelPilotError("Pilot pipeline/config fingerprints changed since this run started; use a new --run-id")
    identifiers = [item.get("task_id") for item in state.get("tasks", []) if isinstance(item, dict)]
    expected_ids = [task["id"] for task in tasks]
    if not all(isinstance(identifier, str) and identifier for identifier in identifiers):
        raise RealModelPilotError("Existing pilot state contains malformed task IDs; refusing ambiguous resume")
    if len(identifiers) != len(state.get("tasks", [])) or len(identifiers) != len(set(identifiers)):
        raise RealModelPilotError("Existing pilot state contains malformed or duplicate task IDs; refusing ambiguous resume")
    if identifiers != expected_ids:
        raise RealModelPilotError("Existing pilot state task order/identity no longer matches the validated catalog; use a new --run-id")


def _model_provenance(state: dict[str, Any]) -> dict[str, Any]:
    model = state.get("model") if isinstance(state.get("model"), dict) else {}
    return {
        "provider": "local_ollama_http_api",
        "model_tag": model.get("requested_tag"),
        "host": model.get("host"),
        "timeout_seconds": model.get("timeout_seconds"),
        "preflight": deepcopy(model.get("preflight", {})),
        "non_claim": "Qwen3-4B is the local selected baseline/provider tag; this pilot does not train, fine-tune, import, or rename it as DukeOTR.",
    }


def _trace_path(paths: PilotPaths, task_id: str, attempt: int) -> Path:
    safe = task_id.replace("/", "_").replace("\\", "_")
    return paths.trace_directory / f"{safe}.attempt-{attempt}.trace.json"


def _terminal_trace_status(trace: dict[str, Any]) -> bool:
    return trace.get("status") in {
        "accepted_for_manual_review_only",
        "rejected_no_auto_promotion",
        "needs_human_review",
    }


def _reconcile_interrupted_attempts(
    *,
    state: dict[str, Any],
    tasks_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    """Inspect a trace that may have finished just before the coordinator was interrupted.

    B/R/F writes each trace atomically. If that write completed but the outer process died before
    its terminal state checkpoint, reuse the persisted actual result rather than automatically
    sampling a second model answer on resume. A persisted error/malformed trace becomes an
    explicit generation failure, which still requires --retry-failed.
    """

    reconciled: list[str] = []
    for item in state.get("tasks", []):
        if not isinstance(item, dict) or item.get("status") != "interrupted":
            continue
        task_id = item.get("task_id")
        trace_name = item.get("trace_path")
        if not isinstance(task_id, str) or not isinstance(trace_name, str) or not Path(trace_name).exists():
            continue
        try:
            trace = read_json(trace_name)
            if not isinstance(trace, dict):
                raise RealModelPilotError("trace root is not an object")
            item["trace_sha256"] = sha256_file(trace_name)
            item["trace_status"] = trace.get("status")
            seed = tasks_by_id.get(task_id)
            if seed is None:
                raise RealModelPilotError("interrupted trace task is absent from the validated pilot catalog")
            materialized = materialize_trace_candidates(
                seed=seed,
                trace=trace,
                trace_path=trace_name,
                run_id=str(state["run_id"]),
                task_set_sha256=str(state["task_set"]["sha256"]),
                model_provenance=_model_provenance(state),
                pipeline_version=state["pipeline_version"],
                attempt=int(item.get("attempts", 1) or 1),
            )
            item["raw_record_ids"] = [record["record_id"] for record in materialized.raw_records]
            item["corrected_record_ids"] = [record["record_id"] for record in materialized.corrected_records]
            item["final_record_ids"] = [record["record_id"] for record in materialized.final_records]
            if _terminal_trace_status(trace):
                item["status"] = "completed"
                item["error"] = None
                item["completed_at"] = trace.get("completed_at") or item.get("completed_at") or utc_now()
                reconciled.append(f"{task_id}: terminal trace recovered without rerunning the model")
            else:
                item["status"] = "generation_failed"
                item["error"] = str(
                    trace.get("terminal_reason")
                    or f"Interrupted attempt persisted non-terminal trace status {trace.get('status')!r}; inspect before --retry-failed"
                )
                reconciled.append(f"{task_id}: persisted error/non-terminal trace marked generation_failed")
        except (OSError, ValueError, RealModelPilotError) as exc:
            item["status"] = "generation_failed"
            item["error"] = f"Interrupted attempt trace is malformed/unreadable: {exc}"
            reconciled.append(f"{task_id}: malformed persisted trace marked generation_failed")
        attempt_rows = item.get("attempt_history")
        if isinstance(attempt_rows, list) and attempt_rows and isinstance(attempt_rows[-1], dict):
            if attempt_rows[-1].get("attempt") == item.get("attempts"):
                attempt_rows[-1].update(
                    {
                        "status": item.get("status"),
                        "completed_at": item.get("completed_at"),
                        "trace_sha256": item.get("trace_sha256"),
                        "trace_status": item.get("trace_status"),
                        "error": item.get("error"),
                        "raw_record_ids": deepcopy(item.get("raw_record_ids", [])),
                        "corrected_record_ids": deepcopy(item.get("corrected_record_ids", [])),
                        "final_record_ids": deepcopy(item.get("final_record_ids", [])),
                    }
                )
    return reconciled


def _write_materialized_artifacts(
    *,
    paths: PilotPaths,
    state: dict[str, Any],
    tasks_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Rebuild append-safe JSONL states from immutable per-attempt traces.

    It is intentionally safe to call after every task: each write is atomic, every trace is
    already persisted by B/R/F, and bad/error traces stay visible in state rather than being
    turned into invented candidate content.
    """

    materializations: list[PilotMaterialization] = []
    errors: list[str] = []
    for item in state.get("tasks", []):
        if not isinstance(item, dict):
            continue
        task_id = item.get("task_id")
        if not isinstance(task_id, str):
            continue
        seed = tasks_by_id.get(task_id)
        if seed is None:
            errors.append(f"Trace task {task_id!r} is absent from the validated pilot catalog")
            continue
        attempt_rows = item.get("attempt_history")
        trace_attempts: list[tuple[int, str]] = []
        if isinstance(attempt_rows, list):
            for row in attempt_rows:
                if not isinstance(row, dict):
                    continue
                attempt = row.get("attempt")
                trace_name = row.get("trace_path")
                if isinstance(attempt, int) and attempt >= 1 and isinstance(trace_name, str):
                    trace_attempts.append((attempt, trace_name))
        if not trace_attempts:
            history = item.get("trace_history")
            if isinstance(history, list):
                trace_attempts = [
                    (index, trace_name)
                    for index, trace_name in enumerate(history, start=1)
                    if isinstance(trace_name, str)
                ]
            elif isinstance(item.get("trace_path"), str):
                trace_attempts = [(int(item.get("attempts", 1) or 1), item["trace_path"])]
        seen_trace_names: set[str] = set()
        for attempt, trace_name in trace_attempts:
            if trace_name in seen_trace_names or not Path(trace_name).exists():
                continue
            seen_trace_names.add(trace_name)
            try:
                trace = read_json(trace_name)
                if not isinstance(trace, dict):
                    raise RealModelPilotError("trace root is not an object")
                materialized = materialize_trace_candidates(
                    seed=seed,
                    trace=trace,
                    trace_path=trace_name,
                    run_id=str(state["run_id"]),
                    task_set_sha256=str(state["task_set"]["sha256"]),
                    model_provenance=_model_provenance(state),
                    pipeline_version=state["pipeline_version"],
                    attempt=attempt,
                )
                materializations.append(materialized)
            except (OSError, ValueError, RealModelPilotError) as exc:
                errors.append(f"{task_id} attempt {attempt}: {exc}")
    raw, corrected, finals = merge_materialized_records(materializations)
    write_jsonl_atomic(paths.raw_candidates, raw)
    write_jsonl_atomic(paths.corrected_candidates, corrected)
    write_jsonl_atomic(paths.final_candidates, finals)
    return raw, corrected, finals, errors


def _write_correction_linkage_artifacts(paths: PilotPaths, final_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Persist an explicit, non-invented subset usable by Factory correction linking.

    The full ``corrected_candidates.jsonl`` remains the canonical capture of every Fixer answer.
    This derived subset contains only corrections whose own persisted metadata satisfies the
    Factory explanation rule. Unsupported corrections stay visible in the source artifact and
    linkage report but cannot make a ledger/failure row claim an explanation that was never
    supplied.
    """

    corrected = list(read_jsonl(paths.corrected_candidates))
    linkable: list[dict[str, Any]] = []
    unlinked: list[dict[str, Any]] = []
    for record in corrected:
        evidence = correction_explanation_evidence(record)
        if evidence.get("status") == "linkable":
            linkable.append(record)
            continue
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        unlinked.append(
            {
                "record_id": record.get("record_id"),
                "parent_record_id": metadata.get("parent_record_id"),
                "correction_round": metadata.get("correction_round"),
                "status": evidence.get("status"),
                "source_field": evidence.get("source_field"),
                "reason": evidence.get("reason"),
            }
        )

    # Existing pre-fix runs have immutable final artifacts without the newer explicit quality
    # hold. Filter only the derived deduplication input so those old records cannot become
    # eligible during repair; the full final JSONL remains untouched and inspectable.
    deduplication_input: list[dict[str, Any]] = []
    terminal_excluded: list[dict[str, Any]] = []
    for record in final_records:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        is_correction = metadata.get("stage") == "corrected" or isinstance(metadata.get("parent_record_id"), str)
        evidence = correction_explanation_evidence(record) if is_correction else {"status": "linkable"}
        if evidence.get("status") == "linkable":
            deduplication_input.append(record)
            continue
        terminal_excluded.append(
            {
                "record_id": record.get("record_id"),
                "parent_record_id": metadata.get("parent_record_id"),
                "status": evidence.get("status"),
                "source_field": evidence.get("source_field"),
                "reason": evidence.get("reason"),
            }
        )
    write_jsonl_atomic(paths.linkable_corrections, linkable)
    write_jsonl_atomic(paths.deduplication_input_candidates, deduplication_input)
    report = {
        "stage": "pilot_correction_linkage",
        "corrected_candidates": str(paths.corrected_candidates),
        "linkable_corrections": str(paths.linkable_corrections),
        "deduplication_input_candidates": str(paths.deduplication_input_candidates),
        "corrected_record_count": len(corrected),
        "linkable_record_count": len(linkable),
        "unlinked_record_count": len(unlinked),
        "unlinked_corrections": unlinked,
        "deduplication_input_record_count": len(deduplication_input),
        "terminal_records_excluded_from_deduplication": terminal_excluded,
        "non_claim": (
            "An unlinked correction is preserved actual Fixer output, not a verified correction. "
            "No change explanation was inferred from answer text, reviewer findings, or a diff."
        ),
    }
    write_json_atomic(paths.correction_linkage_report, report)
    return report


def _run_sidecars(
    *,
    arguments: argparse.Namespace,
    paths: PilotPaths,
    final_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Use existing Factory CLIs programmatically; never call dataset construction."""

    sidecar: dict[str, Any] = {
        "correction_linkage": {"status": "not_run"},
        "deduplication": {"status": "not_run"},
        "ledger": {"status": "not_run"},
        "failure_recording": {"status": "not_run"},
        "failure_analysis": {"status": "not_run"},
    }
    correction_linkage = _write_correction_linkage_artifacts(paths, final_records)
    sidecar["correction_linkage"] = {
        "status": "complete",
        "report": str(paths.correction_linkage_report),
        "linkable_record_count": correction_linkage["linkable_record_count"],
        "unlinked_record_count": correction_linkage["unlinked_record_count"],
        "deduplication_input_record_count": correction_linkage["deduplication_input_record_count"],
        "terminal_excluded_from_deduplication_count": len(correction_linkage["terminal_records_excluded_from_deduplication"]),
    }
    deduplication_input_records = list(read_jsonl(paths.deduplication_input_candidates))
    if deduplication_input_records:
        dedupe_args = argparse.Namespace(
            input=[str(paths.deduplication_input_candidates)],
            output=str(paths.deduplicated_candidates),
            report=str(paths.root / "deduplication_report.json"),
            evaluation=arguments.evaluation,
            config=arguments.pipeline_config,
            threshold=None,
            cross_split_threshold=None,
            # Preserve every terminal state in final_candidates.jsonl, but let the established
            # sidecar exclude static/review-ineligible attempts before duplicate anchoring. An
            # earlier failed retry must not make a later independently accepted response a
            # duplicate merely because their answer text matches.
            include_ineligible=False,
        )
        if deduplicate_examples.run(dedupe_args) != 0:
            raise PilotExecutionError("Existing deduplication sidecar returned a non-zero result")
        sidecar["deduplication"] = {"status": "complete", "report": str(dedupe_args.report)}
    else:
        write_jsonl_atomic(paths.deduplicated_candidates, [])
        write_json_atomic(
            paths.root / "deduplication_report.json",
            {
                "stage": "deduplication",
                "status": "not_run_no_deduplication_eligible_terminal_candidates",
                "non_claim": (
                    "No candidate was invented. Terminal output was absent or every terminal candidate "
                    "was excluded before deduplication by existing quality prerequisites or missing "
                    "actual correction explanation evidence."
                ),
            },
        )
        sidecar["deduplication"] = {"status": "not_run_no_deduplication_eligible_terminal_candidates"}

    # Preserve a ledger row for every actual Builder/Fixer candidate stage so multi-round
    # lineage remains inspectable. Only the correction-link argument is filtered: an empty
    # Fixer explanation must not be attached as if it supplied Factory-valid evidence.
    candidate_inputs = [str(paths.raw_candidates), str(paths.corrected_candidates)]
    ledger_args = argparse.Namespace(
        input=candidate_inputs,
        corrections=[str(paths.linkable_corrections)],
        output=str(paths.ledger),
        report=str(paths.root / "training_factory_ledger_report.json"),
        taxonomy=arguments.taxonomy,
        created_at=None,
    )
    if build_training_ledger.run(ledger_args) != 0:
        raise PilotExecutionError("Existing Training Factory ledger sidecar returned a non-zero result")
    sidecar["ledger"] = {"status": "complete", "report": str(ledger_args.report)}

    # Failure recording accepts a later same-ID dedupe stage and deliberately merges its richer
    # duplicate/split-isolation evidence with the original answer. Keep corrected attempts in
    # failure inputs as actual observed evidence, even when their
    # missing explanation makes them ineligible for a parent-correction link.
    failure_inputs = [str(paths.raw_candidates), str(paths.corrected_candidates), str(paths.deduplicated_candidates)]
    failures_args = argparse.Namespace(
        input=failure_inputs,
        corrections=[str(paths.linkable_corrections)],
        output=str(paths.failures),
        report=str(paths.root / "failure_recording_report.json"),
        taxonomy=arguments.taxonomy,
        created_at=None,
    )
    if record_failures.run(failures_args) != 0:
        raise PilotExecutionError("Existing failure-recording sidecar returned a non-zero result")
    sidecar["failure_recording"] = {"status": "complete", "report": str(failures_args.report)}

    analysis_args = argparse.Namespace(
        input=[str(paths.failures)],
        output=str(paths.failure_analysis),
        taxonomy=arguments.taxonomy,
        config=arguments.factory_config,
        minimum_failure_count=None,
        created_at=None,
    )
    if analyze_failures.run(analysis_args) != 0:
        raise PilotExecutionError("Existing failure-analysis sidecar returned a non-zero result")
    sidecar["failure_analysis"] = {"status": "complete", "report": str(paths.failure_analysis)}
    return sidecar


def _summarize_traces(state: dict[str, Any]) -> dict[str, Any]:
    trace_statuses: Counter[str] = Counter()
    review_statuses: Counter[str] = Counter()
    static_statuses: Counter[str] = Counter()
    fixer_count = 0
    reviewer_findings = 0
    for item in state.get("tasks", []):
        if not isinstance(item, dict):
            continue
        trace_names: list[str] = []
        attempt_rows = item.get("attempt_history")
        if isinstance(attempt_rows, list):
            trace_names.extend(
                row["trace_path"]
                for row in attempt_rows
                if isinstance(row, dict) and isinstance(row.get("trace_path"), str)
            )
        if not trace_names and isinstance(item.get("trace_history"), list):
            trace_names.extend(name for name in item["trace_history"] if isinstance(name, str))
        if not trace_names and isinstance(item.get("trace_path"), str):
            trace_names.append(item["trace_path"])
        for trace_name in dict.fromkeys(trace_names):
            path = Path(trace_name)
            if not path.exists():
                continue
            try:
                trace = read_json(path)
            except (OSError, ValueError):
                continue
            if not isinstance(trace, dict):
                continue
            trace_statuses[str(trace.get("status", "unknown"))] += 1
            rounds = trace.get("rounds")
            if not isinstance(rounds, list):
                continue
            for round_trace in rounds:
                if not isinstance(round_trace, dict):
                    continue
                static = round_trace.get("static_validation")
                if isinstance(static, dict):
                    static_statuses[str(static.get("status", "unknown"))] += 1
                reviewer = round_trace.get("reviewer")
                if isinstance(reviewer, dict):
                    review_statuses[str(reviewer.get("status", "unknown"))] += 1
                    findings = reviewer.get("findings")
                    if isinstance(findings, list):
                        reviewer_findings += len(findings)
                fixer = round_trace.get("fixer")
                if isinstance(fixer, dict) and fixer.get("status") == "complete":
                    fixer_count += 1
    return {
        "trace_attempt_count": sum(trace_statuses.values()),
        "error_trace_attempt_count": trace_statuses.get("error", 0),
        "terminal_trace_statuses": dict(sorted(trace_statuses.items())),
        "review_statuses": dict(sorted(review_statuses.items())),
        "static_validation_statuses": dict(sorted(static_statuses.items())),
        "reviewer_finding_count": reviewer_findings,
        "fixer_response_count": fixer_count,
    }


def _write_final_report(
    *,
    arguments: argparse.Namespace,
    config: dict[str, Any],
    paths: PilotPaths,
    state: dict[str, Any],
    sidecars: dict[str, Any] | None,
    materialization_errors: list[str],
) -> dict[str, Any]:
    terminal = list(read_jsonl(paths.final_candidates)) if paths.final_candidates.exists() else []
    deduplicated = list(read_jsonl(paths.deduplicated_candidates)) if paths.deduplicated_candidates.exists() else []
    eligible, _deduplicated_ineligible = quality_eligible_records(deduplicated)
    deduplicated_by_id = {str(record.get("record_id")): record for record in deduplicated}
    ineligible: dict[str, list[str]] = {}
    for record in terminal:
        record_id = str(record.get("record_id"))
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        is_correction = metadata.get("stage") == "corrected" or isinstance(metadata.get("parent_record_id"), str)
        correction_evidence = correction_explanation_evidence(record) if is_correction else {"status": "linkable"}
        if correction_evidence.get("status") != "linkable":
            ineligible[record_id] = ["correction_explanation_missing_actual_evidence"]
            continue
        pre_dedupe_ok, pre_dedupe_reasons = deduplicate_examples.pre_dedupe_eligibility(record)
        if not pre_dedupe_ok:
            ineligible[record_id] = pre_dedupe_reasons
            continue
        deduplicated_record = deduplicated_by_id.get(record_id)
        if deduplicated_record is None:
            ineligible[record_id] = ["not_emitted_by_deduplication_sidecar"]
            continue
        accepted, reasons = quality_eligible_records([deduplicated_record])
        if not accepted:
            ineligible[record_id] = reasons.get(record_id, ["canonical_final_quality_gate_rejected"])
    write_jsonl_atomic(paths.training_eligible_candidates, eligible)
    failures = list(read_jsonl(paths.failures)) if paths.failures.exists() else []
    correction_linkage_report = read_json(paths.correction_linkage_report) if paths.correction_linkage_report.exists() else {}
    ledger_report = read_json(paths.root / "training_factory_ledger_report.json") if (paths.root / "training_factory_ledger_report.json").exists() else {}
    failure_report = read_json(paths.root / "failure_recording_report.json") if (paths.root / "failure_recording_report.json").exists() else {}
    analysis = read_json(paths.failure_analysis) if paths.failure_analysis.exists() else {}
    failure_summary = analysis.get("summary", {}) if isinstance(analysis, dict) else {}
    category_counts = failure_summary.get("category_counts", {}) if isinstance(failure_summary, dict) else {}
    repeated = [
        {"failure_category": category, "observed_failure_count": count}
        for category, count in sorted(category_counts.items(), key=lambda item: (-item[1], item[0]))
        if isinstance(count, int) and count >= 2
    ]
    state_counts = summarize_state(state)
    trace_summary = _summarize_traces(state)
    attempted = sum(count for status, count in state_counts.items() if status not in {"pending"})
    completed = state_counts.get("completed", 0)
    report = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "kind": "dukeotr_real_model_pilot_report",
        "generated_at": utc_now(),
        "run": {
            "run_id": state.get("run_id"),
            "state": state.get("status"),
            "state_fingerprint": pilot_fingerprint(state),
            "sequential": True,
            "parallelism": 1,
            "task_status_counts": state_counts,
            "attempted_task_count": attempted,
            "completed_task_count": completed,
            "generation_failure_count": state_counts.get("generation_failed", 0),
            "generation_failure_attempt_count": trace_summary["error_trace_attempt_count"],
            "model_trace_attempt_count": trace_summary["trace_attempt_count"],
            "interrupted_task_count": state_counts.get("interrupted", 0),
            "pending_task_count": state_counts.get("pending", 0),
            "max_tasks_this_invocation": arguments.max_tasks,
            "retry_failed_requested": bool(arguments.retry_failed),
            "dry_run": bool(arguments.dry_run),
            "repair_sidecars_requested": bool(arguments.repair_sidecars),
        },
        "model": _model_provenance(state),
        "task_catalog": deepcopy(state.get("task_set", {})),
        "task_results": [
            {
                key: deepcopy(item.get(key))
                for key in (
                    "task_id",
                    "status",
                    "attempts",
                    "trace_path",
                    "trace_sha256",
                    "trace_status",
                    "trace_history",
                    "attempt_history",
                    "raw_record_ids",
                    "corrected_record_ids",
                    "final_record_ids",
                    "started_at",
                    "completed_at",
                    "error",
                )
            }
            for item in state.get("tasks", [])
            if isinstance(item, dict)
        ],
        "evaluation_isolation": deepcopy(state.get("evaluation_isolation", {})),
        "pipeline": deepcopy(state.get("pipeline_version", {})),
        "dataset_version": {
            "target": state.get("pipeline_version", {}).get("dataset_version_target"),
            "status": "planned_not_built_not_promoted",
        },
        "artifact_states": {
            "raw_model_output": {
                "path": str(paths.raw_candidates),
                "count": sum(1 for _ in read_jsonl(paths.raw_candidates)) if paths.raw_candidates.exists() else 0,
                "meaning": "Actual Builder assistant responses reconstructed from persisted B/R/F traces.",
            },
            "reviewed_terminal_output": {
                "path": str(paths.final_candidates),
                "count": sum(1 for _ in read_jsonl(paths.final_candidates)) if paths.final_candidates.exists() else 0,
                "meaning": "Terminal Builder/Fixer candidate with actual trace review/test evidence; a skipped/error reviewer remains visibly ineligible.",
            },
            "correction_linkage_evidence": {
                "path": str(paths.correction_linkage_report),
                "linkable_corrections_path": str(paths.linkable_corrections),
                "deduplication_input_path": str(paths.deduplication_input_candidates),
                "linkable_record_count": correction_linkage_report.get("linkable_record_count", 0),
                "unlinked_record_count": correction_linkage_report.get("unlinked_record_count", 0),
                "unlinked_corrections": deepcopy(correction_linkage_report.get("unlinked_corrections", [])),
                "terminal_records_excluded_from_deduplication": deepcopy(
                    correction_linkage_report.get("terminal_records_excluded_from_deduplication", [])
                ),
                "meaning": "Corrections without their own non-empty Fixer explanation are preserved as actual output but stay visibly unlinked and ineligible; no explanation is inferred.",
            },
            "verified_output": {
                "path": str(paths.deduplicated_candidates),
                "count": len(deduplicated),
                "meaning": "Terminal candidates that first met the existing static/reviewer prerequisites, then passed through deterministic deduplication and held-out wording isolation; excluded terminal states remain in final_candidates.jsonl. This is not a factual-correctness certificate.",
            },
            "training_eligible_output": {
                "path": str(paths.training_eligible_candidates),
                "count": len(eligible),
                "meaning": "Candidates currently satisfying the canonical final quality gate. This is a report-only state, never automatic dataset promotion.",
                "ineligible_reasons_by_record": ineligible,
            },
            "trace_directory": str(paths.trace_directory),
            "state_checkpoint": str(paths.state),
            "manifest_checkpoint": str(paths.manifest),
        },
        "findings_and_fixes": trace_summary,
        "verification": {
            "ledger_report": ledger_report,
            "failure_recording_report": failure_report,
            "stored_failure_count": len(failures),
            "failure_categories": failure_summary.get("category_counts", {}) if isinstance(failure_summary, dict) else {},
            "failure_severity": failure_summary.get("severity_counts", {}) if isinstance(failure_summary, dict) else {},
            "failure_verification": failure_summary.get("verification_counts", {}) if isinstance(failure_summary, dict) else {},
            "repeated_patterns": repeated,
            "recommendations": analysis.get("targeting_recommendations", []) if isinstance(analysis, dict) else [],
            "minimum_failure_count_for_recommendation": analysis.get("minimum_failure_count_for_recommendation") if isinstance(analysis, dict) else None,
            "recommendation_basis": "existing failure analysis over actual stored pilot evidence only, using its configured threshold",
        },
        "sidecars": sidecars or {"status": "not_run"},
        "sidecar_repair_history": deepcopy(state.get("sidecar_repair_history", [])),
        "materialization_errors": materialization_errors,
        "promotion": deepcopy(state.get("promotion", {})),
        "non_claims": [
            "Arena did not run this local model. This report is created only when the user runs the command on their own machine.",
            "No fake response, failure, verification result, model improvement score, adapter, training run, or DukeOTR model release is asserted.",
            "The pilot invokes no dataset builder and writes no final training_data artifact.",
            "A count in this report is bounded by this small predefined pilot set and is not a general measure of Qwen3-4B or DukeOTR capability.",
        ],
        "limitations": config.get("limitations", []),
    }
    write_json_atomic(paths.report, report)
    return report


def _set_final_state(state: dict[str, Any], *, dry_run: bool = False) -> None:
    statuses = summarize_state(state)
    if dry_run:
        state["status"] = "dry_run_complete"
    elif statuses.get("pending", 0) or statuses.get("interrupted", 0):
        state["status"] = "partial_with_generation_failures" if statuses.get("generation_failed", 0) else "partial"
    elif statuses.get("generation_failed", 0):
        state["status"] = "completed_with_generation_failures"
    else:
        state["status"] = "completed"


def _repair_snapshot(paths: PilotPaths, state: dict[str, Any]) -> dict[str, Any]:
    """Copy mutable/derived artifacts before a no-model sidecar repair changes any of them."""

    base = paths.root / "repair_snapshots"
    stamp = utc_now().replace(":", "").replace("-", "").replace(".", "").replace("+", "_")
    snapshot = base / stamp
    suffix = 1
    while snapshot.exists():
        snapshot = base / f"{stamp}-{suffix}"
        suffix += 1
    assert_safe_factory_output_path(snapshot)
    snapshot.mkdir(parents=True, exist_ok=False)
    artifacts = (
        paths.state,
        paths.manifest,
        paths.report,
        paths.raw_candidates,
        paths.corrected_candidates,
        paths.linkable_corrections,
        paths.correction_linkage_report,
        paths.deduplication_input_candidates,
        paths.final_candidates,
        paths.deduplicated_candidates,
        paths.training_eligible_candidates,
        paths.ledger,
        paths.failures,
        paths.failure_analysis,
        paths.root / "deduplication_report.json",
        paths.root / "training_factory_ledger_report.json",
        paths.root / "failure_recording_report.json",
    )
    copied: list[dict[str, str]] = []
    for artifact in artifacts:
        if not artifact.is_file():
            continue
        destination = snapshot / artifact.name
        shutil.copy2(artifact, destination)
        copied.append({"path": str(artifact), "sha256": sha256_file(artifact), "snapshot_copy": str(destination)})
    trace_hashes = [
        {"path": str(path), "sha256": sha256_file(path)}
        for path in sorted(paths.trace_directory.glob("*.trace.json"))
        if path.is_file()
    ]
    manifest = {
        "kind": "dukeotr_real_model_pilot_sidecar_repair_snapshot",
        "created_at": utc_now(),
        "run_id": state.get("run_id"),
        "purpose": "Preserve pre-repair derived artifacts; immutable B/R/F traces are referenced by hash and are not modified.",
        "copied_artifacts": copied,
        "immutable_trace_hashes": trace_hashes,
    }
    manifest_path = snapshot / "snapshot_manifest.json"
    write_json_atomic(manifest_path, manifest)
    return {
        "snapshot_directory": str(snapshot),
        "snapshot_manifest": str(manifest_path),
        "copied_artifacts": copied,
        "immutable_trace_hashes": trace_hashes,
    }


def _persisted_materialized_inputs(paths: PilotPaths) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Load repair inputs without rebuilding or rewriting model-derived candidate artifacts."""

    required = (paths.raw_candidates, paths.corrected_candidates, paths.final_candidates)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RealModelPilotError(
            "--repair-sidecars refuses to reconstruct or overwrite missing model-derived candidate artifacts. "
            "Required persisted files are missing: "
            + ", ".join(missing)
        )
    hashes = {str(path): sha256_file(path) for path in required}
    return list(read_jsonl(paths.final_candidates)), hashes


def _repair_sidecars(
    *,
    arguments: argparse.Namespace,
    config: dict[str, Any],
    paths: PilotPaths,
    state: dict[str, Any],
    repair_pipeline_version: dict[str, Any],
) -> int:
    """Run only deterministic/sidecar work against an older, persisted pilot run.

    This intentionally is *not* a model-generation compatibility override. It never instantiates
    an Ollama client, never invokes B/R/F, never retries a task, and never rewrites the raw,
    corrected, or final candidate artifacts. Normal model-generation resume keeps the existing
    exact pipeline fingerprint guard.
    """

    final_records, source_hashes = _persisted_materialized_inputs(paths)
    snapshot = _repair_snapshot(paths, state)
    prior_report = read_json(paths.report) if paths.report.is_file() else {}
    prior_errors = prior_report.get("materialization_errors", []) if isinstance(prior_report, dict) else []
    repair_event: dict[str, Any] = {
        "kind": "sidecar_repair_no_model",
        "started_at": utc_now(),
        "status": "running",
        "prior_state": state.get("status"),
        "stored_pipeline_version": deepcopy(state.get("pipeline_version", {})),
        "repair_coordinator_pipeline_version": deepcopy(repair_pipeline_version),
        "pipeline_fingerprint_mismatch_allowed_only_for_no_model_repair": state.get("pipeline_version") != repair_pipeline_version,
        "source_candidate_hashes_before": source_hashes,
        "prior_materialization_errors": deepcopy(prior_errors) if isinstance(prior_errors, list) else [],
        **snapshot,
        "non_claim": (
            "This operation reruns no inference and infers no correction explanation. It preserves "
            "the original trace/model provenance and only regenerates derived sidecar evidence."
        ),
    }
    history = state.setdefault("sidecar_repair_history", [])
    if not isinstance(history, list):
        raise RealModelPilotError("Existing pilot state has malformed sidecar_repair_history")
    history.append(repair_event)
    write_pilot_state(paths.state, state)

    try:
        sidecars = _run_sidecars(arguments=arguments, paths=paths, final_records=final_records)
        source_hashes_after = {path: sha256_file(path) for path in source_hashes}
        if source_hashes_after != source_hashes:
            raise PilotExecutionError(
                "Sidecar repair detected a change to a source candidate artifact; preserved snapshot is available and repair is refused."
            )
        trace_hashes_after = [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in sorted(paths.trace_directory.glob("*.trace.json"))
            if path.is_file()
        ]
        if trace_hashes_after != snapshot["immutable_trace_hashes"]:
            raise PilotExecutionError(
                "Sidecar repair detected a change to an immutable B/R/F trace; preserved snapshot is available and repair is refused."
            )
        _set_final_state(state)
        repair_event.update(
            {
                "status": "complete",
                "completed_at": utc_now(),
                "source_candidate_hashes_after": source_hashes_after,
                "immutable_trace_hashes_after": trace_hashes_after,
                "sidecars": deepcopy(sidecars),
            }
        )
        write_pilot_state(paths.state, state)
        _write_final_report(
            arguments=arguments,
            config=config,
            paths=paths,
            state=state,
            sidecars=sidecars,
            materialization_errors=[],
        )
        print(
            f"Sidecar repair completed without Ollama/B/R/F calls. Pilot state: {state['status']}; "
            f"report: {paths.report}; snapshot: {snapshot['snapshot_directory']}"
        )
        # This command's success criterion is sidecar repair, not a claim that prior model
        # attempts were successful. The preserved run state/report still expose any pending or
        # generation_failed tasks, while exit 0 lets an operator distinguish repair success from
        # another sidecar crash.
        return 0
    except (OSError, ValueError, TrainingFactoryError, PilotExecutionError, RealModelPilotError) as exc:
        state["status"] = "sidecar_error"
        repair_event.update({"status": "error", "completed_at": utc_now(), "error": str(exc)})
        write_pilot_state(paths.state, state)
        _write_final_report(
            arguments=arguments,
            config=config,
            paths=paths,
            state=state,
            sidecars={"status": "error", "error": str(exc), "operation": "sidecar_repair_no_model"},
            materialization_errors=[f"Training Factory sidecar repair error: {exc}"],
        )
        print(
            f"Sidecar repair stopped safely without an Ollama/B/R/F call: {exc}\n"
            f"Report: {paths.report}; pre-repair snapshot: {snapshot['snapshot_directory']}",
            file=sys.stderr,
        )
        return 2


def _brf_arguments(arguments: argparse.Namespace, *, task_id: str, trace_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        seed_id=task_id,
        seeds=arguments.tasks,
        evaluation=arguments.evaluation,
        code_book=arguments.code_book,
        config=arguments.brf_config,
        pipeline_config=arguments.pipeline_config,
        output=str(trace_path),
        overwrite=False,
        max_rounds=None,
        # Pin every B/R/F role for this v1 baseline; a future mixed-provider pilot must be a
        # separately versioned workflow, not an unnoticed role-config override.
        model=arguments.model,
        builder_model=arguments.model,
        reviewer_model=arguments.model,
        fixer_model=arguments.model,
        host=arguments.host,
        timeout_seconds=arguments.timeout_seconds,
        dry_run=False,
    )


def run(arguments: argparse.Namespace) -> int:
    validate_run_id(arguments.run_id)
    _require_positive_timeout(arguments.timeout_seconds)
    if arguments.max_tasks is not None and (not isinstance(arguments.max_tasks, int) or arguments.max_tasks < 1):
        raise RealModelPilotError("--max-tasks must be a positive integer when supplied")
    if arguments.repair_sidecars and (arguments.dry_run or arguments.retry_failed or arguments.max_tasks is not None):
        raise RealModelPilotError("--repair-sidecars cannot be combined with --dry-run, --retry-failed, or --max-tasks; it never runs inference")
    if arguments.model != PILOT_MODEL:
        raise RealModelPilotError(
            f"Real Model Pilot v1 is deliberately pinned to {PILOT_MODEL!r}; use a separately versioned future pilot for another provider/model."
        )
    config = _read_config(arguments.config)
    tasks = load_pilot_tasks(arguments.tasks)
    evaluation_tasks = list(read_jsonl(arguments.evaluation))
    pipeline_config = read_json(arguments.pipeline_config)
    if not isinstance(pipeline_config, dict):
        raise RealModelPilotError("Pipeline config must be an object")
    threshold = float(pipeline_config.get("deduplication", {}).get("cross_split_prompt_threshold", 0.93))
    isolation = validate_pilot_tasks(
        tasks,
        evaluation_tasks,
        evaluation_path=arguments.evaluation,
        cross_split_threshold=threshold,
    )
    pipeline_version = _pipeline_version(arguments, config)
    paths = PilotPaths.from_root(arguments.output_root, arguments.run_id)
    _assert_safe_paths(paths)
    _assert_pilot_output_boundary(paths)
    if arguments.repair_sidecars and not paths.state.exists():
        raise RealModelPilotError("--repair-sidecars requires an existing pilot_state.json; it cannot create or infer a model run")
    paths.create_directories()
    tasks_by_id = {task["id"]: task for task in tasks}

    if paths.state.exists():
        state = load_pilot_state(paths.state)
        _state_matches(
            state,
            run_id=arguments.run_id,
            task_path=arguments.tasks,
            tasks=tasks,
            model=arguments.model,
            evaluation_isolation=isolation,
            pipeline_version=pipeline_version,
            allow_pipeline_mismatch_for_sidecar_repair=arguments.repair_sidecars,
        )
        if arguments.repair_sidecars:
            return _repair_sidecars(
                arguments=arguments,
                config=config,
                paths=paths,
                state=state,
                repair_pipeline_version=pipeline_version,
            )
        if arguments.dry_run and any(
            isinstance(item, dict) and int(item.get("attempts", 0) or 0) > 0 for item in state.get("tasks", [])
        ):
            raise RealModelPilotError(
                "Refusing to overwrite an existing model-evidence run with --dry-run; use a new --run-id for a separate static plan."
            )
        recovered = recover_interrupted_tasks(state)
        reconciled = _reconcile_interrupted_attempts(state=state, tasks_by_id=tasks_by_id)
        if recovered:
            print(f"Recovered interrupted task marker(s): {', '.join(recovered)}")
        if reconciled:
            print("Recovered persisted trace result(s): " + "; ".join(reconciled))
        if recovered or reconciled:
            write_pilot_state(paths.state, state)
    else:
        state = initial_pilot_state(
            run_id=arguments.run_id,
            task_path=arguments.tasks,
            tasks=tasks,
            model=arguments.model,
            host=arguments.host,
            timeout_seconds=arguments.timeout_seconds,
            pipeline_version=pipeline_version,
            evaluation_isolation=isolation,
            dry_run=arguments.dry_run,
        )
        write_pilot_state(paths.state, state)

    if arguments.dry_run:
        _set_final_state(state, dry_run=True)
        write_pilot_state(paths.state, state)
        # Do not fabricate empty sidecar evidence during a no-model planning audit.
        _write_final_report(
            arguments=arguments,
            config=config,
            paths=paths,
            state=state,
            sidecars={"status": "not_run_dry_run"},
            materialization_errors=[],
        )
        print(f"Dry run complete: validated {len(tasks)} separate pilot briefs and held-out isolation. No Ollama request was made.\nReport: {paths.report}")
        return 0

    runnable = runnable_task_ids(state, retry_failed=arguments.retry_failed)
    selected = [task for task in tasks if task["id"] in runnable]
    if arguments.max_tasks is not None:
        selected = selected[: arguments.max_tasks]

    if selected:
        # Required real-run preflight. OllamaClient first executes `ollama list` for a local host,
        # then confirms /api/tags. It does not pull, alter, or replace qwen3:4b.
        client = OllamaClient(arguments.host, timeout_seconds=arguments.timeout_seconds)
        # Preserve the resolved local endpoint (including a local OLLAMA_HOST override) that
        # was actually preflighted, rather than recording a guessed default.
        state["model"]["host"] = client.host
        if not client.is_local_host():  # The existing client decides this from --host/OLLAMA_HOST.
            message = (
                "Real Model Pilot v1 is local-only. Use a loopback Ollama host so `ollama list` can attest "
                "the exact existing qwen3:4b tag; remote endpoints are refused."
            )
            state["model"]["preflight"] = {
                "status": "unavailable",
                "checked_at": utc_now(),
                "error": message,
                "required_command": "ollama list",
            }
            state["status"] = "blocked_preflight"
            write_pilot_state(paths.state, state)
            _write_final_report(
                arguments=arguments,
                config=config,
                paths=paths,
                state=state,
                sidecars={"status": "not_run_remote_host_refused"},
                materialization_errors=[],
            )
            print(f"Pilot preflight failed safely: {message}\nState/report: {paths.state} / {paths.report}", file=sys.stderr)
            return 2
        try:
            client.assert_model_present(arguments.model)
        except OllamaError as exc:
            state["model"]["preflight"] = {
                "status": "unavailable",
                "checked_at": utc_now(),
                "error": str(exc),
                "required_command": "ollama list",
            }
            state["status"] = "blocked_preflight"
            write_pilot_state(paths.state, state)
            _write_final_report(
                arguments=arguments,
                config=config,
                paths=paths,
                state=state,
                sidecars={"status": "not_run_preflight_failed"},
                materialization_errors=[],
            )
            print(f"Pilot preflight failed safely: {exc}\nState/report: {paths.state} / {paths.report}", file=sys.stderr)
            return 2
        state["model"]["preflight"] = {
            "status": "present",
            "checked_at": utc_now(),
            "required_command": "ollama list",
            "api_confirmation": "/api/tags",
            "non_mutating": True,
        }
        state["status"] = "running"
        write_pilot_state(paths.state, state)
    elif any(item.get("status") == "generation_failed" for item in state.get("tasks", []) if isinstance(item, dict)) and not arguments.retry_failed:
        print("Generation-failed task(s) remain. They were not retried; pass --retry-failed only after inspecting their trace/state.")

    materialization_errors: list[str] = []
    try:
        for ordinal, task in enumerate(selected, start=1):
            item = task_state(state, task["id"])
            item["status"] = "running"
            item["attempts"] = int(item.get("attempts", 0)) + 1
            item["started_at"] = utc_now()
            item["completed_at"] = None
            item["error"] = None
            trace_path = _trace_path(paths, task["id"], int(item["attempts"]))
            item["trace_path"] = str(trace_path)
            item.setdefault("trace_history", []).append(str(trace_path))
            item.setdefault("attempt_history", []).append(
                {
                    "attempt": int(item["attempts"]),
                    "trace_path": str(trace_path),
                    "status": "running",
                    "started_at": item["started_at"],
                    "completed_at": None,
                    "trace_sha256": None,
                    "trace_status": None,
                    "error": None,
                }
            )
            write_pilot_state(paths.state, state)
            print(f"[{ordinal}/{len(selected)}] {task['id']} — Builder/Tester/Reviewer/Fixer trace attempt {item['attempts']}…", flush=True)

            exit_code: int | None = None
            try:
                exit_code = run_builder_reviewer_fixer.run(_brf_arguments(arguments, task_id=task["id"], trace_path=trace_path))
            except KeyboardInterrupt:
                item["status"] = "interrupted"
                item["completed_at"] = utc_now()
                item["error"] = "Interrupted by user while the task was running. The trace path is preserved; resume to create a new attempt."
                attempt_rows = item.get("attempt_history")
                if isinstance(attempt_rows, list) and attempt_rows and isinstance(attempt_rows[-1], dict):
                    attempt_rows[-1].update(
                        {
                            "status": "interrupted",
                            "completed_at": item["completed_at"],
                            "error": item["error"],
                        }
                    )
                state["status"] = "interrupted"
                write_pilot_state(paths.state, state)
                raise
            except Exception as exc:  # Capture unexpected orchestration/config failures without losing state.
                item["status"] = "generation_failed"
                item["error"] = f"B/R/F invocation exception: {exc}"

            trace: dict[str, Any] | None = None
            if trace_path.exists():
                try:
                    loaded = read_json(trace_path)
                    if not isinstance(loaded, dict):
                        raise RealModelPilotError("trace root is not an object")
                    trace = loaded
                    item["trace_sha256"] = sha256_file(trace_path)
                    item["trace_status"] = trace.get("status")
                except (OSError, ValueError, RealModelPilotError) as exc:
                    item["status"] = "generation_failed"
                    item["error"] = f"Malformed/unreadable persisted trace: {exc}"
            else:
                item["status"] = "generation_failed"
                item["error"] = item.get("error") or f"B/R/F returned {exit_code!r} without a persisted trace"

            if trace is not None:
                try:
                    materialized = materialize_trace_candidates(
                        seed=task,
                        trace=trace,
                        trace_path=trace_path,
                        run_id=arguments.run_id,
                        task_set_sha256=state["task_set"]["sha256"],
                        model_provenance=_model_provenance(state),
                        pipeline_version=state["pipeline_version"],
                        attempt=int(item["attempts"]),
                    )
                    item["raw_record_ids"] = [record["record_id"] for record in materialized.raw_records]
                    item["corrected_record_ids"] = [record["record_id"] for record in materialized.corrected_records]
                    item["final_record_ids"] = [record["record_id"] for record in materialized.final_records]
                except (ValueError, RealModelPilotError) as exc:
                    item["status"] = "generation_failed"
                    item["error"] = f"Trace materialization refused: {exc}"

                if item.get("status") != "generation_failed":
                    if exit_code == 0 and _terminal_trace_status(trace):
                        item["status"] = "completed"
                    else:
                        item["status"] = "generation_failed"
                        item["error"] = str(trace.get("terminal_reason") or f"B/R/F returned {exit_code} with trace status {trace.get('status')!r}")
            item["completed_at"] = utc_now()
            attempt_rows = item.get("attempt_history")
            if isinstance(attempt_rows, list) and attempt_rows and isinstance(attempt_rows[-1], dict):
                current_attempt = attempt_rows[-1]
                if current_attempt.get("attempt") == item.get("attempts"):
                    current_attempt.update(
                        {
                            "status": item.get("status"),
                            "completed_at": item["completed_at"],
                            "trace_sha256": item.get("trace_sha256"),
                            "trace_status": item.get("trace_status"),
                            "error": item.get("error"),
                            "raw_record_ids": deepcopy(item.get("raw_record_ids", [])),
                            "corrected_record_ids": deepcopy(item.get("corrected_record_ids", [])),
                            "final_record_ids": deepcopy(item.get("final_record_ids", [])),
                        }
                    )
            write_pilot_state(paths.state, state)

            # Persist all valid raw/corrected/terminal states after every task, even one ending in
            # reviewer/fixer error. This makes resume and inspection independent of final sidecars.
            _raw, _corrected, _finals, errors = _write_materialized_artifacts(
                paths=paths,
                state=state,
                tasks_by_id=tasks_by_id,
            )
            materialization_errors.extend(errors)
            if errors:
                print("  materialization warning: " + "; ".join(errors), file=sys.stderr)
            print(f"  {task['id']}: {item['status']} (trace: {trace_path})", flush=True)
    except KeyboardInterrupt:
        _set_final_state(state)
        write_pilot_state(paths.state, state)
        _write_final_report(
            arguments=arguments,
            config=config,
            paths=paths,
            state=state,
            sidecars={"status": "not_run_interrupted"},
            materialization_errors=materialization_errors,
        )
        print(f"Pilot interrupted safely. Resume the same --run-id after inspection: {paths.state}", file=sys.stderr)
        return 130

    raw, corrected, finals, errors = _write_materialized_artifacts(paths=paths, state=state, tasks_by_id=tasks_by_id)
    materialization_errors.extend(errors)
    sidecars: dict[str, Any] | None = None
    try:
        sidecars = _run_sidecars(arguments=arguments, paths=paths, final_records=finals)
    except (OSError, ValueError, TrainingFactoryError, PilotExecutionError) as exc:
        state["status"] = "sidecar_error"
        materialization_errors.append(f"Training Factory sidecar error: {exc}")
        write_pilot_state(paths.state, state)
        _write_final_report(
            arguments=arguments,
            config=config,
            paths=paths,
            state=state,
            sidecars={"status": "error", "error": str(exc)},
            materialization_errors=materialization_errors,
        )
        print(f"Pilot evidence persisted, but sidecar processing failed safely: {exc}\nReport: {paths.report}", file=sys.stderr)
        return 2

    _set_final_state(state)
    write_pilot_state(paths.state, state)
    report = _write_final_report(
        arguments=arguments,
        config=config,
        paths=paths,
        state=state,
        sidecars=sidecars,
        materialization_errors=materialization_errors,
    )
    print(
        f"Pilot run state: {state['status']}; completed={report['run']['completed_task_count']}; "
        f"generation_failures={report['run']['generation_failure_count']}; report: {paths.report}"
    )
    # A batch limit is a normal partial success; an observed model/trace failure is deliberately
    # non-zero so automation cannot mistake discovery evidence for a clean successful pilot.
    return 0 if state["status"] in {"completed", "partial"} else 2


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError, TrainingFactoryError, RealModelPilotError) as exc:
        print(f"real model pilot error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
