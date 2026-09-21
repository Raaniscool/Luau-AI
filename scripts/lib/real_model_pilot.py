"""Non-generative orchestration helpers for a small real-local-model pilot.

This module deliberately does not implement Builder, Reviewer, Fixer, Ollama HTTP, failure
classification, deduplication, or final dataset construction.  It validates a separate pilot
brief set, persists resumable pilot state, and translates the existing Builder → Reviewer →
Fixer trace into canonical candidate records so the established Training Factory sidecars can
record actual evidence.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable

from scripts.lib.dedupe import cross_split_prompt_collisions
from scripts.lib.io_utils import canonical_json, read_json, read_jsonl, sha256_file, sha256_text, text_from_message, utc_now, write_json_atomic
from scripts.lib.schema import STATIC_CHECKER_VERSION, clone_for_correction, make_generated_record, quality_gate_status, validate_seed


PILOT_SCHEMA_VERSION = "1.0"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


class RealModelPilotError(ValueError):
    """Raised for a pilot boundary, provenance, or resumability violation."""


@dataclass(frozen=True)
class PilotPaths:
    """All pilot outputs live together outside final training-data paths."""

    root: Path
    trace_directory: Path
    state: Path
    manifest: Path
    report: Path
    raw_candidates: Path
    corrected_candidates: Path
    linkable_corrections: Path
    correction_linkage_report: Path
    deduplication_input_candidates: Path
    final_candidates: Path
    deduplicated_candidates: Path
    training_eligible_candidates: Path
    ledger: Path
    failures: Path
    failure_analysis: Path

    @classmethod
    def from_root(cls, output_root: str | Path, run_id: str) -> "PilotPaths":
        root = Path(output_root) / run_id
        return cls(
            root=root,
            trace_directory=root / "traces",
            state=root / "pilot_state.json",
            manifest=root / "pilot_manifest.json",
            report=root / "pilot_report.json",
            raw_candidates=root / "raw_builder_candidates.jsonl",
            corrected_candidates=root / "corrected_candidates.jsonl",
            linkable_corrections=root / "linkable_corrections.jsonl",
            correction_linkage_report=root / "correction_linkage_report.json",
            deduplication_input_candidates=root / "deduplication_input_candidates.jsonl",
            final_candidates=root / "final_candidates.jsonl",
            deduplicated_candidates=root / "deduplicated_candidates.jsonl",
            training_eligible_candidates=root / "training_eligible_candidates.jsonl",
            ledger=root / "training_factory_ledger.jsonl",
            failures=root / "observed_failures.jsonl",
            failure_analysis=root / "failure_analysis.json",
        )

    def create_directories(self) -> None:
        self.trace_directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PilotMaterialization:
    """Canonical records reconstructed from one persisted B/R/F trace."""

    raw_records: tuple[dict[str, Any], ...]
    corrected_records: tuple[dict[str, Any], ...]
    final_records: tuple[dict[str, Any], ...]
    reviewer_finding_count: int
    fixer_count: int


def correction_explanation_evidence(record: dict[str, Any]) -> dict[str, str]:
    """Classify whether a persisted correction can be linked by Factory sidecars.

    A Fixer response itself remains actual pilot evidence even when it failed to provide the
    explanation required by the Training Factory lineage contract. This helper deliberately
    mirrors that contract's ``changes_made``/``reason`` fallback rather than inventing a
    description from the changed answer, reviewer findings, or a diff.
    """

    metadata = record.get("metadata")
    correction = metadata.get("correction") if isinstance(metadata, dict) else None
    if not isinstance(correction, dict):
        return {
            "status": "ineligible_missing_actual_explanation",
            "reason": "Corrected candidate has no correction metadata object.",
        }
    source_field = "changes_made" if "changes_made" in correction else "reason"
    explanation = correction.get(source_field, [])
    if isinstance(explanation, str) and explanation.strip():
        return {"status": "linkable", "source_field": source_field}
    if isinstance(explanation, list) and explanation and all(isinstance(item, str) and item.strip() for item in explanation):
        return {"status": "linkable", "source_field": source_field}
    return {
        "status": "ineligible_missing_actual_explanation",
        "source_field": source_field,
        "reason": (
            "Fixer correction evidence has no non-empty actual changes_made/reason value; "
            "the coordinator did not infer one from the answer or review."
        ),
    }


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise RealModelPilotError("--run-id must be 1-80 letters, numbers, underscores, or hyphens and start with a letter/number")
    return run_id


def load_pilot_tasks(path: str | Path) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    if not rows:
        raise RealModelPilotError(f"Pilot task set {path} is empty")
    return rows


def _evaluation_collision_view(collisions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only IDs/scores; never serialize a held-out prompt or rubric into pilot output."""

    result: list[dict[str, Any]] = []
    for item in collisions:
        result.append(
            {
                "pilot_task_id": item.get("record_id"),
                "evaluation_task_id": item.get("evaluation_task_id"),
                "similarity": item.get("similarity"),
                "threshold": item.get("threshold"),
            }
        )
    return result


def validate_pilot_tasks(
    tasks: list[dict[str, Any]],
    evaluation_tasks: list[dict[str, Any]],
    *,
    evaluation_path: str | Path,
    cross_split_threshold: float,
) -> dict[str, Any]:
    """Validate task identity and hold-out wording isolation before any Ollama preflight."""

    if not 0 < cross_split_threshold <= 1:
        raise RealModelPilotError("Pilot cross-split threshold must be in (0, 1]")
    identifiers: set[str] = set()
    evaluation_ids = {
        str(task.get("id"))
        for task in evaluation_tasks
        if isinstance(task, dict) and isinstance(task.get("id"), str) and task.get("id").strip()
    }
    errors: list[str] = []
    for index, task in enumerate(tasks, start=1):
        task_id = task.get("id")
        if not isinstance(task_id, str) or not task_id.startswith("pilot-"):
            errors.append(f"Pilot task {index} needs a non-evaluation id beginning with 'pilot-'")
            continue
        if task_id in identifiers:
            errors.append(f"Duplicate pilot task id: {task_id}")
        if task_id in evaluation_ids:
            errors.append(f"Pilot task {task_id}: ID collides with permanently held-out evaluation material")
        identifiers.add(task_id)
        seed_errors = validate_seed(task)
        if seed_errors:
            rendered = "; ".join(f"{item['code']}: {item['message']}" for item in seed_errors)
            errors.append(f"Pilot task {task_id}: {rendered}")
        source = task.get("source")
        if not isinstance(source, dict) or source.get("pilot_only") is not True:
            errors.append(f"Pilot task {task_id}: source.pilot_only must be true")
        if str(task_id).lower().startswith("eval-"):
            errors.append(f"Pilot task {task_id}: evaluation IDs are forbidden")
    if errors:
        raise RealModelPilotError("Invalid pilot task set:\n- " + "\n- ".join(errors))

    placeholders = [
        {
            "record_id": task["id"],
            "source_seed_id": task["id"],
            "messages": [{"role": "user", "content": task["user_request"]}],
        }
        for task in tasks
    ]
    collisions = cross_split_prompt_collisions(placeholders, evaluation_tasks, threshold=cross_split_threshold)
    public_collisions = _evaluation_collision_view(collisions)
    if public_collisions:
        raise RealModelPilotError(
            "Pilot task set has held-out wording collision(s); no model request was made. "
            f"Collision task IDs: {', '.join(str(item['pilot_task_id']) for item in public_collisions)}"
        )
    return {
        "status": "clear",
        "task_count": len(tasks),
        "task_ids": [task["id"] for task in tasks],
        "evaluation_file": str(evaluation_path),
        "evaluation_sha256": sha256_file(evaluation_path),
        "comparison": "local normalized 3-gram Jaccard; held-out prompt/rubric text was never supplied to any model role",
        "threshold": cross_split_threshold,
        "collisions": [],
    }


def initial_pilot_state(
    *,
    run_id: str,
    task_path: str | Path,
    tasks: list[dict[str, Any]],
    model: str,
    host: str | None,
    timeout_seconds: int,
    pipeline_version: dict[str, Any],
    evaluation_isolation: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    """Create persisted state before first local inference so interruption never loses task identity."""

    timestamp = utc_now()
    return {
        "schema_version": PILOT_SCHEMA_VERSION,
        "kind": "dukeotr_real_model_pilot",
        "run_id": run_id,
        "status": "planned",
        "created_at": timestamp,
        "updated_at": timestamp,
        "model": {
            "requested_tag": model,
            "host": host or "http://127.0.0.1:11434",
            "timeout_seconds": timeout_seconds,
            "preflight": {"status": "not_run" if dry_run else "pending"},
        },
        "task_set": {
            "path": str(task_path),
            "sha256": sha256_file(task_path),
            "task_count": len(tasks),
            "task_ids": [task["id"] for task in tasks],
        },
        "evaluation_isolation": evaluation_isolation,
        "pipeline_version": deepcopy(pipeline_version),
        "promotion": {
            "status": "prohibited",
            "reason": (
                "This is a real-model baseline/discovery pilot. Raw, reviewed, and corrected candidates remain outside "
                "final training_data until independent normal gates and an explicit final dataset build occur."
            ),
        },
        "tasks": [
            {
                "task_id": task["id"],
                "status": "pending",
                "attempts": 0,
                "trace_path": None,
                "trace_sha256": None,
                "trace_status": None,
                "trace_history": [],
                "attempt_history": [],
                "raw_record_ids": [],
                "corrected_record_ids": [],
                "final_record_ids": [],
                "started_at": None,
                "completed_at": None,
                "error": None,
            }
            for task in tasks
        ],
    }


def load_pilot_state(path: str | Path) -> dict[str, Any]:
    state = read_json(path)
    if not isinstance(state, dict) or state.get("schema_version") != PILOT_SCHEMA_VERSION:
        raise RealModelPilotError(f"Pilot state {path} has an unsupported schema")
    if state.get("kind") != "dukeotr_real_model_pilot" or not isinstance(state.get("tasks"), list):
        raise RealModelPilotError(f"Pilot state {path} is not a DukeOTR real-model pilot state")
    return state


def write_pilot_state(path: str | Path, state: dict[str, Any]) -> None:
    """Atomically checkpoint mutable state and a separately named inspection manifest.

    The manifest is intentionally a snapshot rather than a second source of truth. Writing both
    before task dispatch and after every terminal/interrupted outcome makes a Windows resume
    auditable even if a console is closed mid-run.
    """

    destination = Path(path)
    state["updated_at"] = utc_now()
    write_json_atomic(destination, state)
    manifest = deepcopy(state)
    manifest["kind"] = "dukeotr_real_model_pilot_manifest"
    manifest["manifest_of"] = destination.name
    manifest["checkpointed_at"] = state["updated_at"]
    write_json_atomic(destination.with_name("pilot_manifest.json"), manifest)


def task_state(state: dict[str, Any], task_id: str) -> dict[str, Any]:
    for item in state.get("tasks", []):
        if isinstance(item, dict) and item.get("task_id") == task_id:
            return item
    raise RealModelPilotError(f"Pilot state has no task {task_id!r}")


def recover_interrupted_tasks(state: dict[str, Any]) -> list[str]:
    """Convert a persisted running marker into an explicit resumable interruption."""

    recovered: list[str] = []
    for item in state.get("tasks", []):
        if isinstance(item, dict) and item.get("status") == "running":
            item["status"] = "interrupted"
            item["completed_at"] = utc_now()
            item["error"] = "Previous process ended while this task was running; resume explicitly to retry it."
            attempts = item.get("attempt_history")
            if isinstance(attempts, list) and attempts and isinstance(attempts[-1], dict):
                attempts[-1].update(
                    {
                        "status": "interrupted",
                        "completed_at": item["completed_at"],
                        "error": item["error"],
                    }
                )
            recovered.append(str(item.get("task_id")))
    if recovered:
        state["status"] = "interrupted"
    return recovered


def runnable_task_ids(state: dict[str, Any], *, retry_failed: bool) -> set[str]:
    runnable = {"pending", "interrupted"}
    if retry_failed:
        runnable.add("generation_failed")
    return {
        str(item["task_id"])
        for item in state.get("tasks", [])
        if isinstance(item, dict) and item.get("status") in runnable and isinstance(item.get("task_id"), str)
    }


def _trace_role_models(trace: dict[str, Any]) -> dict[str, str]:
    configuration = trace.get("configuration")
    models = configuration.get("role_models") if isinstance(configuration, dict) else None
    return models if isinstance(models, dict) else {}


def _trace_role_options(trace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    configuration = trace.get("configuration")
    options = configuration.get("role_options") if isinstance(configuration, dict) else None
    if not isinstance(options, dict):
        return {}
    return {key: value for key, value in options.items() if isinstance(key, str) and isinstance(value, dict)}


def _response_evidence(role_trace: Any) -> dict[str, Any]:
    """Copy bounded role-response provenance already emitted by B/R/F, never invent it."""

    if not isinstance(role_trace, dict):
        return {"status": "not_run"}
    fields = (
        "status",
        "response_sha256",
        "response_characters",
        "elapsed_seconds",
        "response_truncated",
        "error",
    )
    result = {key: deepcopy(role_trace[key]) for key in fields if key in role_trace}
    # Error excerpts are intentionally retained in the original trace, which is linked by hash;
    # do not duplicate potentially large raw text into every derived candidate record.
    return result or {"status": "unknown"}


def _review_quality(round_trace: dict[str, Any], *, checked_at: str) -> dict[str, Any]:
    reviewer = round_trace.get("reviewer")
    if not isinstance(reviewer, dict):
        return {"status": "not_run", "decision": None, "review": None, "checked_at": checked_at}
    status = reviewer.get("status")
    if status == "complete":
        review = {
            key: deepcopy(reviewer.get(key))
            for key in (
                "reported_decisions",
                "decision",
                "dimension_scores",
                "findings",
                "api_claims_to_verify",
                "summary",
                "policy_forced_reasons",
                "passes",
            )
            if key in reviewer
        }
        return {
            "status": "complete",
            "decision": reviewer.get("decision"),
            "review": review,
            "checked_at": checked_at,
            "reviewer": {
                "kind": "ollama",
                "model": None,
                "response_contract": "builder_reviewer_fixer_json_schema",
                "completed_passes": reviewer.get("completed_passes"),
                "requested_passes": reviewer.get("requested_passes"),
            },
        }
    if status == "error":
        return {
            "status": "error",
            "decision": None,
            "review": None,
            "checked_at": checked_at,
            "error": reviewer.get("error", "Reviewer did not complete"),
            "reviewer": {"kind": "ollama", "model": None, "response_contract": "builder_reviewer_fixer_json_schema"},
        }
    return {
        "status": "skipped",
        "decision": None,
        "review": None,
        "checked_at": checked_at,
        "reason": reviewer.get("reason", "Reviewer was not run for this task-appropriate route"),
    }


def _decorate_candidate(
    record: dict[str, Any],
    *,
    trace: dict[str, Any],
    trace_path: str | Path,
    task_set_sha256: str,
    run_id: str,
    model_provenance: dict[str, Any],
    pipeline_version: dict[str, Any],
    stage: str,
    attempt: int,
) -> None:
    metadata = record.setdefault("metadata", {})
    # Keep the canonical stage vocabulary so existing correction/schema/Factory lineage
    # contracts continue to apply; put pilot-specific state in its own namespaced metadata.
    metadata["stage"] = "corrected" if stage == "real_model_pilot_corrected" else "generated"
    metadata["pilot"] = {
        "candidate_state": stage,
        "attempt": attempt,
        "run_id": run_id,
        "task_id": record.get("source_seed_id"),
        "task_set_sha256": task_set_sha256,
        "trace_path": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "trace_status": trace.get("status"),
        "trace_created_at": trace.get("created_at"),
        "trace_completed_at": trace.get("completed_at"),
        "pipeline_version": deepcopy(pipeline_version),
        "model_provenance": deepcopy(model_provenance),
    }


def _apply_trace_quality(record: dict[str, Any], round_trace: dict[str, Any], trace: dict[str, Any]) -> int:
    """Project factual trace evidence into the canonical quality fields without upgrading it."""

    quality = record.setdefault("quality", {})
    static = round_trace.get("static_validation")
    quality["static"] = deepcopy(static) if isinstance(static, dict) else {
        "status": "not_run",
        "issues": [],
        "checker": STATIC_CHECKER_VERSION,
        "checked_at": None,
    }
    checked_at = str(trace.get("completed_at") or trace.get("created_at") or utc_now())
    review = _review_quality(round_trace, checked_at=checked_at)
    role_models = _trace_role_models(trace)
    if isinstance(review.get("reviewer"), dict):
        review["reviewer"]["model"] = role_models.get("reviewer")
    quality["llm_review"] = review
    pilot = record.setdefault("metadata", {}).setdefault("pilot", {})
    pilot["tester_evidence"] = deepcopy(round_trace.get("tester", {"status": "not_run"}))
    pilot["reviewer_response_evidence"] = _response_evidence(round_trace.get("reviewer"))
    isolation = trace.get("evaluation_isolation")
    if isinstance(isolation, dict):
        quality["split_isolation"] = {
            "status": isolation.get("status", "not_run"),
            "collisions": deepcopy(isolation.get("collisions", [])),
            "threshold": isolation.get("threshold"),
            "checked_at": checked_at,
        }
    findings = review.get("review", {}).get("findings", []) if isinstance(review.get("review"), dict) else []
    return len(findings) if isinstance(findings, list) else 0


def materialize_trace_candidates(
    *,
    seed: dict[str, Any],
    trace: dict[str, Any],
    trace_path: str | Path,
    run_id: str,
    task_set_sha256: str,
    model_provenance: dict[str, Any],
    pipeline_version: dict[str, Any],
    attempt: int = 1,
) -> PilotMaterialization:
    """Reconstruct canonical raw/corrected records only from actual persisted trace output.

    This adapter never invents a response: every assistant string is read from a completed
    Builder or Fixer trace field, and a final-answer mismatch is a hard error.
    """

    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        raise RealModelPilotError("Pilot trace attempt must be a positive integer")
    rounds = trace.get("rounds")
    if not isinstance(rounds, list):
        raise RealModelPilotError("B/R/F trace has no rounds list")
    role_models = _trace_role_models(trace)
    role_options = _trace_role_options(trace)
    raw_records: list[dict[str, Any]] = []
    corrected_records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    reviewer_finding_count = 0
    fixer_count = 0

    for index, round_trace in enumerate(rounds, start=1):
        if not isinstance(round_trace, dict):
            continue
        if index == 1:
            builder = round_trace.get("builder")
            answer = builder.get("assistant_response") if isinstance(builder, dict) else None
            if not isinstance(answer, str) or not answer.strip():
                # A malformed/transport-failed Builder has no safely materializable model answer.
                continue
            current = make_generated_record(
                seed,
                answer,
                generator={
                    "kind": "builder_reviewer_fixer",
                    "role": "builder",
                    "model": role_models.get("builder"),
                    "options": deepcopy(role_options.get("builder", {})),
                    "quality_loop_round": index,
                    "pilot_run_id": run_id,
                    "pilot_attempt": attempt,
                },
                # Separate retries are actual distinct local sampling attempts. Retain both
                # canonical identities; the existing dedupe sidecar later marks same-answer
                # retries rather than silently discarding their provenance.
                variant=(attempt * 1000) + index,
                coverage=[],
                generation_notes=["Actual Builder response captured by a non-promoting real-model pilot."],
            )
            _decorate_candidate(
                current,
                trace=trace,
                trace_path=trace_path,
                task_set_sha256=task_set_sha256,
                run_id=run_id,
                model_provenance=model_provenance,
                pipeline_version=pipeline_version,
                stage="real_model_pilot_raw_builder",
                attempt=attempt,
            )
            current["metadata"]["pilot"]["builder_response_evidence"] = _response_evidence(builder)
            raw_records.append(current)
        if current is None:
            continue
        reviewer_finding_count += _apply_trace_quality(current, round_trace, trace)

        fixer = round_trace.get("fixer")
        corrected_answer = fixer.get("assistant_response") if isinstance(fixer, dict) else None
        if not isinstance(corrected_answer, str) or not corrected_answer.strip():
            continue
        correction = {
            "role": "fixer",
            "model": role_models.get("fixer"),
            "options": deepcopy(role_options.get("fixer", {})),
            "trace_round": index,
            "changes_made": deepcopy(fixer.get("changes_made", [])),
            "remaining_assumptions": deepcopy(fixer.get("remaining_assumptions", [])),
            "unresolved_risks": deepcopy(fixer.get("unresolved_risks", [])),
            "addressed_finding_ids": deepcopy(fixer.get("addressed_finding_ids", [])),
            "source_quality_snapshot": deepcopy(current.get("quality", {})),
            "response_evidence": _response_evidence(fixer),
            "source_trace": str(trace_path),
        }
        current = clone_for_correction(current, corrected_answer, correction=correction)
        _decorate_candidate(
            current,
            trace=trace,
            trace_path=trace_path,
            task_set_sha256=task_set_sha256,
            run_id=run_id,
            model_provenance=model_provenance,
            pipeline_version=pipeline_version,
            stage="real_model_pilot_corrected",
            attempt=attempt,
        )
        current["metadata"]["pilot"]["fixer_response_evidence"] = _response_evidence(fixer)
        correction_evidence = correction_explanation_evidence(current)
        current["metadata"]["pilot"]["correction_evidence"] = correction_evidence
        if correction_evidence["status"] != "linkable":
            # Keep the model's actual correction untouched, but make the absence of a real
            # Fixer explanation a visible quality hold. This prevents an otherwise accepting
            # later reviewer from turning unsupported correction text into training-eligible
            # evidence through a sidecar shortcut.
            current.setdefault("quality", {})["correction_evidence"] = {
                "status": "missing_actual_explanation",
                "reason": correction_evidence["reason"],
                "checked_at": str(trace.get("completed_at") or trace.get("created_at") or utc_now()),
            }
        corrected_records.append(current)
        fixer_count += 1

    final_records = [current] if current is not None else []
    trace_final = trace.get("final_candidate")
    if isinstance(trace_final, dict) and final_records:
        expected = text_from_message(trace_final, "assistant")
        actual = text_from_message(final_records[0], "assistant")
        if expected and expected != actual:
            raise RealModelPilotError(
                f"Trace/candidate answer mismatch for {seed.get('id')!r}; refusing to materialize altered pilot output"
            )
    return PilotMaterialization(
        raw_records=tuple(raw_records),
        corrected_records=tuple(corrected_records),
        final_records=tuple(final_records),
        reviewer_finding_count=reviewer_finding_count,
        fixer_count=fixer_count,
    )


def merge_materialized_records(
    materializations: Iterable[PilotMaterialization],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge per-task records and reject accidental duplicate canonical identities."""

    raw: list[dict[str, Any]] = []
    corrected: list[dict[str, Any]] = []
    finals: list[dict[str, Any]] = []
    seen: set[str] = set()
    for materialized in materializations:
        for destination, records in ((raw, materialized.raw_records), (corrected, materialized.corrected_records), (finals, materialized.final_records)):
            for record in records:
                record_id = record.get("record_id")
                if not isinstance(record_id, str) or not record_id:
                    raise RealModelPilotError("Materialized pilot candidate has no record_id")
                key = f"{id(destination)}:{record_id}"
                if key in seen:
                    raise RealModelPilotError(f"Duplicate materialized record id: {record_id}")
                seen.add(key)
                destination.append(record)
    return raw, corrected, finals


def quality_eligible_records(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Expose final-gate status without building a final training dataset."""

    eligible: list[dict[str, Any]] = []
    rejected: dict[str, list[str]] = {}
    for record in records:
        accepted, reasons = quality_gate_status(record)
        if accepted:
            eligible.append(record)
        else:
            rejected[str(record.get("record_id"))] = reasons
    return eligible, rejected


def summarize_state(state: dict[str, Any]) -> dict[str, int]:
    statuses = Counter(
        str(item.get("status", "unknown")) for item in state.get("tasks", []) if isinstance(item, dict)
    )
    return dict(sorted(statuses.items()))


def pilot_fingerprint(state: dict[str, Any]) -> str:
    """Stable state identity excluding timestamps/observed output content."""

    stable = {
        "run_id": state.get("run_id"),
        "model": state.get("model", {}).get("requested_tag") if isinstance(state.get("model"), dict) else None,
        "task_set": state.get("task_set"),
        "pipeline_version": state.get("pipeline_version"),
    }
    return sha256_text(canonical_json(stable))
