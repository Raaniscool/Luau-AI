"""Evidence-first support for the DukeOTR Training Factory.

This module intentionally sits beside, rather than replaces, the canonical candidate
pipeline.  It has no model invocation code.  It records what the Builder → Reviewer/Tester
→ Fixer pipeline actually observed, keeps failures in a separate sidecar database, and makes
review-required source briefs from measured failure categories only.

The helpers are dependency-free so they can run in the constrained data-curation environment.
They are gates and audit aids, not an oracle of Roblox/Luau factual correctness.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import itertools
from pathlib import Path
from typing import Any, Iterable

from scripts.lib.dedupe import cross_split_prompt_collisions, similarity
from scripts.lib.io_utils import canonical_json, sha256_file, sha256_text, text_from_message, utc_now
from scripts.lib.schema import (
    STATIC_CHECKER_VERSION,
    VALID_DIFFICULTIES,
    VALID_TASK_TYPES,
    quality_gate_status,
    record_fingerprint,
)

FACTORY_SCHEMA_VERSION = "1.0"
VALID_TASK_DEPTHS = {"short", "normal", "deep"}
VALID_FAILURE_SEVERITIES = {"minor", "moderate", "major", "critical"}
VALID_DRAFT_STATUSES = {"draft_requires_human_review", "rejected", "promoted"}
VALID_VERIFICATION_STATUSES = {
    "no_verified_correction",
    "rejected_without_verified_correction",
    "correction_present_revalidation_pending_or_failed",
    "corrected_and_revalidated",
}
SEVERITY_RANK = {"minor": 1, "moderate": 2, "major": 3, "critical": 4}
VALID_VERSION_STATUSES = {
    "external_baseline_only",
    "planned_not_built",
    "planned_not_trained",
    "trained_pending_evaluation",
    "evaluated_not_released",
    "released",
    "reserved_not_created",
}


class TrainingFactoryError(ValueError):
    """Raised when a factory artifact is malformed or crosses an isolation boundary."""


def factory_issue(code: str, message: str, severity: str = "error") -> dict[str, str]:
    return {"code": code, "message": message, "severity": severity}


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _string_list(value: Any, *, non_empty: bool = True) -> bool:
    return isinstance(value, list) and (bool(value) or not non_empty) and all(_non_empty_string(item) for item in value)


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("metadata")
    return value if isinstance(value, dict) else {}


def _quality(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("quality")
    return value if isinstance(value, dict) else {}


def candidate_is_evaluation_like(record: dict[str, Any]) -> bool:
    """Use only identifiers/provenance to reject an evaluation candidate without exposing tasks."""
    source_id = str(record.get("source_seed_id", "")).lower()
    record_id = str(record.get("record_id", "")).lower()
    source = _metadata(record).get("source")
    source_text = canonical_json(source).lower() if isinstance(source, dict) else ""
    return source_id.startswith("eval-") or record_id.startswith("eval-") or "evaluation_data" in source_text


def assert_safe_factory_output_path(
    output_path: str | Path,
    *,
    evaluation_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> None:
    """Reject writes into held-out evaluation files/directories before any artifact is created."""
    destination = Path(output_path).resolve()
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
    # Factory artifacts must remain physically distinct from held-out tasks, final training
    # data, and curated reference layers. Source-brief promotion has its own explicit raw-data
    # path, but it is never allowed to overwrite either active catalog below.
    protected_roots = [
        (root / "evaluation_data").resolve(),
        (root / "training_data").resolve(),
        (root / "knowledge_base").resolve(),
        (root / "code_book").resolve(),
        (root / "verified_knowledge").resolve(),
    ]
    protected_files = {
        (root / "raw_data" / "dukeotr_phase1_luau_seed_tasks.jsonl").resolve(),
        (root / "raw_data" / "roblox_luau_seed_tasks.jsonl").resolve(),
    }
    if evaluation_path is not None:
        protected_roots.append(Path(evaluation_path).resolve())
    if destination in protected_files:
        raise TrainingFactoryError(f"Refusing to overwrite an active source catalog with Training Factory output: {destination}")
    for protected in protected_roots:
        try:
            destination.relative_to(protected)
        except ValueError:
            continue
        raise TrainingFactoryError(f"Refusing to write Training Factory output into protected path: {destination}")


def _taxonomy_categories(taxonomy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    categories = taxonomy.get("categories")
    if not isinstance(categories, list):
        raise TrainingFactoryError("Failure taxonomy requires a categories list")
    output: dict[str, dict[str, Any]] = {}
    for entry in categories:
        if not isinstance(entry, dict) or not _non_empty_string(entry.get("id")):
            raise TrainingFactoryError("Every failure taxonomy category needs a non-empty id")
        identifier = entry["id"]
        if identifier in output:
            raise TrainingFactoryError(f"Duplicate failure taxonomy category: {identifier}")
        if entry.get("default_severity") not in VALID_FAILURE_SEVERITIES:
            raise TrainingFactoryError(f"Failure taxonomy category {identifier!r} has invalid default severity")
        output[identifier] = entry
    return output


def validate_failure_taxonomy(taxonomy: dict[str, Any]) -> list[dict[str, str]]:
    """Validate taxonomy shape without requiring a JSON Schema dependency."""
    problems: list[dict[str, str]] = []
    if taxonomy.get("schema_version") != FACTORY_SCHEMA_VERSION:
        problems.append(factory_issue("taxonomy.schema_version", f"Expected {FACTORY_SCHEMA_VERSION!r}"))
    try:
        categories = _taxonomy_categories(taxonomy)
    except TrainingFactoryError as exc:
        problems.append(factory_issue("taxonomy.categories", str(exc)))
        return problems
    rules = taxonomy.get("rules")
    if not isinstance(rules, list):
        problems.append(factory_issue("taxonomy.rules", "Failure taxonomy requires a rules list"))
        return problems
    for index, rule in enumerate(rules, start=1):
        if not isinstance(rule, dict) or rule.get("category") not in categories:
            problems.append(factory_issue("taxonomy.rule_category", f"Rule {index} refers to an unknown category"))
            continue
        for key in ("static_code_prefixes", "keywords"):
            if not _string_list(rule.get(key, []), non_empty=False):
                problems.append(factory_issue("taxonomy.rule_list", f"Rule {index} {key!r} must be a list of strings"))
    return problems


def validate_factory_draft(
    draft: dict[str, Any], *, allowed_categories: set[str] | None = None
) -> list[dict[str, str]]:
    """Validate a review-required targeted source brief, never an answer or final candidate."""
    problems: list[dict[str, str]] = []
    if draft.get("schema_version") != FACTORY_SCHEMA_VERSION:
        problems.append(factory_issue("draft.schema_version", f"Expected schema_version {FACTORY_SCHEMA_VERSION!r}"))
    for key in ("draft_id", "title", "category", "user_request", "created_at"):
        if not _non_empty_string(draft.get(key)):
            problems.append(factory_issue("draft.required", f"Draft requires a non-empty string {key!r}"))
    if allowed_categories is not None and draft.get("category") not in allowed_categories:
        problems.append(factory_issue("draft.category", "Draft category is absent from the controlled Training Factory category configuration"))
    if draft.get("status") not in VALID_DRAFT_STATUSES:
        problems.append(factory_issue("draft.status", "Draft status must be review-required, rejected, or promoted"))
    if draft.get("status") == "draft_requires_human_review" and draft.get("split") != "train":
        problems.append(factory_issue("draft.split", "Review-required drafts must identify train as their intended eventual split"))
    if draft.get("task_type") not in VALID_TASK_TYPES:
        problems.append(factory_issue("draft.task_type", "Draft task_type is missing or unknown"))
    if draft.get("difficulty") not in VALID_DIFFICULTIES:
        problems.append(factory_issue("draft.difficulty", "Draft difficulty is missing or unknown"))
    if draft.get("task_depth") not in VALID_TASK_DEPTHS:
        problems.append(factory_issue("draft.task_depth", "Draft task_depth must be short, normal, or deep"))
    for key in ("requirements", "concepts", "expected_evidence", "avoid"):
        if not _string_list(draft.get(key)):
            problems.append(factory_issue("draft.list", f"Draft {key!r} must be a non-empty list of strings"))
    if not _non_empty_string(draft.get("difficulty_rationale")):
        problems.append(factory_issue("draft.difficulty_rationale", "Draft needs a difficulty rationale tied to construction"))
    if not _non_empty_string(draft.get("task_depth_rationale")):
        problems.append(factory_issue("draft.task_depth_rationale", "Draft needs a task-depth rationale tied to construction"))
    source = draft.get("source")
    if not isinstance(source, dict) or source.get("kind") not in {
        "training_factory_targeted_draft",
        "project_authored_targeted_brief",
    }:
        problems.append(factory_issue("draft.provenance", "Draft must retain targeted-draft or reviewed targeted-brief provenance"))
    targeting = draft.get("targeting")
    if not isinstance(targeting, dict):
        problems.append(factory_issue("draft.targeting", "Draft requires targeting object"))
    else:
        if not _non_empty_string(targeting.get("failure_category")):
            problems.append(factory_issue("draft.failure_category", "Targeting requires a failure_category"))
        if not isinstance(targeting.get("observed_failure_count"), int) or targeting["observed_failure_count"] < 1:
            problems.append(factory_issue("draft.failure_count", "Targeting requires a positive observed_failure_count"))
        if targeting.get("requires_human_approval") is not True:
            problems.append(factory_issue("draft.approval", "Targeted briefs must require human approval"))
    if candidate_is_evaluation_like(
        {
            "record_id": draft.get("draft_id"),
            "source_seed_id": draft.get("draft_id"),
            "metadata": {"source": source},
        }
    ):
        problems.append(factory_issue("draft.evaluation_isolation", "Draft appears to have evaluation provenance"))
    return problems


def _validate_task_context(task: Any, *, prefix: str, allow_unspecified_depth: bool) -> list[dict[str, str]]:
    """Validate task labels carried into failure/lineage sidecars without inventing legacy depth."""
    problems: list[dict[str, str]] = []
    if not isinstance(task, dict) or not _non_empty_string(task.get("prompt")):
        return [factory_issue(f"{prefix}.task", f"{prefix.title()} needs task.prompt")]
    if task.get("task_type") not in VALID_TASK_TYPES:
        problems.append(factory_issue(f"{prefix}.task_type", f"{prefix.title()} task_type is missing or unknown"))
    if task.get("difficulty") not in VALID_DIFFICULTIES:
        problems.append(factory_issue(f"{prefix}.difficulty", f"{prefix.title()} difficulty is missing or unknown"))
    if not _non_empty_string(task.get("category")):
        problems.append(factory_issue(f"{prefix}.category", f"{prefix.title()} needs explicit category or 'unclassified'"))
    allowed_depths = {*VALID_TASK_DEPTHS, "unspecified"} if allow_unspecified_depth else VALID_TASK_DEPTHS
    if task.get("task_depth") not in allowed_depths:
        problems.append(factory_issue(f"{prefix}.task_depth", f"{prefix.title()} task_depth is invalid"))
    for key in ("requirements", "topics"):
        value = task.get(key, [])
        if not _string_list(value, non_empty=False):
            problems.append(factory_issue(f"{prefix}.{key}", f"{prefix.title()} {key} must be a list of strings"))
    return problems


def _non_empty_explanation(value: Any) -> bool:
    return _non_empty_string(value) or (_string_list(value, non_empty=True))


def validate_failure_record(record: dict[str, Any], *, taxonomy: dict[str, Any]) -> list[dict[str, str]]:
    """Validate an observed failure record while preserving unknown agent/model identity honestly."""
    problems: list[dict[str, str]] = []
    categories = _taxonomy_categories(taxonomy)
    if record.get("schema_version") != FACTORY_SCHEMA_VERSION:
        problems.append(factory_issue("failure.schema_version", f"Expected schema_version {FACTORY_SCHEMA_VERSION!r}"))
    for key in ("failure_id", "record_id", "source_seed_id", "answer", "created_at"):
        if not _non_empty_string(record.get(key)):
            problems.append(factory_issue("failure.required", f"Failure record requires non-empty {key!r}"))
    problems.extend(_validate_task_context(record.get("task"), prefix="failure", allow_unspecified_depth=True))
    producing_agent = record.get("producing_agent")
    if not isinstance(producing_agent, dict) or not isinstance(producing_agent.get("known"), bool):
        problems.append(factory_issue("failure.producing_agent", "Failure record needs producing_agent.known boolean"))
    failure_categories = record.get("failure_categories")
    if not _string_list(failure_categories):
        problems.append(factory_issue("failure.categories", "Failure record requires one or more controlled categories"))
    elif any(category not in categories for category in failure_categories):
        problems.append(factory_issue("failure.categories", "Failure record contains a category absent from taxonomy"))
    if record.get("severity") not in VALID_FAILURE_SEVERITIES:
        problems.append(factory_issue("failure.severity", "Failure record severity is invalid"))
    review = record.get("review")
    if not isinstance(review, dict) or not _string_list(review.get("findings"), non_empty=False):
        problems.append(factory_issue("failure.review", "Failure record needs review.findings list"))
    correction = record.get("correction")
    if not isinstance(correction, dict) or "corrected_solution" not in correction:
        problems.append(factory_issue("failure.correction", "Failure record needs correction.corrected_solution (which may be null)"))
    elif correction.get("corrected_record_id") is not None:
        if not _non_empty_string(correction.get("corrected_record_id")) or not _non_empty_string(correction.get("corrected_solution")):
            problems.append(factory_issue("failure.correction_link", "Linked correction needs record ID and non-empty corrected solution"))
        if not _non_empty_explanation(correction.get("explanation")):
            problems.append(factory_issue("failure.correction_explanation", "Linked correction needs actual change explanation evidence"))
    verification = record.get("verification")
    if not isinstance(verification, dict) or verification.get("status") not in VALID_VERIFICATION_STATUSES:
        problems.append(factory_issue("failure.verification", "Failure record needs a controlled verification status"))
    elif verification.get("status") == "corrected_and_revalidated" and not _non_empty_string(verification.get("revalidated_record_id")):
        problems.append(factory_issue("failure.verification_link", "Revalidated correction status requires corrected record ID"))
    provenance = record.get("provenance")
    if not isinstance(provenance, dict) or not _non_empty_string(provenance.get("candidate_artifact")):
        problems.append(factory_issue("failure.provenance", "Failure record needs candidate-artifact provenance"))
    version = record.get("version")
    if not isinstance(version, dict) or version.get("factory_schema_version") != FACTORY_SCHEMA_VERSION:
        problems.append(factory_issue("failure.version", "Failure record needs factory/candidate version metadata"))
    return problems


def validate_ledger_record(record: dict[str, Any], *, taxonomy: dict[str, Any]) -> list[dict[str, str]]:
    """Validate the one-row sidecar that relates Builder, reviewer, correction, and verification."""
    problems: list[dict[str, str]] = []
    categories = _taxonomy_categories(taxonomy)
    if record.get("schema_version") != FACTORY_SCHEMA_VERSION:
        problems.append(factory_issue("ledger.schema_version", f"Expected schema_version {FACTORY_SCHEMA_VERSION!r}"))
    for key in ("ledger_id", "record_id", "created_at"):
        if not _non_empty_string(record.get(key)):
            problems.append(factory_issue("ledger.required", f"Ledger requires non-empty {key!r}"))
    problems.extend(_validate_task_context(record.get("task"), prefix="ledger", allow_unspecified_depth=True))
    builder = record.get("builder")
    if not isinstance(builder, dict) or not _non_empty_string(builder.get("answer")):
        problems.append(factory_issue("ledger.builder", "Ledger needs Builder answer"))
    review = record.get("review")
    if not isinstance(review, dict) or "static" not in review or "llm_review" not in review or "human_review" not in review:
        problems.append(factory_issue("ledger.review", "Ledger needs static, LLM, and human review slots"))
    correction = record.get("correction")
    if not isinstance(correction, dict) or "corrected_output" not in correction or "explanation" not in correction:
        problems.append(factory_issue("ledger.correction", "Ledger needs corrected_output and correction explanation slots"))
    elif correction.get("corrected_record_id") is not None:
        if not _non_empty_string(correction.get("corrected_record_id")) or not _non_empty_string(correction.get("corrected_output")):
            problems.append(factory_issue("ledger.correction_link", "Linked correction needs record ID and non-empty corrected output"))
        if not _non_empty_explanation(correction.get("explanation")):
            problems.append(factory_issue("ledger.correction_explanation", "Linked correction needs actual explanation evidence"))
    analysis = record.get("failure_analysis")
    if not isinstance(analysis, dict) or not isinstance(analysis.get("observed_failure"), bool):
        problems.append(factory_issue("ledger.failure_analysis", "Ledger needs observed_failure boolean"))
    elif not _string_list(analysis.get("failure_categories", []), non_empty=False):
        problems.append(factory_issue("ledger.failure_categories", "Ledger failure categories must be strings"))
    elif any(category not in categories for category in analysis.get("failure_categories", [])):
        problems.append(factory_issue("ledger.failure_categories", "Ledger contains category absent from taxonomy"))
    elif analysis.get("observed_failure") and not analysis.get("failure_categories"):
        problems.append(factory_issue("ledger.failure_categories", "Observed ledger failure needs one or more controlled categories"))
    if isinstance(analysis, dict) and analysis.get("observed_failure") and analysis.get("severity") not in VALID_FAILURE_SEVERITIES:
        problems.append(factory_issue("ledger.failure_severity", "Observed ledger failure needs controlled severity"))
    verification = record.get("verification")
    if not isinstance(verification, dict) or verification.get("status") not in VALID_VERIFICATION_STATUSES:
        problems.append(factory_issue("ledger.verification", "Ledger needs controlled verification status"))
    elif verification.get("status") == "corrected_and_revalidated" and not _non_empty_string(verification.get("revalidated_record_id")):
        problems.append(factory_issue("ledger.verification_link", "Revalidated correction status requires corrected record ID"))
    quality = record.get("quality")
    if not isinstance(quality, dict):
        problems.append(factory_issue("ledger.quality", "Ledger needs quality object"))
    provenance = record.get("provenance")
    if not isinstance(provenance, dict) or not _non_empty_string(provenance.get("candidate_artifact")):
        problems.append(factory_issue("ledger.provenance", "Ledger needs candidate artifact provenance"))
    version = record.get("version")
    if not isinstance(version, dict) or version.get("factory_schema_version") != FACTORY_SCHEMA_VERSION:
        problems.append(factory_issue("ledger.version", "Ledger needs factory/candidate version metadata"))
    return problems


def _finding_text(record: dict[str, Any]) -> list[str]:
    quality = _quality(record)
    findings: list[str] = []
    static = quality.get("static")
    if isinstance(static, dict):
        for item in static.get("issues", []):
            if isinstance(item, dict):
                code = item.get("code")
                message = item.get("message")
                if _non_empty_string(code) or _non_empty_string(message):
                    findings.append(f"static {code or 'finding'}: {message or ''}".strip())
    for reviewer_key in ("llm_review", "human_review"):
        reviewer = quality.get(reviewer_key)
        if not isinstance(reviewer, dict):
            continue
        review = reviewer.get("review")
        if isinstance(review, dict):
            for key in ("blocking_issues", "required_fixes"):
                values = review.get(key, [])
                if isinstance(values, list):
                    findings.extend(f"{reviewer_key} {key}: {item}" for item in values if _non_empty_string(item))
            if _non_empty_string(review.get("summary")):
                findings.append(f"{reviewer_key} summary: {review['summary']}")
        if _non_empty_string(reviewer.get("error")):
            findings.append(f"{reviewer_key} operational error: {reviewer['error']}")
    dedupe = quality.get("deduplication")
    if isinstance(dedupe, dict) and dedupe.get("status") == "duplicate":
        findings.append(f"deduplication: duplicate of {dedupe.get('duplicate_of') or 'unknown record'}")
    isolation = quality.get("split_isolation")
    if isinstance(isolation, dict) and isolation.get("status") not in {None, "clear", "not_run"}:
        findings.append(f"split isolation: {isolation.get('status')}")
    return list(dict.fromkeys(findings))


def _static_codes(record: dict[str, Any]) -> list[str]:
    static = _quality(record).get("static")
    if not isinstance(static, dict):
        return []
    values: list[str] = []
    for item in static.get("issues", []):
        if isinstance(item, dict) and _non_empty_string(item.get("code")):
            values.append(item["code"])
    return values


def observed_failure(record: dict[str, Any]) -> bool:
    """Return true only for stored evidence of a candidate failure, not reviewer outages alone."""
    quality = _quality(record)
    static = quality.get("static")
    if isinstance(static, dict) and static.get("status") == "fail":
        return True
    for key in ("llm_review", "human_review"):
        reviewer = quality.get(key)
        if isinstance(reviewer, dict) and reviewer.get("decision") in {"revise", "reject"}:
            return True
    dedupe = quality.get("deduplication")
    if isinstance(dedupe, dict) and dedupe.get("status") == "duplicate":
        return True
    isolation = quality.get("split_isolation")
    return isinstance(isolation, dict) and isolation.get("status") in {"collision", "fail"}


def infer_failure_categories(record: dict[str, Any], *, taxonomy: dict[str, Any]) -> list[str]:
    """Map stored issue codes/text to taxonomy labels; never invent a measured count."""
    category_map = _taxonomy_categories(taxonomy)
    codes = _static_codes(record)
    combined_text = "\n".join(_finding_text(record)).lower()
    categories: list[str] = []
    existing = _quality(record).get("failure_analysis")
    if isinstance(existing, dict) and isinstance(existing.get("failure_categories"), list):
        for category in existing["failure_categories"]:
            if category in category_map:
                categories.append(category)
    for rule in taxonomy.get("rules", []):
        if not isinstance(rule, dict):
            continue
        category = rule.get("category")
        if category not in category_map:
            continue
        prefixes = rule.get("static_code_prefixes", [])
        keywords = rule.get("keywords", [])
        prefix_match = any(
            isinstance(prefix, str) and any(code.startswith(prefix) for code in codes) for prefix in prefixes
        )
        keyword_match = any(isinstance(keyword, str) and keyword.lower() in combined_text for keyword in keywords)
        if prefix_match or keyword_match:
            categories.append(category)
    quality = _quality(record)
    dedupe = quality.get("deduplication")
    isolation = quality.get("split_isolation")
    if (isinstance(dedupe, dict) and dedupe.get("status") == "duplicate") or (
        isinstance(isolation, dict) and isolation.get("status") in {"collision", "fail"}
    ):
        categories.append("duplicate_or_split_contamination")
    if observed_failure(record) and not categories:
        categories.append("other_observed_failure")
    return list(dict.fromkeys(categories))


def severity_for_failure(record: dict[str, Any], categories: Iterable[str], *, taxonomy: dict[str, Any]) -> str:
    """Use the highest actual evidence severity, with category defaults as conservative labels."""
    category_map = _taxonomy_categories(taxonomy)
    severity = "minor"
    for category in categories:
        entry = category_map.get(category)
        if entry and SEVERITY_RANK[entry["default_severity"]] > SEVERITY_RANK[severity]:
            severity = entry["default_severity"]
    static = _quality(record).get("static")
    if isinstance(static, dict):
        for finding in static.get("issues", []):
            if not isinstance(finding, dict):
                continue
            if finding.get("severity") in {"block", "error"}:
                code = str(finding.get("code", ""))
                candidate = "critical" if code.startswith(("network.", "security.")) else "major"
                if SEVERITY_RANK[candidate] > SEVERITY_RANK[severity]:
                    severity = candidate
    for key in ("llm_review", "human_review"):
        reviewer = _quality(record).get(key)
        if not isinstance(reviewer, dict):
            continue
        candidate = "major" if reviewer.get("decision") == "reject" else "moderate" if reviewer.get("decision") == "revise" else None
        if candidate and SEVERITY_RANK[candidate] > SEVERITY_RANK[severity]:
            severity = candidate
    return severity


def _agent_metadata(record: dict[str, Any]) -> dict[str, Any]:
    generator = _metadata(record).get("generator")
    if not isinstance(generator, dict):
        return {"known": False, "kind": "unknown", "model": None}
    kind = generator.get("kind") if _non_empty_string(generator.get("kind")) else "unknown"
    model = generator.get("model") if _non_empty_string(generator.get("model")) else None
    known = kind != "unknown" or model is not None
    output: dict[str, Any] = {"known": known, "kind": kind, "model": model}
    if "options" in generator:
        output["options"] = deepcopy(generator["options"])
    return output


def _reviewer_revalidated(record: dict[str, Any]) -> bool:
    quality = _quality(record)
    static = quality.get("static")
    static_ok = isinstance(static, dict) and static.get("status") == "pass" and static.get("checker") == STATIC_CHECKER_VERSION
    llm = quality.get("llm_review")
    human = quality.get("human_review")
    reviewer_ok = (isinstance(llm, dict) and llm.get("status") == "complete" and llm.get("decision") == "accept") or (
        isinstance(human, dict) and human.get("status") == "complete" and human.get("decision") == "accept"
    )
    return static_ok and reviewer_ok


def correction_verification(record: dict[str, Any], correction: dict[str, Any] | None) -> dict[str, Any]:
    """Describe correction state truthfully; a correction is not an accepted final example by itself."""
    if correction is None:
        review = _quality(record).get("llm_review")
        if isinstance(review, dict) and review.get("decision") == "reject":
            status = "rejected_without_verified_correction"
        else:
            status = "no_verified_correction"
        return {"status": status, "revalidated_record_id": None, "quality_gate_eligible": False}
    if _reviewer_revalidated(correction):
        eligible, reasons = quality_gate_status(correction)
        return {
            "status": "corrected_and_revalidated",
            "revalidated_record_id": correction.get("record_id"),
            "quality_gate_eligible": eligible,
            "quality_gate_reasons": reasons,
        }
    return {
        "status": "correction_present_revalidation_pending_or_failed",
        "revalidated_record_id": correction.get("record_id"),
        "quality_gate_eligible": False,
    }


def make_failure_record(
    record: dict[str, Any],
    *,
    taxonomy: dict[str, Any],
    candidate_artifact: str,
    correction: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Create a separate auditable failure row from a candidate with actual failure evidence."""
    if not observed_failure(record):
        raise TrainingFactoryError(f"Record {record.get('record_id')!r} has no stored failure evidence")
    categories = infer_failure_categories(record, taxonomy=taxonomy)
    findings = _finding_text(record)
    metadata = _metadata(record)
    correction_metadata = _metadata(correction).get("correction", {}) if isinstance(correction, dict) else {}
    correction_answer = text_from_message(correction, "assistant") if isinstance(correction, dict) else None
    verification = correction_verification(record, correction)
    stable_identity = canonical_json(
        {
            "record_id": record.get("record_id"),
            "categories": categories,
            "answer": text_from_message(record, "assistant"),
        }
    )
    failure_id = f"failure-{str(record.get('record_id', 'unknown'))}-{sha256_text(stable_identity)[:12]}"
    return {
        "schema_version": FACTORY_SCHEMA_VERSION,
        "failure_id": failure_id,
        "record_id": record.get("record_id"),
        "source_seed_id": record.get("source_seed_id"),
        "task": {
            "prompt": text_from_message(record, "user"),
            "task_type": metadata.get("task_type"),
            "category": metadata.get("category", "unclassified"),
            "difficulty": metadata.get("difficulty"),
            "task_depth": metadata.get("task_depth", "unspecified"),
            "requirements": deepcopy(metadata.get("requirements", [])),
            "topics": deepcopy(metadata.get("topics", [])),
        },
        "producing_agent": _agent_metadata(record),
        "answer": text_from_message(record, "assistant"),
        "failure_categories": categories,
        "severity": severity_for_failure(record, categories, taxonomy=taxonomy),
        "review": {
            "findings": findings,
            "static": deepcopy(_quality(record).get("static", {})),
            "llm_review": deepcopy(_quality(record).get("llm_review", {})),
            "human_review": deepcopy(_quality(record).get("human_review", {})),
        },
        "correction": {
            "corrected_record_id": correction.get("record_id") if isinstance(correction, dict) else None,
            "corrected_solution": correction_answer,
            "explanation": deepcopy(correction_metadata.get("changes_made", correction_metadata.get("reason", []))),
            "metadata": deepcopy(correction_metadata) if isinstance(correction_metadata, dict) else {},
        },
        "verification": verification,
        "version": {
            "factory_schema_version": FACTORY_SCHEMA_VERSION,
            "candidate_schema_version": record.get("schema_version"),
            "static_checker_version": _quality(record).get("static", {}).get("checker") if isinstance(_quality(record).get("static"), dict) else None,
            "dataset_version": metadata.get("dataset_version"),
            "planned_dataset_version": "dukeotr_dataset_v1",
        },
        "provenance": {
            "candidate_artifact": str(candidate_artifact),
            "candidate_fingerprint": record_fingerprint(record),
            "candidate_stage": metadata.get("stage", "unknown"),
            "candidate_created_at": metadata.get("created_at"),
            "source": deepcopy(metadata.get("source", {})),
        },
        "created_at": created_at or utc_now(),
    }


def _categorical_counter(records: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        value = record.get(key)
        if _non_empty_string(value):
            counter[value] += 1
    return dict(sorted(counter.items()))


def summarize_failures(records: Iterable[dict[str, Any]], *, taxonomy: dict[str, Any]) -> dict[str, Any]:
    """Calculate, rather than infer, weakness counts from valid stored failure records."""
    category_map = _taxonomy_categories(taxonomy)
    rows = list(records)
    category_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    target_profile_counts: Counter[str] = Counter()
    task_category_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    task_depth_counts: Counter[str] = Counter()
    verification_counts: Counter[str] = Counter()
    for row in rows:
        for category in row.get("failure_categories", []):
            if category in category_map:
                category_counts[category] += 1
                target_profile = category_map[category].get("target_profile")
                if _non_empty_string(target_profile):
                    target_profile_counts[target_profile] += 1
        severity = row.get("severity")
        if severity in VALID_FAILURE_SEVERITIES:
            severity_counts[severity] += 1
        task = row.get("task")
        if isinstance(task, dict):
            if _non_empty_string(task.get("category")):
                task_category_counts[task["category"]] += 1
            if task.get("difficulty") in VALID_DIFFICULTIES:
                difficulty_counts[task["difficulty"]] += 1
            if _non_empty_string(task.get("task_depth")):
                task_depth_counts[task["task_depth"]] += 1
        verification = row.get("verification")
        if isinstance(verification, dict) and _non_empty_string(verification.get("status")):
            verification_counts[verification["status"]] += 1
    counts = dict(sorted(category_counts.items(), key=lambda item: (-item[1], item[0])))
    return {
        "schema_version": FACTORY_SCHEMA_VERSION,
        "kind": "observed_failure_summary",
        "failure_record_count": len(rows),
        "category_counts": counts,
        "severity_counts": dict(sorted(severity_counts.items())),
        "target_profile_counts": dict(sorted(target_profile_counts.items(), key=lambda item: (-item[1], item[0]))),
        "task_category_counts": dict(sorted(task_category_counts.items(), key=lambda item: (-item[1], item[0]))),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "task_depth_counts": dict(sorted(task_depth_counts.items())),
        "verification_counts": dict(sorted(verification_counts.items())),
        "note": "Counts are derived only from stored, validated failure records. Zero or absent categories are not estimates of model weakness.",
    }


def recommend_targeting(
    summary: dict[str, Any],
    *,
    taxonomy: dict[str, Any],
    minimum_failure_count: int = 1,
) -> list[dict[str, Any]]:
    """Return only evidence-backed recommendations; no record means no recommendation."""
    category_map = _taxonomy_categories(taxonomy)
    counts = summary.get("category_counts")
    if not isinstance(counts, dict):
        raise TrainingFactoryError("Failure summary requires category_counts")
    recommendations: list[dict[str, Any]] = []
    for category, count in counts.items():
        if category not in category_map or not isinstance(count, int) or count < minimum_failure_count:
            continue
        entry = category_map[category]
        recommendations.append(
            {
                "failure_category": category,
                "observed_failure_count": count,
                "severity_default": entry["default_severity"],
                "target_profile": entry.get("target_profile"),
                "basis": "stored_failure_records_only",
            }
        )
    return sorted(recommendations, key=lambda item: (-item["observed_failure_count"], item["failure_category"]))


def _depth_for_difficulty(difficulty: str) -> str:
    return {"beginner": "short", "intermediate": "normal", "advanced": "deep"}[difficulty]


def _depth_rationale(depth: str) -> str:
    return {
        "short": "The brief constrains the task to one bounded scenario, one primary failure mode, and a small evidence set.",
        "normal": "The brief requires linked requirements, an edge case, and an explanation of the selected engineering approach.",
        "deep": "The brief requires multiple interacting constraints, explicit assumptions, edge cases, and a tradeoff or verification discussion.",
    }[depth]


def _difficulty_rationale(difficulty: str, profile: dict[str, Any]) -> str:
    requirements = profile.get("difficulty_requirements", {}).get(difficulty, [])
    if not isinstance(requirements, list) or not requirements:
        raise TrainingFactoryError(f"Target profile {profile.get('id')!r} has no requirements for {difficulty!r}")
    return f"{difficulty.title()} construction requires: " + " ".join(str(item) for item in requirements)


def _profile_combinations(profile: dict[str, Any], *, seed: int) -> list[tuple[str, str, str]]:
    forms = profile.get("task_forms")
    scenarios = profile.get("scenarios")
    failure_modes = profile.get("failure_modes")
    if not _string_list(forms) or not _string_list(scenarios) or not _string_list(failure_modes):
        raise TrainingFactoryError(f"Target profile {profile.get('id')!r} needs non-empty task_forms, scenarios, and failure_modes")
    combinations = list(itertools.product(forms, scenarios, failure_modes))
    # The hash ordering is deterministic, seed-sensitive, and avoids a fixed first template bias.
    return sorted(
        combinations,
        key=lambda value: sha256_text(f"{seed}|{profile.get('id')}|{'|'.join(value)}"),
    )


def build_targeted_drafts(
    *,
    profile: dict[str, Any],
    failure_category: str,
    observed_failure_count: int,
    counts_by_difficulty: dict[str, int],
    failure_input_sha256: str,
    seed: int,
    created_at: str,
) -> list[dict[str, Any]]:
    """Deterministically make review-required briefs from one measured category.

    This intentionally generates no answer, no model output, and no final candidate.  A human
    must review and separately promote selected rows before the existing seed/candidate path can
    use them.
    """
    if observed_failure_count < 1:
        raise TrainingFactoryError("Targeted briefs require an observed failure count of at least one")
    if failure_category not in profile.get("failure_categories", []):
        raise TrainingFactoryError(f"Profile {profile.get('id')!r} does not support {failure_category!r}")
    combinations = _profile_combinations(profile, seed=seed)
    base_requirements = profile.get("base_requirements", [])
    expected_evidence = profile.get("expected_evidence", [])
    avoid = profile.get("avoid", [])
    concepts = profile.get("concepts", [])
    instructions = profile.get("form_instructions", {})
    if not all(_string_list(value) for value in (base_requirements, expected_evidence, avoid, concepts)):
        raise TrainingFactoryError(f"Target profile {profile.get('id')!r} has invalid list content")
    if not isinstance(instructions, dict):
        raise TrainingFactoryError(f"Target profile {profile.get('id')!r} needs form_instructions")
    drafts: list[dict[str, Any]] = []
    used_combinations: set[tuple[str, str, str]] = set()
    for difficulty in ("beginner", "intermediate", "advanced"):
        count = counts_by_difficulty.get(difficulty, 0)
        if not isinstance(count, int) or count < 0:
            raise TrainingFactoryError(f"Target count for {difficulty!r} must be a non-negative integer")
        if count == 0:
            continue
        # Reorder for each difficulty, then consume combinations globally. This gives the
        # depth labels distinct task construction rather than repeating the same prompt three
        # times with only a learner-level phrase changed.
        difficulty_order = sorted(
            combinations,
            key=lambda value: sha256_text(f"{seed}|{difficulty}|{profile.get('id')}|{'|'.join(value)}"),
        )
        available = [value for value in difficulty_order if value not in used_combinations]
        if count > len(available):
            raise TrainingFactoryError(
                f"Requested {count} {difficulty} drafts from profile {profile.get('id')!r}, but only {len(available)} distinct combinations remain"
            )
        selected: list[tuple[tuple[str, str, str], str]] = []
        for combination in available:
            task_type, scenario, failure_mode = combination
            instruction = instructions.get(task_type)
            if not _non_empty_string(instruction):
                raise TrainingFactoryError(f"Target profile {profile.get('id')!r} has no instruction for {task_type!r}")
            depth = _depth_for_difficulty(difficulty)
            request = (
                f"{instruction}\n\n"
                f"Scenario: {scenario}.\n"
                f"Observed failure theme to address: {failure_mode}.\n\n"
                f"Respond at a {difficulty} level. Keep the scope suitable for a {depth} task while satisfying every listed requirement. "
                "State assumptions where needed and distinguish verified behavior from assumptions."
            )
            # No superficial wording variants: an accepted prompt must remain below the
            # transparent near-duplicate threshold against every draft already selected.
            if all(similarity(request, prior["user_request"]).jaccard < 0.82 for prior in drafts) and all(
                similarity(request, selected_request).jaccard < 0.82 for _selected_combination, selected_request in selected
            ):
                selected.append((combination, request))
            if len(selected) == count:
                break
        if len(selected) != count:
            raise TrainingFactoryError(
                f"Requested {count} {difficulty} drafts from profile {profile.get('id')!r}, but only {len(selected)} could meet the no-near-duplicate gate"
            )
        used_combinations.update(combination for combination, _request in selected)
        for index, ((task_type, scenario, failure_mode), request) in enumerate(selected, start=1):
            suffix = sha256_text(
                canonical_json(
                    {
                        "profile": profile.get("id"),
                        "category": failure_category,
                        "difficulty": difficulty,
                        "task_type": task_type,
                        "scenario": scenario,
                        "failure_mode": failure_mode,
                        "seed": seed,
                    }
                )
            )[:10]
            draft_id = f"draft-{profile.get('id')}-{difficulty}-{index:03d}-{suffix}"
            difficulty_requirements = profile["difficulty_requirements"][difficulty]
            draft = {
                "schema_version": FACTORY_SCHEMA_VERSION,
                "draft_id": draft_id,
                "status": "draft_requires_human_review",
                "split": "train",
                "title": f"{difficulty.title()} {task_type.replace('_', ' ')}: {scenario}",
                "task_type": task_type,
                "difficulty": difficulty,
                "task_depth": depth,
                "category": profile.get("category"),
                "user_request": request,
                "requirements": [*base_requirements, *difficulty_requirements],
                "concepts": list(concepts),
                "expected_evidence": list(expected_evidence),
                "avoid": list(avoid),
                "difficulty_rationale": _difficulty_rationale(difficulty, profile),
                "task_depth_rationale": _depth_rationale(depth),
                "source": {
                    "kind": "training_factory_targeted_draft",
                    "authorship": "project_authored_deterministic_template",
                    "profile": profile.get("id"),
                    "failure_evidence": {
                        "failure_category": failure_category,
                        "observed_failure_count": observed_failure_count,
                        "failure_input_sha256": failure_input_sha256,
                        "basis": "stored_failure_records_only",
                    },
                },
                "targeting": {
                    "failure_category": failure_category,
                    "observed_failure_count": observed_failure_count,
                    "requires_human_approval": True,
                    "promotion": "not_automatic",
                },
                "created_at": created_at,
            }
            problems = validate_factory_draft(draft)
            if problems:
                raise TrainingFactoryError(f"Generated invalid targeted draft {draft_id}: {problems}")
            drafts.append(draft)
    prompts = [draft["user_request"] for draft in drafts]
    for left_index, left in enumerate(prompts):
        for right in prompts[left_index + 1 :]:
            if similarity(left, right).jaccard >= 0.82:
                raise TrainingFactoryError("Targeted draft generator produced near-duplicate prompt text")
    return drafts


def factory_draft_eval_collisions(
    drafts: Iterable[dict[str, Any]],
    evaluation_tasks: Iterable[dict[str, Any]],
    *,
    threshold: float = 0.93,
) -> list[dict[str, Any]]:
    """Use the existing local-only wording collision guard without exposing held-out content."""
    candidates = [
        {
            "record_id": draft.get("draft_id"),
            "source_seed_id": draft.get("draft_id"),
            "messages": [{"role": "user", "content": draft.get("user_request", "")}],
        }
        for draft in drafts
    ]
    return cross_split_prompt_collisions(candidates, evaluation_tasks, threshold=threshold)


def _correction_round(record: dict[str, Any]) -> int:
    value = _metadata(record).get("correction_round", 0)
    if isinstance(value, bool):
        raise TrainingFactoryError("Correction round must be an integer, not boolean")
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError) as exc:
        raise TrainingFactoryError(f"Correction round is not an integer: {value!r}") from exc
    if parsed < 0:
        raise TrainingFactoryError("Correction round cannot be negative")
    return parsed


def select_corrections(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Choose the latest correction per immutable parent; ambiguity stays visible in provenance."""
    selected: dict[str, dict[str, Any]] = {}
    for record in records:
        metadata = _metadata(record)
        parent = metadata.get("parent_record_id")
        if not _non_empty_string(parent):
            continue
        existing = selected.get(parent)
        if existing is None or _correction_round(record) >= _correction_round(existing):
            selected[parent] = record
    return selected


def make_lineage_ledger_record(
    record: dict[str, Any],
    *,
    taxonomy: dict[str, Any],
    candidate_artifact: str,
    correction: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Flatten a canonical lineage into an audit-only record without changing source candidates."""
    metadata = _metadata(record)
    categories = infer_failure_categories(record, taxonomy=taxonomy) if observed_failure(record) else []
    verification = correction_verification(record, correction)
    ledger_id = f"ledger-{record.get('record_id')}-{sha256_text(candidate_artifact)[:10]}"
    correction_metadata = _metadata(correction).get("correction", {}) if isinstance(correction, dict) else {}
    quality_eligible, quality_reasons = quality_gate_status(record)
    result = {
        "schema_version": FACTORY_SCHEMA_VERSION,
        "ledger_id": ledger_id,
        "record_id": record.get("record_id"),
        "source_seed_id": record.get("source_seed_id"),
        "task": {
            "prompt": text_from_message(record, "user"),
            "task_type": metadata.get("task_type"),
            "category": metadata.get("category", "unclassified"),
            "difficulty": metadata.get("difficulty"),
            "task_depth": metadata.get("task_depth", "unspecified"),
            "requirements": deepcopy(metadata.get("requirements", [])),
            "topics": deepcopy(metadata.get("topics", [])),
        },
        "builder": {
            "answer": text_from_message(record, "assistant"),
            "agent": _agent_metadata(record),
            "record_stage": metadata.get("stage"),
        },
        "review": {
            "static": deepcopy(_quality(record).get("static", {})),
            "llm_review": deepcopy(_quality(record).get("llm_review", {})),
            "human_review": deepcopy(_quality(record).get("human_review", {})),
            "findings": _finding_text(record),
        },
        "correction": {
            "corrected_record_id": correction.get("record_id") if isinstance(correction, dict) else None,
            "corrected_output": text_from_message(correction, "assistant") if isinstance(correction, dict) else None,
            "explanation": deepcopy(correction_metadata.get("changes_made", correction_metadata.get("reason", []))),
            "metadata": deepcopy(correction_metadata) if isinstance(correction_metadata, dict) else {},
        },
        "failure_analysis": {
            "observed_failure": observed_failure(record),
            "failure_categories": categories,
            "severity": severity_for_failure(record, categories, taxonomy=taxonomy) if categories else None,
        },
        "verification": verification,
        "quality": {
            "original_quality_gate_eligible": quality_eligible,
            "original_quality_gate_reasons": quality_reasons,
            "static_checker_version_required": STATIC_CHECKER_VERSION,
        },
        "version": {
            "factory_schema_version": FACTORY_SCHEMA_VERSION,
            "candidate_schema_version": record.get("schema_version"),
            "static_checker_version": _quality(record).get("static", {}).get("checker") if isinstance(_quality(record).get("static"), dict) else None,
            "dataset_version": metadata.get("dataset_version"),
            "planned_dataset_version": "dukeotr_dataset_v1",
            "generator_model": _agent_metadata(record).get("model"),
        },
        "provenance": {
            "candidate_artifact": str(candidate_artifact),
            "candidate_fingerprint": record_fingerprint(record),
            "source": deepcopy(metadata.get("source", {})),
            "candidate_created_at": metadata.get("created_at"),
            "dataset_version": metadata.get("dataset_version"),
            "dataset_version_status": "candidate_unversioned" if metadata.get("dataset_version") is None else "candidate_declared_version",
            "planned_dataset_version": "dukeotr_dataset_v1",
        },
        "created_at": created_at or utc_now(),
    }
    return result


def validate_model_version_registry(registry: dict[str, Any]) -> list[dict[str, str]]:
    """Validate future-model metadata and reject unsupported existence claims structurally."""
    problems: list[dict[str, str]] = []
    if registry.get("schema_version") != FACTORY_SCHEMA_VERSION:
        problems.append(factory_issue("version.schema_version", f"Expected {FACTORY_SCHEMA_VERSION!r}"))
    if registry.get("project") != "DukeOTR":
        problems.append(factory_issue("version.project", "Version registry project must be DukeOTR"))
    versions = registry.get("versions")
    if not isinstance(versions, list) or not versions:
        return [*problems, factory_issue("version.versions", "Version registry requires non-empty versions list")]
    identifiers: set[str] = set()
    for index, entry in enumerate(versions, start=1):
        if not isinstance(entry, dict):
            problems.append(factory_issue("version.entry", f"Version entry {index} is not an object"))
            continue
        identifier = entry.get("id")
        if not _non_empty_string(identifier):
            problems.append(factory_issue("version.id", f"Version entry {index} needs id"))
            continue
        if identifier in identifiers:
            problems.append(factory_issue("version.id", f"Duplicate version id {identifier!r}"))
        identifiers.add(identifier)
        if not _non_empty_string(entry.get("kind")):
            problems.append(factory_issue("version.kind", f"Version {identifier!r} needs kind"))
        status = entry.get("status")
        if status not in VALID_VERSION_STATUSES:
            problems.append(factory_issue("version.status", f"Version {identifier!r} has invalid status {status!r}"))
            continue
        if not _string_list(entry.get("notes")):
            problems.append(factory_issue("version.notes", f"Version {identifier!r} needs non-empty notes"))
        if status in {"planned_not_built", "planned_not_trained", "reserved_not_created"}:
            for artifact_key in ("adapter_artifact", "training_manifest", "evaluation_evidence"):
                if entry.get(artifact_key) not in {None, ""}:
                    problems.append(
                        factory_issue(
                            "version.unsupported_artifact_claim",
                            f"Planned/reserved version {identifier!r} must not declare {artifact_key}",
                        )
                    )
        if status in {"trained_pending_evaluation", "evaluated_not_released", "released"}:
            if not _non_empty_string(entry.get("adapter_artifact")):
                problems.append(factory_issue("version.adapter_evidence", f"{identifier!r} claims a trained state without adapter artifact evidence"))
            if not _sha256(entry.get("adapter_sha256")):
                problems.append(factory_issue("version.adapter_hash", f"{identifier!r} claims a trained state without a hexadecimal SHA-256 adapter hash"))
        if status in {"evaluated_not_released", "released"}:
            if not _non_empty_string(entry.get("evaluation_evidence")):
                problems.append(factory_issue("version.evaluation_evidence", f"{identifier!r} claims evaluation without evidence path"))
            if not _sha256(entry.get("evaluation_evidence_sha256")):
                problems.append(factory_issue("version.evaluation_hash", f"{identifier!r} claims evaluation without a hexadecimal SHA-256 evidence hash"))
    return problems


def registry_file_fingerprint(paths: Iterable[str | Path]) -> str:
    """Create a stable evidence input hash without reading held-out text into a generated brief."""
    digests = []
    for path in sorted((Path(value) for value in paths), key=lambda value: str(value)):
        digests.append({"path": str(path), "sha256": sha256_file(path)})
    return sha256_text(canonical_json(digests))
