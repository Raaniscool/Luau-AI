"""Run a traceable, bounded Builder → Reviewer → Fixer pilot for one DukeOTR train brief.

This is deliberately a *quality-loop artifact generator*, not a training-data constructor.
Even an accepted trace remains outside final training data until the normal independent gates
(static validation, accepting review/human review, deduplication, evaluation isolation, and
explicit dataset construction) have been completed.
"""

from __future__ import annotations

# Support both `python -m scripts.run_builder_reviewer_fixer` and direct Windows invocation.
if __package__ in {None, ""}:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from scripts.lib.builder_reviewer_fixer import (
    BUILDER_RESPONSE_SCHEMA,
    BUILDER_SYSTEM,
    FIXER_RESPONSE_SCHEMA,
    FIXER_SYSTEM,
    REVIEWER_RESPONSE_SCHEMA,
    REVIEWER_SYSTEM,
    builder_prompt,
    fixer_prompt,
    parse_builder,
    parse_fixer,
    parse_reviewer,
    reviewer_prompt,
)
from scripts.lib.code_book import context_view, load_cards, search_cards
from scripts.lib.dedupe import cross_split_prompt_collisions
from scripts.lib.effort_routing import EffortRoute, classify_task
from scripts.lib.io_utils import (
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
    utc_now,
    write_json_atomic,
)
from scripts.lib.ollama import OllamaClient, OllamaError
from scripts.lib.quality import static_validate
from scripts.lib.schema import clone_for_correction, make_generated_record, validate_seed


_TRACE_SCHEMA_VERSION = "1.1"
_DEFAULT_OUTPUT = "reports/builder_reviewer_fixer.trace.json"


class ReviewerPassError(RuntimeError):
    """Preserves completed reviewer-pass diagnostics when a later pass fails."""

    def __init__(self, message: str, pass_traces: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.pass_traces = pass_traces


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run one bounded DukeOTR Builder → Reviewer → Fixer trace; never promotes training data."
    )
    value.add_argument("--seed-id", required=True, help="ID of one project-authored train brief")
    value.add_argument("--seeds", default="raw_data/dukeotr_phase1_luau_seed_tasks.jsonl")
    value.add_argument("--evaluation", default="evaluation_data/roblox_luau_eval.jsonl")
    value.add_argument("--code-book", default="code_book/roblox_luau_cards.jsonl")
    value.add_argument("--config", default="configs/builder_verifier_reviewer.json")
    value.add_argument("--pipeline-config", default="configs/pipeline.json")
    value.add_argument("--output", default=_DEFAULT_OUTPUT)
    value.add_argument("--overwrite", action="store_true", help="Replace an existing trace only after inspection")
    value.add_argument("--max-rounds", type=int, default=None, help="Total candidate rounds, including the initial Builder round")
    value.add_argument("--model", default=None, help="Use this Ollama tag for all roles unless a role-specific model is set")
    value.add_argument("--builder-model", default=None)
    value.add_argument("--reviewer-model", default=None)
    value.add_argument("--fixer-model", default=None)
    value.add_argument("--host", default=None)
    value.add_argument(
        "--timeout-seconds",
        type=int,
        default=600,
        help="Maximum wait for each local Ollama HTTP response/chunk; increase on slow hardware.",
    )
    value.add_argument(
        "--dry-run",
        action="store_true",
        help="Write an isolation-checked execution plan without querying Ollama or producing a candidate",
    )
    return value


def _safe_file_hash(path: str | Path) -> str:
    return sha256_file(path)


def _load_train_seed(path: str, seed_id: str) -> dict[str, Any]:
    matches = [item for item in read_jsonl(path) if item.get("id") == seed_id]
    if not matches:
        raise ValueError(f"No seed with id {seed_id!r} in {path}")
    if len(matches) != 1:
        raise ValueError(f"Seed id {seed_id!r} is not unique in {path}")
    seed = matches[0]
    errors = validate_seed(seed)
    if errors:
        rendered = "; ".join(f"{item['code']}: {item['message']}" for item in errors)
        raise ValueError(f"Seed {seed_id!r} is not a valid train brief: {rendered}")
    if seed.get("split", "train") != "train":
        raise ValueError("Builder → Reviewer → Fixer only accepts a train split source brief")
    return seed


def _code_book_context(seed: dict[str, Any], path: str, limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if limit < 0:
        raise ValueError("Code Book context limit must be zero or positive")
    if limit == 0:
        return [], []
    cards = load_cards(path)
    query = " ".join(
        [
            str(seed.get("title", "")),
            str(seed.get("user_request", "")),
            *[str(item) for item in seed.get("concepts", [])],
        ]
    )
    matches = search_cards(cards, query, limit=limit)
    context = [context_view(match) for match in matches]
    provenance = [
        {
            "id": item["id"],
            "revision": item["revision"],
            "status": item["status"],
            "source_urls": [source["url"] for source in item["sources"]],
        }
        for item in context
    ]
    return context, provenance


def _isolation_report(seed: dict[str, Any], evaluation_path: str, threshold: float) -> dict[str, Any]:
    """Compare wording locally without passing any held-out material to a model role."""
    if not 0 < threshold <= 1:
        raise ValueError("Evaluation isolation threshold must be in (0, 1]")
    placeholder = {
        "record_id": f"brf-isolation-{seed['id']}",
        "source_seed_id": seed["id"],
        "messages": [{"role": "user", "content": seed["user_request"]}],
    }
    tasks = list(read_jsonl(evaluation_path))
    collisions = cross_split_prompt_collisions([placeholder], tasks, threshold=threshold)
    # No held-out prompt or rubric text is put into trace data or role context.
    public_collisions = [
        {
            "evaluation_task_id": item["evaluation_task_id"],
            "similarity": item["similarity"],
            "threshold": item["threshold"],
        }
        for item in collisions
    ]
    return {
        "status": "clear" if not public_collisions else "collision",
        "evaluation_file": evaluation_path,
        "evaluation_sha256": _safe_file_hash(evaluation_path),
        "comparison": "local wording-level normalized 3-gram Jaccard; held-out text was not supplied to any model role",
        "threshold": threshold,
        "collisions": public_collisions,
    }


def _role_options(config: dict[str, Any], role: str) -> dict[str, Any]:
    orchestrator = config.get("orchestrator", {})
    raw = orchestrator.get("role_options", {}).get(role, {}) if isinstance(orchestrator, dict) else {}
    if not isinstance(raw, dict):
        raise ValueError(f"orchestrator.role_options.{role} must be an object")
    return deepcopy(raw)


def _role_models(arguments: argparse.Namespace, config: dict[str, Any], pipeline_config: dict[str, Any]) -> dict[str, str]:
    orchestrator = config.get("orchestrator", {})
    config_models = orchestrator.get("role_models", {}) if isinstance(orchestrator, dict) else {}
    if not isinstance(config_models, dict):
        raise ValueError("orchestrator.role_models must be an object")
    base = arguments.model or str(config_models.get("default") or pipeline_config.get("base_ollama_model", "qwen3:4b"))
    return {
        "builder": arguments.builder_model or str(config_models.get("builder") or base),
        "reviewer": arguments.reviewer_model or str(config_models.get("reviewer") or base),
        "fixer": arguments.fixer_model or str(config_models.get("fixer") or base),
    }


def _minimums(config: dict[str, Any]) -> dict[str, int]:
    raw = config.get("orchestrator", {}).get("minimum_dimension_scores", {})
    expected = ("correctness", "security", "api_validity", "requirements", "english", "code_quality")
    if not isinstance(raw, dict):
        raise ValueError("orchestrator.minimum_dimension_scores must be an object")
    result: dict[str, int] = {}
    for key in expected:
        score = raw.get(key)
        if not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"orchestrator.minimum_dimension_scores.{key} must be an integer from 1 to 5")
        result[key] = score
    return result


def _resolve_max_rounds(arguments: argparse.Namespace, config: dict[str, Any], route: EffortRoute) -> int:
    """Bound explicit overrides by both the global and task-appropriate route caps."""
    orchestrator = config.get("orchestrator", {})
    if not isinstance(orchestrator, dict):
        raise ValueError("orchestrator configuration must be an object")
    global_cap = orchestrator.get("maximum_max_rounds", 4)
    cap = min(route.maximum_rounds, global_cap) if isinstance(global_cap, int) else 0
    value = arguments.max_rounds if arguments.max_rounds is not None else route.default_max_rounds
    if not isinstance(value, int) or not isinstance(cap, int) or value < 1 or cap < 1 or value > cap:
        raise ValueError(
            f"max rounds for the {route.level} route must be an integer from 1 to {cap}; "
            "choose a higher-risk route only when the task actually warrants it"
        )
    return value


def _static_category(code: str) -> str:
    if code.startswith(("security.", "network.", "datastore.")):
        return "security"
    if code.startswith("api."):
        return "api"
    if code.startswith("coverage."):
        return "requirements"
    if code.startswith("style."):
        return "style"
    if code.startswith("performance."):
        return "code_quality"
    if code.startswith("answer."):
        return "code_quality"
    return "correctness"


def _failure_reasons(static: dict[str, Any], review: dict[str, Any] | None) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    for item in static.get("issues", []):
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity", "warning"))
        reasons.append(
            {
                "stage": "static_validation",
                "id": str(item.get("code", "static.unknown")),
                "category": _static_category(str(item.get("code", ""))),
                "severity": "block" if severity in {"block", "error"} else "minor",
                "message": str(item.get("message", "Static checker finding")),
                "evidence": item.get("evidence"),
                "required_fix": "Resolve this deterministic validation finding before promotion.",
            }
        )
    if review:
        for item in review.get("findings", []):
            if isinstance(item, dict):
                reasons.append({"stage": "reviewer", **item})
        for item in review.get("policy_forced_reasons", []):
            rendered = str(item)
            category = next(
                (
                    candidate
                    for candidate in ("api", "security", "correctness", "requirements", "english", "code_quality")
                    if candidate in rendered
                ),
                "requirements",
            )
            reasons.append(
                {
                    "stage": "review_policy",
                    "id": rendered,
                    "category": category,
                    "severity": "major",
                    "message": "The reviewer acceptance did not satisfy the configured API or dimension policy.",
                    "evidence": None,
                    "required_fix": "Resolve every uncertain API claim and raise every below-policy review dimension before acceptance.",
                }
            )
    return reasons


def _response_audit(response: Any) -> dict[str, Any]:
    return {
        "elapsed_seconds": round(float(response.elapsed_seconds), 3),
        "response_sha256": sha256_text(response.content),
        "response_characters": len(response.content),
    }


def _error_audit(response: Any | None, error: Exception) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "error", "error": str(error)}
    if response is not None:
        result.update(
            {
                "response_sha256": sha256_text(response.content),
                "response_excerpt": response.content[:8000],
                "response_truncated": len(response.content) > 8000,
                "elapsed_seconds": round(float(response.elapsed_seconds), 3),
            }
        )
    return result


def _task_appropriate_tester(record: dict[str, Any], route: EffortRoute) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the existing deterministic validator with the route's relevant effort budget.

    Static validation remains a safety floor; the routing record explains which semantic checks
    are requested from a reviewer and which irrelevant expensive scopes are skipped.
    """
    static = static_validate(record, minimum_assistant_characters=route.minimum_assistant_characters)
    tester = {
        "status": "pass" if static.get("status") == "pass" else "fail",
        "kind": "deterministic_task_appropriate_tester",
        "selected_checks": list(route.selected_checks),
        "skipped_checks": list(route.skipped_checks),
        "minimum_assistant_characters": route.minimum_assistant_characters,
        "static_checker": static.get("checker"),
        "issue_count": len(static.get("issues", [])),
    }
    return static, tester


def _aggregate_reviews(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine independent reviewer passes conservatively for one candidate round."""
    if not reviews:
        raise ValueError("Cannot aggregate zero reviewer passes")
    findings: list[dict[str, Any]] = []
    api_claims: list[str] = []
    policy_reasons: list[str] = []
    scores: dict[str, int] = {}
    decisions: list[str] = []
    summaries: list[str] = []
    for index, review in enumerate(reviews, start=1):
        decisions.append(str(review["decision"]))
        summaries.append(str(review["summary"]))
        for key, score in review["dimension_scores"].items():
            scores[key] = min(scores.get(key, score), score)
        for finding in review.get("findings", []):
            clone = dict(finding)
            # Preserve established finding IDs on the standard one-pass route; deeper routes
            # namespace IDs so Fixer acknowledgement remains unambiguous across passes.
            if len(reviews) > 1:
                clone["id"] = f"r{index}-{clone['id']}"
            findings.append(clone)
        for claim in review.get("api_claims_to_verify", []):
            if claim not in api_claims:
                api_claims.append(claim)
        policy_reasons.extend(f"r{index}:{reason}" for reason in review.get("policy_forced_reasons", []))
    if "reject" in decisions:
        decision = "reject"
    elif "revise" in decisions:
        decision = "revise"
    else:
        decision = "accept"
    return {
        "reported_decisions": decisions,
        "decision": decision,
        "dimension_scores": scores,
        "findings": findings,
        "api_claims_to_verify": api_claims,
        "summary": "\n".join(f"Pass {index}: {summary}" for index, summary in enumerate(summaries, start=1)),
        "policy_forced_reasons": policy_reasons,
    }


def _review_candidate(
    *,
    seed: dict[str, Any],
    candidate_answer: str,
    static_issues: list[dict[str, Any]],
    code_book_context: list[dict[str, Any]],
    route: EffortRoute,
    client: OllamaClient,
    model: str,
    options: dict[str, Any],
    minimums: dict[str, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the route's reviewer depth and retain each response in the trace."""
    complete_reviews: list[dict[str, Any]] = []
    pass_traces: list[dict[str, Any]] = []
    for review_pass in range(1, route.reviewer_passes + 1):
        response = None
        try:
            response = client.generate(
                model=model,
                system=REVIEWER_SYSTEM,
                prompt=reviewer_prompt(
                    seed,
                    candidate_answer,
                    static_issues,
                    code_book_context,
                    selected_checks=list(route.selected_checks),
                    review_pass=review_pass,
                    total_review_passes=route.reviewer_passes,
                ),
                options=options,
                think=False,
                response_format=REVIEWER_RESPONSE_SCHEMA,
            )
            review = parse_reviewer(response.content, minimums=minimums)
            complete_reviews.append(review)
            pass_traces.append({"pass": review_pass, "status": "complete", **review, **_response_audit(response)})
            # A hard rejection has already established the terminal policy; further expensive
            # passes cannot make it acceptable and are intentionally skipped.
            if review["decision"] == "reject":
                break
        except (OllamaError, ValueError) as exc:
            pass_traces.append({"pass": review_pass, **_error_audit(response, exc)})
            raise ReviewerPassError(str(exc), pass_traces) from exc
    aggregate = _aggregate_reviews(complete_reviews)
    return aggregate, {
        "status": "complete",
        "requested_passes": route.reviewer_passes,
        "completed_passes": len(complete_reviews),
        "selected_checks": list(route.selected_checks),
        "skipped_checks": list(route.skipped_checks),
        "passes": pass_traces,
        **aggregate,
    }


def _candidate(seed: dict[str, Any], answer: str, *, model: str, options: dict[str, Any], round_number: int) -> dict[str, Any]:
    record = make_generated_record(
        seed,
        answer,
        generator={
            "kind": "builder_reviewer_fixer",
            "role": "builder" if round_number == 1 else "fixer",
            "model": model,
            "options": options,
            "quality_loop_round": round_number,
        },
        variant=round_number,
        coverage=[],
        generation_notes=["Created inside a non-promoting Builder → Reviewer → Fixer trace."],
    )
    record["metadata"]["stage"] = "brf_candidate"
    return record


def _base_trace(
    *,
    arguments: argparse.Namespace,
    config: dict[str, Any],
    seed: dict[str, Any],
    seed_path: str,
    code_book_path: str,
    code_book_provenance: list[dict[str, Any]],
    isolation: dict[str, Any],
    route: EffortRoute,
    max_rounds: int,
    role_models: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": _TRACE_SCHEMA_VERSION,
        "project_name": "DukeOTR",
        "trace_kind": "builder_reviewer_fixer_quality_loop",
        "created_at": utc_now(),
        "status": "running",
        "source": {
            "seed_id": seed["id"],
            "seed_file": seed_path,
            "seed_sha256": _safe_file_hash(seed_path),
            "seed_split": seed.get("split", "train"),
            "source_kind": seed.get("source", {}).get("kind"),
        },
        "configuration": {
            "role_contract_file": arguments.config,
            "role_contract_sha256": _safe_file_hash(arguments.config),
            "role_contract_status": config.get("status"),
            "max_rounds": max_rounds,
            "timeout_seconds": getattr(arguments, "timeout_seconds", 600),
            "role_models": role_models,
            "role_options": {role: _role_options(config, role) for role in ("builder", "reviewer", "fixer")},
        },
        "routing": route.as_dict(),
        "code_book": {
            "file": code_book_path,
            "sha256": _safe_file_hash(code_book_path),
            "retrieved_cards": code_book_provenance,
            "policy": "reference_context_only_manual_curation_only",
        },
        "evaluation_isolation": isolation,
        "promotion": {
            "status": "prohibited",
            "reason": (
                "This trace is not a training-data artifact and does not satisfy independent validation, "
                "deduplication, human review, final-dataset construction, training, or held-out evaluation gates."
            ),
        },
        "rounds": [],
    }


def _write(trace: dict[str, Any], output: str) -> None:
    trace["completed_at"] = utc_now()
    write_json_atomic(output, trace)


def run(arguments: argparse.Namespace) -> int:
    # Keep programmatic callers made before the additive timeout flag compatible with the
    # established default, while the CLI/parser always supplies an explicit value.
    timeout_seconds = getattr(arguments, "timeout_seconds", 600)
    if not isinstance(timeout_seconds, int) or timeout_seconds < 1:
        raise ValueError("--timeout-seconds must be a positive integer")
    arguments.timeout_seconds = timeout_seconds
    output = Path(arguments.output)
    if output.exists() and not arguments.overwrite:
        raise ValueError(f"Refusing to overwrite existing trace {output}; use a new path or --overwrite after review")
    config = read_json(arguments.config)
    if config.get("project_name") != "DukeOTR":
        raise ValueError("Builder → Reviewer → Fixer config must identify project_name DukeOTR")
    pipeline_config = read_json(arguments.pipeline_config)
    seed = _load_train_seed(arguments.seeds, arguments.seed_id)
    route = classify_task(seed, config)
    max_rounds = _resolve_max_rounds(arguments, config, route)
    orchestrator = config.get("orchestrator", {})
    if not isinstance(orchestrator, dict):
        raise ValueError("orchestrator configuration must be an object")
    code_book_context, code_book_provenance = _code_book_context(
        seed, arguments.code_book, route.code_book_context_limit
    )
    cross_threshold = float(pipeline_config.get("deduplication", {}).get("cross_split_prompt_threshold", 0.93))
    isolation = _isolation_report(seed, arguments.evaluation, cross_threshold)
    role_models = _role_models(arguments, config, pipeline_config)
    trace = _base_trace(
        arguments=arguments,
        config=config,
        seed=seed,
        seed_path=arguments.seeds,
        code_book_path=arguments.code_book,
        code_book_provenance=code_book_provenance,
        isolation=isolation,
        route=route,
        max_rounds=max_rounds,
        role_models=role_models,
    )
    if isolation["status"] != "clear":
        trace["status"] = "blocked_by_evaluation_isolation"
        trace["terminal_reason"] = "The selected train brief has wording-level collision(s) with held-out evaluation prompts."
        _write(trace, str(output))
        print(f"Trace blocked by held-out evaluation isolation; report: {output}", file=sys.stderr)
        return 2

    if arguments.dry_run:
        trace["status"] = "planned_no_inference"
        planned_roles = ["builder", "task_appropriate_tester"]
        planned_roles.extend(f"reviewer_pass_{index}" for index in range(1, route.reviewer_passes + 1))
        trace["planned_rounds"] = [
            {
                "round": 1,
                "roles": planned_roles,
                "selected_checks": list(route.selected_checks),
                "skipped_checks": list(route.skipped_checks),
                "conditional_next_step": (
                    "accept immediately after the configured checks pass"
                    if max_rounds == 1
                    else "fixer then re-tester/reviewer only if the configured checks request revision and a round remains"
                ),
            }
        ]
        trace["terminal_reason"] = "Dry run: no Ollama preflight, inference, candidate generation, or promotion occurred."
        _write(trace, str(output))
        print(f"Builder → Reviewer → Fixer plan written to {output}; no model was called.")
        return 0

    minimums = _minimums(config)
    review_static_failures = orchestrator.get("review_static_failures", True)
    repair_rejections = orchestrator.get("repair_rejections", False)
    if not isinstance(review_static_failures, bool) or not isinstance(repair_rejections, bool):
        raise ValueError("orchestrator review_static_failures and repair_rejections must be booleans")
    client = OllamaClient(arguments.host, timeout_seconds=arguments.timeout_seconds)
    # This invokes the exact local registration preflight (including `ollama list` for a local
    # host) and never downloads or changes a model. Delay a distinct Fixer model's preflight
    # until a repair is actually needed, so an early-pass response does not touch that role.
    preflighted_models: set[str] = set()

    def ensure_model_preflight(role: str) -> None:
        model = role_models[role]
        if model not in preflighted_models:
            client.assert_model_present(model)
            preflighted_models.add(model)

    ensure_model_preflight("builder")

    builder_options = _role_options(config, "builder")
    reviewer_options = _role_options(config, "reviewer")
    fixer_options = _role_options(config, "fixer")
    current_record: dict[str, Any] | None = None
    current_answer: str | None = None

    for round_number in range(1, max_rounds + 1):
        round_trace: dict[str, Any] = {"round": round_number}
        if round_number == 1:
            response = None
            try:
                response = client.generate(
                    model=role_models["builder"],
                    system=BUILDER_SYSTEM,
                    prompt=builder_prompt(seed, code_book_context),
                    options=builder_options,
                    think=False,
                    response_format=BUILDER_RESPONSE_SCHEMA,
                )
                builder = parse_builder(response.content)
                current_answer = builder["assistant_response"]
                current_record = _candidate(
                    seed,
                    current_answer,
                    model=role_models["builder"],
                    options=builder_options,
                    round_number=round_number,
                )
                round_trace["builder"] = {"status": "complete", **builder, **_response_audit(response)}
            except (OllamaError, ValueError) as exc:
                round_trace["builder"] = _error_audit(response, exc)
                trace["rounds"].append(round_trace)
                trace["status"] = "error"
                trace["terminal_reason"] = "Builder failed; no candidate was promoted."
                _write(trace, str(output))
                print(f"Builder failed; trace written to {output}", file=sys.stderr)
                return 2
        else:
            if current_record is None or current_answer is None:
                raise RuntimeError("Internal error: fixer round has no prior candidate")

        assert current_record is not None and current_answer is not None
        static, tester = _task_appropriate_tester(current_record, route)
        round_trace["static_validation"] = static
        round_trace["tester"] = tester
        should_review = route.reviewer_passes > 0 and (review_static_failures or static.get("status") == "pass")
        review: dict[str, Any] | None = None
        if should_review:
            try:
                ensure_model_preflight("reviewer")
                review, reviewer_trace = _review_candidate(
                    seed=seed,
                    candidate_answer=current_answer,
                    static_issues=list(static.get("issues", [])),
                    code_book_context=code_book_context,
                    route=route,
                    client=client,
                    model=role_models["reviewer"],
                    options=reviewer_options,
                    minimums=minimums,
                )
                round_trace["reviewer"] = reviewer_trace
            except (ReviewerPassError, OllamaError) as exc:
                round_trace["reviewer"] = {
                    "status": "error",
                    "requested_passes": route.reviewer_passes,
                    "passes": getattr(exc, "pass_traces", []),
                    "error": str(exc),
                }
                round_trace["failure_reasons"] = _failure_reasons(static, None)
                trace["rounds"].append(round_trace)
                trace["status"] = "error"
                trace["terminal_reason"] = "Independent reviewer failed; no candidate was promoted."
                _write(trace, str(output))
                print(f"Reviewer failed; trace written to {output}", file=sys.stderr)
                return 2
        else:
            reason = (
                "The simple route requires no model reviewer after its task-appropriate deterministic tester passes."
                if route.reviewer_passes == 0 and static.get("status") == "pass"
                else "Configured route does not review a static-failing candidate; record remains ineligible."
            )
            round_trace["reviewer"] = {"status": "skipped", "reason": reason}

        decision = review["decision"] if review else ("accept" if static.get("status") == "pass" and route.reviewer_passes == 0 else "revise")
        policy_overrides: list[str] = []
        if static.get("status") != "pass" and decision == "accept":
            decision = "revise"
            policy_overrides.append("accept_downgraded_static_validation_failed")
        round_trace["effective_decision"] = decision
        round_trace["policy_overrides"] = policy_overrides
        round_trace["failure_reasons"] = _failure_reasons(static, review)

        if decision == "accept" and static.get("status") == "pass":
            round_trace["early_pass"] = True
            round_trace["early_pass_reason"] = (
                "The Builder response met the route's configured task-appropriate checks; no Fixer or additional round was invoked."
            )
            trace["rounds"].append(round_trace)
            trace["status"] = "accepted_for_manual_review_only"
            trace["terminal_reason"] = (
                "The Builder response passed its configured task-appropriate tester/reviewer checks, so the loop stopped early. "
                "The trace remains non-promoting and requires independent pipeline gates before any dataset decision."
            )
            trace["final_candidate"] = current_record
            _write(trace, str(output))
            print(f"Trace accepted for manual review only: {output}. It was not promoted to training data.")
            return 0

        can_fix = decision == "revise" or (decision == "reject" and repair_rejections)
        if not can_fix:
            trace["rounds"].append(round_trace)
            trace["status"] = "rejected_no_auto_promotion"
            trace["terminal_reason"] = "Reviewer rejected the candidate; automatic promotion and implicit repair are disabled."
            trace["final_candidate"] = current_record
            _write(trace, str(output))
            print(f"Trace rejected without promotion: {output}")
            return 0
        if round_number >= max_rounds:
            trace["rounds"].append(round_trace)
            trace["status"] = "needs_human_review"
            trace["terminal_reason"] = "Maximum configured review/fix rounds reached without a passing independent review."
            trace["final_candidate"] = current_record
            _write(trace, str(output))
            print(f"Trace reached its bounded repair limit and needs human review: {output}")
            return 0

        response = None
        try:
            ensure_model_preflight("fixer")
            response = client.generate(
                model=role_models["fixer"],
                system=FIXER_SYSTEM,
                prompt=fixer_prompt(
                    seed,
                    current_answer,
                    review or {},
                    list(static.get("issues", [])),
                    code_book_context,
                    selected_checks=list(route.selected_checks),
                ),
                options=fixer_options,
                think=False,
                response_format=FIXER_RESPONSE_SCHEMA,
            )
            fixer = parse_fixer(response.content, reviewer_findings=list((review or {}).get("findings", [])))
            correction = {
                "role": "fixer",
                "model": role_models["fixer"],
                "options": fixer_options,
                "source_quality_snapshot": current_record.get("quality", {}),
                "review_summary": (review or {}).get("summary"),
                "review_findings": (review or {}).get("findings", []),
                "changes_made": fixer["changes_made"],
                "remaining_assumptions": fixer["remaining_assumptions"],
                "unresolved_risks": fixer["unresolved_risks"],
                "addressed_finding_ids": fixer["addressed_finding_ids"],
            }
            current_record = clone_for_correction(current_record, fixer["assistant_response"], correction=correction)
            current_record["metadata"]["stage"] = "brf_candidate"
            current_answer = fixer["assistant_response"]
            round_trace["fixer"] = {"status": "complete", **fixer, **_response_audit(response)}
        except (OllamaError, ValueError) as exc:
            round_trace["fixer"] = _error_audit(response, exc)
            trace["rounds"].append(round_trace)
            trace["status"] = "error"
            trace["terminal_reason"] = "Fixer failed; no candidate was promoted."
            trace["final_candidate"] = current_record
            _write(trace, str(output))
            print(f"Fixer failed; trace written to {output}", file=sys.stderr)
            return 2
        trace["rounds"].append(round_trace)

    raise RuntimeError("Internal error: bounded loop ended without a terminal trace state")


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError, OllamaError) as exc:
        print(f"builder/reviewer/fixer error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
