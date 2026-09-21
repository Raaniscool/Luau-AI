from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.lib.io_utils import read_json, read_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic
from scripts.lib.ollama import OllamaUnavailableError
from scripts.lib.real_model_pilot import load_pilot_tasks, validate_pilot_tasks
from scripts.lib.schema import STATIC_CHECKER_VERSION
from scripts.run_real_model_pilot import main


RAW_ANSWER = """Use a server-owned validation path rather than trusting client input.

```luau
local function isAllowed(value)
    return type(value) == "string" and #value > 0
end
```

The server validates the value before changing protected state."""

FIXED_ANSWER = """The client may request an action, but the server owns validation and protected state.

```luau
local function isAllowed(value)
    return type(value) == "string" and #value > 0
end
```

Validate the request on the server before any protected change."""


def pilot_task(index: int, *, task_id: str | None = None, prompt: str | None = None) -> dict[str, object]:
    return {
        "id": task_id or f"pilot-offline-{index:03d}",
        "split": "train",
        "title": f"Offline pilot task {index}",
        "task_type": "code_generation",
        "difficulty": "intermediate",
        "category": "offline_test",
        "task_depth": "normal",
        "user_request": prompt or f"Explain a unique local validation pattern for offline pilot task {index}.",
        "requirements": ["Include a small Luau code example"],
        "concepts": ["Luau", "validation"],
        "expected_evidence": ["a validation function"],
        "avoid": ["unrelated services"],
        "tags": ["offline", "pilot"],
        "source": {"kind": "project_authored", "pilot_only": True},
    }


def accept_trace(arguments, *, answer: str = RAW_ANSWER) -> int:
    trace = {
        "schema_version": "1.1",
        "status": "accepted_for_manual_review_only",
        "created_at": "2026-09-20T00:00:00Z",
        "completed_at": "2026-09-20T00:01:00Z",
        "configuration": {
            "role_models": {"builder": "qwen3:4b", "reviewer": "qwen3:4b", "fixer": "qwen3:4b"},
            "role_options": {"builder": {"seed": 3407}, "reviewer": {"seed": 3407}, "fixer": {"seed": 3407}},
            "timeout_seconds": arguments.timeout_seconds,
        },
        "evaluation_isolation": {"status": "clear", "collisions": [], "threshold": 0.93},
        "rounds": [
            {
                "round": 1,
                "builder": {"status": "complete", "assistant_response": answer, "response_sha256": "actual-trace-hash"},
                "static_validation": {
                    "status": "pass",
                    "issues": [],
                    "checker": STATIC_CHECKER_VERSION,
                    "checked_at": "2026-09-20T00:01:00Z",
                },
                "reviewer": {
                    "status": "complete",
                    "decision": "accept",
                    "reported_decisions": ["accept"],
                    "dimension_scores": {
                        "correctness": 5,
                        "security": 5,
                        "api_validity": 5,
                        "requirements": 5,
                        "english": 5,
                        "code_quality": 5,
                    },
                    "findings": [],
                    "api_claims_to_verify": [],
                    "summary": "Mocked complete review for offline orchestration testing.",
                    "policy_forced_reasons": [],
                    "completed_passes": 1,
                    "requested_passes": 1,
                },
            }
        ],
        "final_candidate": {"messages": [{"role": "assistant", "content": answer}]},
    }
    write_json_atomic(arguments.output, trace)
    return 0


def skipped_reviewer_trace(arguments) -> int:
    accept_trace(arguments)
    trace = read_json(arguments.output)
    trace["rounds"][0]["reviewer"] = {
        "status": "skipped",
        "reason": "Mocked simple-route deterministic pass; no independent model review was run.",
    }
    write_json_atomic(arguments.output, trace)
    return 0


def error_trace(arguments) -> int:
    trace = {
        "schema_version": "1.1",
        "status": "error",
        "terminal_reason": "Independent reviewer failed; no candidate was promoted.",
        "created_at": "2026-09-20T00:00:00Z",
        "completed_at": "2026-09-20T00:01:00Z",
        "configuration": {
            "role_models": {"builder": "qwen3:4b", "reviewer": "qwen3:4b", "fixer": "qwen3:4b"},
            "role_options": {"builder": {}, "reviewer": {}, "fixer": {}},
            "timeout_seconds": arguments.timeout_seconds,
        },
        "evaluation_isolation": {"status": "clear", "collisions": [], "threshold": 0.93},
        "rounds": [
            {
                "round": 1,
                "builder": {"status": "complete", "assistant_response": RAW_ANSWER},
                "static_validation": {
                    "status": "pass",
                    "issues": [],
                    "checker": STATIC_CHECKER_VERSION,
                    "checked_at": "2026-09-20T00:01:00Z",
                },
                "reviewer": {"status": "error", "error": "mocked local review transport timeout"},
            }
        ],
        "final_candidate": {"messages": [{"role": "assistant", "content": RAW_ANSWER}]},
    }
    write_json_atomic(arguments.output, trace)
    return 2


def corrected_failure_trace(arguments) -> int:
    trace = {
        "schema_version": "1.1",
        "status": "accepted_for_manual_review_only",
        "created_at": "2026-09-20T00:00:00Z",
        "completed_at": "2026-09-20T00:02:00Z",
        "configuration": {
            "role_models": {"builder": "qwen3:4b", "reviewer": "qwen3:4b", "fixer": "qwen3:4b"},
            "role_options": {"builder": {"seed": 3407}, "reviewer": {"seed": 3407}, "fixer": {"seed": 3407}},
            "timeout_seconds": arguments.timeout_seconds,
        },
        "evaluation_isolation": {"status": "clear", "collisions": [], "threshold": 0.93},
        "rounds": [
            {
                "round": 1,
                "builder": {"status": "complete", "assistant_response": RAW_ANSWER},
                "static_validation": {
                    "status": "fail",
                    "issues": [
                        {
                            "code": "security.trust_client_price",
                            "severity": "block",
                            "message": "The first response trusted a client-supplied price.",
                        }
                    ],
                    "checker": STATIC_CHECKER_VERSION,
                    "checked_at": "2026-09-20T00:01:00Z",
                },
                "reviewer": {
                    "status": "complete",
                    "decision": "revise",
                    "reported_decisions": ["revise"],
                    "dimension_scores": {
                        "correctness": 3,
                        "security": 1,
                        "api_validity": 4,
                        "requirements": 3,
                        "english": 4,
                        "code_quality": 3,
                    },
                    "findings": [{"id": "security-1", "severity": "major", "message": "Do not trust client price."}],
                    "api_claims_to_verify": [],
                    "summary": "Mocked security finding.",
                    "policy_forced_reasons": [],
                    "completed_passes": 1,
                    "requested_passes": 1,
                },
                "fixer": {
                    "status": "complete",
                    "assistant_response": FIXED_ANSWER,
                    "changes_made": ["Moved price validation to the server."],
                    "remaining_assumptions": [],
                    "unresolved_risks": [],
                    "addressed_finding_ids": ["security-1"],
                },
            },
            {
                "round": 2,
                "static_validation": {
                    "status": "pass",
                    "issues": [],
                    "checker": STATIC_CHECKER_VERSION,
                    "checked_at": "2026-09-20T00:02:00Z",
                },
                "reviewer": {
                    "status": "complete",
                    "decision": "accept",
                    "reported_decisions": ["accept"],
                    "dimension_scores": {
                        "correctness": 5,
                        "security": 5,
                        "api_validity": 5,
                        "requirements": 5,
                        "english": 5,
                        "code_quality": 4,
                    },
                    "findings": [],
                    "api_claims_to_verify": [],
                    "summary": "Mocked corrected review.",
                    "policy_forced_reasons": [],
                    "completed_passes": 1,
                    "requested_passes": 1,
                },
            },
        ],
        "final_candidate": {"messages": [{"role": "assistant", "content": FIXED_ANSWER}]},
    }
    write_json_atomic(arguments.output, trace)
    return 0


def empty_correction_explanation_trace(arguments) -> int:
    """Legacy-shaped trace: a real Fixer answer with no actual change explanation."""

    corrected_failure_trace(arguments)
    trace = read_json(arguments.output)
    trace["rounds"][0]["fixer"]["changes_made"] = []
    write_json_atomic(arguments.output, trace)
    return 0


class RealModelPilotOfflineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tasks_path = self.root / "pilot_tasks.jsonl"
        self.evaluation_path = self.root / "held_out.jsonl"
        self.evaluation_path.write_text("", encoding="utf-8")
        self.output_root = self.root / "reports"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_tasks(self, tasks: list[dict[str, object]]) -> None:
        self.tasks_path.write_text("".join(json.dumps(task) + "\n" for task in tasks), encoding="utf-8")

    def command(self, run_id: str, *extra: str) -> list[str]:
        return [
            "--run-id",
            run_id,
            "--tasks",
            str(self.tasks_path),
            "--evaluation",
            str(self.evaluation_path),
            "--output-root",
            str(self.output_root),
            *extra,
        ]

    def paths(self, run_id: str) -> Path:
        return self.output_root / run_id

    def test_catalog_is_predefined_separate_and_covers_requested_pilot_breadth(self) -> None:
        tasks = load_pilot_tasks("pilot_data/real_model_pilot_v1.jsonl")
        evaluation = list(read_jsonl("evaluation_data/roblox_luau_eval.jsonl"))
        isolation = validate_pilot_tasks(
            tasks,
            evaluation,
            evaluation_path="evaluation_data/roblox_luau_eval.jsonl",
            cross_split_threshold=0.93,
        )
        self.assertGreaterEqual(len(tasks), 10)
        self.assertLessEqual(len(tasks), 20)
        self.assertTrue(all(str(task["id"]).startswith("pilot-") for task in tasks))
        self.assertTrue(all(task["source"]["pilot_only"] is True for task in tasks))
        joined = " ".join(" ".join(str(part) for part in task.get("tags", [])) for task in tasks).lower()
        for term in ("english", "luau", "roblox", "script", "localscript", "remoteevent", "security", "ui", "debugging", "architecture", "command-bar", "requirements"):
            self.assertIn(term, joined)
        self.assertEqual(isolation["status"], "clear")
        self.assertEqual(isolation["task_count"], len(tasks))

    def test_dry_run_is_no_model_static_audit(self) -> None:
        self.write_tasks([pilot_task(1)])
        with patch("scripts.run_real_model_pilot.OllamaClient") as client:
            code = main(self.command("dry-audit", "--dry-run"))
        state = read_json(self.paths("dry-audit") / "pilot_state.json")
        manifest = read_json(self.paths("dry-audit") / "pilot_manifest.json")
        report = read_json(self.paths("dry-audit") / "pilot_report.json")
        self.assertEqual(code, 0)
        client.assert_not_called()
        self.assertEqual(state["status"], "dry_run_complete")
        self.assertEqual(manifest["kind"], "dukeotr_real_model_pilot_manifest")
        self.assertEqual(manifest["status"], "dry_run_complete")
        self.assertEqual(report["model"]["preflight"]["status"], "not_run")
        self.assertEqual(report["evaluation_isolation"]["status"], "clear")
        self.assertEqual(report["sidecars"]["status"], "not_run_dry_run")

    def test_missing_model_preflight_refuses_before_brf_and_persists_actionable_state(self) -> None:
        self.write_tasks([pilot_task(1)])
        with patch(
            "scripts.run_real_model_pilot.OllamaClient.assert_model_present",
            side_effect=OllamaUnavailableError("`ollama list` completed, but 'qwen3:4b' is not registered"),
        ) as preflight, patch("scripts.run_real_model_pilot.run_builder_reviewer_fixer.run") as brf:
            code = main(self.command("missing-model"))
        state = read_json(self.paths("missing-model") / "pilot_state.json")
        report = read_json(self.paths("missing-model") / "pilot_report.json")
        self.assertEqual(code, 2)
        preflight.assert_called_once_with("qwen3:4b")
        brf.assert_not_called()
        self.assertEqual(state["status"], "blocked_preflight")
        self.assertEqual(state["model"]["preflight"]["required_command"], "ollama list")
        self.assertEqual(report["sidecars"]["status"], "not_run_preflight_failed")
        self.assertEqual(list((self.paths("missing-model") / "traces").glob("*.trace.json")), [])

    def test_v1_refuses_a_non_qwen3_4b_model_before_preflight(self) -> None:
        self.write_tasks([pilot_task(1)])
        with patch("scripts.run_real_model_pilot.OllamaClient.assert_model_present") as preflight:
            code = main(self.command("wrong-model", "--model", "another-local-tag"))
        self.assertEqual(code, 2)
        preflight.assert_not_called()
        self.assertFalse((self.paths("wrong-model") / "pilot_state.json").exists())

    def test_remote_host_is_refused_before_ollama_preflight_or_generation(self) -> None:
        self.write_tasks([pilot_task(1)])
        with patch("scripts.run_real_model_pilot.OllamaClient.assert_model_present") as preflight, patch(
            "scripts.run_real_model_pilot.run_builder_reviewer_fixer.run"
        ) as brf:
            code = main(self.command("remote-refusal", "--host", "https://example.invalid:11434"))
        state = read_json(self.paths("remote-refusal") / "pilot_state.json")
        report = read_json(self.paths("remote-refusal") / "pilot_report.json")
        self.assertEqual(code, 2)
        preflight.assert_not_called()
        brf.assert_not_called()
        self.assertEqual(state["status"], "blocked_preflight")
        self.assertEqual(report["sidecars"]["status"], "not_run_remote_host_refused")

    def test_duplicate_pilot_ids_refuse_before_preflight(self) -> None:
        self.write_tasks([pilot_task(1, task_id="pilot-duplicate"), pilot_task(2, task_id="pilot-duplicate")])
        with patch("scripts.run_real_model_pilot.OllamaClient.assert_model_present") as preflight:
            code = main(self.command("duplicate-ids"))
        self.assertEqual(code, 2)
        preflight.assert_not_called()
        self.assertFalse((self.paths("duplicate-ids") / "pilot_state.json").exists())

    def test_output_cannot_be_redirected_into_final_training_or_held_out_layers(self) -> None:
        self.write_tasks([pilot_task(1)])
        for destination in ("training_data/pilot-output-refusal", "evaluation_data/pilot-output-refusal"):
            with self.subTest(destination=destination), patch("scripts.run_real_model_pilot.OllamaClient.assert_model_present") as preflight:
                code = main(
                    [
                        "--run-id",
                        "protected-output",
                        "--tasks",
                        str(self.tasks_path),
                        "--evaluation",
                        str(self.evaluation_path),
                        "--output-root",
                        destination,
                    ]
                )
            self.assertEqual(code, 2)
            preflight.assert_not_called()
            self.assertFalse((Path(destination) / "protected-output").exists())

    def test_held_out_collision_refuses_without_model_context(self) -> None:
        prompt = "This is deliberately the exact held out wording for pilot isolation refusal."
        self.write_tasks([pilot_task(1, prompt=prompt)])
        self.evaluation_path.write_text(json.dumps({"id": "eval-private-001", "prompt": prompt}) + "\n", encoding="utf-8")
        with patch("scripts.run_real_model_pilot.OllamaClient.assert_model_present") as preflight:
            code = main(self.command("isolation-refusal"))
        self.assertEqual(code, 2)
        preflight.assert_not_called()
        self.assertFalse((self.paths("isolation-refusal") / "pilot_state.json").exists())

    def test_evaluation_id_collision_refuses_even_without_prompt_text_disclosure(self) -> None:
        self.write_tasks([pilot_task(1, task_id="pilot-shared-id")])
        self.evaluation_path.write_text(
            json.dumps({"id": "pilot-shared-id", "prompt": "Different held-out prompt text."}) + "\n",
            encoding="utf-8",
        )
        with patch("scripts.run_real_model_pilot.OllamaClient.assert_model_present") as preflight:
            code = main(self.command("evaluation-id-collision"))
        self.assertEqual(code, 2)
        preflight.assert_not_called()
        self.assertFalse((self.paths("evaluation-id-collision") / "pilot_state.json").exists())

    def test_sequential_partial_completion_resumes_without_repeating_completed_task(self) -> None:
        self.write_tasks([pilot_task(1), pilot_task(2)])
        calls: list[str] = []
        brf_arguments: list[object] = []

        def fake(arguments) -> int:
            calls.append(arguments.seed_id)
            brf_arguments.append(arguments)
            return accept_trace(arguments)

        with patch("scripts.run_real_model_pilot.OllamaClient.assert_model_present"), patch(
            "scripts.run_real_model_pilot.run_builder_reviewer_fixer.run", side_effect=fake
        ):
            first = main(self.command("resume", "--max-tasks", "1", "--timeout-seconds", "47"))
            second = main(self.command("resume", "--timeout-seconds", "47"))
        state = read_json(self.paths("resume") / "pilot_state.json")
        report = read_json(self.paths("resume") / "pilot_report.json")
        self.assertEqual(first, 0)
        self.assertEqual(second, 0)
        self.assertEqual(calls, ["pilot-offline-001", "pilot-offline-002"])
        self.assertTrue(all(item.builder_model == item.reviewer_model == item.fixer_model == "qwen3:4b" for item in brf_arguments))
        self.assertEqual(state["status"], "completed")
        self.assertEqual([item["attempts"] for item in state["tasks"]], [1, 1])
        self.assertEqual(report["run"]["completed_task_count"], 2)
        traces = sorted((self.paths("resume") / "traces").glob("*.trace.json"))
        self.assertEqual(len(traces), 2)
        self.assertEqual(read_json(traces[0])["configuration"]["timeout_seconds"], 47)

    def test_error_trace_preserves_actual_builder_output_but_requires_explicit_retry(self) -> None:
        self.write_tasks([pilot_task(1)])
        with patch("scripts.run_real_model_pilot.OllamaClient.assert_model_present"), patch(
            "scripts.run_real_model_pilot.run_builder_reviewer_fixer.run", side_effect=error_trace
        ) as brf:
            first = main(self.command("retry"))
            no_retry = main(self.command("retry"))
        state = read_json(self.paths("retry") / "pilot_state.json")
        raw = list(read_jsonl(self.paths("retry") / "raw_builder_candidates.jsonl"))
        self.assertEqual(first, 2)
        self.assertEqual(no_retry, 2)
        self.assertEqual(brf.call_count, 1)
        self.assertEqual(state["tasks"][0]["status"], "generation_failed")
        self.assertEqual(len(raw), 1)
        self.assertEqual(raw[0]["messages"][2]["content"], RAW_ANSWER)
        self.assertEqual(raw[0]["quality"]["llm_review"]["status"], "error")

        with patch("scripts.run_real_model_pilot.OllamaClient.assert_model_present"), patch(
            "scripts.run_real_model_pilot.run_builder_reviewer_fixer.run", side_effect=accept_trace
        ) as retry_brf:
            retried = main(self.command("retry", "--retry-failed"))
        state = read_json(self.paths("retry") / "pilot_state.json")
        self.assertEqual(retried, 0)
        self.assertEqual(retry_brf.call_count, 1)
        retry_raw = list(read_jsonl(self.paths("retry") / "raw_builder_candidates.jsonl"))
        retry_report = read_json(self.paths("retry") / "pilot_report.json")
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["tasks"][0]["attempts"], 2)
        self.assertEqual(len(state["tasks"][0]["trace_history"]), 2)
        self.assertEqual(len(state["tasks"][0]["attempt_history"]), 2)
        self.assertEqual(len(retry_raw), 2)
        self.assertEqual(retry_report["run"]["generation_failure_attempt_count"], 1)
        self.assertTrue((self.paths("retry") / "traces" / "pilot-offline-001.attempt-1.trace.json").exists())
        self.assertTrue((self.paths("retry") / "traces" / "pilot-offline-001.attempt-2.trace.json").exists())

    def test_interruption_is_marked_and_a_later_resume_uses_a_new_attempt(self) -> None:
        self.write_tasks([pilot_task(1)])

        def interrupted(_arguments) -> int:
            raise KeyboardInterrupt()

        with patch("scripts.run_real_model_pilot.OllamaClient.assert_model_present"), patch(
            "scripts.run_real_model_pilot.run_builder_reviewer_fixer.run", side_effect=interrupted
        ):
            stopped = main(self.command("interrupt"))
        state = read_json(self.paths("interrupt") / "pilot_state.json")
        self.assertEqual(stopped, 130)
        self.assertEqual(state["tasks"][0]["status"], "interrupted")
        self.assertEqual(state["tasks"][0]["attempts"], 1)

        with patch("scripts.run_real_model_pilot.OllamaClient.assert_model_present"), patch(
            "scripts.run_real_model_pilot.run_builder_reviewer_fixer.run", side_effect=accept_trace
        ):
            resumed = main(self.command("interrupt"))
        state = read_json(self.paths("interrupt") / "pilot_state.json")
        self.assertEqual(resumed, 0)
        self.assertEqual(state["tasks"][0]["status"], "completed")
        self.assertEqual(state["tasks"][0]["attempts"], 2)
        self.assertTrue((self.paths("interrupt") / "traces" / "pilot-offline-001.attempt-2.trace.json").exists())

    def test_resume_reuses_terminal_trace_written_before_interruption_without_second_sampling(self) -> None:
        self.write_tasks([pilot_task(1)])

        def trace_then_interrupt(arguments) -> int:
            accept_trace(arguments)
            raise KeyboardInterrupt()

        with patch("scripts.run_real_model_pilot.OllamaClient.assert_model_present"), patch(
            "scripts.run_real_model_pilot.run_builder_reviewer_fixer.run", side_effect=trace_then_interrupt
        ):
            stopped = main(self.command("recover-terminal"))
        self.assertEqual(stopped, 130)
        with patch("scripts.run_real_model_pilot.OllamaClient.assert_model_present") as preflight, patch(
            "scripts.run_real_model_pilot.run_builder_reviewer_fixer.run"
        ) as brf:
            resumed = main(self.command("recover-terminal"))
        state = read_json(self.paths("recover-terminal") / "pilot_state.json")
        raw = list(read_jsonl(self.paths("recover-terminal") / "raw_builder_candidates.jsonl"))
        self.assertEqual(resumed, 0)
        preflight.assert_not_called()
        brf.assert_not_called()
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["tasks"][0]["attempts"], 1)
        self.assertEqual(len(raw), 1)

    def test_malformed_trace_is_a_generation_failure_and_never_fabricates_a_candidate(self) -> None:
        self.write_tasks([pilot_task(1)])

        def malformed(arguments) -> int:
            write_json_atomic(arguments.output, {"status": "accepted_for_manual_review_only", "rounds": "not-a-list"})
            return 0

        with patch("scripts.run_real_model_pilot.OllamaClient.assert_model_present"), patch(
            "scripts.run_real_model_pilot.run_builder_reviewer_fixer.run", side_effect=malformed
        ):
            code = main(self.command("malformed"))
        state = read_json(self.paths("malformed") / "pilot_state.json")
        raw = list(read_jsonl(self.paths("malformed") / "raw_builder_candidates.jsonl"))
        report = read_json(self.paths("malformed") / "pilot_report.json")
        self.assertEqual(code, 2)
        self.assertEqual(state["tasks"][0]["status"], "generation_failed")
        self.assertEqual(raw, [])
        self.assertTrue(any("materialization" in message.lower() or "rounds" in message.lower() for message in report["materialization_errors"]))

    def test_simple_route_reviewer_skip_is_processed_but_stays_training_ineligible(self) -> None:
        self.write_tasks([pilot_task(1)])
        with patch("scripts.run_real_model_pilot.OllamaClient.assert_model_present"), patch(
            "scripts.run_real_model_pilot.run_builder_reviewer_fixer.run", side_effect=skipped_reviewer_trace
        ):
            code = main(self.command("reviewer-skip"))
        root = self.paths("reviewer-skip")
        final = list(read_jsonl(root / "final_candidates.jsonl"))
        eligible = list(read_jsonl(root / "training_eligible_candidates.jsonl"))
        report = read_json(root / "pilot_report.json")
        self.assertEqual(code, 0)
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0]["quality"]["llm_review"]["status"], "skipped")
        self.assertEqual(eligible, [])
        self.assertIn("no_accepting_reviewer", next(iter(report["artifact_states"]["training_eligible_output"]["ineligible_reasons_by_record"].values())))

    def test_correction_failure_lineage_and_eligibility_are_evidence_based_not_auto_promotion(self) -> None:
        self.write_tasks([pilot_task(1)])
        with patch("scripts.run_real_model_pilot.OllamaClient.assert_model_present"), patch(
            "scripts.run_real_model_pilot.run_builder_reviewer_fixer.run", side_effect=corrected_failure_trace
        ):
            code = main(self.command("lineage"))
        root = self.paths("lineage")
        raw = list(read_jsonl(root / "raw_builder_candidates.jsonl"))
        corrected = list(read_jsonl(root / "corrected_candidates.jsonl"))
        final = list(read_jsonl(root / "final_candidates.jsonl"))
        eligible = list(read_jsonl(root / "training_eligible_candidates.jsonl"))
        linkable = list(read_jsonl(root / "linkable_corrections.jsonl"))
        failures = list(read_jsonl(root / "observed_failures.jsonl"))
        ledger = list(read_jsonl(root / "training_factory_ledger.jsonl"))
        report = read_json(root / "pilot_report.json")
        self.assertEqual(code, 0)
        self.assertEqual(len(raw), 1)
        self.assertEqual(raw[0]["metadata"]["pilot"]["candidate_state"], "real_model_pilot_raw_builder")
        self.assertEqual(raw[0]["metadata"]["pilot"]["model_provenance"]["provider"], "local_ollama_http_api")
        self.assertEqual(len(raw[0]["metadata"]["pilot"]["trace_sha256"]), 64)
        self.assertEqual(len(corrected), 1)
        self.assertEqual(corrected[0]["metadata"]["parent_record_id"], raw[0]["record_id"])
        self.assertEqual(corrected[0]["metadata"]["stage"], "corrected")
        self.assertEqual(final[0]["messages"][2]["content"], FIXED_ANSWER)
        self.assertEqual(len(eligible), 1)
        self.assertEqual(eligible[0]["metadata"]["pilot"]["model_provenance"]["model_tag"], "qwen3:4b")
        self.assertEqual(linkable[0]["record_id"], corrected[0]["record_id"])
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["correction"]["corrected_record_id"], corrected[0]["record_id"])
        self.assertEqual(failures[0]["verification"]["status"], "corrected_and_revalidated")
        self.assertEqual(len(ledger), 2)
        raw_ledger = next(row for row in ledger if row["record_id"] == raw[0]["record_id"])
        self.assertEqual(raw_ledger["correction"]["corrected_record_id"], corrected[0]["record_id"])
        self.assertEqual(report["promotion"]["status"], "prohibited")
        self.assertIn("never automatic dataset promotion", report["artifact_states"]["training_eligible_output"]["meaning"])
        self.assertFalse((self.root / "training_data").exists())

    def test_empty_fixer_explanation_is_preserved_unlinked_and_never_crashes_sidecars(self) -> None:
        self.write_tasks([pilot_task(1)])
        with patch("scripts.run_real_model_pilot.OllamaClient.assert_model_present"), patch(
            "scripts.run_real_model_pilot.run_builder_reviewer_fixer.run", side_effect=empty_correction_explanation_trace
        ):
            code = main(self.command("empty-correction"))
        root = self.paths("empty-correction")
        corrected = list(read_jsonl(root / "corrected_candidates.jsonl"))
        final = list(read_jsonl(root / "final_candidates.jsonl"))
        linkable = list(read_jsonl(root / "linkable_corrections.jsonl"))
        deduplicated = list(read_jsonl(root / "deduplicated_candidates.jsonl"))
        eligible = list(read_jsonl(root / "training_eligible_candidates.jsonl"))
        ledger = list(read_jsonl(root / "training_factory_ledger.jsonl"))
        linkage = read_json(root / "correction_linkage_report.json")
        report = read_json(root / "pilot_report.json")
        self.assertEqual(code, 0)
        self.assertEqual(len(corrected), 1)
        self.assertEqual(corrected[0]["metadata"]["correction"]["changes_made"], [])
        self.assertEqual(corrected[0]["metadata"]["pilot"]["correction_evidence"]["status"], "ineligible_missing_actual_explanation")
        self.assertEqual(corrected[0]["quality"]["correction_evidence"]["status"], "missing_actual_explanation")
        self.assertEqual(len(final), 1)
        self.assertEqual(linkable, [])
        self.assertEqual(deduplicated, [])
        self.assertEqual(eligible, [])
        self.assertEqual(len(ledger), 2)
        self.assertTrue(all(row["correction"]["corrected_record_id"] is None for row in ledger))
        self.assertEqual(linkage["unlinked_record_count"], 1)
        self.assertEqual(linkage["terminal_records_excluded_from_deduplication"][0]["record_id"], final[0]["record_id"])
        self.assertEqual(
            report["artifact_states"]["training_eligible_output"]["ineligible_reasons_by_record"][final[0]["record_id"]],
            ["correction_explanation_missing_actual_evidence"],
        )
        self.assertEqual(report["sidecars"]["failure_analysis"]["status"], "complete")

    def test_sidecar_repair_refuses_a_changed_held_out_evaluation_fingerprint(self) -> None:
        self.write_tasks([pilot_task(1)])
        with patch("scripts.run_real_model_pilot.OllamaClient.assert_model_present"), patch(
            "scripts.run_real_model_pilot.run_builder_reviewer_fixer.run", side_effect=accept_trace
        ):
            self.assertEqual(main(self.command("repair-eval-fingerprint")), 0)
        root = self.paths("repair-eval-fingerprint")
        raw_path = root / "raw_builder_candidates.jsonl"
        raw_hash = sha256_file(raw_path)
        self.evaluation_path.write_text(
            json.dumps({"id": "eval-new", "prompt": "A held-out cardinal-direction puzzle with unrelated wording."}) + "\n",
            encoding="utf-8",
        )
        with patch("scripts.run_real_model_pilot.OllamaClient") as client, patch(
            "scripts.run_real_model_pilot.run_builder_reviewer_fixer.run"
        ) as brf:
            code = main(self.command("repair-eval-fingerprint", "--repair-sidecars"))
        self.assertEqual(code, 2)
        client.assert_not_called()
        brf.assert_not_called()
        self.assertEqual(sha256_file(raw_path), raw_hash)
        self.assertFalse((root / "repair_snapshots").exists())

    def test_sidecar_repair_is_no_model_audited_and_preserves_legacy_source_artifacts(self) -> None:
        self.write_tasks([pilot_task(1)])
        with patch("scripts.run_real_model_pilot.OllamaClient.assert_model_present"), patch(
            "scripts.run_real_model_pilot.run_builder_reviewer_fixer.run", side_effect=empty_correction_explanation_trace
        ):
            initial = main(self.command("repair-legacy"))
        self.assertEqual(initial, 0)
        root = self.paths("repair-legacy")

        # Simulate the pre-fix persisted artifact shape from the Windows run: it contains an
        # actual empty Fixer explanation but no newer quality-hold metadata. Do not alter trace.
        final_path = root / "final_candidates.jsonl"
        final = list(read_jsonl(final_path))
        final[0]["quality"].pop("correction_evidence", None)
        write_jsonl_atomic(final_path, final)
        state_path = root / "pilot_state.json"
        state = read_json(state_path)
        state["status"] = "sidecar_error"
        state["pipeline_version"]["pilot_runner"]["sha256"] = "legacy-pilot-runner-hash"
        write_json_atomic(state_path, state)
        write_json_atomic(root / "pilot_manifest.json", {**state, "kind": "dukeotr_real_model_pilot_manifest"})
        old_report = read_json(root / "pilot_report.json")
        old_report["materialization_errors"] = ["Training Factory sidecar error: legacy empty correction evidence"]
        write_json_atomic(root / "pilot_report.json", old_report)
        source_paths = (root / "raw_builder_candidates.jsonl", root / "corrected_candidates.jsonl", final_path)
        source_hashes = {str(path): sha256_file(path) for path in source_paths}

        # The normal generation resume still refuses a changed pipeline fingerprint. The repair
        # path is the only deliberate exception and it runs no model-facing code.
        with patch("scripts.run_real_model_pilot.OllamaClient") as client, patch(
            "scripts.run_real_model_pilot.run_builder_reviewer_fixer.run"
        ) as brf:
            blocked = main(self.command("repair-legacy"))
            repaired = main(self.command("repair-legacy", "--repair-sidecars"))
        self.assertEqual(blocked, 2)
        self.assertEqual(repaired, 0)
        client.assert_not_called()
        brf.assert_not_called()
        self.assertEqual({str(path): sha256_file(path) for path in source_paths}, source_hashes)

        repaired_state = read_json(state_path)
        repaired_report = read_json(root / "pilot_report.json")
        self.assertEqual(repaired_state["status"], "completed")
        self.assertEqual(len(repaired_state["sidecar_repair_history"]), 1)
        repair_event = repaired_state["sidecar_repair_history"][0]
        self.assertEqual(repair_event["status"], "complete")
        self.assertTrue(repair_event["pipeline_fingerprint_mismatch_allowed_only_for_no_model_repair"])
        self.assertEqual(repair_event["immutable_trace_hashes_after"], repair_event["immutable_trace_hashes"])
        snapshot = Path(repair_event["snapshot_directory"])
        self.assertTrue((snapshot / "snapshot_manifest.json").exists())
        self.assertEqual(read_json(snapshot / "pilot_state.json")["status"], "sidecar_error")
        self.assertEqual(repaired_report["sidecars"]["correction_linkage"]["unlinked_record_count"], 1)
        self.assertEqual(repaired_report["artifact_states"]["verified_output"]["count"], 0)
        self.assertEqual(repaired_report["sidecar_repair_history"][0]["prior_state"], "sidecar_error")


if __name__ == "__main__":
    unittest.main()
