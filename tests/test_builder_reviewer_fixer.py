from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from scripts.lib.builder_reviewer_fixer import parse_fixer, parse_reviewer
from scripts.lib.effort_routing import EffortRoute
from scripts.lib.io_utils import read_json
from scripts.lib.ollama import OllamaResponse
from scripts.run_builder_reviewer_fixer import main


ANSWER = """A local variable belongs to the block where it is declared, which limits accidental reuse.
`nil` means no value is currently stored; it is not the same thing as `false`.

```luau
local playerName = "Ari"
local score = 12
local isReady = true
local pendingReward = nil

if isReady then
    local label = playerName .. " has " .. score
    print(label, pendingReward)
end
```

A global-looking assignment can be visible more broadly, so a local is usually safer when the
value is only needed by this script or block. Globals are not automatically forbidden, but they
need deliberate ownership and naming."""


def builder_response() -> str:
    return json.dumps(
        {
            "assistant_response": ANSWER,
            "assumptions": ["The lesson is intentionally pure Luau."],
            "security_notes": [],
            "test_plan": ["Run the snippet and inspect the printed values."],
            "code_book_card_ids": ["cb-luau-types-tables-functions"],
        }
    )


def accept_review() -> str:
    return json.dumps(
        {
            "decision": "accept",
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
            "summary": "The answer is scoped, accurate, and meets the pure-Luau brief.",
        }
    )


def revise_review() -> str:
    return json.dumps(
        {
            "decision": "revise",
            "dimension_scores": {
                "correctness": 5,
                "security": 5,
                "api_validity": 5,
                "requirements": 5,
                "english": 2,
                "code_quality": 4,
            },
            "findings": [
                {
                    "id": "english-1",
                    "category": "english",
                    "severity": "major",
                    "message": "Clarify the relationship between nil and false.",
                    "evidence": "The first sentence is terse.",
                    "required_fix": "State explicitly that nil and false are distinct values.",
                }
            ],
            "api_claims_to_verify": [],
            "summary": "One language clarity correction is needed.",
        }
    )


def complex_route() -> EffortRoute:
    return EffortRoute(
        level="complex",
        reasons=("high_risk:client_server_security",),
        risk_labels=("client_server_security",),
        selected_checks=("structure_and_safety", "requirements", "english_clarity", "luau_code", "roblox_api", "client_server_security", "testability"),
        skipped_checks=("persistence_economy", "lifecycle_performance"),
        reviewer_passes=2,
        default_max_rounds=3,
        maximum_rounds=4,
        minimum_assistant_characters=220,
        code_book_context_limit=6,
    )


def simple_route() -> EffortRoute:
    return EffortRoute(
        level="simple",
        reasons=("narrow_low_risk_simple_signal",),
        risk_labels=(),
        selected_checks=("structure_and_safety",),
        skipped_checks=("requirements", "english_clarity", "luau_code", "roblox_api", "client_server_security", "persistence_economy", "lifecycle_performance", "testability"),
        reviewer_passes=0,
        default_max_rounds=1,
        maximum_rounds=1,
        minimum_assistant_characters=24,
        code_book_context_limit=0,
    )


class BuilderReviewerFixerTests(unittest.TestCase):
    def test_dry_run_writes_isolated_non_inference_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plan.json"
            with patch("scripts.run_builder_reviewer_fixer.OllamaClient") as client:
                code = main(["--seed-id", "dukeotr-phase1-001", "--dry-run", "--output", str(output)])
            trace = read_json(output)
        self.assertEqual(code, 0)
        self.assertEqual(trace["status"], "planned_no_inference")
        self.assertEqual(trace["evaluation_isolation"]["status"], "clear")
        self.assertEqual(trace["promotion"]["status"], "prohibited")
        self.assertEqual(trace["rounds"], [])
        client.assert_not_called()

    @patch("scripts.run_builder_reviewer_fixer.OllamaClient.assert_model_present")
    @patch("scripts.run_builder_reviewer_fixer.OllamaClient.generate")
    def test_simple_fast_path_skips_reviewer_and_fixer_after_tester_pass(self, generate_mock, preflight_mock) -> None:
        generate_mock.return_value = OllamaResponse(content=builder_response(), raw={}, elapsed_seconds=0.1)
        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.run_builder_reviewer_fixer.classify_task", return_value=simple_route()
        ):
            output = Path(directory) / "trace.json"
            code = main(["--seed-id", "dukeotr-phase1-001", "--output", str(output)])
            trace = read_json(output)
        self.assertEqual(code, 0)
        self.assertEqual(trace["routing"]["level"], "simple")
        self.assertNotIn("roblox_api", trace["routing"]["selected_checks"])
        self.assertEqual(trace["code_book"]["retrieved_cards"], [])
        self.assertEqual(trace["rounds"][0]["tester"]["status"], "pass")
        self.assertEqual(trace["rounds"][0]["reviewer"]["status"], "skipped")
        self.assertTrue(trace["rounds"][0]["early_pass"])
        self.assertNotIn("fixer", trace["rounds"][0])
        self.assertEqual(generate_mock.call_count, 1)
        preflight_mock.assert_called_once_with("qwen3:4b")
        self.assertEqual(trace["promotion"]["status"], "prohibited")

    @patch("scripts.run_builder_reviewer_fixer.OllamaClient.assert_model_present")
    @patch("scripts.run_builder_reviewer_fixer.OllamaClient.generate")
    def test_complex_route_uses_deeper_independent_reviewer_passes(self, generate_mock, preflight_mock) -> None:
        generate_mock.side_effect = [
            OllamaResponse(content=builder_response(), raw={}, elapsed_seconds=0.1),
            OllamaResponse(content=accept_review(), raw={}, elapsed_seconds=0.1),
            OllamaResponse(content=accept_review(), raw={}, elapsed_seconds=0.1),
        ]
        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.run_builder_reviewer_fixer.classify_task", return_value=complex_route()
        ):
            output = Path(directory) / "trace.json"
            code = main(["--seed-id", "dukeotr-phase1-001", "--output", str(output)])
            trace = read_json(output)
        self.assertEqual(code, 0)
        reviewer = trace["rounds"][0]["reviewer"]
        self.assertEqual(trace["routing"]["level"], "complex")
        self.assertEqual(reviewer["requested_passes"], 2)
        self.assertEqual(reviewer["completed_passes"], 2)
        self.assertEqual(len(reviewer["passes"]), 2)
        self.assertIn("client_server_security", reviewer["selected_checks"])
        self.assertTrue(trace["rounds"][0]["early_pass"])
        self.assertEqual(generate_mock.call_count, 3)
        preflight_mock.assert_called_once_with("qwen3:4b")

    @patch("scripts.run_builder_reviewer_fixer.OllamaClient.assert_model_present")
    @patch("scripts.run_builder_reviewer_fixer.OllamaClient.generate")
    def test_complex_repair_loop_is_bounded_after_retesting(self, generate_mock, preflight_mock) -> None:
        fixed_answer = ANSWER.replace(
            "`nil` means no value is currently stored; it is not the same thing as `false`.",
            "`nil` means no value is currently stored, while `false` is a boolean value; they are distinct values.",
        )
        fixer = json.dumps(
            {
                "assistant_response": fixed_answer,
                "changes_made": ["Made the nil versus false distinction explicit."],
                "remaining_assumptions": [],
                "unresolved_risks": [],
                "addressed_finding_ids": ["r1-english-1", "r2-english-1"],
            }
        )
        generate_mock.side_effect = [
            OllamaResponse(content=builder_response(), raw={}, elapsed_seconds=0.1),
            OllamaResponse(content=revise_review(), raw={}, elapsed_seconds=0.1),
            OllamaResponse(content=revise_review(), raw={}, elapsed_seconds=0.1),
            OllamaResponse(content=fixer, raw={}, elapsed_seconds=0.1),
            OllamaResponse(content=revise_review(), raw={}, elapsed_seconds=0.1),
            OllamaResponse(content=revise_review(), raw={}, elapsed_seconds=0.1),
        ]
        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.run_builder_reviewer_fixer.classify_task", return_value=complex_route()
        ):
            output = Path(directory) / "trace.json"
            code = main(["--seed-id", "dukeotr-phase1-001", "--output", str(output), "--max-rounds", "2"])
            trace = read_json(output)
        self.assertEqual(code, 0)
        self.assertEqual(trace["status"], "needs_human_review")
        self.assertEqual(len(trace["rounds"]), 2)
        self.assertIn("fixer", trace["rounds"][0])
        self.assertNotIn("fixer", trace["rounds"][1])
        self.assertEqual(trace["rounds"][1]["reviewer"]["completed_passes"], 2)
        self.assertEqual(generate_mock.call_count, 6)
        preflight_mock.assert_called_once_with("qwen3:4b")

    def test_evaluation_collision_blocks_before_any_generation_or_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evaluation = Path(directory) / "held_out.jsonl"
            output = Path(directory) / "trace.json"
            evaluation.write_text(
                json.dumps(
                    {
                        "id": "heldout-collision",
                        "prompt": "Teach a new Luau learner the difference between a local variable, a global-looking assignment, and nil. Use a tiny score-and-player-name example, then answer why a local is usually safer.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with patch("scripts.run_builder_reviewer_fixer.OllamaClient") as client:
                code = main(
                    [
                        "--seed-id",
                        "dukeotr-phase1-001",
                        "--evaluation",
                        str(evaluation),
                        "--dry-run",
                        "--output",
                        str(output),
                    ]
                )
            trace = read_json(output)
        self.assertEqual(code, 2)
        self.assertEqual(trace["status"], "blocked_by_evaluation_isolation")
        self.assertEqual(trace["promotion"]["status"], "prohibited")
        client.assert_not_called()

    @patch("scripts.run_builder_reviewer_fixer.OllamaClient.assert_model_present")
    @patch("scripts.run_builder_reviewer_fixer.OllamaClient.generate")
    def test_accepting_trace_remains_manual_review_only(self, generate_mock, preflight_mock) -> None:
        generate_mock.side_effect = [
            OllamaResponse(content=builder_response(), raw={}, elapsed_seconds=0.1),
            OllamaResponse(content=accept_review(), raw={}, elapsed_seconds=0.1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trace.json"
            code = main(["--seed-id", "dukeotr-phase1-001", "--output", str(output)])
            trace = read_json(output)
        self.assertEqual(code, 0)
        self.assertEqual(trace["status"], "accepted_for_manual_review_only")
        self.assertEqual(trace["promotion"]["status"], "prohibited")
        self.assertEqual(len(trace["rounds"]), 1)
        self.assertEqual(trace["rounds"][0]["effective_decision"], "accept")
        self.assertEqual(trace["final_candidate"]["metadata"]["stage"], "brf_candidate")
        self.assertEqual(generate_mock.call_count, 2)
        preflight_mock.assert_called_once_with("qwen3:4b")

    @patch("scripts.run_builder_reviewer_fixer.OllamaClient.assert_model_present")
    @patch("scripts.run_builder_reviewer_fixer.OllamaClient.generate")
    def test_early_pass_does_not_preflight_or_invoke_a_distinct_fixer(self, generate_mock, preflight_mock) -> None:
        generate_mock.side_effect = [
            OllamaResponse(content=builder_response(), raw={}, elapsed_seconds=0.1),
            OllamaResponse(content=accept_review(), raw={}, elapsed_seconds=0.1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trace.json"
            code = main(
                [
                    "--seed-id",
                    "dukeotr-phase1-001",
                    "--builder-model",
                    "builder-test",
                    "--reviewer-model",
                    "reviewer-test",
                    "--fixer-model",
                    "fixer-test",
                    "--output",
                    str(output),
                ]
            )
            trace = read_json(output)
        self.assertEqual(code, 0)
        self.assertTrue(trace["rounds"][0]["early_pass"])
        self.assertEqual(generate_mock.call_count, 2)
        self.assertEqual([call.args[0] for call in preflight_mock.call_args_list], ["builder-test", "reviewer-test"])
        self.assertNotIn("fixer", trace["rounds"][0])

    @patch("scripts.run_builder_reviewer_fixer.OllamaClient.assert_model_present")
    @patch("scripts.run_builder_reviewer_fixer.OllamaClient.generate")
    def test_revise_is_fixed_then_independently_reviewed_again(self, generate_mock, preflight_mock) -> None:
        revise_review = json.dumps(
            {
                "decision": "revise",
                "dimension_scores": {
                    "correctness": 5,
                    "security": 5,
                    "api_validity": 5,
                    "requirements": 5,
                    "english": 2,
                    "code_quality": 4,
                },
                "findings": [
                    {
                        "id": "english-1",
                        "category": "english",
                        "severity": "major",
                        "message": "Clarify the relationship between nil and false.",
                        "evidence": "The first sentence is terse.",
                        "required_fix": "State explicitly that nil and false are distinct values.",
                    }
                ],
                "api_claims_to_verify": [],
                "summary": "One language clarity correction is needed.",
            }
        )
        fixed_answer = ANSWER.replace("`nil` means no value is currently stored; it is not the same thing as `false`.", "`nil` means no value is currently stored, while `false` is a boolean value; they are distinct values.")
        fixer = json.dumps(
            {
                "assistant_response": fixed_answer,
                "changes_made": ["Made the nil versus false distinction explicit."],
                "remaining_assumptions": ["The learner can run a basic Luau snippet."],
                "unresolved_risks": [],
                "addressed_finding_ids": ["english-1"],
            }
        )
        generate_mock.side_effect = [
            OllamaResponse(content=builder_response(), raw={}, elapsed_seconds=0.1),
            OllamaResponse(content=revise_review, raw={}, elapsed_seconds=0.1),
            OllamaResponse(content=fixer, raw={}, elapsed_seconds=0.1),
            OllamaResponse(content=accept_review(), raw={}, elapsed_seconds=0.1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trace.json"
            code = main(["--seed-id", "dukeotr-phase1-001", "--output", str(output), "--max-rounds", "2"])
            trace = read_json(output)
        self.assertEqual(code, 0)
        self.assertEqual(trace["status"], "accepted_for_manual_review_only")
        self.assertEqual(len(trace["rounds"]), 2)
        self.assertEqual(trace["rounds"][0]["effective_decision"], "revise")
        self.assertEqual(trace["rounds"][0]["fixer"]["addressed_finding_ids"], ["english-1"])
        self.assertEqual(trace["rounds"][1]["effective_decision"], "accept")
        self.assertEqual(trace["final_candidate"]["metadata"]["correction_round"], 1)
        self.assertEqual(generate_mock.call_count, 4)
        preflight_mock.assert_called_once_with("qwen3:4b")

    def test_fixer_must_explicitly_address_every_major_or_block_finding(self) -> None:
        with self.assertRaisesRegex(ValueError, "every reviewer block/major"):
            parse_fixer(
                json.dumps(
                    {
                        "assistant_response": "A corrected answer.",
                        "changes_made": [],
                        "remaining_assumptions": [],
                        "unresolved_risks": ["Needs product decision."],
                        "addressed_finding_ids": [],
                    }
                ),
                reviewer_findings=[
                    {
                        "id": "security-1",
                        "category": "security",
                        "severity": "block",
                        "message": "Client input is trusted.",
                        "evidence": "amount from remote",
                        "required_fix": "Validate and decide on the server.",
                    }
                ],
            )

    def test_reviewer_accept_is_downgraded_when_major_finding_exists(self) -> None:
        review = parse_reviewer(
            json.dumps(
                {
                    "decision": "accept",
                    "dimension_scores": {
                        "correctness": 5,
                        "security": 5,
                        "api_validity": 5,
                        "requirements": 5,
                        "english": 5,
                        "code_quality": 5,
                    },
                    "findings": [
                        {
                            "id": "api-1",
                            "category": "api",
                            "severity": "major",
                            "message": "The API name requires correction.",
                            "evidence": "UnknownService",
                            "required_fix": "Use a documented service name.",
                        }
                    ],
                    "api_claims_to_verify": ["UnknownService"],
                    "summary": "Contradictory accept should be downgraded.",
                }
            ),
            minimums={
                "correctness": 4,
                "security": 4,
                "api_validity": 4,
                "requirements": 4,
                "english": 3,
                "code_quality": 3,
            },
        )
        self.assertEqual(review["reported_decision"], "accept")
        self.assertEqual(review["decision"], "revise")
        self.assertIn("accept_downgraded_due_to_block_or_major_finding", review["policy_forced_reasons"])


if __name__ == "__main__":
    unittest.main()
