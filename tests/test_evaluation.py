from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.lib.evaluation import deterministic_regression_flags
from scripts.lib.io_utils import read_jsonl
from scripts.lib.ollama import OllamaResponse
from scripts.score_evaluation import apply_deterministic_regression_guard, main as score_main, parse_score


class EvaluationTests(unittest.TestCase):
    def test_score_requires_all_rubric_criteria_and_recomputes_total(self) -> None:
        task = {
            "rubric": [
                {"id": "a", "description": "a", "max_points": 40},
                {"id": "b", "description": "b", "max_points": 60},
            ]
        }
        parsed = parse_score(
            '{"overall_score":99,"criterion_scores":[{"id":"a","points":30,"max_points":40,"evidence":"x"},{"id":"b","points":50,"max_points":60,"evidence":"y"}],"critical_failures":[],"missing_requirements":[],"strengths":[],"verdict":"pass"}',
            task,
        )
        self.assertEqual(parsed["overall_score"], 80.0)
        self.assertEqual(parsed["reported_overall_score"], 99.0)

    @patch("scripts.score_evaluation.OllamaClient.generate")
    @patch("scripts.score_evaluation.OllamaClient.assert_model_present")
    def test_score_requests_json_mode_and_preserves_an_unparseable_judge_response(self, preflight_mock, generate_mock) -> None:
        generate_mock.return_value = OllamaResponse(content="judge explanation without JSON", raw={}, elapsed_seconds=0.25)
        task = next(item for item in read_jsonl("evaluation_data/roblox_luau_eval.jsonl") if item["id"] == "eval-remoteevent-secure-001")
        answer_record = {
            "schema_version": "1.0",
            "run_id": "test-run",
            "run_kind": "baseline",
            "task_id": task["id"],
            "model": "qwen3:4b",
            "generation_options": {},
            "status": "complete",
            "answer": "RemoteEvents are inherently secure, and clients cannot call RemoteEvent:FireServer.",
            "error": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            answers = Path(directory) / "answers.jsonl"
            output = Path(directory) / "scores.jsonl"
            answers.write_text(json.dumps(answer_record) + "\n", encoding="utf-8")
            code = score_main(["--answers", str(answers), "--output", str(output)])
            self.assertEqual(code, 2)
            score_record = next(read_jsonl(output))
        self.assertEqual(score_record["status"], "error")
        self.assertEqual(score_record["judge_response"], "judge explanation without JSON")
        self.assertEqual(len(score_record["deterministic_regression_flags"]), 2)
        self.assertEqual(generate_mock.call_args.kwargs["response_format"], "json")
        preflight_mock.assert_called_once_with("qwen3:4b")

    def test_remoteevent_baseline_task_rejects_known_factual_errors(self) -> None:
        task = next(item for item in read_jsonl("evaluation_data/roblox_luau_eval.jsonl") if item["id"] == "eval-remoteevent-secure-001")
        must_not = " ".join(task["must_not"])
        self.assertIn("cannot use RemoteEvent:FireServer", must_not)
        self.assertIn("automatically validates or secures", must_not)
        self.assertIn("server must call RemoteEvent:FireServer", must_not)
        self.assertIn("baseline_regression_focus", task)
        flags = deterministic_regression_flags(
            task,
            "RemoteEvents are inherently secure, and clients cannot call RemoteEvent:FireServer.",
        )
        self.assertEqual(len(flags), 2)
        judge_score = {"verdict": "pass", "critical_failures": []}
        applied_flags = apply_deterministic_regression_guard(
            task,
            "RemoteEvents are inherently secure, and clients cannot call RemoteEvent:FireServer.",
            judge_score,
        )
        self.assertEqual(applied_flags, flags)
        self.assertEqual(judge_score["verdict"], "fail")
        self.assertEqual(len(judge_score["critical_failures"]), 2)
        variant_flags = deterministic_regression_flags(
            task,
            "Only the server can call FireServer, and a RemoteEvent automatically validates client input.",
        )
        self.assertEqual(len(variant_flags), 2)
        baseline_style_flags = deterministic_regression_flags(
            task,
            "A RemoteEvent is a secure, server-controlled mechanism. The server must explicitly fire "
            "RemoteEvent:FireServer, and RemoteEvents only let the server trigger actions.",
        )
        self.assertEqual(len(baseline_style_flags), 2)
        corrected_flags = deterministic_regression_flags(
            task,
            "It is incorrect to say clients cannot call RemoteEvent:FireServer. RemoteEvents are not inherently secure.",
        )
        self.assertEqual(corrected_flags, [])


if __name__ == "__main__":
    unittest.main()
