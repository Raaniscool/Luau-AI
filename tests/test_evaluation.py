from __future__ import annotations

import unittest

from scripts.lib.evaluation import deterministic_regression_flags
from scripts.lib.io_utils import read_jsonl
from scripts.score_evaluation import apply_deterministic_regression_guard, parse_score


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
