from __future__ import annotations

import unittest

from scripts.score_evaluation import parse_score


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


if __name__ == "__main__":
    unittest.main()
