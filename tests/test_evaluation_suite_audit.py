from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_evaluation_suite import main
from scripts.lib.io_utils import read_json


class EvaluationSuiteAuditTests(unittest.TestCase):
    def test_mature_held_out_suite_passes_all_coverage_and_taxonomy_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation_audit.json"
            code = main(["--strict", "--require-mature-target", "--output", str(output)])
            report = read_json(output)
        self.assertEqual(code, 0)
        self.assertEqual(report["project_name"], "DukeOTR")
        self.assertGreaterEqual(report["task_count"], 100)
        self.assertEqual(report["current_floor_status"], "pass")
        self.assertEqual(report["mature_suite_status"], "mature_target_reached")
        self.assertEqual(report["remaining_to_mature_minimum"], 0)
        self.assertEqual(report["missing_mature_tracks"], [])
        self.assertEqual(report["missing_authoring_forms"], [])
        self.assertEqual(report["missing_response_depths"], [])
        self.assertEqual(report["missing_difficulties"], [])
        self.assertEqual(report["tasks_missing_taxonomy_metadata"], [])
        self.assertGreater(report["response_depth_counts"]["short"], 0)
        self.assertGreater(report["response_depth_counts"]["deep"], 0)
        self.assertEqual(report["suite_policy"], "permanently_held_out_not_generation_or_training_input")

    def test_strict_mode_catches_a_floor_regression_without_scoring_a_model(self) -> None:
        plan = read_json("evaluation_data/coverage_plan.json")
        plan["current_suite_floor"] = 101
        with tempfile.TemporaryDirectory() as directory:
            plan_path = Path(directory) / "coverage_plan.json"
            output = Path(directory) / "evaluation_audit.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            code = main(["--plan", str(plan_path), "--strict", "--output", str(output)])
            report = read_json(output)
        self.assertEqual(code, 2)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["current_floor_status"], "fail")
        self.assertEqual(report["mature_suite_status"], "mature_target_reached")


if __name__ == "__main__":
    unittest.main()
