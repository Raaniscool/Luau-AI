from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_evaluation_suite import main
from scripts.lib.io_utils import read_json


class EvaluationSuiteAuditTests(unittest.TestCase):
    def test_current_suite_passes_floor_without_claiming_mature_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation_audit.json"
            code = main(["--strict", "--output", str(output)])
            report = read_json(output)
        self.assertEqual(code, 0)
        self.assertEqual(report["project_name"], "DukeOTR")
        self.assertEqual(report["task_count"], 24)
        self.assertEqual(report["current_floor_status"], "pass")
        self.assertEqual(report["mature_suite_status"], "expansion_in_progress")
        self.assertEqual(report["remaining_to_mature_minimum"], 76)
        self.assertEqual(report["suite_policy"], "permanently_held_out_not_generation_or_training_input")

    def test_mature_target_flag_refuses_to_pretend_twenty_four_tasks_are_enough(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation_audit.json"
            code = main(["--require-mature-target", "--output", str(output)])
            report = read_json(output)
        self.assertEqual(code, 2)
        self.assertEqual(report["status"], "fail")
        self.assertIn("client_server_security", report["missing_mature_tracks"])


if __name__ == "__main__":
    unittest.main()
