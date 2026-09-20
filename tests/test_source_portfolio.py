from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.audit_source_portfolio import main
from scripts.lib.io_utils import read_json


class SourcePortfolioTests(unittest.TestCase):
    def test_combined_source_portfolio_has_required_forms_and_no_held_out_collision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "portfolio.json"
            code = main(["--strict", "--output", str(output)])
            report = read_json(output)
        self.assertEqual(code, 0)
        self.assertEqual(report["project_name"], "DukeOTR")
        self.assertEqual(report["source_brief_count"], 131)
        self.assertEqual(report["missing_required_task_forms"], [])
        self.assertEqual(report["evaluation_wording_collisions"], [])
        self.assertEqual(report["source_brief_count_status"], "pass")
        self.assertEqual(report["task_type_concentration_status"], "pass")
        for form in ("bad_answer_critique", "completion", "subtle_bug_analysis", "runtime_reasoning", "insecurity_analysis"):
            self.assertTrue(report["required_task_form_evidence"][form], form)


if __name__ == "__main__":
    unittest.main()
