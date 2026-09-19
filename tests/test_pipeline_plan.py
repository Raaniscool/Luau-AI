from __future__ import annotations

import json
import tempfile
import unittest
from uuid import uuid4
from pathlib import Path

from scripts.generate_examples import main as generate_main
from scripts.run_pipeline import main, stage_paths


class PipelinePlanTests(unittest.TestCase):
    def test_dry_run_uses_phase1_catalog_and_versioned_stage_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "plan.json"
            run_id = f"dukeotr_test_{uuid4().hex}"
            code = main(
                [
                    "--dry-run",
                    "--run-id",
                    run_id,
                    "--limit",
                    "3",
                    "--report",
                    str(report),
                ]
            )
            self.assertEqual(code, 0)
            plan = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(plan["stage"], "dukeotr_phase_1_pipeline")
        self.assertEqual(plan["seeds"], "raw_data/dukeotr_phase1_luau_seed_tasks.jsonl")
        self.assertEqual(plan["stage_paths"], stage_paths(run_id))
        self.assertIn("--output-dir", plan["commands"][-1])
        self.assertIn("--strict", plan["commands"][-1])

    def test_generator_preserves_an_existing_candidate_output_before_model_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.jsonl"
            output.write_text('{"existing": true}\n', encoding="utf-8")
            code = generate_main(["--output", str(output), "--limit", "1"])
            self.assertEqual(code, 2)
            self.assertEqual(output.read_text(encoding="utf-8"), '{"existing": true}\n')

    def test_unsafe_run_identifier_is_rejected_before_any_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "plan.json"
            code = main(["--dry-run", "--run-id", "../unsafe", "--report", str(report)])
            self.assertEqual(code, 2)
            self.assertFalse(report.exists())


if __name__ == "__main__":
    unittest.main()
