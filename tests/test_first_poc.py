from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_first_poc import effective_local_host, main


class FirstPocTests(unittest.TestCase):
    def test_dry_run_writes_a_non_training_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "poc"
            code = main(["--dry-run", "--host", "http://127.0.0.1:11434", "--output-dir", str(output_dir)])
            self.assertEqual(code, 0)
            report = json.loads((output_dir / "first_poc_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "planned_not_executed")
            self.assertEqual(report["model"], "qwen3:4b")
            self.assertIn("No training data is generated.", report["guarantees"])

    def test_poc_rejects_remote_host_that_ollama_list_cannot_attest(self) -> None:
        with self.assertRaises(ValueError):
            effective_local_host("http://192.0.2.25:11434")


if __name__ == "__main__":
    unittest.main()
