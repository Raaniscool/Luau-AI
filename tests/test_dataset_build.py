from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_datasets import main as build_main
from scripts.lib.io_utils import read_jsonl, write_jsonl_atomic
from tests.helpers import reviewed_record


class DatasetBuildTests(unittest.TestCase):
    def test_build_keeps_eval_out_of_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "dedup.jsonl"
            evaluation = root / "eval.jsonl"
            output = root / "training"
            record = reviewed_record(prompt="Describe a secure orchard harvest request.")
            write_jsonl_atomic(source, [record])
            task = {
                "schema_version": "1.0",
                "id": "eval-held-out",
                "split": "evaluation",
                "category": "test",
                "difficulty": "beginner",
                "prompt": "A completely separate held-out question about a lighthouse.",
                "rubric": [{"id": "correct", "description": "correct", "max_points": 100}],
                "must_not": [],
            }
            write_jsonl_atomic(evaluation, [task])
            code = build_main(["--input", str(source), "--evaluation", str(evaluation), "--output-dir", str(output)])
            self.assertEqual(code, 0)
            final = list(read_jsonl(output / "final_dataset.jsonl"))
            self.assertEqual(len(final), 1)
            self.assertNotIn("eval-held-out", json.dumps(final))
            manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["accepted_records"], 1)


if __name__ == "__main__":
    unittest.main()
