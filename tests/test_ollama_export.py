from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.lib.io_utils import read_json
from scripts.prepare_ollama_modelfile import main as prepare_main


class OllamaExportTests(unittest.TestCase):
    def test_prepare_defaults_to_dukeotr_candidate_and_release_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "verified-dukeotr-adapter.gguf"
            adapter.write_bytes(b"GGUF")
            output_dir = root / "ollama"
            code = prepare_main(
                [
                    "--base",
                    "C:/models/matching-qwen3-4b.gguf",
                    "--adapter",
                    str(adapter),
                    "--output-dir",
                    str(output_dir),
                ]
            )
            self.assertEqual(code, 0)
            modelfile = (output_dir / "Modelfile").read_text(encoding="utf-8")
            readme = (output_dir / "README.txt").read_text(encoding="utf-8")
            manifest = read_json(output_dir / "dukeotr_model_identity.json")

        self.assertIn("Public model identity: DukeOTR", modelfile)
        self.assertIn("Planned candidate tag: dukeotr-v1", modelfile)
        self.assertIn("Planned stable release alias: dukeotr", modelfile)
        self.assertIn("You are DukeOTR", modelfile)
        self.assertIn("ollama create dukeotr-v1", readme)
        self.assertIn("ollama run dukeotr", readme)
        self.assertEqual(manifest["status"], "prepared_not_created")
        self.assertEqual(manifest["public_display_name"], "DukeOTR")
        self.assertEqual(manifest["candidate_ollama_tag"], "dukeotr-v1")
        self.assertEqual(manifest["planned_release_ollama_tag"], "dukeotr")

    def test_prepare_refuses_non_dukeotr_public_tag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapter.gguf"
            adapter.write_bytes(b"GGUF")
            code = prepare_main(
                [
                    "--base",
                    "matching-base.gguf",
                    "--adapter",
                    str(adapter),
                    "--output-dir",
                    str(root / "ollama"),
                    "--name",
                    "qwen-specialized-v1",
                ]
            )
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
