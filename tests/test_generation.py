from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.generate_examples import main as generate_main
from scripts.lib.io_utils import read_json, read_jsonl, sha256_text
from scripts.lib.ollama import OllamaResponse
from scripts.lib.prompts import GENERATION_RESPONSE_SCHEMA


class GenerationTests(unittest.TestCase):
    @patch("scripts.generate_examples.OllamaClient.generate")
    @patch("scripts.generate_examples.OllamaClient.show", return_value={"details": {"family": "qwen3"}})
    @patch("scripts.generate_examples.OllamaClient.assert_model_present")
    def test_generation_requests_json_schema_and_writes_candidate(self, preflight_mock, show_mock, generate_mock) -> None:
        generate_mock.return_value = OllamaResponse(
            content=(
                '{"assistant_response":"Use a ServerScript and validate the request on the server.",'
                '"coverage":["server authority"],"self_check":["included a direct answer"]}'
            ),
            raw={},
            elapsed_seconds=0.25,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate.jsonl"
            report = Path(directory) / "candidate.report.json"
            code = generate_main(
                [
                    "--limit",
                    "1",
                    "--output",
                    str(output),
                    "--report",
                    str(report),
                ]
            )
            self.assertEqual(code, 0)
            records = list(read_jsonl(output))
            generation_report = read_json(report)

        self.assertEqual(len(records), 1)
        self.assertEqual(generation_report["response_contract"], "ollama_json_schema")
        self.assertEqual(generation_report["response_schema"], GENERATION_RESPONSE_SCHEMA)
        self.assertEqual(generate_mock.call_args.kwargs["response_format"], GENERATION_RESPONSE_SCHEMA)
        preflight_mock.assert_called_once_with("qwen3:4b")
        show_mock.assert_called_once_with("qwen3:4b")

    @patch("scripts.generate_examples.OllamaClient.generate")
    @patch("scripts.generate_examples.OllamaClient.show", return_value={})
    @patch("scripts.generate_examples.OllamaClient.assert_model_present")
    def test_generation_preserves_bounded_malformed_response_diagnostics(self, preflight_mock, show_mock, generate_mock) -> None:
        raw_response = "This is a normal prose answer, not the required JSON envelope."
        generate_mock.return_value = OllamaResponse(content=raw_response, raw={}, elapsed_seconds=0.5)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate.jsonl"
            report = Path(directory) / "candidate.report.json"
            code = generate_main(
                [
                    "--limit",
                    "1",
                    "--output",
                    str(output),
                    "--report",
                    str(report),
                ]
            )
            self.assertEqual(code, 2)
            generation_report = read_json(report)

        self.assertEqual(generation_report["generated_records"], 0)
        failure = generation_report["failures"][0]
        self.assertEqual(failure["model_response_excerpt"], raw_response)
        self.assertEqual(failure["model_response_sha256"], sha256_text(raw_response))
        self.assertFalse(failure["model_response_truncated"])
        self.assertEqual(failure["elapsed_seconds"], 0.5)
        self.assertEqual(generate_mock.call_args.kwargs["response_format"], GENERATION_RESPONSE_SCHEMA)
        preflight_mock.assert_called_once_with("qwen3:4b")
        show_mock.assert_called_once_with("qwen3:4b")


if __name__ == "__main__":
    unittest.main()
