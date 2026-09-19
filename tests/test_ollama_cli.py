from __future__ import annotations

import json
import subprocess
import unittest
from unittest.mock import patch

from scripts.lib.ollama import OllamaClient
from scripts.lib.ollama_cli import OllamaCliError, ensure_ollama_model, ollama_list


class OllamaCliTests(unittest.TestCase):
    @patch("scripts.lib.ollama_cli.subprocess.run")
    def test_preflight_parses_existing_qwen_tag_without_download(self, run_mock) -> None:
        run_mock.return_value = subprocess.CompletedProcess(
            args=["ollama", "list"],
            returncode=0,
            stdout="NAME              ID              SIZE      MODIFIED\nqwen3:4b          abc123          2.6 GB    1 hour ago\n",
            stderr="",
        )
        preflight = ensure_ollama_model("qwen3:4b")
        self.assertEqual(preflight.model, "qwen3:4b")
        self.assertEqual(preflight.available_models, ("qwen3:4b",))
        run_mock.assert_called_once_with(
            ["ollama", "list"], check=False, capture_output=True, text=True, timeout=30
        )

    @patch("scripts.lib.ollama.ensure_ollama_model")
    @patch.object(OllamaClient, "model_names")
    def test_local_client_runs_cli_preflight_before_api_tag_check(self, model_names_mock, preflight_mock) -> None:
        call_order: list[str] = []

        def cli_preflight(model: str) -> None:
            self.assertEqual(model, "qwen3:4b")
            call_order.append("cli")

        def api_tags() -> set[str]:
            call_order.append("api")
            return {"qwen3:4b"}

        preflight_mock.side_effect = cli_preflight
        model_names_mock.side_effect = api_tags
        OllamaClient().assert_model_present("qwen3:4b")
        self.assertEqual(call_order, ["cli", "api"])

    @patch("scripts.lib.ollama.urlopen")
    def test_generate_aggregates_streamed_chunks_for_slow_local_inference(self, urlopen_mock) -> None:
        class FakeStream:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def __iter__(self):
                return iter(
                    [
                        b'{"model":"qwen3:4b","response":"secure ","done":false}\n',
                        b'{"model":"qwen3:4b","response":"answer","done":false}\n',
                        b'{"model":"qwen3:4b","response":"","done":true,"eval_count":2}\n',
                    ]
                )

        urlopen_mock.return_value = FakeStream()
        response = OllamaClient().generate(model="qwen3:4b", prompt="test", think=False)
        self.assertEqual(response.content, "secure answer")
        self.assertEqual(response.raw["eval_count"], 2)
        request = urlopen_mock.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertTrue(payload["stream"])
        self.assertFalse(payload["think"])

    @patch("scripts.lib.ollama_cli.subprocess.run")
    def test_missing_tag_fails_without_suggesting_automatic_download(self, run_mock) -> None:
        run_mock.return_value = subprocess.CompletedProcess(
            args=["ollama", "list"], returncode=0, stdout="NAME ID SIZE MODIFIED\nother:latest x 1 GB now\n", stderr=""
        )
        with self.assertRaises(OllamaCliError) as captured:
            ensure_ollama_model("qwen3:4b")
        self.assertIn("Do not download", str(captured.exception))


if __name__ == "__main__":
    unittest.main()
