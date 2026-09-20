"""Mocked-core tests for the native DukeOTR desktop application's non-UI behavior."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from threading import Event
from typing import Any, Iterator

from desktop_app.core.app_core import DukeOTRApplicationCore
from desktop_app.core.builder_adapter import plan_builder_request
from desktop_app.core.models import AppMode, AppSettings, ChatMessage, ChatRequest, Conversation, ModelInfo, ProviderStatus, StreamChunk
from desktop_app.core.modes import ModeSubmission, compose_messages, format_user_content, system_instruction
from desktop_app.core.ollama_provider import OllamaProvider, UrllibOllamaTransport
from desktop_app.core.provider import GenerationCancelled, ModelProvider, ModelUnavailableError, ProviderProtocolError, ProviderUnavailableError
from desktop_app.core.storage import ConversationStore, SettingsStore


class FakeTransport:
    def __init__(self, *, models: list[dict[str, Any]] | None = None, stream: list[dict[str, Any]] | None = None) -> None:
        self.models = models if models is not None else [{"name": "qwen3:4b", "size": 2_500_000_000}]
        self.stream = stream if stream is not None else [
            {"model": "qwen3:4b", "message": {"role": "assistant", "content": "Hello"}, "done": False},
            {"model": "qwen3:4b", "message": {"role": "assistant", "content": " world"}, "done": False},
            {"model": "qwen3:4b", "done": True, "eval_count": 2},
        ]
        self.requests: list[tuple[str, str, dict[str, Any] | None]] = []

    def request_json(self, method: str, url: str, payload: dict[str, Any] | None, timeout_seconds: float) -> dict[str, Any]:
        self.requests.append((method, url, payload))
        return {"models": self.models}

    def stream_json_lines(
        self, url: str, payload: dict[str, Any], timeout_seconds: float, cancel_event: Event
    ) -> Iterator[dict[str, Any]]:
        self.requests.append(("POST", url, payload))
        for item in self.stream:
            if cancel_event.is_set():
                raise GenerationCancelled("Generation stopped by the user.")
            yield item


class UnavailableTransport(FakeTransport):
    def request_json(self, method: str, url: str, payload: dict[str, Any] | None, timeout_seconds: float) -> dict[str, Any]:
        raise ProviderUnavailableError("Cannot reach Ollama for this test.")


class FakeHTTPResponse:
    """Enough of urllib's response API to exercise JSON/JSONL parsing without a server."""

    def __init__(self, *, body: bytes = b"", lines: list[bytes] | None = None) -> None:
        self.body = body
        self.lines = lines or []
        self.closed = False

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def read(self) -> bytes:
        return self.body

    def __iter__(self) -> Iterator[bytes]:
        return iter(self.lines)

    def close(self) -> None:
        self.closed = True


class FakeProvider(ModelProvider):
    provider_id = "fake"

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.models = (ModelInfo("qwen3:4b"), ModelInfo("another:latest"))
        self.requests: list[ChatRequest] = []

    def status(self) -> ProviderStatus:
        return ProviderStatus(True, self.endpoint, "Fake provider connected", self.models)

    def list_models(self) -> tuple[ModelInfo, ...]:
        return self.models

    def validate_model(self, model: str) -> None:
        if model not in {item.name for item in self.models}:
            raise ModelUnavailableError(f"No fake model {model}")

    def stream_chat(self, request: ChatRequest, cancel_event: Event) -> Iterator[StreamChunk]:
        self.validate_model(request.model)
        self.requests.append(request)
        if cancel_event.is_set():
            raise GenerationCancelled("Generation stopped by the user.")
        yield StreamChunk(content="mock ", model=request.model)
        if cancel_event.is_set():
            raise GenerationCancelled("Generation stopped by the user.")
        yield StreamChunk(content="answer", model=request.model)
        yield StreamChunk(done=True, model=request.model)


class OllamaProviderTests(unittest.TestCase):
    def test_model_discovery_uses_tags_and_truthful_qwen_tag(self) -> None:
        transport = FakeTransport(models=[{"name": "llama3:latest"}, {"name": "qwen3:4b", "size": 42}])
        provider = OllamaProvider("http://127.0.0.1:11434", transport=transport)

        models = provider.list_models()
        status = provider.status()

        self.assertEqual([item.name for item in models], ["llama3:latest", "qwen3:4b"])
        self.assertEqual(models[1].display_name, "Qwen3-4B (qwen3:4b)")
        self.assertTrue(status.connected)
        self.assertIn("2 models", status.message)
        self.assertTrue(all(request[1].endswith("/api/tags") for request in transport.requests))

    def test_stream_parses_ollama_jsonl_and_builds_chat_payload(self) -> None:
        transport = FakeTransport()
        provider = OllamaProvider("127.0.0.1:11434", transport=transport)
        request = ChatRequest(
            model="qwen3:4b",
            messages=({"role": "user", "content": "hello"},),
            temperature=0.25,
            num_ctx=8192,
            num_predict=400,
        )

        chunks = list(provider.stream_chat(request, Event()))
        payload = transport.requests[-1][2]

        self.assertEqual("".join(chunk.content for chunk in chunks), "Hello world")
        self.assertTrue(chunks[-1].done)
        self.assertEqual(chunks[-1].metadata["eval_count"], 2)
        self.assertEqual(payload["model"], "qwen3:4b")
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hello"}])
        self.assertEqual(payload["options"]["temperature"], 0.25)

    def test_stdlib_transport_parses_raw_jsonl_and_sends_json_request(self) -> None:
        response = FakeHTTPResponse(
            lines=[
                b'{"message":{"role":"assistant","content":"first"},"done":false}\n',
                b'{"message":{"role":"assistant","content":" second"},"done":false}\n',
                b'{"done":true,"eval_count":2}\n',
            ]
        )
        transport = UrllibOllamaTransport()
        cancel = Event()
        with patch("desktop_app.core.ollama_provider.urlopen", return_value=response) as opened:
            parsed = list(
                transport.stream_json_lines(
                    "http://127.0.0.1:11434/api/chat",
                    {"model": "qwen3:4b", "stream": True},
                    5,
                    cancel,
                )
            )
        request = opened.call_args.args[0]
        self.assertEqual(json.loads(request.data.decode("utf-8"))["model"], "qwen3:4b")
        self.assertEqual(parsed[-1]["eval_count"], 2)
        self.assertTrue(response.closed)

    def test_stdlib_transport_cancels_and_closes_stream(self) -> None:
        response = FakeHTTPResponse(lines=[b'{"message":{"content":"first"},"done":false}\n', b'{"done":true}\n'])
        transport = UrllibOllamaTransport()
        cancel = Event()
        with patch("desktop_app.core.ollama_provider.urlopen", return_value=response):
            iterator = transport.stream_json_lines("http://127.0.0.1:11434/api/chat", {"model": "qwen3:4b"}, 5, cancel)
            self.assertEqual(next(iterator)["message"]["content"], "first")
            cancel.set()
            with self.assertRaises(GenerationCancelled):
                next(iterator)
        self.assertTrue(response.closed)

    def test_stdlib_transport_reports_malformed_jsonl(self) -> None:
        response = FakeHTTPResponse(lines=[b"not-json\n"])
        with patch("desktop_app.core.ollama_provider.urlopen", return_value=response):
            with self.assertRaisesRegex(ProviderProtocolError, "malformed streaming JSON"):
                list(UrllibOllamaTransport().stream_json_lines("http://127.0.0.1:11434/api/chat", {"model": "qwen3:4b"}, 5, Event()))
        self.assertTrue(response.closed)

    def test_model_absence_has_actionable_error_and_never_downloads(self) -> None:
        provider = OllamaProvider("http://localhost:11434", transport=FakeTransport(models=[{"name": "llama3:latest"}]))
        with self.assertRaisesRegex(ModelUnavailableError, "ollama list"):
            provider.validate_model("qwen3:4b")

    def test_provider_rejects_nonlocal_endpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "local loopback"):
            OllamaProvider("https://example.invalid")

    def test_unavailable_status_is_not_reported_as_connected(self) -> None:
        provider = OllamaProvider("http://127.0.0.1:11434", transport=UnavailableTransport())
        status = provider.status()
        self.assertFalse(status.connected)
        self.assertIn("Cannot reach Ollama", status.message)

    def test_bad_stream_without_done_marker_is_an_error(self) -> None:
        provider = OllamaProvider(
            "http://127.0.0.1:11434",
            transport=FakeTransport(stream=[{"message": {"role": "assistant", "content": "partial"}, "done": False}]),
        )
        with self.assertRaisesRegex(ProviderProtocolError, "completion marker"):
            list(provider.stream_chat(ChatRequest("qwen3:4b", ({"role": "user", "content": "x"},), 0.2, 8192, 100), Event()))

    def test_cancelled_generation_is_explicit(self) -> None:
        provider = OllamaProvider("http://127.0.0.1:11434", transport=FakeTransport())
        cancel = Event()
        cancel.set()
        with self.assertRaises(GenerationCancelled):
            list(provider.stream_chat(ChatRequest("qwen3:4b", ({"role": "user", "content": "x"},), 0.2, 8192, 100), cancel))


class PersistenceTests(unittest.TestCase):
    def test_settings_and_history_round_trip_locally(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings_store = SettingsStore(temporary)
            settings = AppSettings(temperature=0.4, num_ctx=4096, data_directory=temporary)
            settings_store.save(settings)
            loaded_settings = settings_store.load()
            self.assertEqual(loaded_settings.temperature, 0.4)
            self.assertEqual(loaded_settings.num_ctx, 4096)

            conversation = Conversation(mode=AppMode.CODE)
            conversation.add_message(ChatMessage("user", "make a module", context="structured request", mode=AppMode.CODE))
            conversation.add_message(ChatMessage("assistant", "```lua\nreturn {}\n```", model="qwen3:4b", mode=AppMode.CODE))
            history = ConversationStore(temporary)
            history.save_all([conversation])
            restored = history.load_all()

            self.assertEqual(len(restored), 1)
            self.assertEqual(restored[0].mode, AppMode.CODE)
            self.assertEqual(restored[0].messages[0].context, "structured request")
            self.assertEqual(restored[0].messages[1].model, "qwen3:4b")
            self.assertTrue((Path(temporary) / "settings.json").exists())
            self.assertTrue((Path(temporary) / "conversations.json").exists())

    def test_remote_ollama_endpoint_is_rejected_to_keep_history_local(self) -> None:
        with self.assertRaisesRegex(ValueError, "local Ollama loopback"):
            AppSettings(ollama_url="https://example.invalid").validate()

    def test_corrupt_history_is_quarantined_instead_of_silently_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "conversations.json"
            path.write_text("not json", encoding="utf-8")
            store = ConversationStore(temporary)
            self.assertEqual(store.load_all(), [])
            self.assertIsNotNone(store.last_warning)
            self.assertFalse(path.exists())
            self.assertTrue(list(Path(temporary).glob("conversations.json.corrupt*")))


class ApplicationCoreTests(unittest.TestCase):
    def _core(self, temporary: str) -> tuple[DukeOTRApplicationCore, FakeProvider]:
        captured: list[FakeProvider] = []

        def factory(endpoint: str) -> FakeProvider:
            provider = FakeProvider(endpoint)
            captured.append(provider)
            return provider

        core = DukeOTRApplicationCore(data_directory=temporary, provider_factory=factory)
        return core, captured[-1]

    def test_model_switch_changes_next_request_without_ui_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            core, provider = self._core(temporary)
            core.select_model("another:latest")
            conversation = core.new_conversation(AppMode.CHAT)
            prepared = core.prepare_generation(conversation, ModeSubmission(mode=AppMode.CHAT, prompt="hello"))
            self.assertEqual(prepared.request.model, "another:latest")
            core.record_user_message(conversation, prepared.user_message)
            self.assertEqual("".join(chunk.content for chunk in core.stream_generation(prepared.request, Event())), "mock answer")
            self.assertEqual(provider.requests[-1].model, "another:latest")

    def test_model_endpoint_switch_replaces_provider_and_preserves_local_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            core, old_provider = self._core(temporary)
            conversation = core.new_conversation()
            core.record_user_message(conversation, ChatMessage("user", "local only"))
            changed = AppSettings(ollama_url="http://127.0.0.1:22444", data_directory=temporary)
            core.update_settings(changed)
            self.assertIsNot(core.provider, old_provider)
            self.assertEqual(core.provider.endpoint, "http://127.0.0.1:22444")
            self.assertEqual(core.get_conversation(conversation.id).messages[0].content, "local only")

    def test_core_only_persists_assistant_after_it_has_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            core, _provider = self._core(temporary)
            conversation = core.new_conversation()
            prepared = core.prepare_generation(conversation, ModeSubmission(mode=AppMode.CHAT, prompt="A question"))
            core.record_user_message(conversation, prepared.user_message)
            self.assertEqual([message.role for message in conversation.messages], ["user"])
            core.record_assistant_message(conversation, "Answer", model="qwen3:4b", mode=AppMode.CHAT, interrupted=True)
            reloaded = ConversationStore(temporary).load_all()[0]
            self.assertTrue(reloaded.messages[-1].interrupted)


class ModeInstructionTests(unittest.TestCase):
    def test_every_visible_mode_has_explicit_instruction(self) -> None:
        for mode in AppMode:
            with self.subTest(mode=mode):
                instruction = system_instruction(mode)
                self.assertIn(f"Mode: {mode.label}", instruction)
                self.assertIn("held-out evaluation", instruction)
                self.assertIn("fine-tuned", instruction)

    def test_code_and_debug_mode_context_is_structured(self) -> None:
        code = ModeSubmission(mode=AppMode.CODE, prompt="make this safer", code="local x = 1", action="Improve")
        content = format_user_content(code)
        self.assertIn("Requested action: Improve", content)
        self.assertIn("Luau code", content)
        debug = ModeSubmission(mode=AppMode.DEBUG, prompt="help", code="error('x')", error="x", expected="works", actual="fails")
        messages = compose_messages([ChatMessage("assistant", "Earlier")], debug)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("Observed error", messages[-1]["content"])

    def test_debug_mode_rejects_missing_observation(self) -> None:
        with self.assertRaisesRegex(ValueError, "error message or observed behavior"):
            ModeSubmission(mode=AppMode.DEBUG, prompt="debug this", code="print('x')").validate()

    def test_builder_mode_reuses_existing_deterministic_routing_for_planning_only(self) -> None:
        context = plan_builder_request("Build a RemoteEvent currency purchase flow with server validation.")
        self.assertEqual(context.level, "complex")
        self.assertIn("client_server_security", context.selected_checks)
        self.assertIn("client_server_security", context.risk_labels)
        self.assertIn("does not invoke", context.summary())


if __name__ == "__main__":
    unittest.main()
