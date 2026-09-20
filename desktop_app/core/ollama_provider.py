"""Streaming local Ollama provider for the native DukeOTR desktop application.

The repository already uses Ollama's local HTTP API in ``scripts.lib.ollama`` for data tooling.
This application-specific provider keeps the same dependency-free approach but exposes response
chunks and cancellation for the interactive desktop UI.
"""

from __future__ import annotations

import json
from collections.abc import Iterator as IteratorABC
from threading import Event
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from desktop_app.core.models import ChatRequest, ModelInfo, ProviderStatus, StreamChunk
from desktop_app.core.provider import (
    GenerationCancelled,
    ModelProvider,
    ModelUnavailableError,
    ProviderProtocolError,
    ProviderUnavailableError,
)


class OllamaTransport(Protocol):
    """Small injectable transport used to test API parsing without an Ollama process."""

    def request_json(self, method: str, url: str, payload: dict[str, Any] | None, timeout_seconds: float) -> dict[str, Any]:
        ...

    def stream_json_lines(
        self, url: str, payload: dict[str, Any], timeout_seconds: float, cancel_event: Event
    ) -> IteratorABC[dict[str, Any]]:
        ...


class UrllibOllamaTransport:
    """No-extra-dependency HTTP transport for Ollama's documented local API."""

    @staticmethod
    def _request(method: str, url: str, payload: dict[str, Any] | None) -> Request:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(url, data=data, method=method)
        request.add_header("Accept", "application/json")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        return request

    @staticmethod
    def _http_error(url: str, exc: HTTPError) -> ProviderProtocolError | ModelUnavailableError:
        body = exc.read().decode("utf-8", errors="replace")
        detail = body[:1000]
        lowered = detail.lower()
        if exc.code == 404 or "model" in lowered and ("not found" in lowered or "does not exist" in lowered):
            return ModelUnavailableError(
                "The selected model is unavailable in Ollama. Choose a listed local model or verify the exact installed tag."
            )
        return ProviderProtocolError(f"Ollama returned HTTP {exc.code} for {url}: {detail or 'no response detail'}")

    def request_json(self, method: str, url: str, payload: dict[str, Any] | None, timeout_seconds: float) -> dict[str, Any]:
        try:
            with urlopen(self._request(method, url, payload), timeout=timeout_seconds) as response:  # noqa: S310 - configured local provider endpoint
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            raise self._http_error(url, exc) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ProviderUnavailableError(
                "Ollama is not running or cannot be reached. Start Ollama, then try again. "
                f"Configured URL: {url.rsplit('/api/', 1)[0]}. ({exc})"
            ) from exc
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderProtocolError("Ollama returned malformed JSON while checking its API.") from exc
        if not isinstance(value, dict):
            raise ProviderProtocolError("Ollama returned an unexpected JSON response shape.")
        if isinstance(value.get("error"), str) and value["error"].strip():
            error = value["error"].strip()
            if "model" in error.lower() and ("not found" in error.lower() or "does not exist" in error.lower()):
                raise ModelUnavailableError(f"The selected model is unavailable: {error}")
            raise ProviderProtocolError(f"Ollama reported an error: {error}")
        return value

    def stream_json_lines(
        self, url: str, payload: dict[str, Any], timeout_seconds: float, cancel_event: Event
    ) -> IteratorABC[dict[str, Any]]:
        response = None
        try:
            response = urlopen(self._request("POST", url, payload), timeout=timeout_seconds)  # noqa: S310 - configured local provider endpoint
            for raw_line in response:
                if cancel_event.is_set():
                    raise GenerationCancelled("Generation stopped by the user.")
                if not raw_line.strip():
                    continue
                try:
                    item = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ProviderProtocolError("Ollama returned malformed streaming JSON.") from exc
                if not isinstance(item, dict):
                    raise ProviderProtocolError("Ollama returned an unexpected streamed response shape.")
                if isinstance(item.get("error"), str) and item["error"].strip():
                    error = item["error"].strip()
                    if "model" in error.lower() and "not found" in error.lower():
                        raise ModelUnavailableError(f"The selected model is unavailable: {error}")
                    raise ProviderProtocolError(f"Ollama reported a generation error: {error}")
                yield item
        except HTTPError as exc:
            raise self._http_error(url, exc) from exc
        except GenerationCancelled:
            raise
        except (URLError, TimeoutError, OSError) as exc:
            if cancel_event.is_set():
                raise GenerationCancelled("Generation stopped by the user.") from exc
            raise ProviderUnavailableError(
                "The Ollama connection was interrupted. Confirm Ollama is running and retry the request."
            ) from exc
        finally:
            # Closing the response terminates the local HTTP stream when Stop is pressed.
            if response is not None:
                response.close()


class OllamaProvider(ModelProvider):
    provider_id = "ollama"

    def __init__(self, base_url: str, *, timeout_seconds: float = 600.0, transport: OllamaTransport | None = None) -> None:
        normalized = base_url.strip().rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            normalized = f"http://{normalized}"
        parsed = urlparse(normalized)
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}:
            raise ValueError("OllamaProvider accepts only a local loopback Ollama endpoint.")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("OllamaProvider endpoint must be a base URL without a path, query, or fragment.")
        self.base_url = normalized
        self.timeout_seconds = timeout_seconds
        self._transport = transport or UrllibOllamaTransport()

    @property
    def _tags_url(self) -> str:
        return f"{self.base_url}/api/tags"

    @property
    def _chat_url(self) -> str:
        return f"{self.base_url}/api/chat"

    def list_models(self) -> tuple[ModelInfo, ...]:
        payload = self._transport.request_json("GET", self._tags_url, None, self.timeout_seconds)
        raw_models = payload.get("models")
        if not isinstance(raw_models, list):
            raise ProviderProtocolError("Ollama /api/tags returned no models list.")
        models: list[ModelInfo] = []
        for item in raw_models:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            size = item.get("size")
            models.append(
                ModelInfo(
                    name=name.strip(),
                    size_bytes=size if isinstance(size, int) and size >= 0 else None,
                    modified_at=item.get("modified_at") if isinstance(item.get("modified_at"), str) else None,
                    digest=item.get("digest") if isinstance(item.get("digest"), str) else None,
                )
            )
        return tuple(sorted(models, key=lambda item: item.name.casefold()))

    def status(self) -> ProviderStatus:
        try:
            models = self.list_models()
        except ProviderUnavailableError as exc:
            return ProviderStatus(False, self.base_url, str(exc), ())
        except ProviderProtocolError as exc:
            return ProviderStatus(False, self.base_url, f"Ollama responded, but the API response was invalid: {exc}", ())
        except Exception as exc:  # Provider status must never fabricate a successful connection after an internal fault.
            return ProviderStatus(False, self.base_url, f"Could not check the local Ollama endpoint: {exc}", ())
        detail = f"Ollama connected · {len(models)} model{'s' if len(models) != 1 else ''} discovered"
        return ProviderStatus(True, self.base_url, detail, models)

    def validate_model(self, model: str) -> None:
        wanted = model.strip()
        if not wanted:
            raise ModelUnavailableError("Choose a model before sending a message.")
        names = {item.name for item in self.list_models()}
        if wanted not in names:
            available = ", ".join(sorted(names)) or "none"
            raise ModelUnavailableError(
                f"Model {wanted!r} is not installed in Ollama. Available local models: {available}. "
                "Run `ollama list` to verify tags; the app will not download a replacement automatically."
            )

    def stream_chat(self, request: ChatRequest, cancel_event: Event) -> Iterator[StreamChunk]:
        if cancel_event.is_set():
            raise GenerationCancelled("Generation stopped by the user.")
        self.validate_model(request.model)
        if cancel_event.is_set():
            raise GenerationCancelled("Generation stopped by the user.")
        saw_done = False
        saw_content = False
        for item in self._transport.stream_json_lines(
            self._chat_url,
            request.as_ollama_payload(),
            self.timeout_seconds,
            cancel_event,
        ):
            message = item.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if content is not None and not isinstance(content, str):
                raise ProviderProtocolError("Ollama streamed a message with non-text content.")
            done = item.get("done") is True
            if isinstance(content, str) and content:
                saw_content = True
                yield StreamChunk(content=content, done=False, model=item.get("model") if isinstance(item.get("model"), str) else request.model)
            if done:
                saw_done = True
                yield StreamChunk(
                    content="",
                    done=True,
                    model=item.get("model") if isinstance(item.get("model"), str) else request.model,
                    metadata={
                        key: item[key]
                        for key in ("total_duration", "load_duration", "prompt_eval_count", "eval_count")
                        if key in item
                    },
                )
                break
        if cancel_event.is_set() and not saw_done:
            raise GenerationCancelled("Generation stopped by the user.")
        if not saw_done:
            raise ProviderProtocolError("Ollama closed the response before a completion marker was received.")
        if not saw_content:
            raise ProviderProtocolError("Ollama completed the response without assistant text.")
