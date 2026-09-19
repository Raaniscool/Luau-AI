"""Minimal Ollama HTTP client.

Using the local HTTP API keeps the data pipeline independent of the optional `ollama`
Python package and works on a Windows machine where Ollama is already installed.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from scripts.lib.ollama_cli import OllamaCliError, ensure_ollama_model


class OllamaError(RuntimeError):
    pass


class OllamaUnavailableError(OllamaError):
    pass


@dataclass(frozen=True)
class OllamaResponse:
    content: str
    raw: dict[str, Any]
    elapsed_seconds: float


class OllamaClient:
    def __init__(self, host: str | None = None, *, timeout_seconds: int = 600) -> None:
        configured_host = host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
        if not configured_host.startswith(("http://", "https://")):
            configured_host = f"http://{configured_host}"
        self.host = configured_host.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def tags(self) -> dict[str, Any]:
        return self._request("/api/tags", None, method="GET")

    def model_names(self) -> set[str]:
        tags = self.tags()
        names = set()
        for model in tags.get("models", []):
            if isinstance(model, dict) and isinstance(model.get("name"), str):
                names.add(model["name"])
        return names

    def assert_model_present(self, model: str) -> None:
        """Verify the local CLI registration before making an API generation request.

        This intentionally never pulls/downloads a model. For a local default host it first
        runs `ollama list`, then confirms the same tag through the Ollama API. A remote host
        cannot reliably use this machine's CLI, so only the API check applies there.
        """
        if self._is_local_host():
            try:
                ensure_ollama_model(model)
            except OllamaCliError as exc:
                raise OllamaUnavailableError(str(exc)) from exc
        names = self.model_names()
        if model not in names:
            available = ", ".join(sorted(names)) or "(none)"
            raise OllamaUnavailableError(
                f"Ollama is reachable at {self.host}, but model {model!r} is not registered by its API. "
                f"Available: {available}. Do not download a replacement automatically; confirm the existing tag or choose --model."
            )

    def _is_local_host(self) -> bool:
        parsed = urlparse(self.host)
        return parsed.hostname in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}

    def show(self, model: str) -> dict[str, Any]:
        return self._request("/api/show", {"name": model})

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None = None,
        options: dict[str, Any] | None = None,
        think: bool | None = False,
    ) -> OllamaResponse:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            # Streaming keeps a slow CPU-only local generation from waiting for one giant
            # HTTP response. We still aggregate the chunks before returning to callers.
            "stream": True,
            "options": options or {},
        }
        if system:
            payload["system"] = system
        # Current Ollama supports this for thinking-capable models; older servers safely
        # ignore unknown JSON fields. It prevents internal reasoning markup in datasets.
        if think is not None:
            payload["think"] = think
        started = time.perf_counter()
        result = self._request_with_think_fallback("/api/generate", payload, think)
        elapsed = time.perf_counter() - started
        content = result.get("response")
        if not isinstance(content, str):
            raise OllamaError("Ollama /api/generate returned no string response")
        return OllamaResponse(content=content, raw=result, elapsed_seconds=elapsed)

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
        think: bool | None = False,
    ) -> OllamaResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options or {},
        }
        if think is not None:
            payload["think"] = think
        started = time.perf_counter()
        result = self._request_with_think_fallback("/api/chat", payload, think)
        elapsed = time.perf_counter() - started
        message = result.get("message", {})
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise OllamaError("Ollama /api/chat returned no string message.content")
        return OllamaResponse(content=content, raw=result, elapsed_seconds=elapsed)

    def _request_with_think_fallback(
        self, endpoint: str, payload: dict[str, Any], think: bool | None
    ) -> dict[str, Any]:
        """Retry older Ollama servers that reject the newer optional `think` field."""
        request = self._stream_request if payload.get("stream") else self._request
        try:
            return request(endpoint, payload)
        except OllamaError as exc:
            message = str(exc).lower()
            if think is not None and "think" in payload and ("unknown field" in message or "invalid" in message):
                legacy_payload = dict(payload)
                legacy_payload.pop("think", None)
                return request(endpoint, legacy_payload)
            raise

    def _build_request(self, endpoint: str, payload: dict[str, Any] | None, *, method: str = "POST") -> Request:
        url = f"{self.host}{endpoint}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(url, data=data, method=method)
        request.add_header("Accept", "application/json")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        return request

    def _request(self, endpoint: str, payload: dict[str, Any] | None, *, method: str = "POST") -> dict[str, Any]:
        request = self._build_request(endpoint, payload, method=method)
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310 - local configured endpoint
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama request {endpoint} failed with HTTP {exc.code}: {body[:1000]}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise OllamaUnavailableError(
                f"Cannot reach Ollama at {self.host}. Start Ollama and confirm its local API is available. ({exc})"
            ) from exc
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OllamaError(f"Ollama returned non-JSON for {endpoint}: {raw[:500]!r}") from exc
        if not isinstance(result, dict):
            raise OllamaError(f"Ollama returned an unexpected JSON shape for {endpoint}")
        if result.get("error"):
            raise OllamaError(f"Ollama error for {endpoint}: {result['error']}")
        return result

    def _stream_request(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Aggregate Ollama's JSONL generation stream into the normal response shape.

        A chunked local response prevents a slow CPU-only generation from exceeding a
        whole-response socket wait. The configured timeout still bounds the wait for each
        individual stream chunk and initial model-load response.
        """
        request = self._build_request(endpoint, payload)
        chunks: list[str] = []
        final: dict[str, Any] | None = None
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310 - local configured endpoint
                for raw_line in response:
                    if not raw_line.strip():
                        continue
                    try:
                        item = json.loads(raw_line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        preview = raw_line[:500].decode("utf-8", errors="replace")
                        raise OllamaError(f"Ollama returned invalid JSONL for {endpoint}: {preview!r}") from exc
                    if not isinstance(item, dict):
                        raise OllamaError(f"Ollama returned an unexpected streamed JSON shape for {endpoint}")
                    if item.get("error"):
                        raise OllamaError(f"Ollama error for {endpoint}: {item['error']}")
                    piece = item.get("response")
                    if isinstance(piece, str):
                        chunks.append(piece)
                    if item.get("done") is True:
                        final = item
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama request {endpoint} failed with HTTP {exc.code}: {body[:1000]}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise OllamaUnavailableError(
                f"Cannot reach Ollama at {self.host} or receive its next generation chunk. "
                f"Confirm the local API is available and increase the request timeout if the machine is slow. ({exc})"
            ) from exc
        if final is None:
            raise OllamaError(f"Ollama stream for {endpoint} ended before its final done record")
        result = dict(final)
        result["response"] = "".join(chunks)
        return result
