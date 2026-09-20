"""Typed local domain records shared by the DukeOTR desktop UI and providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4


APP_SCHEMA_VERSION = "1.0"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:4b"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class AppMode(str, Enum):
    CHAT = "chat"
    CODE = "code"
    REVIEW = "review"
    DEBUG = "debug"
    BUILDER = "builder"
    SECURITY = "security"

    @property
    def label(self) -> str:
        return {
            AppMode.CHAT: "Chat",
            AppMode.CODE: "Code",
            AppMode.REVIEW: "Review",
            AppMode.DEBUG: "Debug",
            AppMode.BUILDER: "Builder",
            AppMode.SECURITY: "Security",
        }[self]


@dataclass
class AppSettings:
    """User-owned local settings; only documented Ollama options are sent to the provider."""

    ollama_url: str = DEFAULT_OLLAMA_URL
    default_model: str = DEFAULT_MODEL
    temperature: float = 0.2
    num_ctx: int = 8192
    num_predict: int = 1200
    theme: str = "system"
    response_streaming: bool = True
    data_directory: str | None = None

    def validate(self) -> None:
        url = self.ollama_url.strip()
        if not url.startswith(("http://", "https://")):
            raise ValueError("Ollama URL must begin with http:// or https://")
        parsed = urlparse(url)
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}:
            raise ValueError("DukeOTR accepts only a local Ollama loopback URL; remote/cloud endpoints are not supported.")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("Ollama URL must be an API base URL without a path, query, or fragment")
        if not self.default_model.strip():
            raise ValueError("Default model cannot be empty")
        if not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        if not 256 <= self.num_ctx <= 131072:
            raise ValueError("Context window must be between 256 and 131072")
        if not 1 <= self.num_predict <= 32768:
            raise ValueError("Maximum response tokens must be between 1 and 32768")
        if self.theme not in {"system", "dark", "light"}:
            raise ValueError("Theme must be system, dark, or light")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": APP_SCHEMA_VERSION,
            "ollama_url": self.ollama_url.strip().rstrip("/"),
            "default_model": self.default_model.strip(),
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "theme": self.theme,
            "response_streaming": self.response_streaming,
            "data_directory": self.data_directory,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AppSettings":
        if not isinstance(value, dict):
            raise ValueError("Settings must be a JSON object")
        settings = cls(
            ollama_url=str(value.get("ollama_url", DEFAULT_OLLAMA_URL)),
            default_model=str(value.get("default_model", DEFAULT_MODEL)),
            temperature=float(value.get("temperature", 0.2)),
            num_ctx=int(value.get("num_ctx", 8192)),
            num_predict=int(value.get("num_predict", 1200)),
            theme=str(value.get("theme", "system")),
            response_streaming=bool(value.get("response_streaming", True)),
            data_directory=value.get("data_directory") if isinstance(value.get("data_directory"), str) else None,
        )
        settings.validate()
        return settings


@dataclass
class ChatMessage:
    role: str
    content: str
    created_at: str = field(default_factory=utc_now)
    model: str | None = None
    mode: AppMode | None = None
    # Context can retain structured mode fields/code while content stays readable in the transcript.
    context: str | None = None
    interrupted: bool = False

    def validate(self) -> None:
        if self.role not in {"user", "assistant", "system"}:
            raise ValueError("Message role must be user, assistant, or system")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("Message content cannot be empty")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
            "model": self.model,
            "mode": self.mode.value if self.mode else None,
            "context": self.context,
            "interrupted": self.interrupted,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ChatMessage":
        if not isinstance(value, dict):
            raise ValueError("Conversation message must be an object")
        raw_mode = value.get("mode")
        try:
            mode = AppMode(raw_mode) if raw_mode is not None else None
        except ValueError as exc:
            raise ValueError(f"Unknown message mode: {raw_mode!r}") from exc
        message = cls(
            role=str(value.get("role", "")),
            content=str(value.get("content", "")),
            created_at=str(value.get("created_at", utc_now())),
            model=value.get("model") if isinstance(value.get("model"), str) else None,
            mode=mode,
            context=value.get("context") if isinstance(value.get("context"), str) else None,
            interrupted=bool(value.get("interrupted", False)),
        )
        message.validate()
        return message


@dataclass
class Conversation:
    id: str = field(default_factory=lambda: str(uuid4()))
    title: str = "New conversation"
    mode: AppMode = AppMode.CHAT
    messages: list[ChatMessage] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def touch(self) -> None:
        self.updated_at = utc_now()

    def add_message(self, message: ChatMessage) -> None:
        message.validate()
        self.messages.append(message)
        self.touch()
        if self.title == "New conversation" and message.role == "user":
            compact = " ".join(message.content.strip().split())
            self.title = (compact[:54].rstrip() + "…") if len(compact) > 55 else compact

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "mode": self.mode.value,
            "messages": [message.as_dict() for message in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Conversation":
        if not isinstance(value, dict):
            raise ValueError("Conversation must be an object")
        try:
            mode = AppMode(value.get("mode", AppMode.CHAT.value))
        except ValueError as exc:
            raise ValueError(f"Unknown conversation mode: {value.get('mode')!r}") from exc
        identifier = value.get("id")
        title = value.get("title")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("Conversation has no id")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("Conversation has no title")
        raw_messages = value.get("messages", [])
        if not isinstance(raw_messages, list):
            raise ValueError("Conversation messages must be a list")
        return cls(
            id=identifier,
            title=title.strip(),
            mode=mode,
            messages=[ChatMessage.from_dict(item) for item in raw_messages],
            created_at=str(value.get("created_at", utc_now())),
            updated_at=str(value.get("updated_at", utc_now())),
        )


@dataclass(frozen=True)
class ModelInfo:
    name: str
    size_bytes: int | None = None
    modified_at: str | None = None
    digest: str | None = None

    @property
    def display_name(self) -> str:
        # The technical tag remains visible and authoritative. This helper is merely friendly
        # display text; it never renames qwen3:4b to DukeOTR.
        return "Qwen3-4B (qwen3:4b)" if self.name == "qwen3:4b" else self.name


@dataclass(frozen=True)
class ProviderStatus:
    connected: bool
    endpoint: str
    message: str
    models: tuple[ModelInfo, ...] = ()


@dataclass(frozen=True)
class StreamChunk:
    content: str = ""
    done: bool = False
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: tuple[dict[str, str], ...]
    temperature: float
    num_ctx: int
    num_predict: int
    stream: bool = True

    def as_ollama_payload(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": list(self.messages),
            "stream": self.stream,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }
