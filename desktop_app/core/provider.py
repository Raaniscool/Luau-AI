"""Provider boundary for local and future DukeOTR model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from threading import Event
from typing import Iterator

from desktop_app.core.models import ChatRequest, ModelInfo, ProviderStatus, StreamChunk


class ModelProviderError(RuntimeError):
    """Base error shown to users as a useful provider/application message."""


class ProviderUnavailableError(ModelProviderError):
    """The configured provider cannot be reached."""


class ModelUnavailableError(ModelProviderError):
    """The provider is reachable but the selected tag is absent."""


class ProviderProtocolError(ModelProviderError):
    """The provider returned malformed or incomplete data."""


class GenerationCancelled(ModelProviderError):
    """A user cancelled a streamed generation."""


class ModelProvider(ABC):
    """The application only depends on this interface, not a specific model/vendor."""

    provider_id: str = "abstract"

    @abstractmethod
    def status(self) -> ProviderStatus:
        """Return actual reachability and discovered models; never fabricate connected state."""

    @abstractmethod
    def list_models(self) -> tuple[ModelInfo, ...]:
        """Discover provider model tags."""

    @abstractmethod
    def stream_chat(self, request: ChatRequest, cancel_event: Event) -> Iterator[StreamChunk]:
        """Yield response chunks. Closing/cancelling the iterator must stop local receipt."""

    @abstractmethod
    def validate_model(self, model: str) -> None:
        """Raise ModelUnavailableError when the selected model is absent."""
