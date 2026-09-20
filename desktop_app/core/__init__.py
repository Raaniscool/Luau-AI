"""UI-independent core of the DukeOTR desktop application."""

from desktop_app.core.app_core import DukeOTRApplicationCore
from desktop_app.core.ollama_provider import OllamaProvider
from desktop_app.core.provider import ModelProvider

__all__ = ["DukeOTRApplicationCore", "ModelProvider", "OllamaProvider"]
