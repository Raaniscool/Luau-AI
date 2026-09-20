"""DukeOTR native desktop application package.

The package is the local interface around a selectable model provider. It is not a model,
training system, adapter, or a claim that Qwen3-4B has become DukeOTR.
"""

from desktop_app.core.app_core import DukeOTRApplicationCore
from desktop_app.core.models import AppMode, AppSettings, ChatMessage, Conversation

__all__ = ["AppMode", "AppSettings", "ChatMessage", "Conversation", "DukeOTRApplicationCore"]
