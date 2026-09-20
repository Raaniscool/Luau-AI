"""UI-independent application orchestration for the DukeOTR desktop client."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from desktop_app.core.models import AppMode, AppSettings, ChatMessage, ChatRequest, Conversation, ProviderStatus, StreamChunk
from desktop_app.core.modes import ModeSubmission, compose_messages, format_user_content
from desktop_app.core.ollama_provider import OllamaProvider
from desktop_app.core.provider import ModelProvider
from desktop_app.core.storage import ConversationStore, SettingsStore


@dataclass(frozen=True)
class PreparedGeneration:
    request: ChatRequest
    user_message: ChatMessage


class DukeOTRApplicationCore:
    """Coordinates local storage, visible modes, and a swappable model provider.

    This class has no tkinter dependency, so provider/persistence behavior can be tested without
    a desktop display or a real Ollama instance.
    """

    def __init__(
        self,
        *,
        data_directory: str | Path | None = None,
        provider_factory: Callable[[str], ModelProvider] | None = None,
        settings_store: SettingsStore | None = None,
        conversation_store: ConversationStore | None = None,
    ) -> None:
        self.settings_store = settings_store or SettingsStore(data_directory)
        self.settings = self.settings_store.load()
        active_directory = Path(self.settings.data_directory or self.settings_store.directory)
        self.conversation_store = conversation_store or ConversationStore(active_directory)
        self.conversations = self.conversation_store.load_all()
        self._provider_factory = provider_factory or (lambda endpoint: OllamaProvider(endpoint))
        self.provider = self._provider_factory(self.settings.ollama_url)

    @property
    def storage_warnings(self) -> tuple[str, ...]:
        return tuple(
            warning
            for warning in (self.settings_store.last_warning, self.conversation_store.last_warning)
            if warning
        )

    def refresh_provider_status(self) -> ProviderStatus:
        return self.provider.status()

    def new_conversation(self, mode: AppMode = AppMode.CHAT) -> Conversation:
        conversation = Conversation(mode=mode)
        self.conversations.insert(0, conversation)
        try:
            self._save_all()
        except Exception:
            self.conversations = [item for item in self.conversations if item.id != conversation.id]
            raise
        return conversation

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        return next((item for item in self.conversations if item.id == conversation_id), None)

    def delete_conversation(self, conversation_id: str) -> None:
        previous = self.conversations
        self.conversations = [item for item in self.conversations if item.id != conversation_id]
        try:
            self._save_all()
        except Exception:
            self.conversations = previous
            raise

    def save_conversation(self, conversation: Conversation) -> None:
        previous = self.conversations
        replacement = {item.id: item for item in self.conversations}
        replacement[conversation.id] = conversation
        self.conversations = sorted(replacement.values(), key=lambda item: item.updated_at, reverse=True)
        try:
            self._save_all()
        except Exception:
            self.conversations = previous
            raise

    def update_settings(self, settings: AppSettings) -> None:
        settings.validate()
        settings.ollama_url = settings.ollama_url.strip().rstrip("/")
        settings.default_model = settings.default_model.strip()
        # Read from the active provider when possible so callers can safely edit the current
        # settings object before passing it back to the core.
        old_endpoint = str(getattr(self.provider, "base_url", self.settings.ollama_url)).rstrip("/")
        endpoint_changed = settings.ollama_url.rstrip("/") != old_endpoint
        replacement_provider = self._provider_factory(settings.ollama_url) if endpoint_changed else self.provider
        self.settings_store.save(settings)
        self.settings = settings
        self.provider = replacement_provider

    def select_model(self, model: str) -> None:
        model = model.strip()
        if not model:
            raise ValueError("Choose a model before saving.")
        previous = self.settings.default_model
        self.settings.default_model = model
        try:
            self.settings_store.save(self.settings)
        except Exception:
            self.settings.default_model = previous
            raise

    def prepare_generation(self, conversation: Conversation, submission: ModeSubmission) -> PreparedGeneration:
        """Build one request before any user message is persisted or provider work begins."""

        submission.validate()
        request_context = format_user_content(submission)
        display = submission.prompt.strip()
        if not display:
            display = f"[{submission.mode.label} request with attached {submission.language} code]"
        elif submission.code.strip():
            display += f"\n\n[Attached {submission.language} code]"
        user_message = ChatMessage(
            role="user",
            content=display,
            context=request_context,
            mode=submission.mode,
        )
        request = ChatRequest(
            model=self.settings.default_model,
            messages=compose_messages(conversation.messages, submission),
            temperature=self.settings.temperature,
            num_ctx=self.settings.num_ctx,
            num_predict=self.settings.num_predict,
            stream=self.settings.response_streaming,
        )
        return PreparedGeneration(request=request, user_message=user_message)

    def record_user_message(self, conversation: Conversation, message: ChatMessage) -> None:
        previous_mode = conversation.mode
        previous_title = conversation.title
        previous_updated_at = conversation.updated_at
        conversation.mode = message.mode or conversation.mode
        conversation.add_message(message)
        try:
            self.save_conversation(conversation)
        except Exception:
            if conversation.messages and conversation.messages[-1] is message:
                conversation.messages.pop()
            conversation.mode = previous_mode
            conversation.title = previous_title
            conversation.updated_at = previous_updated_at
            raise

    def record_assistant_message(
        self,
        conversation: Conversation,
        content: str,
        *,
        model: str,
        mode: AppMode,
        interrupted: bool = False,
    ) -> ChatMessage:
        message = ChatMessage(
            role="assistant",
            content=content,
            model=model,
            mode=mode,
            interrupted=interrupted,
        )
        previous_updated_at = conversation.updated_at
        conversation.add_message(message)
        try:
            self.save_conversation(conversation)
        except Exception:
            if conversation.messages and conversation.messages[-1] is message:
                conversation.messages.pop()
            conversation.updated_at = previous_updated_at
            raise
        return message

    def stream_generation(self, request: ChatRequest, cancel_event: Event) -> Iterator[StreamChunk]:
        """Delegate to the provider only after the UI has committed a local user message."""

        yield from self.provider.stream_chat(request, cancel_event)

    def _save_all(self) -> None:
        self.conversations = sorted(self.conversations, key=lambda item: item.updated_at, reverse=True)
        self.conversation_store.save_all(self.conversations)
