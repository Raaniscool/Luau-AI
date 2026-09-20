"""Local-only JSON persistence for DukeOTR settings and conversation history."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from desktop_app.core.models import APP_SCHEMA_VERSION, AppSettings, Conversation


class LocalStorageError(RuntimeError):
    """Saving user-owned local state failed; callers should keep the UI usable and explain it."""


def default_data_directory() -> Path:
    """Return a per-user directory without putting prompt/history data in the repository."""

    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if root:
            return Path(root) / "DukeOTR"
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "dukeotr"
    return Path.home() / ".local" / "share" / "dukeotr"


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.chmod(temporary, 0o600)
    except OSError:
        # Windows ACLs govern this more precisely; lack of chmod support must not lose history.
        pass
    os.replace(temporary, path)


def _quarantine_bad_json(path: Path) -> str:
    """Keep corrupt user data for recovery instead of silently overwriting it."""

    suffix = ".corrupt"
    candidate = path.with_name(path.name + suffix)
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}{suffix}.{counter}")
        counter += 1
    try:
        os.replace(path, candidate)
        return f"Unreadable local data was preserved as {candidate.name}; a fresh file will be used."
    except OSError:
        return "Unreadable local data could not be moved; a fresh in-memory state will be used until saving succeeds."


class SettingsStore:
    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory).expanduser() if directory is not None else default_data_directory()
        self.path = self.directory / "settings.json"
        self.last_warning: str | None = None

    def load(self) -> AppSettings:
        self.last_warning = None
        if not self.path.exists():
            return AppSettings(data_directory=str(self.directory))
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            settings = AppSettings.from_dict(value)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.last_warning = f"Settings could not be read ({exc}). " + _quarantine_bad_json(self.path)
            return AppSettings(data_directory=str(self.directory))
        # The active store is the source of truth for the current run. This prevents a stale
        # saved directory field from redirecting reads somewhere surprising.
        settings.data_directory = str(self.directory)
        return settings

    def save(self, settings: AppSettings) -> None:
        settings.data_directory = str(self.directory)
        try:
            _atomic_write_json(self.path, settings.as_dict())
        except OSError as exc:
            raise LocalStorageError(
                f"Could not save local DukeOTR settings at {self.path}. Check free disk space and folder permissions. ({exc})"
            ) from exc


class ConversationStore:
    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory).expanduser() if directory is not None else default_data_directory()
        self.path = self.directory / "conversations.json"
        self.last_warning: str | None = None

    def load_all(self) -> list[Conversation]:
        self.last_warning = None
        if not self.path.exists():
            return []
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            if not isinstance(value, dict) or not isinstance(value.get("conversations"), list):
                raise ValueError("Conversation history has an unexpected schema")
            conversations = [Conversation.from_dict(item) for item in value["conversations"]]
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.last_warning = f"Conversation history could not be read ({exc}). " + _quarantine_bad_json(self.path)
            return []
        # Most recently updated first gives the sidebar useful ordering without changing IDs.
        return sorted(conversations, key=lambda item: item.updated_at, reverse=True)

    def save_all(self, conversations: list[Conversation]) -> None:
        identifiers = [item.id for item in conversations]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Cannot persist duplicate conversation IDs")
        try:
            _atomic_write_json(
                self.path,
                {
                    "schema_version": APP_SCHEMA_VERSION,
                    "conversations": [item.as_dict() for item in conversations],
                },
            )
        except OSError as exc:
            raise LocalStorageError(
                f"Could not save local DukeOTR history at {self.path}. Check free disk space and folder permissions. ({exc})"
            ) from exc

    def upsert(self, conversation: Conversation) -> list[Conversation]:
        conversations = self.load_all()
        replacement = {item.id: item for item in conversations}
        replacement[conversation.id] = conversation
        updated = sorted(replacement.values(), key=lambda item: item.updated_at, reverse=True)
        self.save_all(updated)
        return updated

    def delete(self, conversation_id: str) -> list[Conversation]:
        conversations = [item for item in self.load_all() if item.id != conversation_id]
        self.save_all(conversations)
        return conversations
