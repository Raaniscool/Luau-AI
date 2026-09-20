"""Future boundary for an explicit, user-authorized Roblox Studio integration.

No transport is implemented here. DukeOTR must not pretend it can inspect or modify a Studio
place merely because a desktop client is running.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StudioIntegrationStatus:
    available: bool
    message: str


@dataclass(frozen=True)
class StudioRequest:
    """Narrow, reviewable envelope for a future local plugin/companion channel."""

    operation: str
    payload: dict[str, Any]
    user_confirmed: bool


class RobloxStudioBridge(ABC):
    """A future implementation must obtain explicit consent for each sensitive operation."""

    @abstractmethod
    def status(self) -> StudioIntegrationStatus:
        ...

    @abstractmethod
    def send(self, request: StudioRequest) -> dict[str, Any]:
        """Perform a documented local operation or raise a clear integration error."""


class UnconfiguredRobloxStudioBridge(RobloxStudioBridge):
    """Truthful default instead of a fake Studio integration."""

    def status(self) -> StudioIntegrationStatus:
        return StudioIntegrationStatus(
            available=False,
            message="Roblox Studio integration is not configured. DukeOTR has not read, modified, or connected to Studio.",
        )

    def send(self, request: StudioRequest) -> dict[str, Any]:
        if not request.user_confirmed:
            raise PermissionError("A future Studio operation requires explicit user confirmation.")
        raise RuntimeError(
            "Roblox Studio integration is a documented future boundary only; no plugin, local companion, or place connection is configured."
        )
