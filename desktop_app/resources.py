"""Paths for DukeOTR's checked-in UI resources in source and PyInstaller builds."""

from __future__ import annotations

import sys
from pathlib import Path


def configure_windows_app_id() -> None:
    """Give the frozen Windows process a stable taskbar/shortcut application identity.

    Failure is deliberately non-fatal: the native executable icon still remains available, and
    source launches on platforms other than Windows do not need an AppUserModelID.
    """

    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("DukeOTR.Desktop")
    except (AttributeError, OSError):
        pass


def application_resource_path(*parts: str) -> Path:
    """Return a bundled desktop-app resource without relying on the working directory.

    PyInstaller extracts data files to ``sys._MEIPASS`` at runtime.  Source launches resolve
    from this package directory instead, so ``python -m desktop_app`` remains the development
    workflow and a packaged ``DukeOTR.exe`` can be launched from a shortcut or any directory.
    """

    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root) / "desktop_app" / Path(*parts)
    return Path(__file__).resolve().parent / Path(*parts)


def application_icon_path() -> Path:
    """Return the original DukeOTR Windows icon included in the portable package."""

    return application_resource_path("assets", "dukeotr.ico")
