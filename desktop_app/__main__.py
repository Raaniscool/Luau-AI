"""Run DukeOTR as a native desktop application with ``python -m desktop_app``."""

from __future__ import annotations

from desktop_app.resources import configure_windows_app_id


def main() -> None:
    """Load tkinter lazily so a missing desktop runtime gets an actionable message."""

    configure_windows_app_id()
    try:
        from desktop_app.ui import run_desktop_app
    except ModuleNotFoundError as exc:
        if exc.name == "tkinter":
            raise SystemExit(
                "DukeOTR needs Python's tkinter/Tcl-Tk desktop component. On Windows, rerun the CPython installer "
                "and enable Tcl/Tk, then start `python -m desktop_app` again."
            ) from exc
        raise
    run_desktop_app()


if __name__ == "__main__":
    main()
