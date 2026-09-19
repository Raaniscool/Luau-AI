"""Local Ollama CLI preflight used before model-facing pipeline stages."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


class OllamaCliError(RuntimeError):
    pass


@dataclass(frozen=True)
class OllamaCliPreflight:
    model: str
    available_models: tuple[str, ...]
    raw_output: str


def ollama_list(*, timeout_seconds: int = 30) -> tuple[tuple[str, ...], str]:
    """Run exactly `ollama list` and return its discovered model tags and raw output."""
    try:
        completed = subprocess.run(
            ["ollama", "list"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise OllamaCliError(
            "The `ollama` command is not available on PATH. This check does not download anything; "
            "start/check your existing Ollama installation, then run `ollama list`."
        ) from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OllamaCliError(f"Could not run `ollama list`: {exc}") from exc
    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        raise OllamaCliError(f"`ollama list` failed with exit code {completed.returncode}: {output.strip()}")
    models: list[str] = []
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    for line in lines[1:]:  # First line is normally: NAME ID SIZE MODIFIED
        first_column = line.split()[0] if line.split() else ""
        if first_column:
            models.append(first_column)
    return tuple(models), output.strip()


def ensure_ollama_model(model: str, *, timeout_seconds: int = 30) -> OllamaCliPreflight:
    models, output = ollama_list(timeout_seconds=timeout_seconds)
    if model not in models:
        available = ", ".join(models) or "(none)"
        raise OllamaCliError(
            f"`ollama list` completed, but {model!r} is not registered. Available: {available}. "
            "Do not download a replacement automatically; confirm the exact existing local tag first."
        )
    return OllamaCliPreflight(model=model, available_models=models, raw_output=output)
