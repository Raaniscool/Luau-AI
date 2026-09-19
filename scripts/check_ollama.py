"""Verify the existing local Ollama registration without downloading or running a model."""

from __future__ import annotations

# Support both `python -m scripts.check_ollama` and direct Windows invocation.
if __package__ in {None, ""}:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import sys
from pathlib import Path

from scripts.lib.io_utils import utc_now, write_json_atomic
from scripts.lib.ollama_cli import OllamaCliError, ensure_ollama_model


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Run `ollama list` and verify an existing local model tag")
    value.add_argument("--model", default="qwen3:4b")
    value.add_argument("--timeout", type=int, default=30)
    value.add_argument("--report", default=None, help="Optional local JSON preflight report")
    return value


def run(arguments: argparse.Namespace) -> int:
    if arguments.timeout < 1:
        raise ValueError("--timeout must be at least one second")
    preflight = ensure_ollama_model(arguments.model, timeout_seconds=arguments.timeout)
    print("`ollama list` succeeded.")
    print(preflight.raw_output)
    print(f"Verified existing local model tag: {preflight.model}")
    if arguments.report:
        write_json_atomic(
            Path(arguments.report),
            {
                "stage": "ollama_cli_preflight",
                "checked_at": utc_now(),
                "model": preflight.model,
                "available_models": list(preflight.available_models),
                "raw_ollama_list": preflight.raw_output,
                "action_taken": "verification_only_no_download_no_inference",
            },
        )
        print(f"Preflight report: {arguments.report}")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (ValueError, OSError, OllamaCliError) as exc:
        print(f"Ollama preflight error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
