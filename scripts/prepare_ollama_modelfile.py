"""Prepare, but do not silently import, an Ollama Modelfile for a trained adapter.

Compatibility is strict: an adapter must be paired with the exact base architecture and
weights it was trained against. In particular, do not assume a QLoRA adapter trained from
`Qwen/Qwen3-4B` is compatible with the downloaded quantized `qwen3:4b` Ollama tag.
"""

from __future__ import annotations

# Support both `python -m scripts.name` and `python scripts/name.py` from the repo.
if __package__ in {None, ""}:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import sys
from pathlib import Path

from scripts.lib.io_utils import utc_now


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Write an explicitly reviewed Ollama Modelfile")
    value.add_argument("--base", required=True, help="Exact matching Ollama base tag or matching local GGUF/model path")
    value.add_argument("--adapter", required=True, help="Converted adapter path (prefer a verified GGUF adapter)")
    value.add_argument("--output-dir", default="models/ollama")
    value.add_argument("--name", default="dukeotr-v1-qwen3-4b")
    value.add_argument("--system", default="You are a careful Roblox and Luau engineering assistant. Keep servers authoritative and validate untrusted client input.")
    value.add_argument("--allow-safetensors-template", action="store_true", help="Write a template for safetensors only after independently verifying compatibility")
    return value


def run(arguments: argparse.Namespace) -> int:
    adapter = Path(arguments.adapter)
    if not adapter.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter}")
    suffix = adapter.suffix.lower()
    if suffix != ".gguf" and not arguments.allow_safetensors_template:
        raise ValueError(
            "Refusing to imply direct QLoRA/safetensors compatibility. Convert/merge and verify first, "
            "then provide a GGUF adapter, or use --allow-safetensors-template after checking Ollama support."
        )
    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    modelfile = output_dir / "Modelfile"
    contents = f'''# Generated {utc_now()} -- review base/adapter compatibility before `ollama create`.
# Adapter: {adapter.resolve()}
# Base: {arguments.base}
# The adapter must have been trained against this exact base; mismatches cause erratic output.
FROM {arguments.base}
ADAPTER {adapter.resolve()}
PARAMETER temperature 0.2
PARAMETER num_ctx 8192
SYSTEM """{arguments.system}"""
'''
    modelfile.write_text(contents, encoding="utf-8")
    readme = output_dir / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "Ollama export preparation only; no model was created by this script.",
                f"Suggested command after compatibility review:",
                f"  ollama create {arguments.name} -f {modelfile.name}",
                f"  ollama run {arguments.name}",
                "",
                "Important: a QLoRA adapter trained from Hugging Face Qwen/Qwen3-4B is not automatically compatible",
                "with an arbitrary quantized Ollama qwen3:4b tag. Prefer a verified matching GGUF conversion or merge",
                "against the exact source base and evaluate the result against the held-out suite before use.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {modelfile}; no `ollama create` command was executed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Ollama export preparation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
