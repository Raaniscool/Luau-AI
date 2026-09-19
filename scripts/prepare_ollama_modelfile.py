"""Prepare, but do not import, an Ollama Modelfile for a future DukeOTR artifact.

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

from scripts.lib.io_utils import utc_now, write_json_atomic


DEFAULT_DUKEOTR_CANDIDATE_TAG = "dukeotr-v1"
DEFAULT_DUKEOTR_RELEASE_ALIAS = "dukeotr"
DEFAULT_DUKEOTR_SYSTEM_PROMPT = (
    "You are DukeOTR, a careful Roblox and Luau engineering assistant. "
    "Keep servers authoritative and validate untrusted client input."
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Write an explicitly reviewed DukeOTR Ollama Modelfile")
    value.add_argument("--base", required=True, help="Exact matching Ollama base tag or matching local GGUF/model path")
    value.add_argument("--adapter", required=True, help="Converted adapter path (prefer a verified GGUF adapter)")
    value.add_argument("--output-dir", default="models/ollama")
    value.add_argument(
        "--name",
        default=DEFAULT_DUKEOTR_CANDIDATE_TAG,
        help="Planned DukeOTR candidate tag; no Ollama model is created by this script",
    )
    value.add_argument(
        "--release-alias",
        default=DEFAULT_DUKEOTR_RELEASE_ALIAS,
        help="Planned stable DukeOTR alias after the candidate passes the release gate",
    )
    value.add_argument("--system", default=DEFAULT_DUKEOTR_SYSTEM_PROMPT)
    value.add_argument("--allow-safetensors-template", action="store_true", help="Write a template for safetensors only after independently verifying compatibility")
    return value


def _require_dukeotr_tag(value: str, argument_name: str) -> None:
    if not value.casefold().startswith("dukeotr"):
        raise ValueError(f"{argument_name} must use a DukeOTR-facing tag, not {value!r}")


def run(arguments: argparse.Namespace) -> int:
    adapter = Path(arguments.adapter)
    if not adapter.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter}")
    _require_dukeotr_tag(arguments.name, "--name")
    _require_dukeotr_tag(arguments.release_alias, "--release-alias")
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
# Public model identity: DukeOTR
# Planned candidate tag: {arguments.name}
# Planned stable release alias: {arguments.release_alias}
# Adapter: {adapter.resolve()}
# Technical base/provenance: {arguments.base}
# The adapter must have been trained against this exact base; mismatches cause erratic output.
FROM {arguments.base}
ADAPTER {adapter.resolve()}
PARAMETER temperature 0.2
PARAMETER num_ctx 8192
SYSTEM """{arguments.system}"""
'''
    modelfile.write_text(contents, encoding="utf-8")
    identity_manifest = output_dir / "dukeotr_model_identity.json"
    write_json_atomic(
        identity_manifest,
        {
            "schema_version": "1.0",
            "status": "prepared_not_created",
            "public_display_name": "DukeOTR",
            "candidate_ollama_tag": arguments.name,
            "planned_release_ollama_tag": arguments.release_alias,
            "adapter_path": str(adapter.resolve()),
            "technical_base_reference": arguments.base,
            "compatibility_requirement": "The adapter must be trained against the exact exported base architecture and weights.",
            "release_gate": [
                "quality-gated dataset manifest",
                "completed compatible export",
                "held-out evaluation comparison against the preserved base-model baseline",
                "no unresolved critical security regression",
                "human release approval",
            ],
            "created_at": utc_now(),
        },
    )
    readme = output_dir / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "DukeOTR export preparation only; no Ollama model was created by this script.",
                f"Prepared candidate identity: DukeOTR ({arguments.name})",
                f"Reserved stable release alias: DukeOTR ({arguments.release_alias})",
                "",
                "Only after compatibility review, create the versioned candidate:",
                f"  ollama create {arguments.name} -f {modelfile.name}",
                f"  ollama run {arguments.name}",
                "",
                f"Only after the held-out evaluation and human release gate, create the stable DukeOTR alias:",
                f"  ollama create {arguments.release_alias} -f {modelfile.name}",
                f"  ollama run {arguments.release_alias}",
                "",
                "Important: a QLoRA adapter trained from Hugging Face Qwen/Qwen3-4B is not automatically compatible",
                "with an arbitrary quantized Ollama qwen3:4b tag. Prefer a verified matching GGUF conversion or merge",
                "against the exact source base, preserve provenance, and evaluate the result against the held-out suite before release.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Prepared {modelfile} and {identity_manifest}; no `ollama create` command was executed. "
        f"Planned DukeOTR candidate: {arguments.name}."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Ollama export preparation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
