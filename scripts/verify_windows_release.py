#!/usr/bin/env python3
"""Verify the static layout of a portable DukeOTR PyInstaller release.

This intentionally does not launch the GUI or contact Ollama.  It confirms that the Windows
executable, runtime icon, and Builder routing configuration were actually collected, and that no
model/checkpoint artifact was accidentally included.  Run it after a Windows PyInstaller build.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Iterable


_REQUIRED_RESOURCE_SUFFIXES = (
    ("desktop_app", "assets", "dukeotr.ico"),
    ("configs", "builder_verifier_reviewer.json"),
)
_MODEL_SUFFIXES = {".gguf", ".safetensors", ".onnx", ".pt", ".pth", ".ckpt", ".bin"}
_MODEL_NAME_PREFIXES = ("adapter_model", "pytorch_model", "model-")


def _relative_matches(path: Path, package_directory: Path, suffix: tuple[str, ...]) -> bool:
    try:
        parts = path.relative_to(package_directory).parts
    except ValueError:
        return False
    return len(parts) >= len(suffix) and tuple(part.casefold() for part in parts[-len(suffix) :]) == tuple(part.casefold() for part in suffix)


def _find_resource(package_directory: Path, suffix: tuple[str, ...]) -> Path | None:
    leaf = suffix[-1]
    for candidate in package_directory.rglob(leaf):
        if candidate.is_file() and _relative_matches(candidate, package_directory, suffix):
            return candidate
    return None


def _model_artifacts(package_directory: Path) -> Iterable[Path]:
    for candidate in package_directory.rglob("*"):
        if not candidate.is_file():
            continue
        name = candidate.name.casefold()
        if candidate.suffix.casefold() in _MODEL_SUFFIXES or name.startswith(_MODEL_NAME_PREFIXES):
            yield candidate


def _valid_windows_icon(path: Path) -> bool:
    try:
        raw = path.read_bytes()
        reserved, icon_type, image_count = struct.unpack("<HHH", raw[:6])
    except (OSError, struct.error):
        return False
    return reserved == 0 and icon_type == 1 and image_count >= 1


def _valid_json_file(path: Path) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(value, dict)


def validate_release(package_directory: str | Path) -> list[str]:
    """Return human-readable packaging violations; an empty list means the layout is sound."""

    root = Path(package_directory)
    errors: list[str] = []
    executable = root / "DukeOTR.exe"
    if not root.is_dir():
        return [f"Portable package directory does not exist: {root}"]
    if not executable.is_file() or executable.stat().st_size == 0:
        errors.append(f"Expected non-empty Windows executable is missing: {executable}")

    resources: dict[tuple[str, ...], Path] = {}
    for suffix in _REQUIRED_RESOURCE_SUFFIXES:
        resource = _find_resource(root, suffix)
        if resource is None:
            errors.append(f"Required packaged resource is missing: {'/'.join(suffix)}")
        else:
            resources[suffix] = resource

    icon = resources.get(("desktop_app", "assets", "dukeotr.ico"))
    if icon is not None and not _valid_windows_icon(icon):
        errors.append(f"Packaged DukeOTR icon is not a valid ICO file: {icon.relative_to(root)}")
    routing_config = resources.get(("configs", "builder_verifier_reviewer.json"))
    if routing_config is not None and not _valid_json_file(routing_config):
        errors.append(f"Packaged Builder routing configuration is not valid JSON: {routing_config.relative_to(root)}")

    forbidden = sorted(_model_artifacts(root))
    for artifact in forbidden:
        errors.append(f"Model/checkpoint artifact must not be bundled: {artifact.relative_to(root)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a portable DukeOTR Windows release without launching it.")
    parser.add_argument("--package-dir", type=Path, required=True, help="Path to the generated DukeOTR portable folder")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable result")
    args = parser.parse_args()

    package_directory = args.package_dir.resolve()
    errors = validate_release(package_directory)
    payload = {
        "kind": "dukeotr_portable_release_verification",
        "package_directory": str(package_directory),
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "non_claim": "This verifies package layout only. It does not launch the Windows executable or contact Ollama.",
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif errors:
        print("DukeOTR portable release verification failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
    else:
        print(f"DukeOTR portable release layout verified: {package_directory}")
        print("This check did not launch DukeOTR.exe or contact Ollama.")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
