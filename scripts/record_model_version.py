"""Validate and update DukeOTR future-version metadata without performing model actions.

This utility records only provided metadata. It cannot train, inspect Ollama, create adapters,
export weights, run evaluation, or turn a planned DukeOTR identity into a model claim.
"""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from scripts.lib.io_utils import read_json, sha256_file, write_json_atomic
from scripts.lib.training_factory import TrainingFactoryError, assert_safe_factory_output_path, validate_model_version_registry


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Validate or explicitly update DukeOTR version metadata only")
    value.add_argument("--registry", default="training_factory/model_versions.json")
    value.add_argument("--check", action="store_true", help="Validate registry only")
    value.add_argument("--entry", default=None, help="JSON file containing one complete version metadata object")
    value.add_argument("--output", default=None, help="Output registry path required when --entry is supplied")
    value.add_argument("--replace-id", default=None, help="Explicitly replace an existing version ID instead of adding a new ID")
    value.add_argument(
        "--verify-evidence-files",
        action="store_true",
        help="Required for trained/evaluated/released status metadata; hash the supplied local artifact/evaluation evidence files before recording",
    )
    return value


def _verify_claimed_evidence(entry: dict[str, Any], *, required: bool) -> None:
    status = entry.get("status")
    actual_statuses = {"trained_pending_evaluation", "evaluated_not_released", "released"}
    if status not in actual_statuses:
        return
    if not required:
        raise ValueError(
            "A trained/evaluated/released version claim requires --verify-evidence-files on the machine holding the actual artifacts"
        )
    checks = [("adapter_artifact", "adapter_sha256")]
    if status in {"evaluated_not_released", "released"}:
        checks.append(("evaluation_evidence", "evaluation_evidence_sha256"))
    for path_key, hash_key in checks:
        artifact = entry.get(path_key)
        declared_hash = entry.get(hash_key)
        if not isinstance(artifact, str) or not artifact:
            raise ValueError(f"{path_key} must be a non-empty local file path")
        path = Path(artifact)
        if not path.is_file():
            raise ValueError(f"Evidence file does not exist or is not a regular file: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != declared_hash:
            raise ValueError(f"SHA-256 mismatch for {path_key}: declared metadata does not match {path}")


def run(arguments: argparse.Namespace) -> int:
    registry = read_json(arguments.registry)
    if not isinstance(registry, dict):
        raise ValueError("Version registry must be a JSON object")
    problems = validate_model_version_registry(registry)
    if problems:
        raise ValueError(f"Invalid version registry: {problems}")
    if arguments.check:
        if arguments.entry or arguments.output or arguments.replace_id or arguments.verify_evidence_files:
            raise ValueError("--check cannot be combined with update arguments")
        print(f"Version registry is valid: {arguments.registry}")
        return 0
    if not arguments.entry or not arguments.output:
        raise ValueError("Supply --check, or both --entry and --output for an explicit metadata update")
    assert_safe_factory_output_path(arguments.output)
    entry = read_json(arguments.entry)
    if not isinstance(entry, dict):
        raise ValueError("Version entry must be a JSON object")
    identifier = entry.get("id")
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("Version entry requires non-empty id")
    _verify_claimed_evidence(entry, required=arguments.verify_evidence_files)
    clone = deepcopy(registry)
    versions = clone["versions"]
    existing_indexes = [index for index, row in enumerate(versions) if isinstance(row, dict) and row.get("id") == identifier]
    if existing_indexes:
        if arguments.replace_id != identifier:
            raise ValueError(
                f"Version ID {identifier!r} already exists; use --replace-id {identifier} to replace metadata explicitly"
            )
        versions[existing_indexes[0]] = entry
    else:
        if arguments.replace_id:
            raise ValueError("--replace-id may only be used for an existing matching ID")
        versions.append(entry)
    updated_problems = validate_model_version_registry(clone)
    if updated_problems:
        raise ValueError(f"Refusing invalid version metadata update: {updated_problems}")
    write_json_atomic(arguments.output, clone)
    print(
        f"Wrote validated DukeOTR version metadata to {arguments.output}. "
        "This records metadata only; it does not train, export, import, or release any model."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError, TrainingFactoryError) as exc:
        print(f"model version metadata error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
