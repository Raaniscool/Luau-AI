"""Create a transfer bundle for a future DukeOTR CUDA/cloud training machine.

The bundle contains only a verified, versioned final dataset plus the training configuration
and dependency list. It does not download a base model, invoke Ollama, train an adapter, or
create an Ollama model. The ZIP is intentionally ignored by Git so reviewed training data and
large artifacts remain outside repository history.
"""

from __future__ import annotations

# Support both `python -m scripts.name` and `python scripts/name.py` from the repo.
if __package__ in {None, ""}:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from scripts.lib.io_utils import canonical_json, read_json, sha256_file, utc_now

ROOT = Path(__file__).resolve().parents[1]
_REQUIRED_DATASET_FILES = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "final": "final_dataset.jsonl",
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Package a verified DukeOTR dataset for a future CUDA/cloud training machine")
    value.add_argument("--config", default="configs/qlora_sft.json")
    value.add_argument(
        "--dataset-dir",
        default=None,
        help="Defaults to the parent directory of dataset_manifest in the training config",
    )
    value.add_argument(
        "--output",
        default="artifacts/dukeotr_v1_training_bundle.zip",
        help="Ignored ZIP output; transfer it separately from Git to the training machine",
    )
    value.add_argument("--overwrite", action="store_true", help="Replace an existing bundle only after reviewing it")
    return value


def _safe_archive_path(value: str | Path) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Archive path must be relative and must not contain '..': {value}")
    return path.as_posix()


def _config_archive_path(config_path: Path) -> str:
    """Keep ad-hoc local configs portable without embedding absolute workstation paths."""
    try:
        return _safe_archive_path(config_path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return _safe_archive_path(Path("configs") / config_path.name)


def repository_provenance() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        return {"git_revision": revision, "working_tree_clean": not bool(dirty)}
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return {"git_revision": None, "working_tree_clean": None}


def validate_bundle_inputs(config: dict[str, Any], dataset_dir: Path) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    expected_version = config.get("planned_dataset_version")
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Dataset manifest must be a JSON object: {manifest_path}")
    if manifest.get("stage") != "final_dataset_creation":
        raise ValueError("Dataset manifest was not created by the final-dataset stage")
    if manifest.get("dataset_version_status") != "created" or manifest.get("dataset_version") != expected_version:
        raise ValueError(
            f"Dataset is not the required finalized version {expected_version!r}; "
            "build it with --dataset-version only after quality review."
        )
    hashes = manifest.get("file_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("Dataset manifest has no file hashes; rebuild it with the current final-dataset builder.")
    paths = {role: dataset_dir / filename for role, filename in _REQUIRED_DATASET_FILES.items()}
    for role, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required {role} dataset artifact is missing: {path}")
        if hashes.get(role) != sha256_file(path):
            raise ValueError(f"SHA-256 mismatch for {role} dataset artifact: {path}")
    counts = manifest.get("partition_counts", {})
    if not isinstance(counts, dict):
        raise ValueError("Dataset manifest partition_counts is missing")
    return manifest_path, manifest, paths


def write_bundle(output: Path, *, config_path: Path, config: dict[str, Any], dataset_dir: Path) -> None:
    manifest_path, dataset_manifest, dataset_paths = validate_bundle_inputs(config, dataset_dir)
    configured_dataset_dir = Path(str(config["dataset_manifest"])).parent
    try:
        archive_dataset_dir = _safe_archive_path(configured_dataset_dir)
    except ValueError:
        archive_dataset_dir = _safe_archive_path(Path("training_data") / dataset_dir.name)
    config_archive_path = _config_archive_path(config_path)
    requirements = ROOT / "requirements" / "training.txt"
    pyproject = ROOT / "pyproject.toml"
    for path in (requirements, pyproject):
        if not path.exists():
            raise FileNotFoundError(f"Required reproducibility file is missing: {path}")

    bundle_manifest = {
        "schema_version": "1.0",
        "status": "prepared_for_transfer_not_trained",
        "created_at": utc_now(),
        "public_model_identity": config.get("project_name", "DukeOTR"),
        "dataset": {
            "version": dataset_manifest.get("dataset_version"),
            "manifest_sha256": sha256_file(manifest_path),
            "file_sha256": dict(dataset_manifest.get("file_sha256", {})),
            "partition_counts": dataset_manifest.get("partition_counts"),
            "quality_gate": dataset_manifest.get("quality_gate"),
            "archive_directory": archive_dataset_dir,
        },
        "training": {
            "config_archive_path": config_archive_path,
            "config_sha256": sha256_file(config_path),
            "base_model": config.get("base_model"),
            "base_model_revision_requested": config.get("base_model_revision"),
            "base_model_revision_policy": config.get("base_model_revision_policy"),
            "method": config.get("method"),
            "planned_adapter_version": config.get("planned_adapter_version"),
            "planned_candidate_ollama_tag": config.get("planned_candidate_ollama_tag"),
        },
        "repository_provenance": repository_provenance(),
        "transfer_instructions": [
            "Clone the recorded Git revision on a suitable CUDA/cloud training machine.",
            "Extract this ZIP at the cloned repository root so training_data and configs preserve their archive paths.",
            "Run python -m scripts.preflight_hardware --config configs/qlora_sft.json --require-suitable.",
            "Run python -m scripts.train_qlora --config configs/qlora_sft.json first without --execute to record a plan.",
            "Only then run the explicitly reviewed --execute command; this bundle contains no model weights or trained adapter.",
        ],
    }
    bundle_readme = "\n".join(
        [
            "DukeOTR training transfer bundle — prepared only, not trained.",
            "",
            "This ZIP contains a hash-verified final dataset, its manifest, the QLoRA configuration,",
            "and Python dependency declarations. It does not contain Qwen weights, an adapter, checkpoints,",
            "an Ollama model, or a training result.",
            "",
            "On the CUDA/cloud training machine:",
            "  1. Clone the repository at the git revision recorded in training_bundle_manifest.json.",
            "  2. Extract this archive at the repository root.",
            "  3. Follow docs/TRAINING_MACHINE_RUNBOOK.md.",
            "  4. Verify hashes with the training plan before using --execute.",
            "",
            "Do not commit this bundle or its contents to Git.",
        ]
    ) + "\n"

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", suffix=".zip", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.write(manifest_path, f"{archive_dataset_dir}/dataset_manifest.json")
            for role, source in dataset_paths.items():
                archive.write(source, f"{archive_dataset_dir}/{_REQUIRED_DATASET_FILES[role]}")
            archive.write(config_path, config_archive_path)
            archive.write(requirements, "requirements/training.txt")
            archive.write(pyproject, "pyproject.toml")
            archive.writestr("training_bundle_manifest.json", canonical_json(bundle_manifest) + "\n")
            archive.writestr("README.txt", bundle_readme)
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def run(arguments: argparse.Namespace) -> int:
    config_path = Path(arguments.config)
    config = read_json(config_path)
    if not isinstance(config, dict):
        raise ValueError(f"Training config must be a JSON object: {config_path}")
    configured_manifest = config.get("dataset_manifest")
    if not isinstance(configured_manifest, str) or not configured_manifest:
        raise ValueError("Training config requires dataset_manifest")
    dataset_dir = Path(arguments.dataset_dir) if arguments.dataset_dir else Path(configured_manifest).parent
    output = Path(arguments.output)
    if output.exists() and not arguments.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing training bundle: {output}. Review it or use --overwrite deliberately.")
    write_bundle(output, config_path=config_path, config=config, dataset_dir=dataset_dir)
    print(f"Prepared hash-verified DukeOTR training bundle: {output}. No model download, training, adapter, or Ollama import was performed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, FileExistsError, ValueError, OSError, zipfile.BadZipFile) as exc:
        print(f"training bundle error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
