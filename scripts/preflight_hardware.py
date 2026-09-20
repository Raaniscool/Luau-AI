"""Inspect whether this machine is appropriate for the configured QLoRA experiment.

The check deliberately requires a CUDA-capable NVIDIA GPU for the shipped training path.
It does not treat CPU fallback or a small shared-memory integrated GPU as a successful
fine-tuning environment.
"""

from __future__ import annotations

# Support both `python -m scripts.name` and `python scripts/name.py` from the repo.
if __package__ in {None, ""}:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import ctypes
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.lib.io_utils import read_json, utc_now, write_json_atomic


def system_ram_gb() -> float | None:
    try:
        if sys.platform.startswith("win"):
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return round(status.ullTotalPhys / 1024**3, 2)
        elif Path("/proc/meminfo").exists():
            values = Path("/proc/meminfo").read_text(encoding="utf-8")
            for line in values.splitlines():
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) * 1024 / 1024**3, 2)
    except (OSError, ValueError, AttributeError):
        pass
    return None


def nvidia_gpus() -> list[dict[str, Any]]:
    command = ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    gpus: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) >= 2:
            try:
                memory_mib = int(fields[1])
            except ValueError:
                memory_mib = None
            gpus.append(
                {
                    "name": fields[0],
                    "memory_mib": memory_mib,
                    "memory_gib": round(memory_mib / 1024, 2) if memory_mib is not None else None,
                    "driver_version": fields[2] if len(fields) > 2 else None,
                }
            )
    return gpus


def torch_details() -> dict[str, Any]:
    try:
        import torch  # type: ignore

        details: dict[str, Any] = {
            "installed": True,
            "version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
        }
        if torch.cuda.is_available():
            details["device_count"] = torch.cuda.device_count()
            details["devices"] = [
                {
                    "name": torch.cuda.get_device_name(index),
                    "memory_gib": round(torch.cuda.get_device_properties(index).total_memory / 1024**3, 2),
                }
                for index in range(torch.cuda.device_count())
            ]
        return details
    except ImportError:
        return {"installed": False, "cuda_available": False}
    except Exception as exc:  # pragma: no cover - driver mismatches are environment-specific
        return {"installed": True, "cuda_available": False, "error": str(exc)}


def assess_hardware(minimum_cuda_vram_gb: float, minimum_system_ram_gib: float | None = None) -> dict[str, Any]:
    """Return a serializable assessment used by CLI and training entry points.

    CUDA/VRAM is the hard guard for the shipped QLoRA path. Host RAM is recorded as a
    separately visible recommendation because a workload may technically start below the
    planning target yet be too fragile to endorse as a normal DukeOTR run.
    """
    torch_info = torch_details()
    gpus = nvidia_gpus()
    detected_vram = [gpu["memory_gib"] for gpu in gpus if isinstance(gpu.get("memory_gib"), (int, float))]
    if torch_info.get("cuda_available"):
        detected_vram.extend(
            item["memory_gib"] for item in torch_info.get("devices", []) if isinstance(item.get("memory_gib"), (int, float))
        )
    largest_vram = max(detected_vram) if detected_vram else 0.0
    system_ram = system_ram_gb()
    suitable = bool(torch_info.get("cuda_available")) and largest_vram >= minimum_cuda_vram_gb
    meets_ram_recommendation = (
        minimum_system_ram_gib is None or system_ram is None or system_ram >= minimum_system_ram_gib
    )
    if suitable and meets_ram_recommendation:
        recommendation = (
            "Suitable for a conservative local QLoRA pilot. Start with the configured batch size of 1, "
            "gradient checkpointing, and a small subset; monitor VRAM, host RAM, and evaluation quality."
        )
    elif suitable:
        recommendation = (
            f"CUDA/VRAM meets the QLoRA guard, but detected system RAM ({system_ram:.2f} GiB) is below the "
            f"configured {minimum_system_ram_gib:.0f} GiB planning recommendation. A run may be fragile; "
            "increase host RAM or proceed only after an explicitly measured pilot."
        )
    elif torch_info.get("cuda_available"):
        recommendation = (
            f"CUDA is present but the largest detected VRAM ({largest_vram:.2f} GiB) is below the "
            f"configured {minimum_cuda_vram_gb:.0f} GiB recommendation. Use a larger cloud/GPU machine or reduce "
            "the experiment only after measuring memory safely."
        )
    else:
        recommendation = (
            "No suitable NVIDIA CUDA training device was detected. This machine can run the local Ollama "
            "generation/review/evaluation pipeline, but use cloud or another CUDA GPU machine for QLoRA training."
        )
    return {
        "checked_at": utc_now(),
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "python": sys.version.split()[0]},
        "system_ram_gib": system_ram,
        "minimum_recommended_system_ram_gib": minimum_system_ram_gib,
        "meets_recommended_system_ram": meets_ram_recommendation,
        "nvidia_gpus": gpus,
        "torch": torch_info,
        "minimum_recommended_cuda_vram_gib": minimum_cuda_vram_gb,
        "largest_detected_cuda_vram_gib": largest_vram,
        "suitable_for_configured_qlora": suitable,
        "local_supported_work": [
            "Ollama inference", "example generation", "LLM review", "validation", "dataset assembly", "baseline and candidate evaluation"
        ],
        "cloud_or_suitable_cuda_required_for": ["QLoRA/LoRA adapter training", "checkpointing", "merged model export"],
        "recommendation": recommendation,
        "important_note": "An Intel integrated GPU with approximately 2 GB shared/available VRAM is not treated as adequate for this QLoRA configuration.",
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Preflight the configured QLoRA training hardware")
    value.add_argument("--config", default="configs/qlora_sft.json")
    value.add_argument("--output", default="reports/hardware_preflight.json")
    value.add_argument("--require-suitable", action="store_true", help="Exit nonzero if QLoRA should not be started here")
    value.add_argument("--print-json", action="store_true")
    return value


def run(arguments: argparse.Namespace) -> int:
    config = read_json(arguments.config)
    report = assess_hardware(
        float(config.get("minimum_recommended_cuda_vram_gb", 16)),
        float(config["minimum_recommended_system_ram_gib"]) if config.get("minimum_recommended_system_ram_gib") is not None else None,
    )
    report["config"] = str(arguments.config)
    write_json_atomic(arguments.output, report)
    if arguments.print_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(report["recommendation"])
        print(f"Hardware report: {arguments.output}")
    return 0 if report["suitable_for_configured_qlora"] or not arguments.require_suitable else 2


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"hardware preflight error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
