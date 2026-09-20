"""Clean bridge from interactive Builder mode to existing DukeOTR routing policy.

This bridge deliberately reuses only deterministic planning metadata. The repository's complete
Builder -> Reviewer/Tester -> Fixer runner operates on curated train briefs with isolation and
provenance controls; it is not silently run on arbitrary conversation text.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.lib.effort_routing import EffortRoute, classify_task


@dataclass(frozen=True)
class BuilderPlanningContext:
    level: str
    selected_checks: tuple[str, ...]
    risk_labels: tuple[str, ...]
    reviewer_passes: int
    note: str

    def summary(self) -> str:
        checks = ", ".join(self.selected_checks) or "baseline structure and safety"
        risks = ", ".join(self.risk_labels) or "none detected"
        return (
            f"Adaptive route: {self.level}. Planning checks: {checks}. Risk labels: {risks}. "
            f"The curated-trace policy would request {self.reviewer_passes} reviewer pass(es). {self.note}"
        )


def _repository_root() -> Path:
    # PyInstaller exposes bundled configuration files under _MEIPASS; source checkouts keep
    # them at the repository root. This is a resource lookup only, never a model artifact path.
    bundled_root = getattr(sys, "_MEIPASS", None)
    if bundled_root:
        return Path(bundled_root)
    return Path(__file__).resolve().parents[2]


def _runtime_seed(prompt: str) -> dict[str, Any]:
    return {
        "id": "desktop-interactive-request",
        "title": "Interactive DukeOTR Builder request",
        "user_request": prompt,
        "task_type": "code_generation",
        "difficulty": "intermediate",
        "concepts": [],
        "tags": ["desktop_interactive", "not_training_data"],
        "requirements": [],
        "expected_evidence": [],
    }


def plan_builder_request(prompt: str, config_path: str | Path | None = None) -> BuilderPlanningContext:
    """Classify a local Builder prompt without creating a dataset record or hitting Ollama."""

    path = Path(config_path) if config_path is not None else _repository_root() / "configs" / "builder_verifier_reviewer.json"
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        route: EffortRoute = classify_task(_runtime_seed(prompt), config)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return BuilderPlanningContext(
            level="unavailable",
            selected_checks=(),
            risk_labels=(),
            reviewer_passes=0,
            note=(
                "Existing routing metadata is unavailable in this installation; the request remains an interactive "
                f"first pass only. ({exc})"
            ),
        )
    return BuilderPlanningContext(
        level=route.level,
        selected_checks=route.selected_checks,
        risk_labels=route.risk_labels,
        reviewer_passes=route.reviewer_passes,
        note=(
            "This desktop request does not invoke the curated Builder -> Reviewer/Tester -> Fixer trace or create training data."
        ),
    )
