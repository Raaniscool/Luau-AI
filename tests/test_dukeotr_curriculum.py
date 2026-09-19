from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_dukeotr_curriculum import REQUIRED_CONCEPTS, REQUIRED_INSTRUCTIONAL_MODES, audit, main
from scripts.lib.io_utils import read_jsonl
from scripts.lib.prompts import TASK_TYPE_RESPONSE_GUIDANCE, generation_prompt
from scripts.lib.schema import VALID_TASK_TYPES, validate_seed


class DukeOTRCurriculumTests(unittest.TestCase):
    def test_phase1_catalog_passes_curriculum_and_isolation_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "phase1_audit.json"
            code = main(["--strict", "--output", str(report)])
            self.assertEqual(code, 0)
            payload = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(payload["project_name"], "DukeOTR")
        self.assertEqual(payload["status"], "pass")
        self.assertEqual(payload["source_brief_count"], 45)
        self.assertEqual(payload["identity_errors"], [])
        self.assertEqual(payload["missing_concepts"], [])
        self.assertEqual(payload["missing_instructional_modes"], [])
        self.assertEqual(payload["evaluation_id_collisions"], [])
        self.assertEqual(payload["evaluation_prompt_collisions"], [])

    def test_catalog_records_and_requested_modes_are_schema_supported(self) -> None:
        seeds = list(read_jsonl("raw_data/dukeotr_phase1_luau_seed_tasks.jsonl"))
        self.assertEqual(len(seeds), 45)
        self.assertTrue(set(REQUIRED_INSTRUCTIONAL_MODES).issubset(VALID_TASK_TYPES))
        for seed in seeds:
            self.assertEqual(validate_seed(seed), [], seed["id"])
        terms = {term.casefold() for seed in seeds for term in [*seed["concepts"], *seed["tags"]]}
        self.assertTrue({term.casefold() for term in REQUIRED_CONCEPTS}.issubset(terms))

    def test_each_supported_instructional_mode_has_generation_guidance(self) -> None:
        self.assertEqual(set(TASK_TYPE_RESPONSE_GUIDANCE), VALID_TASK_TYPES)
        seeds = list(read_jsonl("raw_data/dukeotr_phase1_luau_seed_tasks.jsonl"))
        for seed in seeds:
            guidance = TASK_TYPE_RESPONSE_GUIDANCE[seed["task_type"]]
            self.assertIn(guidance, generation_prompt(seed, variant=1))

    def test_audit_rejects_evaluation_prompt_collision(self) -> None:
        seeds = list(read_jsonl("raw_data/dukeotr_phase1_luau_seed_tasks.jsonl"))
        evaluation = list(read_jsonl("evaluation_data/roblox_luau_eval.jsonl"))
        seeds[0] = dict(seeds[0])
        seeds[0]["user_request"] = evaluation[0]["prompt"]
        project_config = json.loads(Path("configs/dukeotr_project.json").read_text(encoding="utf-8"))
        report = audit(
            seeds,
            evaluation,
            project_config,
            seed_path="test-seeds.jsonl",
            evaluation_path="test-evaluation.jsonl",
            project_config_path="test-project.json",
        )
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["evaluation_prompt_collisions"], ["dukeotr-phase1-001"])


if __name__ == "__main__":
    unittest.main()
