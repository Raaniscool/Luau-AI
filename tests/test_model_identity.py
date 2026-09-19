from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.lib.prompts import DUKEOTR_EVALUATION_SYSTEM, EVALUATION_SYSTEM, evaluation_system_for_model


class ModelIdentityTests(unittest.TestCase):
    def test_project_contract_reserves_dukeotr_candidate_and_release_tags(self) -> None:
        project = json.loads(Path("configs/dukeotr_project.json").read_text(encoding="utf-8"))
        identity = project["model_identity"]
        self.assertEqual(project["project_name"], "DukeOTR")
        self.assertEqual(identity["display_name"], "DukeOTR")
        self.assertEqual(identity["planned_first_adapter_version"], "dukeotr_v1")
        self.assertEqual(identity["planned_versioned_ollama_tag"], "dukeotr-v1")
        self.assertEqual(identity["planned_release_ollama_tag"], "dukeotr")
        self.assertEqual(identity["status"], "planned_not_trained")

    def test_candidate_evaluation_uses_dukeotr_identity_without_changing_baseline(self) -> None:
        candidate_system, candidate_identity = evaluation_system_for_model("dukeotr-v1", run_kind="candidate")
        baseline_system, baseline_identity = evaluation_system_for_model("qwen3:4b", run_kind="baseline")
        mislabeled_baseline_system, mislabeled_baseline_identity = evaluation_system_for_model("dukeotr", run_kind="baseline")

        self.assertEqual(candidate_system, DUKEOTR_EVALUATION_SYSTEM)
        self.assertEqual(candidate_identity, "DukeOTR")
        self.assertIn("You are DukeOTR", candidate_system)
        self.assertEqual(baseline_system, EVALUATION_SYSTEM)
        self.assertEqual(baseline_identity, "unbranded_base_or_external")
        self.assertEqual(mislabeled_baseline_system, EVALUATION_SYSTEM)
        self.assertEqual(mislabeled_baseline_identity, "unbranded_base_or_external")


if __name__ == "__main__":
    unittest.main()
