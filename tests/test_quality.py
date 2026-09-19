from __future__ import annotations

import unittest

from scripts.lib.quality import parse_review, static_validate
from scripts.lib.schema import quality_gate_status
from tests.helpers import good_answer, reviewed_record


class QualityTests(unittest.TestCase):
    def test_correct_negated_trust_client_guidance_is_not_a_blocker(self) -> None:
        result = static_validate(reviewed_record())
        self.assertEqual(result["status"], "pass")
        self.assertFalse(any(item["code"] == "security.trust_client" for item in result["issues"]))

    def test_unsafe_dynamic_execution_is_blocked(self) -> None:
        record = reviewed_record()
        record["messages"][2]["content"] = good_answer() + "\n```luau\nloadstring(payload)()\n```"
        result = static_validate(record)
        self.assertEqual(result["status"], "fail")
        self.assertTrue(any(item["code"] == "security.loadstring" and item["severity"] == "block" for item in result["issues"]))

    def test_reviewer_policy_downgrades_low_security_acceptance(self) -> None:
        review = parse_review(
            '{"decision":"accept","scores":{"accuracy":5,"security":2,"requirement_coverage":5,"pedagogy":4},"blocking_issues":[],"required_fixes":[],"strengths":["clear"],"api_claims_to_verify":[],"summary":"x"}',
            minimums={"accuracy": 4, "security": 4, "pedagogy": 3},
        )
        self.assertEqual(review["reported_decision"], "accept")
        self.assertEqual(review["decision"], "revise")

    def test_final_gate_requires_deduplication(self) -> None:
        record = reviewed_record()
        record["quality"]["deduplication"]["status"] = "not_run"
        ok, reasons = quality_gate_status(record)
        self.assertFalse(ok)
        self.assertIn("deduplication_not_unique", reasons)


if __name__ == "__main__":
    unittest.main()
