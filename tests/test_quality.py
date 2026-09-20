from __future__ import annotations

import unittest

from scripts.lib.io_utils import read_jsonl
from scripts.lib.quality import parse_review, static_validate
from scripts.lib.schema import STATIC_CHECKER_VERSION, make_generated_record, quality_gate_status
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

    def test_known_remoteevent_baseline_hallucinations_are_blocked(self) -> None:
        record = reviewed_record()
        record["messages"][2]["content"] = (
            "RemoteEvents are inherently secure, and clients cannot call RemoteEvent:FireServer. "
            "This intentionally long incorrect explanation is only a test fixture and has enough characters "
            "to pass the minimum answer-length check while the factual networking checks reject it."
        )
        result = static_validate(record)
        codes = {item["code"] for item in result["issues"] if item["severity"] == "block"}
        self.assertIn("network.false_claim_client_cannot_fireserver", codes)
        self.assertIn("network.false_claim_remote_inherently_secure", codes)

    def test_remoteevent_variant_false_claims_are_blocked_but_quoted_corrections_are_allowed(self) -> None:
        record = reviewed_record()
        record["messages"][2]["content"] = (
            "Only the server can call FireServer, and a RemoteEvent automatically validates client input. "
            "This intentionally long incorrect explanation is only a test fixture and has enough characters "
            "to pass the minimum answer-length check while deterministic networking checks reject it."
        )
        blocked = static_validate(record)
        codes = {item["code"] for item in blocked["issues"] if item["severity"] == "block"}
        self.assertIn("network.false_claim_client_cannot_fireserver", codes)
        self.assertIn("network.false_claim_remote_inherently_secure", codes)

        record["messages"][2]["content"] = (
            "A RemoteEvent is a secure, server-controlled mechanism. The server must explicitly fire "
            "RemoteEvent:FireServer, and RemoteEvents only let the server trigger actions. This intentionally "
            "long incorrect explanation is a fixture that verifies this baseline-style wording is blocked too."
        )
        baseline_style = static_validate(record)
        baseline_codes = {item["code"] for item in baseline_style["issues"] if item["severity"] == "block"}
        self.assertIn("network.false_claim_client_cannot_fireserver", baseline_codes)
        self.assertIn("network.false_claim_remote_inherently_secure", baseline_codes)

        record["messages"][2]["content"] = (
            'The statement "Clients cannot call RemoteEvent:FireServer" is false. RemoteEvents do not automatically '\
            "validate client input; validate requests on the server. This corrected explanation is deliberately "
            "long enough for the static test fixture and describes the safe server-authoritative behavior."
        )
        corrected = static_validate(record)
        corrected_codes = {item["code"] for item in corrected["issues"]}
        self.assertNotIn("network.false_claim_client_cannot_fireserver", corrected_codes)
        self.assertNotIn("network.false_claim_remote_inherently_secure", corrected_codes)

    def test_known_hallucinated_player_api_is_blocked(self) -> None:
        record = reviewed_record()
        record["messages"][2]["content"] = good_answer() + "\n```luau\nif player:IsAuthenticated() then\n\tprint('ok')\nend\n```"
        result = static_validate(record)
        self.assertTrue(any(item["code"] == "api.nonexistent_player_is_authenticated" for item in result["issues"]))

    def test_client_context_cannot_connect_on_server_event_or_treat_localplayer_as_character(self) -> None:
        record = reviewed_record()
        record["messages"][2]["content"] = """This is deliberately invalid instructional code used to lock in the observed pilot regression.

```luau
-- ServerScriptService (authoritative)
local PlayerScore = 0

-- StarterPlayerScripts (client-side)
local player = game.Players:GetPlayerFromCharacter(game.Players.LocalPlayer)
local RemoteEvent = game.ReplicatedStorage:WaitForChild("ScoreUpdate")
RemoteEvent:OnServerEvent:Connect(function()
    PlayerScore = PlayerScore + 1
end)
```

The validator must reject this rather than letting a reviewer accept it."""
        result = static_validate(record)
        codes = {item["code"] for item in result["issues"]}
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["checker"], STATIC_CHECKER_VERSION)
        self.assertIn("network.onserverevent_client_context", codes)
        self.assertIn("api.getplayerfromcharacter_localplayer", codes)

    def test_current_static_checker_is_required_for_final_eligibility(self) -> None:
        record = reviewed_record()
        record["quality"]["static"]["checker"] = "static-v1"
        accepted, reasons = quality_gate_status(record)
        self.assertFalse(accepted)
        self.assertIn("static_validation_checker_version_not_current", reasons)

    def test_phase1_literal_code_evidence_is_enforced(self) -> None:
        phase1_seed = next(item for item in read_jsonl("raw_data/dukeotr_phase1_luau_seed_tasks.jsonl") if item["id"] == "dukeotr-phase1-001")
        record = make_generated_record(
            phase1_seed,
            """This deliberately incomplete beginner answer has a number and a string but omits the
required boolean and nil literals from its code block.

```luau
local playerName = "Raani"
local score = 0
print(playerName, score)
```

It must not be treated as satisfying the authored source brief just because the prose mentions types.""",
            generator={"kind": "test"},
            variant=1,
        )
        result = static_validate(record)
        codes = {item["code"] for item in result["issues"]}
        self.assertEqual(result["status"], "fail")
        self.assertIn("coverage.required_code_pattern_boolean_literal", codes)
        self.assertIn("coverage.required_code_pattern_nil_literal", codes)

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
