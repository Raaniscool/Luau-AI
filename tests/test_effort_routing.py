from __future__ import annotations

import unittest

from scripts.lib.effort_routing import classify_task
from scripts.lib.io_utils import read_json


class EffortRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = read_json("configs/builder_verifier_reviewer.json")

    def test_low_risk_greeting_uses_minimal_non_api_path(self) -> None:
        route = classify_task(
            {
                "id": "routing-simple-greeting",
                "title": "Greeting",
                "user_request": "Hello!",
                "task_type": "question_answer",
                "difficulty": "beginner",
                "concepts": [],
                "tags": ["conversation"],
                "requirements": [],
                "expected_evidence": [],
            },
            self.config,
        )
        self.assertEqual(route.level, "simple")
        self.assertEqual(route.reviewer_passes, 0)
        self.assertEqual(route.default_max_rounds, 1)
        self.assertEqual(route.code_book_context_limit, 0)
        self.assertEqual(route.selected_checks, ("structure_and_safety",))
        self.assertNotIn("roblox_api", route.selected_checks)
        self.assertNotIn("client_server_security", route.selected_checks)

    def test_normal_luau_request_gets_code_checks_not_networking_checks(self) -> None:
        route = classify_task(
            {
                "id": "routing-normal-luau",
                "title": "Typed score helper",
                "user_request": "Write a small Luau function that returns the larger of two scores.",
                "task_type": "code_generation",
                "difficulty": "intermediate",
                "concepts": ["Luau functions", "numbers"],
                "tags": ["luau", "functions"],
                "requirements": ["Use a clear return value."],
                "expected_evidence": ["A focused function."],
            },
            self.config,
        )
        self.assertEqual(route.level, "normal")
        self.assertEqual(route.reviewer_passes, 1)
        self.assertIn("luau_code", route.selected_checks)
        self.assertNotIn("client_server_security", route.selected_checks)
        self.assertNotIn("persistence_economy", route.selected_checks)

    def test_remote_request_escalates_to_complex_security_path(self) -> None:
        route = classify_task(
            {
                "id": "routing-complex-remote",
                "title": "Secure purchase RemoteEvent",
                "user_request": "Build a secure RemoteEvent purchase request from a client to a server.",
                "task_type": "requirements_implementation",
                "difficulty": "intermediate",
                "concepts": ["RemoteEvent", "server validation"],
                "tags": ["remoteevents", "security", "client-server"],
                "requirements": ["Validate item ID.", "Use server-owned prices."],
                "expected_evidence": ["Server validation order."],
            },
            self.config,
        )
        self.assertEqual(route.level, "complex")
        self.assertEqual(route.reviewer_passes, 2)
        self.assertGreaterEqual(route.default_max_rounds, 3)
        self.assertIn("client_server_security", route.selected_checks)
        self.assertIn("testability", route.selected_checks)
        self.assertIn("client_server_security", route.risk_labels)

    def test_positive_high_risk_requirement_escalates_but_negative_scope_does_not(self) -> None:
        positive = classify_task(
            {
                "id": "routing-positive-requirement",
                "title": "Reward action",
                "user_request": "Implement the requested feature.",
                "task_type": "requirements_implementation",
                "difficulty": "intermediate",
                "concepts": [],
                "tags": [],
                "requirements": ["Validate every RemoteEvent request on the server."],
                "expected_evidence": [],
            },
            self.config,
        )
        negative = classify_task(
            {
                "id": "routing-negative-scope",
                "title": "Pure Luau scope lesson",
                "user_request": "Teach local variables with a tiny Luau example.",
                "task_type": "question_answer",
                "difficulty": "intermediate",
                "concepts": ["local scope"],
                "tags": ["luau"],
                "requirements": ["Do not introduce RemoteEvents or client/server placement."],
                "expected_evidence": [],
            },
            self.config,
        )
        self.assertEqual(positive.level, "complex")
        self.assertIn("client_server_security", positive.risk_labels)
        self.assertEqual(negative.level, "normal")
        self.assertNotIn("client_server_security", negative.risk_labels)

    def test_ambiguous_request_never_uses_simple_fast_path(self) -> None:
        route = classify_task(
            {
                "id": "routing-ambiguous",
                "title": "Unclear wording",
                "user_request": "What is this? The requirement is unclear.",
                "task_type": "question_answer",
                "difficulty": "beginner",
                "concepts": [],
                "tags": [],
                "requirements": [],
                "expected_evidence": [],
            },
            self.config,
        )
        self.assertEqual(route.level, "complex")
        self.assertIn("ambiguous:unclear", route.reasons)
        self.assertEqual(route.as_dict()["level"], "complex")


if __name__ == "__main__":
    unittest.main()
