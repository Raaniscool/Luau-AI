from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from datetime import date
from unittest.mock import patch
from pathlib import Path

from scripts.audit_verified_knowledge import main as audit_main
from scripts.query_verified_knowledge import main as query_main
from scripts.lib.io_utils import read_json, sha256_file
from scripts.lib.request_routing import route_user_request
from scripts.lib.verified_knowledge import (
    load_entries,
    load_source_catalog,
    match_fast_answer,
    render_answer,
    source_index,
    validate_entry,
)
from scripts.lib.verified_knowledge_audit import audit_verified_knowledge


class VerifiedKnowledgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.knowledge_config = read_json("configs/verified_knowledge.json")
        cls.effort_config = read_json("configs/builder_verifier_reviewer.json")
        cls.entries = load_entries(cls.knowledge_config["entries_file"])
        cls.sources = source_index(load_source_catalog(cls.knowledge_config["sources_file"]))

    def _match(self, prompt: str) -> dict:
        return match_fast_answer(
            prompt,
            entries=self.entries,
            sources=self.sources,
            matcher_config=self.knowledge_config["matcher"],
            as_of=date(2026, 9, 19),
        )

    def _route(self, prompt: str) -> dict:
        return route_user_request(
            prompt,
            effort_config=self.effort_config,
            knowledge_config=self.knowledge_config,
            entries=self.entries,
            sources=self.sources,
            as_of=date(2026, 9, 19),
        )

    def test_initial_requested_concepts_are_source_checked_and_valid(self) -> None:
        expected = {
            "vk-roblox-event",
            "vk-rbxscript-signal-connect",
            "vk-remote-event",
            "vk-remote-function",
            "vk-script",
            "vk-local-script",
            "vk-module-script",
            "vk-replicated-storage",
            "vk-server-script-service",
            "vk-server-storage",
            "vk-starter-player",
            "vk-starter-gui",
            "vk-players-service",
            "vk-player",
            "vk-character",
            "vk-humanoid",
            "vk-instance",
            "vk-instance-new",
            "vk-find-first-child",
            "vk-wait-for-child",
            "vk-vector3",
            "vk-cframe",
            "vk-color3",
            "vk-tween-service",
            "vk-run-service",
            "vk-user-input-service",
            "vk-context-action-service",
        }
        self.assertEqual({entry["id"] for entry in self.entries}, expected)
        self.assertEqual(len(self.entries), 27)
        self.assertEqual(len(self.sources), 19)
        supported = set(self.knowledge_config["supported_api_names"])
        rejected = set(self.knowledge_config["rejected_api_names"])
        for entry in self.entries:
            errors = validate_entry(
                entry,
                sources=self.sources,
                allowed_api_names=supported,
                rejected_api_names=rejected,
                known_code_book_ids={
                    "cb-client-server-replication-containers",
                    "cb-instances-hierarchy-service-access",
                    "cb-luau-values-scope-nil-operators",
                    "cb-modules-oop-service-boundaries",
                    "cb-remoteevents-client-server-authority",
                    "cb-remotefunctions-response-boundaries",
                    "cb-roblox-event-connections-lifecycle",
                    "cb-roblox-players-character-lifecycle",
                    "cb-roblox-ui-input-buttons",
                    "cb-services-instances-events",
                    "cb-tween-runservice-frame-performance",
                    "cb-ui-client-view-authority",
                    "cb-workspace-raycasting-physics-validation",
                },
            )
            self.assertEqual(errors, [], entry["id"])
            self.assertTrue({"quick", "normal", "deep"}.issubset(entry["explanations"]), entry["id"])
            self.assertEqual(entry["training_policy"], "not_training_data_or_auto_promotion")
            self.assertEqual(entry["evaluation_policy"], "not_sourced_from_held_out_evaluation")

    def test_semantic_event_phrasings_are_the_same_verified_fast_path(self) -> None:
        for prompt in ("What is a Roblox event?", "What’s an event in Roblox?", "Explain Roblox events."):
            result = self._match(prompt)
            self.assertEqual(result["status"], "matched", prompt)
            self.assertEqual(result["concept_id"], "vk-roblox-event")
            self.assertEqual(result["depth"], "normal")
            self.assertIn("Roblox Creator Hub", result["sources"][0]["publisher"])
            self.assertIn("create.roblox.com", result["sources"][0]["url"])

    def test_depth_selection_is_curated_and_source_attributed(self) -> None:
        quick = self._match("Briefly explain Roblox events.")
        normal = self._match("Explain Roblox events.")
        deep = self._match("Explain Roblox events in detail.")
        self.assertEqual((quick["status"], quick["depth"]), ("matched", "quick"))
        self.assertEqual((normal["status"], normal["depth"]), ("matched", "normal"))
        self.assertEqual((deep["status"], deep["depth"]), ("matched", "deep"))
        self.assertNotIn("**Example", quick["answer"])
        self.assertIn("**Verified source(s)**", quick["answer"])
        self.assertIn("**Example", normal["answer"])
        self.assertIn("**Common misconception(s)**", deep["answer"])
        self.assertIn("**Verified source(s)**", deep["answer"])

    def test_connect_alternate_wording_and_normal_remote_question_fast_path(self) -> None:
        connect = self._route("What does Connect do in Roblox?")
        connect_method = self._route("What is :Connect()?")
        remote = self._route("What is a RemoteEvent?")
        self.assertEqual(connect["route"], "verified_knowledge_fast_answer")
        self.assertEqual(connect["fast_answer"]["concept_id"], "vk-rbxscript-signal-connect")
        self.assertEqual(connect_method["route"], "verified_knowledge_fast_answer")
        self.assertEqual(connect_method["fast_answer"]["concept_id"], "vk-rbxscript-signal-connect")
        self.assertEqual(remote["route"], "verified_knowledge_fast_answer")
        self.assertEqual(remote["fast_answer"]["concept_id"], "vk-remote-event")
        self.assertIsNone(remote["adaptive_route"])

    def test_fast_path_skips_effort_classifier_and_fallback_calls_existing_classifier(self) -> None:
        with patch("scripts.lib.request_routing.classify_task") as classify:
            fast = route_user_request(
                "What is a Roblox event?",
                effort_config=self.effort_config,
                knowledge_config=self.knowledge_config,
                entries=self.entries,
                sources=self.sources,
                as_of=date(2026, 9, 19),
            )
        self.assertEqual(fast["route"], "verified_knowledge_fast_answer")
        classify.assert_not_called()

        from scripts.lib.effort_routing import classify_task as existing_classifier

        with patch("scripts.lib.request_routing.classify_task", wraps=existing_classifier) as classify:
            fallback = route_user_request(
                "What is a banana?",
                effort_config=self.effort_config,
                knowledge_config=self.knowledge_config,
                entries=self.entries,
                sources=self.sources,
                as_of=date(2026, 9, 19),
            )
        self.assertEqual(fallback["route"], "adaptive_pipeline")
        classify.assert_called_once()
        self.assertEqual(fallback["adaptive_route"]["level"], "normal")

    def test_ambiguous_multi_concept_and_detailed_client_server_requests_escalate(self) -> None:
        cases = {
            "What is this event?": "requires_adaptive_pipeline",
            "What is RemoteEvent and RemoteFunction?": "multiple_concepts_or_ambiguous_match",
            "Explain detailed RemoteEvent behavior between the client and server.": "requires_adaptive_pipeline",
        }
        for prompt, reason in cases.items():
            result = self._route(prompt)
            self.assertEqual(result["route"], "adaptive_pipeline", prompt)
            self.assertEqual(result["fast_answer"]["reason"], reason)
            self.assertIsNotNone(result["adaptive_route"])
        complex_route = self._route("Explain detailed RemoteEvent behavior between the client and server.")
        self.assertEqual(complex_route["adaptive_route"]["level"], "complex")
        self.assertIn("client_server_security", complex_route["adaptive_route"]["selected_checks"])

    def test_custom_code_unrelated_and_unsupported_depth_fall_back_to_existing_router(self) -> None:
        for prompt, expected_reason in (
            ("Write a custom RemoteEvent inventory purchase system.", "requires_adaptive_pipeline"),
            ("What is a banana?", "no_confident_verified_concept_match"),
            ("Explain RemoteEvent in detail.", "requested_depth_not_verified:deep"),
        ):
            result = self._route(prompt)
            self.assertEqual(result["route"], "adaptive_pipeline", prompt)
            self.assertEqual(result["fast_answer"]["reason"], expected_reason)
            self.assertIn("level", result["adaptive_route"])
        custom = self._route("Write a custom RemoteEvent inventory purchase system.")
        self.assertEqual(custom["adaptive_route"]["level"], "complex")
        self.assertIn("client_server_security", custom["adaptive_route"]["selected_checks"])
        remote_entry = next(entry for entry in self.entries if entry["id"] == "vk-remote-event")
        with self.assertRaises(ValueError):
            render_answer(remote_entry, "deep", self.sources)

    def test_strict_audit_passes_and_reports_held_out_isolation_without_prompt_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "verified-audit.json"
            code = audit_main(["--strict", "--output", str(report_path)])
            report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["library"]["entry_count"], 27)
        self.assertEqual(report["library"]["source_count"], 19)
        self.assertEqual(len(report["library"]["entries_sha256"]), 64)
        self.assertEqual(len(report["library"]["sources_sha256"]), 64)
        self.assertEqual(report["evaluation_isolation"]["status"], "clear")
        self.assertEqual(report["evaluation_isolation"]["collision_count"], 0)
        self.assertEqual(report["evaluation_isolation"]["collisions"], [])
        rendered = json.dumps(report)
        self.assertNotIn('"prompt"', rendered)
        self.assertNotIn('"rubric"', rendered)

    def test_strict_audit_identifies_stale_entries(self) -> None:
        report = audit_verified_knowledge(
            self.knowledge_config,
            as_of=date(2027, 3, 19),
            strict=True,
        )
        self.assertEqual(report["status"], "fail")
        self.assertEqual(len(report["freshness"]["stale_entry_ids"]), 27)
        self.assertTrue(any(item["code"] == "freshness.stale_entry" for item in report["errors"]))

    def test_unsupported_api_name_is_rejected_by_entry_audit(self) -> None:
        broken = copy.deepcopy(self.entries[0])
        broken["api_names"].append("RemoteEvent:SecureFireServer")
        errors = validate_entry(
            broken,
            sources=self.sources,
            allowed_api_names=set(self.knowledge_config["supported_api_names"]),
            rejected_api_names=set(self.knowledge_config["rejected_api_names"]),
            known_code_book_ids=None,
        )
        self.assertTrue(any(item["code"] == "entry.rejected_api" for item in errors))

    def test_output_clis_cannot_overwrite_held_out_suite(self) -> None:
        evaluation = self.knowledge_config["evaluation_file"]
        before = sha256_file(evaluation)
        with contextlib.redirect_stderr(io.StringIO()):
            query_code = query_main(
                [
                    "--prompt",
                    "What is a Roblox event?",
                    "--output",
                    evaluation,
                ]
            )
            audit_code = audit_main(["--strict", "--output", evaluation])
            with tempfile.TemporaryDirectory() as directory:
                alternate_evaluation = Path(directory) / "not-held-out.jsonl"
                alternate_evaluation.write_text('{"id":"temporary","prompt":"unused"}\n', encoding="utf-8")
                bypass_attempt_code = audit_main(
                    ["--strict", "--evaluation", str(alternate_evaluation), "--output", evaluation]
                )
        after = sha256_file(evaluation)
        self.assertEqual(query_code, 2)
        self.assertEqual(audit_code, 2)
        self.assertEqual(bypass_attempt_code, 2)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
