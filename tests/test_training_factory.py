from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_failures import main as analyze_failures_main
from scripts.build_training_ledger import main as ledger_main
from scripts.generate_targeted_briefs import main as targeted_main
from scripts.promote_targeted_briefs import main as promote_main
from scripts.record_failures import main as record_failures_main
from scripts.record_model_version import main as record_model_version_main
from scripts.lib.io_utils import read_json, read_jsonl, write_json_atomic, write_jsonl_atomic
from scripts.lib.schema import STATIC_CHECKER_VERSION, clone_for_correction, make_generated_record, quality_gate_status, validate_seed
from scripts.lib.training_factory import (
    TrainingFactoryError,
    assert_safe_factory_output_path,
    build_targeted_drafts,
    factory_draft_eval_collisions,
    infer_failure_categories,
    make_failure_record,
    make_lineage_ledger_record,
    observed_failure,
    recommend_targeting,
    summarize_failures,
    validate_factory_draft,
    validate_failure_record,
    validate_ledger_record,
    validate_model_version_registry,
)
from tests.helpers import good_answer, reviewed_record, seed


ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = read_json(ROOT / "training_factory" / "failure_taxonomy.json")
TEMPLATES = read_json(ROOT / "training_factory" / "targeted_brief_templates.json")
VERSIONS = read_json(ROOT / "training_factory" / "model_versions.json")


def failed_record() -> dict:
    record = reviewed_record("train-factory-failure", "Review a RemoteEvent reward handler that trusts a client reward amount.")
    record["quality"]["static"] = {
        "status": "fail",
        "checker": STATIC_CHECKER_VERSION,
        "checked_at": "2026-09-20T00:00:00Z",
        "issues": [
            {
                "code": "security.trust_client",
                "message": "Server trusts client-controlled reward amount",
                "severity": "block",
            }
        ],
    }
    record["quality"]["llm_review"] = {
        "status": "complete",
        "decision": "revise",
        "checked_at": "2026-09-20T00:00:00Z",
        "review": {
            "blocking_issues": ["Do not accept a client-controlled reward amount."],
            "required_fixes": ["Look up the reward on the server and validate eligibility."],
            "strengths": [],
            "api_claims_to_verify": [],
            "summary": "The trust boundary is unsafe.",
            "scores": {"accuracy": 2, "security": 1, "requirement_coverage": 3, "pedagogy": 3},
        },
    }
    return record


def accepted_correction(parent: dict) -> dict:
    correction = clone_for_correction(
        parent,
        good_answer(),
        correction={"changes_made": ["Moved reward lookup and eligibility validation to the server."], "reason": "Fix trust boundary"},
    )
    correction["quality"]["static"] = {
        "status": "pass",
        "checker": STATIC_CHECKER_VERSION,
        "issues": [],
        "checked_at": "2026-09-20T00:00:00Z",
    }
    correction["quality"]["llm_review"] = {
        "status": "complete",
        "decision": "accept",
        "checked_at": "2026-09-20T00:00:00Z",
        "review": {"scores": {"accuracy": 5, "security": 5, "requirement_coverage": 5, "pedagogy": 4}},
    }
    correction["quality"]["deduplication"] = {
        "status": "unique",
        "duplicate_of": None,
        "similarity": None,
        "checked_at": "2026-09-20T00:00:00Z",
    }
    return correction


class TrainingFactoryLibraryTests(unittest.TestCase):
    def test_failure_record_preserves_actual_evidence_and_unknowns_honestly(self) -> None:
        record = failed_record()
        record["metadata"].pop("generator")
        self.assertTrue(observed_failure(record))
        categories = infer_failure_categories(record, taxonomy=TAXONOMY)
        self.assertIn("trusting_client_data", categories)
        failure = make_failure_record(
            record,
            taxonomy=TAXONOMY,
            candidate_artifact="validated_data/pilot.jsonl",
            created_at="2026-09-20T00:00:00Z",
        )
        self.assertFalse(failure["producing_agent"]["known"])
        self.assertEqual(failure["producing_agent"]["model"], None)
        self.assertEqual(failure["severity"], "critical")
        self.assertIn("trusting_client_data", failure["failure_categories"])
        self.assertEqual(failure["verification"]["status"], "no_verified_correction")
        self.assertFalse(validate_failure_record(failure, taxonomy=TAXONOMY))

    def test_failure_record_links_fix_and_revalidation_without_auto_acceptance(self) -> None:
        original = failed_record()
        correction = accepted_correction(original)
        failure = make_failure_record(
            original,
            taxonomy=TAXONOMY,
            candidate_artifact="validated_data/original.jsonl",
            correction=correction,
            created_at="2026-09-20T00:00:00Z",
        )
        self.assertEqual(failure["correction"]["corrected_record_id"], correction["record_id"])
        self.assertEqual(failure["verification"]["status"], "corrected_and_revalidated")
        # The failure stays a failure record even if the distinct corrected candidate passed its gates.
        self.assertTrue(failure["verification"]["quality_gate_eligible"])
        self.assertIn("Moved reward lookup", " ".join(failure["correction"]["explanation"]))

    def test_ledger_contains_builder_review_correction_and_quality_state(self) -> None:
        original = failed_record()
        correction = accepted_correction(original)
        ledger = make_lineage_ledger_record(
            original,
            taxonomy=TAXONOMY,
            candidate_artifact="validated_data/original.jsonl",
            correction=correction,
            created_at="2026-09-20T00:00:00Z",
        )
        self.assertEqual(ledger["builder"]["answer"], good_answer())
        self.assertEqual(ledger["correction"]["corrected_output"], good_answer())
        self.assertTrue(ledger["failure_analysis"]["observed_failure"])
        self.assertEqual(ledger["verification"]["status"], "corrected_and_revalidated")
        self.assertIsNone(ledger["provenance"]["dataset_version"])
        self.assertEqual(ledger["provenance"]["planned_dataset_version"], "dukeotr_dataset_v1")
        self.assertEqual(ledger["version"]["factory_schema_version"], "1.0")
        self.assertFalse(validate_ledger_record(ledger, taxonomy=TAXONOMY))

    def test_statistics_and_recommendations_come_from_rows_only(self) -> None:
        record = failed_record()
        failure = make_failure_record(
            record,
            taxonomy=TAXONOMY,
            candidate_artifact="validated_data/original.jsonl",
            created_at="2026-09-20T00:00:00Z",
        )
        summary = summarize_failures([failure], taxonomy=TAXONOMY)
        self.assertEqual(summary["failure_record_count"], 1)
        self.assertEqual(summary["category_counts"]["trusting_client_data"], 1)
        recommendations = recommend_targeting(summary, taxonomy=TAXONOMY)
        observed = {item["failure_category"] for item in recommendations}
        self.assertIn("trusting_client_data", observed)
        self.assertNotIn("incorrect_cframe_reasoning", observed)
        self.assertEqual(recommend_targeting(summary, taxonomy=TAXONOMY, minimum_failure_count=2), [])

    def test_targeted_drafts_are_deterministic_depth_labeled_and_review_required(self) -> None:
        profile = next(item for item in TEMPLATES["profiles"] if item["id"] == "remote_security")
        kwargs = {
            "profile": profile,
            "failure_category": "trusting_client_data",
            "observed_failure_count": 3,
            "counts_by_difficulty": {"beginner": 1, "intermediate": 1, "advanced": 1},
            "failure_input_sha256": "a" * 64,
            "seed": 3407,
            "created_at": "2026-09-20T00:00:00Z",
        }
        first = build_targeted_drafts(**kwargs)
        second = build_targeted_drafts(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual([item["task_depth"] for item in first], ["short", "normal", "deep"])
        self.assertTrue(all(item["status"] == "draft_requires_human_review" for item in first))
        self.assertTrue(all(item["targeting"]["requires_human_approval"] for item in first))
        self.assertTrue(all(not validate_factory_draft(item) for item in first))
        with self.assertRaises(TrainingFactoryError):
            build_targeted_drafts(**{**kwargs, "observed_failure_count": 0})

    def test_targeted_draft_local_held_out_collision_is_detected(self) -> None:
        profile = next(item for item in TEMPLATES["profiles"] if item["id"] == "remote_security")
        draft = build_targeted_drafts(
            profile=profile,
            failure_category="trusting_client_data",
            observed_failure_count=1,
            counts_by_difficulty={"beginner": 1, "intermediate": 0, "advanced": 0},
            failure_input_sha256="b" * 64,
            seed=1,
            created_at="2026-09-20T00:00:00Z",
        )[0]
        collisions = factory_draft_eval_collisions(
            [draft],
            [{"id": "eval-private", "prompt": draft["user_request"]}],
        )
        self.assertEqual(len(collisions), 1)
        self.assertEqual(collisions[0]["record_id"], draft["draft_id"])

    def test_version_registry_rejects_unsupported_trained_claim(self) -> None:
        self.assertFalse(validate_model_version_registry(VERSIONS))
        invalid = copy.deepcopy(VERSIONS)
        entry = next(item for item in invalid["versions"] if item["id"] == "dukeotr_v1")
        entry["status"] = "trained_pending_evaluation"
        entry["adapter_artifact"] = "models/imaginary_adapter"
        errors = validate_model_version_registry(invalid)
        self.assertTrue(any(item["code"] == "version.adapter_hash" for item in errors))

    def test_output_guard_refuses_protected_dataset_and_active_catalog_paths(self) -> None:
        for path in (
            ROOT / "evaluation_data" / "must_not_write.jsonl",
            ROOT / "training_data" / "must_not_write.jsonl",
            ROOT / "raw_data" / "dukeotr_phase1_luau_seed_tasks.jsonl",
        ):
            with self.subTest(path=path), self.assertRaises(TrainingFactoryError):
                assert_safe_factory_output_path(path)

    def test_malformed_factory_contracts_and_unresolved_critical_gate_are_rejected(self) -> None:
        failure = make_failure_record(
            failed_record(),
            taxonomy=TAXONOMY,
            candidate_artifact="validated_data/original.jsonl",
            created_at="2026-09-20T00:00:00Z",
        )
        failure["failure_categories"] = ["invented_category"]
        self.assertTrue(any(item["code"] == "failure.categories" for item in validate_failure_record(failure, taxonomy=TAXONOMY)))
        profile = next(item for item in TEMPLATES["profiles"] if item["id"] == "remote_security")
        draft = build_targeted_drafts(
            profile=profile,
            failure_category="trusting_client_data",
            observed_failure_count=1,
            counts_by_difficulty={"beginner": 1, "intermediate": 0, "advanced": 0},
            failure_input_sha256="c" * 64,
            seed=3,
            created_at="2026-09-20T00:00:00Z",
        )[0]
        draft["task_depth"] = "unbounded"
        self.assertTrue(any(item["code"] == "draft.task_depth" for item in validate_factory_draft(draft)))
        draft["task_depth"] = "short"
        self.assertTrue(
            any(item["code"] == "draft.category" for item in validate_factory_draft(draft, allowed_categories={"luau_language"}))
        )
        candidate = reviewed_record("train-unresolved-critical")
        candidate["quality"]["failure_analysis"] = {"unresolved_critical": True, "failure_categories": ["insecure_remote_handling"]}
        accepted, reasons = quality_gate_status(candidate)
        self.assertFalse(accepted)
        self.assertIn("unresolved_critical_failure", reasons)

    def test_optional_factory_seed_fields_flow_to_canonical_candidate(self) -> None:
        source = seed("train-depth")
        source.update(
            {
                "category": "client_server_security",
                "task_depth": "normal",
                "difficulty_rationale": "Two linked validation requirements make this intermediate.",
                "task_depth_rationale": "The brief requires validation and an edge case.",
            }
        )
        self.assertFalse(validate_seed(source))
        candidate = make_generated_record(source, good_answer(), generator={"kind": "test"}, variant=1)
        self.assertEqual(candidate["metadata"]["category"], "client_server_security")
        self.assertEqual(candidate["metadata"]["task_depth"], "normal")


class TrainingFactoryScriptTests(unittest.TestCase):
    def test_failure_to_targeted_brief_to_human_promotion_flow_stays_separate(self) -> None:
        original = failed_record()
        correction = accepted_correction(original)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.jsonl"
            corrections = root / "corrections.jsonl"
            failures = root / "failures.jsonl"
            ledger = root / "ledger.jsonl"
            analysis = root / "analysis.json"
            drafts = root / "drafts.jsonl"
            decisions = root / "decisions.jsonl"
            promoted = root / "promoted.jsonl"
            write_jsonl_atomic(candidates, [original])
            write_jsonl_atomic(corrections, [correction])
            self.assertEqual(
                ledger_main(
                    [
                        "--input",
                        str(candidates),
                        "--corrections",
                        str(corrections),
                        "--output",
                        str(ledger),
                        "--created-at",
                        "2026-09-20T00:00:00Z",
                    ]
                ),
                0,
            )
            self.assertEqual(
                record_failures_main(
                    [
                        "--input",
                        str(candidates),
                        "--corrections",
                        str(corrections),
                        "--output",
                        str(failures),
                        "--created-at",
                        "2026-09-20T00:00:00Z",
                    ]
                ),
                0,
            )
            recorded = list(read_jsonl(failures))
            self.assertEqual(len(recorded), 1)
            self.assertEqual(
                analyze_failures_main(["--input", str(failures), "--output", str(analysis), "--created-at", "2026-09-20T00:00:00Z"]),
                0,
            )
            report = read_json(analysis)
            self.assertEqual(report["summary"]["category_counts"]["trusting_client_data"], 1)
            self.assertEqual(
                targeted_main(
                    [
                        "--failure-input",
                        str(failures),
                        "--failure-category",
                        "trusting_client_data",
                        "--beginner",
                        "1",
                        "--output",
                        str(drafts),
                        "--seed",
                        "3407",
                        "--created-at",
                        "2026-09-20T00:00:00Z",
                    ]
                ),
                0,
            )
            draft_rows = list(read_jsonl(drafts))
            self.assertEqual(len(draft_rows), 1)
            self.assertEqual(draft_rows[0]["status"], "draft_requires_human_review")
            write_jsonl_atomic(
                decisions,
                [
                    {
                        "draft_id": draft_rows[0]["draft_id"],
                        "decision": "accept",
                        "reviewer": "factory-test-reviewer",
                        "notes": "Distinct scenario and explicit server-authority learning goal.",
                    }
                ],
            )
            self.assertEqual(
                promote_main(
                    [
                        "--drafts",
                        str(drafts),
                        "--decisions",
                        str(decisions),
                        "--output",
                        str(promoted),
                        "--created-at",
                        "2026-09-20T00:00:00Z",
                    ]
                ),
                0,
            )
            source_rows = list(read_jsonl(promoted))
            self.assertEqual(len(source_rows), 1)
            self.assertFalse(validate_seed(source_rows[0]))
            self.assertEqual(source_rows[0]["source"]["kind"], "project_authored")
            self.assertEqual(source_rows[0]["source"]["origin"], "training_factory_targeted_draft")
            # Source promotion creates a separate brief, not a canonical candidate or final dataset.
            self.assertNotIn("messages", source_rows[0])
            self.assertEqual(len(list(read_jsonl(ledger))), 1)

    def test_version_recorder_refuses_trained_claim_without_local_evidence_verification(self) -> None:
        entry = {
            "id": "dukeotr_test_claim",
            "kind": "adapter",
            "status": "trained_pending_evaluation",
            "notes": ["Test only; no model artifact is created."],
            "adapter_artifact": "models/nonexistent_adapter",
            "adapter_sha256": "a" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry_path = root / "entry.json"
            output = root / "versions.json"
            write_json_atomic(entry_path, entry)
            self.assertEqual(
                record_model_version_main(["--entry", str(entry_path), "--output", str(output)]),
                2,
            )
            self.assertFalse(output.exists())

    def test_failure_recording_merges_same_candidate_stage_for_duplicate_evidence(self) -> None:
        original = reviewed_record("train-duplicate-stage", "Explain a distinct client/server request.")
        deduplicated = copy.deepcopy(original)
        deduplicated["quality"]["deduplication"] = {
            "status": "duplicate",
            "duplicate_of": "gen-earlier",
            "similarity": 0.99,
            "checked_at": "2026-09-20T00:00:00Z",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validated = root / "validated.jsonl"
            dedupe_stage = root / "deduplicated.jsonl"
            output = root / "failures.jsonl"
            write_jsonl_atomic(validated, [original])
            write_jsonl_atomic(dedupe_stage, [deduplicated])
            self.assertEqual(
                record_failures_main(["--input", str(validated), "--input", str(dedupe_stage), "--output", str(output)]),
                0,
            )
            rows = list(read_jsonl(output))
            self.assertEqual(len(rows), 1)
            self.assertIn("duplicate_or_split_contamination", rows[0]["failure_categories"])
            self.assertEqual(rows[0]["provenance"]["candidate_artifact"], str(dedupe_stage))

    def test_analysis_rejects_duplicate_failure_ids(self) -> None:
        failure = make_failure_record(
            failed_record(),
            taxonomy=TAXONOMY,
            candidate_artifact="validated_data/original.jsonl",
            created_at="2026-09-20T00:00:00Z",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failures = root / "duplicate_failures.jsonl"
            report = root / "analysis.json"
            write_jsonl_atomic(failures, [failure, failure])
            self.assertEqual(analyze_failures_main(["--input", str(failures), "--output", str(report)]), 2)
            self.assertFalse(report.exists())

    def test_targeted_script_refuses_unobserved_category(self) -> None:
        original = failed_record()
        failure = make_failure_record(
            original,
            taxonomy=TAXONOMY,
            candidate_artifact="validated_data/original.jsonl",
            created_at="2026-09-20T00:00:00Z",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failures = root / "failures.jsonl"
            output = root / "drafts.jsonl"
            write_jsonl_atomic(failures, [failure])
            result = targeted_main(
                [
                    "--failure-input",
                    str(failures),
                    "--failure-category",
                    "incorrect_cframe_reasoning",
                    "--beginner",
                    "1",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(result, 2)
            self.assertFalse(output.exists())

    def test_failure_script_refuses_evaluation_output(self) -> None:
        original = failed_record()
        with tempfile.TemporaryDirectory() as directory:
            candidates = Path(directory) / "candidates.jsonl"
            write_jsonl_atomic(candidates, [original])
            forbidden = ROOT / "evaluation_data" / "factory-refusal-test.jsonl"
            self.assertFalse(forbidden.exists())
            result = record_failures_main(["--input", str(candidates), "--output", str(forbidden)])
            self.assertEqual(result, 2)
            self.assertFalse(forbidden.exists())


if __name__ == "__main__":
    unittest.main()
