from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_code_book import main as audit_main
from scripts.lib.code_book import context_view, load_cards, search_cards, validate_card


class CodeBookTests(unittest.TestCase):
    def test_source_checked_cards_are_structurally_valid(self) -> None:
        cards = load_cards("code_book/roblox_luau_cards.jsonl")
        self.assertGreaterEqual(len(cards), 12)
        for card in cards:
            errors = [finding for finding in validate_card(card) if finding["severity"] == "error"]
            self.assertEqual(errors, [], card["id"])
            self.assertEqual(card["training_policy"], "manual_curation_only")

    def test_remote_query_returns_factual_regression_card(self) -> None:
        cards = load_cards("code_book/roblox_luau_cards.jsonl")
        matches = search_cards(cards, "secure RemoteEvent client FireServer", limit=5)
        ids = [match["card"]["id"] for match in matches]
        self.assertIn("cb-remoteevents-client-server-authority", ids)
        remote = next(match["card"] for match in matches if match["card"]["id"] == "cb-remoteevents-client-server-authority")
        statements = " ".join(claim["statement"] for claim in remote["claims"])
        self.assertIn("FireServer", statements)
        self.assertIn("not make client input trustworthy", statements)
        context = context_view(next(match for match in matches if match["card"]["id"] == "cb-remoteevents-client-server-authority"))
        self.assertEqual(context["patterns"][0]["id"], "request-validate-mutate-notify")

    def test_audit_reports_complete_required_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "audit.json"
            code = audit_main(["--strict", "--output", str(report)])
            self.assertEqual(code, 0)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "pass")
            self.assertEqual(payload["missing_required_domains"], {})


if __name__ == "__main__":
    unittest.main()
