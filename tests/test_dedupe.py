from __future__ import annotations

import unittest

from scripts.lib.dedupe import cross_split_prompt_collisions, mark_deduplicated, similarity
from tests.helpers import reviewed_record


class DedupeTests(unittest.TestCase):
    def test_exact_duplicate_is_marked(self) -> None:
        first = reviewed_record("train-one")
        second = reviewed_record("train-two")
        # Use a distinct record id to model two generator records with identical content.
        second["record_id"] = "gen-train-two-v1-same"
        records, audit = mark_deduplicated([first, second], threshold=0.82)
        self.assertEqual(records[0]["quality"]["deduplication"]["status"], "unique")
        self.assertEqual(records[1]["quality"]["deduplication"]["status"], "duplicate")
        self.assertEqual(len(audit), 1)

    def test_near_similarity_is_bounded(self) -> None:
        result = similarity("a secure server validates item ids", "a secure server validates item ids before grants")
        self.assertGreater(result.jaccard, 0)
        self.assertLessEqual(result.jaccard, 1)

    def test_cross_split_prompt_collision_is_detected(self) -> None:
        prompt = "Unique evaluation wording about a blue vault door."
        record = reviewed_record(prompt=prompt)
        tasks = [{"id": "eval-x", "prompt": prompt}]
        collisions = cross_split_prompt_collisions([record], tasks, threshold=0.93)
        self.assertEqual(collisions[0]["evaluation_task_id"], "eval-x")


if __name__ == "__main__":
    unittest.main()
