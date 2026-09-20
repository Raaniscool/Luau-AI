from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.build_datasets import main as build_main
from scripts.lib.io_utils import read_jsonl, write_jsonl_atomic
from scripts.prepare_training_bundle import main as bundle_main
from scripts.train_qlora import validate_dataset_manifest, validate_training_records, verify_training_inputs_for_plan
from tests.helpers import reviewed_record


class TrainingBundleTests(unittest.TestCase):
    def _evaluation_task(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "id": "eval-held-out",
            "split": "evaluation",
            "category": "test",
            "difficulty": "beginner",
            "prompt": "A completely separate held-out question about a lighthouse.",
            "rubric": [{"id": "correct", "description": "correct", "max_points": 100}],
            "must_not": [],
        }

    def _build_versioned_dataset(self, root: Path) -> Path:
        source = root / "deduplicated.jsonl"
        evaluation = root / "evaluation.jsonl"
        output = root / "dukeotr_dataset_v1"
        records = [
            reviewed_record(seed_id=f"train-test-{index}", prompt=f"Explain secure orchard harvest request {index}.")
            for index in range(6)
        ]
        write_jsonl_atomic(source, records)
        write_jsonl_atomic(evaluation, [self._evaluation_task()])
        code = build_main(
            [
                "--input",
                str(source),
                "--evaluation",
                str(evaluation),
                "--output-dir",
                str(output),
                "--dataset-version",
                "dukeotr_dataset_v1",
                "--strict",
            ]
        )
        self.assertEqual(code, 0)
        return output

    def _config(self, root: Path) -> Path:
        config = json.loads(Path("configs/qlora_sft.json").read_text(encoding="utf-8"))
        # The archive paths intentionally match the normal repository layout. The source data
        # itself lives in a temporary directory only for this unit test.
        config["dataset_manifest"] = "training_data/dukeotr_dataset_v1/dataset_manifest.json"
        config["training_file"] = "training_data/dukeotr_dataset_v1/train.jsonl"
        config["validation_file"] = "training_data/dukeotr_dataset_v1/validation.jsonl"
        path = root / "qlora_test.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_bundle_contains_verified_versioned_data_without_model_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = self._build_versioned_dataset(root)
            config = self._config(root)
            bundle = root / "dukeotr_training_bundle.zip"
            code = bundle_main(
                [
                    "--config",
                    str(config),
                    "--dataset-dir",
                    str(dataset),
                    "--output",
                    str(bundle),
                ]
            )
            self.assertEqual(code, 0)
            with zipfile.ZipFile(bundle) as archive:
                names = set(archive.namelist())
                transfer = json.loads(archive.read("training_bundle_manifest.json"))

        self.assertIn("training_data/dukeotr_dataset_v1/dataset_manifest.json", names)
        self.assertIn("training_data/dukeotr_dataset_v1/train.jsonl", names)
        self.assertIn("training_data/dukeotr_dataset_v1/validation.jsonl", names)
        self.assertIn("training_data/dukeotr_dataset_v1/final_dataset.jsonl", names)
        self.assertIn("requirements/training.txt", names)
        self.assertEqual(transfer["status"], "prepared_for_transfer_not_trained")
        self.assertEqual(transfer["dataset"]["version"], "dukeotr_dataset_v1")
        self.assertFalse(any(name.endswith((".gguf", ".safetensors", ".bin", ".pt")) for name in names))

    def test_trainer_manifest_guard_detects_tampered_final_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = self._build_versioned_dataset(root)
            config = json.loads(Path("configs/qlora_sft.json").read_text(encoding="utf-8"))
            config["dataset_manifest"] = str(dataset / "dataset_manifest.json")
            config["training_file"] = str(dataset / "train.jsonl")
            config["validation_file"] = str(dataset / "validation.jsonl")
            train_records = validate_training_records(dataset / "train.jsonl", "train", expected_partition="train")
            validation_records = validate_training_records(
                dataset / "validation.jsonl", "validation", expected_partition="development"
            )
            provenance = validate_dataset_manifest(
                config,
                train_path=dataset / "train.jsonl",
                validation_path=dataset / "validation.jsonl",
                train_records=train_records,
                validation_records=validation_records,
            )
            self.assertEqual(provenance["dataset_version"], "dukeotr_dataset_v1")
            self.assertEqual(verify_training_inputs_for_plan(config)["status"], "verified")

            with (dataset / "train.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                validate_dataset_manifest(
                    config,
                    train_path=dataset / "train.jsonl",
                    validation_path=dataset / "validation.jsonl",
                    train_records=list(read_jsonl(dataset / "train.jsonl")),
                    validation_records=list(read_jsonl(dataset / "validation.jsonl")),
                )
            self.assertEqual(verify_training_inputs_for_plan(config)["status"], "invalid")


if __name__ == "__main__":
    unittest.main()
