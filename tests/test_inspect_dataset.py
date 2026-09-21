"""以獨立模擬資料驗證檢查器，不更動官方資料。"""

import csv
import tempfile
import unittest
from pathlib import Path

from inspect_dataset import inspect_dataset


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / "dataset_A"
        (self.root / "audio").mkdir(parents=True)
        self.rows = []
        for split in ("train", "validation", "test"):
            relative = f"audio/A_{split}.wav"
            (self.root / relative).touch()  # 本步只檢查存在，不解碼音訊。
            self.rows.append({
                "sample_id": f"A_{split}", "split": split,
                "label": "" if split == "test" else "1960s",
                "audio_path": relative, "duration_seconds": "30",
                "sample_rate": "24000", "sha256": "0" * 64,
            })

    def inspect(self):
        with (self.root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        return inspect_dataset(self.root)

    def test_hidden_test_labels_are_expected(self):
        result = self.inspect()
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["splits"]["test"]["missing_labels"], 1)

    def test_duplicate_id_across_splits_is_detected(self):
        self.rows[1]["sample_id"] = self.rows[0]["sample_id"]
        result = self.inspect()
        self.assertEqual(result["duplicate_ids"], {"A_train": 2})
        self.assertEqual(result["split_id_overlaps"]["train / validation"], ["A_train"])
        self.assertTrue(result["errors"])

    def test_missing_file_is_detected(self):
        self.rows[0]["audio_path"] = "audio/missing.wav"
        result = self.inspect()
        self.assertEqual(result["missing_audio"], ["audio/missing.wav"])
        self.assertTrue(result["errors"])

    def test_missing_training_label_is_an_error(self):
        self.rows[0]["label"] = ""
        self.assertTrue(any("train" in error and "缺少標籤" in error
                            for error in self.inspect()["errors"]))

    def test_unexpected_split_and_test_label_are_detected(self):
        self.rows[0]["split"] = "training"
        self.rows[2]["label"] = "1960s"
        result = self.inspect()
        self.assertEqual(result["unexpected_splits"], ["training"])
        self.assertTrue(any("test 出現非空標籤" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
