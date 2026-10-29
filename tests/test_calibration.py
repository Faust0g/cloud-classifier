from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cloud_classifier.calibration import (
    TARGET_PRECISION,
    _negative_log_likelihood,
    _smallest_threshold,
    acquire_final_evaluation_lock,
    fit_temperature,
    read_split_rows,
)
from cloud_classifier.prediction import classify_logits


class TemperatureScalingTests(unittest.TestCase):
    def test_temperature_reduces_nll_on_overconfident_logits(self) -> None:
        logits = np.asarray(
            [[8.0, 0.0], [8.0, 0.0], [0.0, 8.0], [0.0, 8.0]]
        )
        labels = np.asarray([0, 1, 1, 1])

        temperature = fit_temperature(logits, labels)

        self.assertGreater(temperature, 1)
        self.assertLess(
            _negative_log_likelihood(logits, labels, temperature),
            _negative_log_likelihood(logits, labels, 1),
        )


class ThresholdTests(unittest.TestCase):
    def test_firm_and_pair_thresholds_meet_target_precision(self) -> None:
        confidences = np.asarray([0.96, 0.92, 0.88, 0.84, 0.80, 0.76])
        correct_top1 = np.asarray([True, True, True, True, True, False])
        pair_sums = np.asarray([0.99, 0.96, 0.93, 0.90, 0.87, 0.84])
        correct_top2 = np.asarray([True, True, True, True, False, False])

        firm_threshold = _smallest_threshold(confidences, correct_top1)
        pair_threshold = _smallest_threshold(pair_sums, correct_top2)

        self.assertIsNotNone(firm_threshold)
        self.assertIsNotNone(pair_threshold)
        firm_selected = confidences >= firm_threshold
        pair_selected = pair_sums >= pair_threshold
        self.assertGreaterEqual(
            np.mean(correct_top1[firm_selected]), TARGET_PRECISION
        )
        self.assertGreaterEqual(
            np.mean(correct_top2[pair_selected]), TARGET_PRECISION
        )

    def test_returns_no_threshold_when_target_is_unreachable(self) -> None:
        self.assertIsNone(
            _smallest_threshold(
                np.asarray([0.9, 0.8]),
                np.asarray([False, True]),
            )
        )


class SharedClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.class_ids = ("cumulus", "cirrus", "no_nube")

    def test_classifies_firm_pair_retry_and_no_cloud(self) -> None:
        firm = classify_logits(
            [4.0, 1.0, 0.0], 1.0, 0.8, 0.9, self.class_ids
        )
        pair = classify_logits(
            [4.0, 3.0, 0.0], 1.0, 0.99, 0.9, self.class_ids
        )
        retry = classify_logits(
            [1.0, 0.9, 0.8], 1.0, 0.99, 0.99, self.class_ids
        )
        no_cloud = classify_logits(
            [0.0, 0.0, 4.0], 1.0, 0.8, 0.9, self.class_ids
        )

        self.assertEqual(firm.decision, "firme")
        self.assertEqual(pair.decision, "par")
        self.assertEqual(retry.decision, "repetir")
        self.assertEqual(no_cloud.decision, "no_nube")


class SplitIsolationTests(unittest.TestCase):
    def test_calibration_reader_ignores_non_calibration_row_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partition_path = root / "partitions.json"
            exclusion_path = root / "exclusions.json"
            rows = [
                {"split": "train", "path": "../never-open-train.jpg"},
                {"split": "test", "class_id": "invalid"},
                {
                    "split": "calibration",
                    "path": "data/calibration.jpg",
                    "class_id": "cumulus",
                    "source": "ccsn",
                },
            ]
            partition_bytes = json.dumps(rows).encode("utf-8")
            partition_path.write_bytes(partition_bytes)
            exclusion_path.write_text(
                json.dumps(
                    {
                        "exclusions_version": 1,
                        "partition_sha256": hashlib.sha256(partition_bytes).hexdigest(),
                        "groups": [],
                    }
                ),
                encoding="utf-8",
            )

            selected, _, _ = read_split_rows(
                partition_path,
                exclusion_path,
                "calibration",
                ("cumulus", "cirrus", "no_nube"),
            )

        self.assertEqual([row["path"] for row in selected], ["data/calibration.jpg"])


class FinalEvaluationLockTests(unittest.TestCase):
    def test_existing_lock_prevents_a_second_final_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "calibration.json"
            lock = root / "evaluation.lock"
            calibration.write_text("{}", encoding="utf-8")
            lock.write_text("already consumed", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                acquire_final_evaluation_lock(lock, calibration, True)

    def test_final_evaluation_requires_explicit_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "calibration.json"
            calibration.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "--confirm-final"):
                acquire_final_evaluation_lock(
                    root / "evaluation.lock", calibration, False
                )


if __name__ == "__main__":
    unittest.main()
