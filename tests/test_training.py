from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from cloud_classifier.preprocessing import load_model_contract
from cloud_classifier.training import (
    EarlyStopping,
    TrainingDataError,
    _augment_training_image,
    _records_from_training_rows,
    _verify_pretrained_normalization,
    class_order_from_contract,
    create_validation_assignment,
    load_or_create_validation_assignment,
    read_training_rows,
)


def separated_phash(index: int) -> str:
    value = 0
    for bit in range(6):
        if index & (1 << bit):
            value |= ((1 << 9) - 1) << (bit * 9)
    return f"{value:016x}"


def training_row(
    index: int,
    *,
    class_id: str = "cirrus",
    author_id: str = "desconocido",
    phash64: str | None = None,
    split: str = "train",
) -> dict[str, str]:
    return {
        "path": f"data/synthetic/{index}.jpg",
        "class_id": class_id,
        "source": "ccsn",
        "author_id": author_id,
        "phash64": phash64 or separated_phash(index),
        "split": split,
    }


class ValidationAssignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.class_ids = class_order_from_contract(load_model_contract())
        self.rows = [
            training_row(index, class_id=self.class_ids[index % 2])
            for index in range(40)
        ]

    def test_same_seed_produces_same_validation(self) -> None:
        first = create_validation_assignment(self.rows, "partition-hash", self.class_ids, 42)
        second = create_validation_assignment(self.rows, "partition-hash", self.class_ids, 42)

        self.assertEqual(first, second)

    def test_validation_does_not_concentrate_multi_image_groups(self) -> None:
        rows = []
        image_index = 0
        group_sizes = [1] * 80 + [2] * 10 + [3] * 5
        for group_index, group_size in enumerate(group_sizes):
            author_id = (
                "desconocido" if group_size == 1 else f"autor_{group_index:03d}"
            )
            for _ in range(group_size):
                rows.append(
                    training_row(
                        image_index,
                        class_id=self.class_ids[image_index % len(self.class_ids)],
                        author_id=author_id,
                        phash64=separated_phash(image_index),
                    )
                )
                image_index += 1

        assignment = create_validation_assignment(
            rows, "partition-hash", self.class_ids, 42
        )
        validation_sizes = assignment["group_size_counts"]["validation"]
        remainder_sizes = assignment["group_size_counts"]["training_remainder"]
        validation_multi_image_fraction = sum(
            count for size, count in validation_sizes.items() if size != "1"
        ) / sum(validation_sizes.values())
        remainder_multi_image_fraction = sum(
            count for size, count in remainder_sizes.items() if size != "1"
        ) / sum(remainder_sizes.values())

        self.assertLessEqual(
            abs(validation_multi_image_fraction - remainder_multi_image_fraction),
            0.08,
        )
        self.assertEqual(
            set(assignment["class_group_counts"]["validation"]),
            set(self.class_ids),
        )

    def test_validation_never_splits_similarity_group(self) -> None:
        rows = [
            training_row(index, phash64=separated_phash(index + 10))
            for index in range(30)
        ]
        rows[0]["author_id"] = "autor_001"
        rows[1]["author_id"] = "autor_001"
        assignment = create_validation_assignment(rows, "partition-hash", self.class_ids, 42)
        validation_paths = set(assignment["validation_paths"])
        group_validation_paths = {rows[0]["path"], rows[1]["path"]}.intersection(
            validation_paths
        )
        self.assertNotIn(
            group_validation_paths,
            ({rows[0]["path"]}, {rows[1]["path"]}),
        )

    def test_assignment_is_versioned_and_rejects_changed_partition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            validation_path = Path(directory) / "validation-v1.json"
            assignment = load_or_create_validation_assignment(
                validation_path, self.rows, "partition-hash", self.class_ids, 42
            )
            self.assertEqual(
                json.loads(validation_path.read_text(encoding="utf-8")), assignment
            )
            with self.assertRaises(TrainingDataError):
                load_or_create_validation_assignment(
                    validation_path, self.rows, "changed-hash", self.class_ids, 42
                )

    def test_non_train_rows_are_not_returned_or_validated_as_images(self) -> None:
        rows = self.rows + [
            {
                "split": "calibration",
                "path": "data/must-not-open-calibration.jpg",
                "class_id": "not-a-contract-class",
            },
            {
                "split": "test",
                "path": "data/must-not-open-test.jpg",
                "class_id": "not-a-contract-class",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            partition_path = Path(directory) / "partitions.json"
            partition_path.write_text(json.dumps(rows), encoding="utf-8")
            with patch(
                "cloud_classifier.training.Image.open",
                side_effect=AssertionError("Las imágenes no se deben abrir aquí."),
            ):
                loaded_rows, _ = read_training_rows(partition_path)

        self.assertEqual(len(loaded_rows), len(self.rows))
        self.assertTrue(all(row["split"] == "train" for row in loaded_rows))
        records = _records_from_training_rows(loaded_rows, self.class_ids)
        self.assertTrue(
            all("must-not-open" not in record.path for record in records)
        )

    def test_training_row_with_class_outside_contract_fails(self) -> None:
        with self.assertRaisesRegex(TrainingDataError, "Clase fuera del orden"):
            _records_from_training_rows(
                [training_row(1, class_id="unknown_class")], self.class_ids
            )

    def test_unknown_classes_in_non_train_rows_are_ignored(self) -> None:
        non_train = training_row(1, class_id="unknown_class", split="test")
        records = _records_from_training_rows([self.rows[0]], self.class_ids)

        self.assertEqual(len(records), 1)
        self.assertEqual(non_train["split"], "test")


class EarlyStoppingTests(unittest.TestCase):
    def test_selects_expected_epoch_using_macro_f1_then_loss(self) -> None:
        stopping = EarlyStopping(patience=2)
        curves = [
            (0.60, 1.20),
            (0.72, 1.10),
            (0.72, 0.95),
            (0.70, 0.80),
            (0.71, 0.90),
        ]
        stopped = False
        for epoch, (macro_f1, validation_loss) in enumerate(curves, start=1):
            _, stopped = stopping.observe(epoch, macro_f1, validation_loss)
            if stopped:
                break

        self.assertTrue(stopped)
        self.assertEqual(stopping.best_epoch, 3)
        self.assertEqual(stopping.stale_epochs, 2)


class TrainingAugmentationTests(unittest.TestCase):
    def test_horizontal_mirror_does_not_flip_vertically(self) -> None:
        class FixedRandom:
            def uniform(self, minimum: float, maximum: float) -> float:
                return 1.0

            def randint(self, minimum: int, maximum: int) -> int:
                return 0

            def random(self) -> float:
                return 0.0

        image = Image.new("RGB", (4, 2))
        image.putdata(
            [
                (255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
                (255, 255, 0),
                (255, 0, 255),
                (0, 255, 255),
                (255, 255, 255),
                (0, 0, 0),
            ]
        )
        augmented = _augment_training_image(image, FixedRandom())
        try:
            self.assertEqual(augmented.getpixel((0, 0)), image.getpixel((3, 0)))
            self.assertEqual(augmented.getpixel((0, 1)), image.getpixel((3, 1)))
        finally:
            augmented.close()
            image.close()


class ClassOrderTests(unittest.TestCase):
    def test_class_order_is_read_from_contract_ids_without_a_second_order(self) -> None:
        contract = load_model_contract()
        expected = tuple(item["id"] for item in contract["classes"])

        self.assertEqual(class_order_from_contract(contract), expected)

    def test_malformed_contract_order_fails(self) -> None:
        duplicate_classes = [{"id": f"class-{index}"} for index in range(11)]
        duplicate_classes[-1]["id"] = duplicate_classes[0]["id"]
        with self.assertRaises(TrainingDataError):
            class_order_from_contract({"classes": duplicate_classes})


class PretrainedConfigurationTests(unittest.TestCase):
    def test_pretrained_normalization_must_match_shared_contract(self) -> None:
        contract = load_model_contract()
        normalization = contract["input"]["normalization"]
        _verify_pretrained_normalization(
            {"mean": normalization["mean"], "std": normalization["std"]},
            contract,
        )
        with self.assertRaisesRegex(TrainingDataError, "mean"):
            _verify_pretrained_normalization(
                {"mean": [0.0, 0.0, 0.0], "std": normalization["std"]},
                contract,
            )


if __name__ == "__main__":
    unittest.main()
