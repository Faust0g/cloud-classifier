from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from cloud_classifier.partitioning import build_similarity_groups
from cloud_classifier.preprocessing import load_model_contract
from cloud_classifier.training import (
    EarlyStopping,
    TrainingDataError,
    VALIDATION_MANIFEST,
    _augment_training_image,
    _parse_args,
    _records_from_training_rows,
    _verify_pretrained_normalization,
    class_order_from_contract,
    create_validation_assignment,
    create_validation_v2_assignment,
    filter_excluded_training_rows,
    load_validation_v2_assignment,
    load_or_create_validation_assignment,
    prepare_training_validation_rows,
    read_exclusion_manifest,
    read_training_rows,
    _load_preprocessed_image,
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


class ValidationV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.class_ids = class_order_from_contract(load_model_contract())
        self.rows = [
            training_row(index, class_id=self.class_ids[index % len(self.class_ids)])
            for index in range(220)
        ]
        self.partition_sha256 = "partition-hash"
        self.exclusion_sha256 = "exclusion-hash"

    def _group_id_for_path(
        self, rows: list[dict[str, str]], path: str
    ) -> str:
        records = _records_from_training_rows(rows, self.class_ids)
        target_id = next(record.image_id for record in records if record.path == path)
        return next(
            group.group_id
            for group in build_similarity_groups(records)
            if target_id in group.image_ids
        )

    def _exclusion_rows(
        self, rows: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        group_id = self._group_id_for_path(rows, rows[0]["path"])
        return [
            {
                "group_id": group_id,
                "path": row["path"],
                "class_id": row["class_id"],
                "source": row["source"],
                "split": row["split"],
            }
            for row in rows
        ]

    def test_v2_is_deterministic_and_contains_only_included_train_rows(self) -> None:
        excluded = [
            training_row(900, class_id=self.class_ids[0], author_id="author_conflict"),
            training_row(
                901,
                class_id=self.class_ids[1],
                author_id="author_conflict",
                phash64="ffffffffffffffff",
            ),
        ]
        excluded[0]["phash64"] = "ffffffffffffffff"
        rows = self.rows + excluded + [
            training_row(902, split="calibration"),
            training_row(903, split="test"),
        ]
        train_rows = [row for row in rows if row["split"] == "train"]
        exclusions = self._exclusion_rows(excluded)
        included_rows = filter_excluded_training_rows(
            train_rows, exclusions, self.class_ids
        )

        first = create_validation_v2_assignment(
            included_rows,
            self.partition_sha256,
            self.exclusion_sha256,
            self.class_ids,
            42,
        )
        second = create_validation_v2_assignment(
            included_rows,
            self.partition_sha256,
            self.exclusion_sha256,
            self.class_ids,
            42,
        )
        validation_paths = set(first["validation_paths"])

        self.assertEqual(first, second)
        self.assertEqual(first["validation_version"], 2)
        self.assertTrue(validation_paths.issubset({row["path"] for row in included_rows}))
        self.assertFalse(
            validation_paths.intersection(row["path"] for row in excluded)
        )
        self.assertTrue(all(row["split"] == "train" for row in included_rows))

    def test_v2_rejects_changes_to_either_manifest_hash(self) -> None:
        assignment = create_validation_v2_assignment(
            self.rows,
            self.partition_sha256,
            self.exclusion_sha256,
            self.class_ids,
            42,
        )
        with tempfile.TemporaryDirectory() as directory:
            validation_path = Path(directory) / "validation-v2.json"
            validation_path.write_text(json.dumps(assignment), encoding="utf-8")
            loaded = load_validation_v2_assignment(
                validation_path,
                self.rows,
                self.partition_sha256,
                self.exclusion_sha256,
                self.class_ids,
                42,
            )
            self.assertEqual(loaded, assignment)
            for partition_hash, exclusion_hash in (
                ("changed-partition", self.exclusion_sha256),
                (self.partition_sha256, "changed-exclusion"),
            ):
                with self.subTest(
                    partition_hash=partition_hash, exclusion_hash=exclusion_hash
                ):
                    with self.assertRaises(TrainingDataError):
                        load_validation_v2_assignment(
                            validation_path,
                            self.rows,
                            partition_hash,
                            exclusion_hash,
                            self.class_ids,
                            42,
                        )

    def test_v2_rejects_v1_manifest_with_clear_message(self) -> None:
        legacy_assignment = create_validation_assignment(
            self.rows, self.partition_sha256, self.class_ids, 42
        )
        with tempfile.TemporaryDirectory() as directory:
            validation_path = Path(directory) / "validation-v1.json"
            validation_path.write_text(json.dumps(legacy_assignment), encoding="utf-8")
            with self.assertRaisesRegex(TrainingDataError, "validation-v1 es solo historial"):
                load_validation_v2_assignment(
                    validation_path,
                    self.rows,
                    self.partition_sha256,
                    self.exclusion_sha256,
                    self.class_ids,
                    42,
                )

    def test_v2_never_splits_groups_and_does_not_concentrate_multiimage_groups(self) -> None:
        rows = []
        image_index = 0
        group_sizes = [1] * 110 + [2] * 15 + [3] * 5
        for group_index, group_size in enumerate(group_sizes):
            author_id = (
                "desconocido" if group_size == 1 else f"author_{group_index:03d}"
            )
            group_class = self.class_ids[group_index % len(self.class_ids)]
            for _ in range(group_size):
                rows.append(
                    training_row(
                        image_index,
                        class_id=group_class,
                        author_id=author_id,
                        phash64=separated_phash(image_index),
                    )
                )
                image_index += 1
        conflicting_group = [
            training_row(
                image_index,
                class_id=self.class_ids[0],
                author_id="author_excluded",
                phash64="ffffffffffffffff",
            ),
            training_row(
                image_index + 1,
                class_id=self.class_ids[1],
                author_id="author_excluded",
                phash64="ffffffffffffffff",
            ),
        ]
        exclusion_rows = self._exclusion_rows(conflicting_group)
        included_rows = filter_excluded_training_rows(
            rows + conflicting_group, exclusion_rows, self.class_ids
        )

        assignment = create_validation_v2_assignment(
            included_rows,
            self.partition_sha256,
            self.exclusion_sha256,
            self.class_ids,
            42,
        )
        validation_paths = set(assignment["validation_paths"])
        records = _records_from_training_rows(included_rows, self.class_ids)
        records_by_id = {record.image_id: record for record in records}
        for group in build_similarity_groups(records):
            member_paths = {
                records_by_id[image_id].path for image_id in group.image_ids
            }
            selected = member_paths.intersection(validation_paths)
            self.assertIn(len(selected), (0, len(member_paths)))

        validation_sizes = assignment["group_size_counts"]["validation"]
        remainder_sizes = assignment["group_size_counts"]["training_remainder"]
        validation_multi_fraction = sum(
            count for size, count in validation_sizes.items() if size != "1"
        ) / sum(validation_sizes.values())
        remainder_multi_fraction = sum(
            count for size, count in remainder_sizes.items() if size != "1"
        ) / sum(remainder_sizes.values())
        self.assertLessEqual(abs(validation_multi_fraction - remainder_multi_fraction), 0.08)

    def test_v2_requires_train_rows_and_fixed_seed(self) -> None:
        with self.assertRaisesRegex(TrainingDataError, "solo admite filas"):
            create_validation_v2_assignment(
                self.rows + [training_row(999, split="test")],
                self.partition_sha256,
                self.exclusion_sha256,
                self.class_ids,
                42,
            )
        with self.assertRaisesRegex(TrainingDataError, "semilla fija 42"):
            create_validation_v2_assignment(
                self.rows,
                self.partition_sha256,
                self.exclusion_sha256,
                self.class_ids,
                41,
            )

    def test_training_defaults_to_v2_manifest(self) -> None:
        self.assertEqual(VALIDATION_MANIFEST.as_posix(), "manifests/validation-v2.json")
        self.assertEqual(
            _parse_args([]).validation_manifest.as_posix(),
            "manifests/validation-v2.json",
        )


class ExclusionManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.class_ids = class_order_from_contract(load_model_contract())

    def _exclusion_document(
        self, rows: list[dict[str, str]], partition_sha256: str = "partition-hash"
    ) -> dict[str, object]:
        return {
            "exclusions_version": 1,
            "partition_sha256": partition_sha256,
            "groups": [
                {
                    "group_id": "group_conflicting_labels",
                    "reason": "misma foto, etiquetas contradictorias",
                    "images": [
                        {
                            "path": row["path"],
                            "class_id": row["class_id"],
                            "source": row["source"],
                            "split": row["split"],
                        }
                        for row in rows
                    ],
                    "evidence": {"pairs": []},
                }
            ],
        }

    def _group_id_for_path(
        self, rows: list[dict[str, str]], path: str
    ) -> str:
        records = _records_from_training_rows(rows, self.class_ids)
        target_id = next(record.image_id for record in records if record.path == path)
        group = next(
            group
            for group in build_similarity_groups(records)
            if target_id in group.image_ids
        )
        return group.group_id

    def test_conflicting_group_is_excluded_but_single_class_group_is_kept(self) -> None:
        conflicting_rows = [
            training_row(1, class_id=self.class_ids[0], author_id="autor_conflict"),
            training_row(2, class_id=self.class_ids[1], author_id="autor_conflict"),
        ]
        same_class_rows = [
            training_row(3, class_id=self.class_ids[2], author_id="autor_same"),
            training_row(4, class_id=self.class_ids[2], author_id="autor_same"),
        ]
        group_id = self._group_id_for_path(
            conflicting_rows + same_class_rows, conflicting_rows[0]["path"]
        )
        exclusion_rows = [
            {
                "group_id": group_id,
                "path": row["path"],
                "class_id": row["class_id"],
                "source": row["source"],
                "split": row["split"],
            }
            for row in conflicting_rows
        ]

        remaining = filter_excluded_training_rows(
            conflicting_rows + same_class_rows, exclusion_rows, self.class_ids
        )

        self.assertEqual(
            {row["path"] for row in remaining},
            {row["path"] for row in same_class_rows},
        )

    def test_exclusion_manifest_must_match_partition_hash(self) -> None:
        rows = [
            training_row(1, class_id=self.class_ids[0]),
            training_row(2, class_id=self.class_ids[1]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            exclusion_path = Path(directory) / "exclusions.json"
            exclusion_path.write_text(
                json.dumps(self._exclusion_document(rows)), encoding="utf-8"
            )

            with self.assertRaisesRegex(TrainingDataError, "no corresponde"):
                read_exclusion_manifest(
                    exclusion_path, "different-partition-hash", self.class_ids
                )

    def test_incomplete_exclusion_cannot_split_similarity_group(self) -> None:
        rows = [
            training_row(1, class_id=self.class_ids[0], author_id="autor_conflict"),
            training_row(2, class_id=self.class_ids[1], author_id="autor_conflict"),
        ]
        group_id = self._group_id_for_path(rows, rows[0]["path"])
        exclusion_rows = [
            {
                "group_id": group_id,
                "path": rows[0]["path"],
                "class_id": rows[0]["class_id"],
                "source": rows[0]["source"],
                "split": rows[0]["split"],
            }
        ]

        with self.assertRaisesRegex(TrainingDataError, "divide el grupo"):
            filter_excluded_training_rows(rows, exclusion_rows, self.class_ids)

    def test_excluded_train_and_validation_images_are_never_opened(self) -> None:
        excluded_rows = [
            training_row(1, class_id=self.class_ids[0], author_id="autor_conflict"),
            training_row(2, class_id=self.class_ids[1], author_id="autor_conflict"),
        ]
        validation_row = training_row(3, class_id=self.class_ids[2])
        training_row_kept = training_row(4, class_id=self.class_ids[3])
        all_rows = excluded_rows + [validation_row, training_row_kept]
        group_id = self._group_id_for_path(all_rows, excluded_rows[0]["path"])
        exclusion_rows = [
            {
                "group_id": group_id,
                "path": row["path"],
                "class_id": row["class_id"],
                "source": row["source"],
                "split": row["split"],
            }
            for row in excluded_rows
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for row in (validation_row, training_row_kept):
                image_path = root / row["path"]
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (300, 300), "white").save(image_path)

            training_rows, validation_rows = prepare_training_validation_rows(
                all_rows,
                {validation_row["path"]},
                exclusion_rows,
                self.class_ids,
            )
            opened_paths: list[str] = []
            original_open = Image.open

            def track_open(path: str | Path, *args: object, **kwargs: object):
                opened_paths.append(Path(path).resolve().as_posix())
                return original_open(path, *args, **kwargs)

            with patch("cloud_classifier.training.Image.open", side_effect=track_open):
                for row in training_rows:
                    _load_preprocessed_image(row, root, augment=False)
                for row in validation_rows:
                    _load_preprocessed_image(row, root, augment=False)

        self.assertEqual(
            set(opened_paths),
            {
                (root / validation_row["path"]).resolve().as_posix(),
                (root / training_row_kept["path"]).resolve().as_posix(),
            },
        )
        self.assertFalse(
            {row["path"] for row in training_rows + validation_rows}
            & {row["path"] for row in excluded_rows}
        )

    def test_training_does_not_read_double_label_manifest(self) -> None:
        rows = [
            training_row(1, class_id=self.class_ids[0]),
            training_row(2, class_id=self.class_ids[1]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partition_path = root / "partitions.json"
            exclusion_path = root / "exclusions.json"
            double_label_path = root / "double-label-v1.json"
            partition_path.write_text(json.dumps(rows), encoding="utf-8")
            partition_sha256 = hashlib.sha256(partition_path.read_bytes()).hexdigest()
            exclusion_path.write_text(
                json.dumps(self._exclusion_document(rows, partition_sha256)),
                encoding="utf-8",
            )
            double_label_path.write_text("not read", encoding="utf-8")

            read_paths: list[Path] = []
            original_read_bytes = Path.read_bytes

            def track_read_bytes(path: Path) -> bytes:
                read_paths.append(path)
                return original_read_bytes(path)

            with patch.object(Path, "read_bytes", track_read_bytes):
                training_rows, _ = read_training_rows(partition_path)
                read_exclusion_manifest(
                    exclusion_path, partition_sha256, self.class_ids
                )

        self.assertEqual(len(training_rows), 2)
        self.assertNotIn(double_label_path, read_paths)


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
