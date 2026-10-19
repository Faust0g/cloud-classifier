from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cloud_classifier.partitioning import (
    FrozenAssignmentConflict,
    ImageRecord,
    PartitionError,
    SimilarityGroup,
    _validate_class_directories,
    assign_groups,
    build_similarity_groups,
    check_future_trap_hash_conflicts,
    extend_assignments,
    hamming_distance,
    load_or_create_author_ids,
    versioned_manifest_rows,
)


def record(
    image_id: str,
    class_id: str = "cirrus",
    source: str = "ccsn",
    author_id: str = "desconocido",
    phash64: str = "0000000000000000",
) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        path=f"data/{image_id}.jpg",
        class_id=class_id,
        source=source,
        author_id=author_id,
        phash64=phash64,
    )


def singleton_groups(records: list[ImageRecord]) -> list[SimilarityGroup]:
    return [
        SimilarityGroup(group_id=f"group_{item.image_id}", image_ids=(item.image_id,))
        for item in records
    ]


class SimilarityGroupingTests(unittest.TestCase):
    def test_hamming_distance_boundary_is_inclusive_at_eight(self) -> None:
        self.assertEqual(hamming_distance(0, 0xFF), 8)
        self.assertEqual(hamming_distance(0, 0x1FF), 9)
        records = [
            record("first", phash64="0000000000000000"),
            record("within", phash64="00000000000000ff"),
            record("outside", phash64="00000000000001ff"),
        ]

        groups = build_similarity_groups(records)
        members = {frozenset(group.image_ids) for group in groups}
        self.assertIn(frozenset({"first", "within", "outside"}), members)

    def test_similarity_is_transitive(self) -> None:
        records = [
            record("a", phash64="0000000000000000"),
            record("b", phash64="00000000000000ff"),
            record("c", phash64="000000000000ffff"),
        ]

        groups = build_similarity_groups(records)

        self.assertEqual(len(groups), 1)
        self.assertEqual(set(groups[0].image_ids), {"a", "b", "c"})
        self.assertEqual(hamming_distance(0, 0xFFFF), 16)

    def test_shared_known_author_joins_images_with_dissimilar_hashes(self) -> None:
        records = [
            record("a", author_id="autor_001", phash64="0000000000000000"),
            record("b", author_id="autor_001", phash64="ffffffffffffffff"),
        ]

        groups = build_similarity_groups(records)

        self.assertEqual(len(groups), 1)

    def test_unknown_author_does_not_join_distinct_images_by_author(self) -> None:
        records = [
            record("a", phash64="0000000000000000"),
            record("b", phash64="ffffffffffffffff"),
        ]

        groups = build_similarity_groups(records)

        self.assertEqual(len(groups), 2)


class AssignmentTests(unittest.TestCase):
    def make_class_records(self, class_id: str, count: int) -> list[ImageRecord]:
        return [
            record(
                f"{class_id}-{index}",
                class_id=class_id,
                source="ccsn" if index % 2 == 0 else "sociedad",
                phash64=f"{index:016x}",
            )
            for index in range(count)
        ]

    def test_same_seed_produces_same_partition(self) -> None:
        records = self.make_class_records("cirrus", 40)
        groups = singleton_groups(records)

        first, first_large = assign_groups(records, groups, seed=42)
        second, second_large = assign_groups(records, groups, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(first_large, second_large)

    def test_groups_are_never_split_and_classes_are_stratified(self) -> None:
        records = [
            *self.make_class_records("cirrus", 40),
            *self.make_class_records("cumulus", 40),
        ]
        groups = singleton_groups(records)
        assignments, _ = assign_groups(records, groups, seed=42)

        for group in groups:
            self.assertEqual(
                len({assignments[image_id] for image_id in group.image_ids}), 1
            )
        for class_id in ("cirrus", "cumulus"):
            splits = {
                split: sum(
                    assignments[item.image_id] == split
                    for item in records
                    if item.class_id == class_id
                )
                for split in ("train", "calibration", "test")
            }
            self.assertGreater(splits["train"], splits["calibration"])
            self.assertGreater(splits["train"], splits["test"])
            self.assertLessEqual(abs(splits["calibration"] - splits["test"]), 2)
            self.assertLessEqual(abs(splits["train"] - 28), 2)
            self.assertLessEqual(abs(splits["calibration"] - 6), 1)
            self.assertLessEqual(abs(splits["test"] - 6), 1)

    def test_large_group_is_assigned_wholly_to_training(self) -> None:
        records = [
            record("large-a", phash64="0000000000000000"),
            record("large-b", phash64="ffffffffffffffff"),
            *self.make_class_records("cirrus", 18),
        ]
        groups = [
            SimilarityGroup("large", ("large-a", "large-b")),
            *singleton_groups(records[2:]),
        ]

        assignments, large_ids = assign_groups(records, groups, seed=42)

        self.assertEqual(large_ids, {"large"})
        self.assertEqual(assignments["large-a"], "train")
        self.assertEqual(assignments["large-b"], "train")

    def test_multi_image_groups_are_not_concentrated_in_evaluation_splits(self) -> None:
        records = [
            *self.make_class_records("cirrus", 150),
            *[
                record(
                    f"pair-{pair}-{side}",
                    class_id="cirrus",
                    source="ccsn",
                    phash64=f"{1000 + pair * 2 + side:016x}",
                )
                for pair in range(50)
                for side in range(2)
            ],
        ]
        groups = [
            *singleton_groups(records[:150]),
            *[
                SimilarityGroup(
                    f"pair-{pair}",
                    (f"pair-{pair}-0", f"pair-{pair}-1"),
                )
                for pair in range(50)
            ],
        ]

        assignments, _ = assign_groups(records, groups, seed=42)
        multi_group_counts = {
            split: sum(
                assignments[group.image_ids[0]] == split
                for group in groups
                if len(group.image_ids) > 1
            )
            for split in ("train", "calibration", "test")
        }

        self.assertEqual(sum(multi_group_counts.values()), 50)
        self.assertLessEqual(abs(multi_group_counts["train"] - 35), 1)
        self.assertLessEqual(abs(multi_group_counts["calibration"] - 7.5), 1)
        self.assertLessEqual(abs(multi_group_counts["test"] - 7.5), 1)

    def test_extension_preserves_frozen_assignment_for_connected_image(self) -> None:
        frozen = [record("old", phash64="0000000000000000")]
        added = [record("new", phash64="0000000000000001")]

        assignments = extend_assignments(frozen, {"old": "test"}, added)

        self.assertEqual(assignments["old"], "test")
        self.assertEqual(assignments["new"], "test")

    def test_extension_fails_if_new_component_connects_frozen_splits(self) -> None:
        frozen = [
            record("old-train", phash64="0000000000000000"),
            record("old-test", phash64="ffffffffffffffff"),
        ]
        additions = [
            record(f"bridge-{index}", phash64=f"{((1 << (index * 8)) - 1):016x}")
            for index in range(1, 8)
        ]

        with self.assertRaises(FrozenAssignmentConflict):
            extend_assignments(
                frozen,
                {"old-train": "train", "old-test": "test"},
                additions,
            )

    def test_future_trap_hash_conflicts_are_reported_at_threshold(self) -> None:
        frozen = [record("frozen", phash64="0000000000000000")]
        traps = [
            record("trap-match", source="traps", phash64="00000000000000ff"),
            record("trap-clear", source="traps", phash64="00000000000001ff"),
        ]

        conflicts = check_future_trap_hash_conflicts(frozen, traps)

        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["trap_path"], "data/trap-match.jpg")
        self.assertEqual(conflicts[0]["hamming_distance"], 8)

    def test_future_traps_cannot_be_assigned_without_human_review(self) -> None:
        frozen = [record("frozen", phash64="0000000000000000")]
        trap = record("trap", source="traps", phash64="00000000000000ff")

        with self.assertRaisesRegex(
            PartitionError, r"trap\.jpg.*frozen\.jpg.*Hamming=8"
        ):
            extend_assignments(frozen, {"frozen": "train"}, [trap])


class VersionedManifestTests(unittest.TestCase):
    def test_manifest_contains_only_approved_fields_and_relative_paths(self) -> None:
        item = record("image-1", author_id="autor_007")

        rows = versioned_manifest_rows([item], {"image-1": "calibration"})

        self.assertEqual(
            rows,
            [
                {
                    "path": "data/image-1.jpg",
                    "class_id": "cirrus",
                    "source": "ccsn",
                    "author_id": "autor_007",
                    "phash64": "0000000000000000",
                    "split": "calibration",
                }
            ],
        )
        self.assertEqual(
            set(rows[0]),
            {"path", "class_id", "source", "author_id", "phash64", "split"},
        )

    def test_manifest_rejects_author_names_and_absolute_paths(self) -> None:
        with self.assertRaisesRegex(PartitionError, "ID de autor inválido"):
            versioned_manifest_rows(
                [record("named", author_id="A Person")], {"named": "train"}
            )
        absolute = ImageRecord(
            image_id="absolute",
            path="C:/private/image.jpg",
            class_id="cirrus",
            source="ccsn",
            author_id="desconocido",
            phash64="0000000000000000",
        )
        with self.assertRaisesRegex(PartitionError, "debe ser relativa"):
            versioned_manifest_rows([absolute], {"absolute": "train"})


class InventorySupportTests(unittest.TestCase):
    def test_unknown_class_directory_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "known").mkdir()
            (root / "unknown-class").mkdir()

            with self.assertRaisesRegex(PartitionError, "unknown-class"):
                _validate_class_directories(root, {"known"}, "synthetic")

    def test_author_ids_remain_stable_when_mapping_is_extended(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mapping_path = Path(temporary) / "author_ids.json"
            first_mapping = load_or_create_author_ids(
                ["Zoe Author", "Alice Author"], mapping_path
            )
            saved_document = json.loads(mapping_path.read_text(encoding="utf-8"))
            second_mapping = load_or_create_author_ids(
                ["New Author", "Alice Author", "Zoe Author"], mapping_path
            )

        self.assertEqual(first_mapping["alice author"], second_mapping["alice author"])
        self.assertEqual(first_mapping["zoe author"], second_mapping["zoe author"])
        self.assertTrue(all(entry["id"].startswith("autor_") for entry in saved_document["authors"]))


if __name__ == "__main__":
    unittest.main()
