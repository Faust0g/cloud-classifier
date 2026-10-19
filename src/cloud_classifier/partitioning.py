from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps


PARTITION_SEED = 42
SPLIT_RATIOS = {"train": 0.70, "calibration": 0.15, "test": 0.15}
PHASH_HAMMING_THRESHOLD = 8
LARGE_GROUP_EVAL_TARGET_FRACTION = Fraction(1, 3)
CALIBRATION_WARNING_MIN_IMAGES = 30
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
OPEN_IMAGES_ATTRIBUTION_PATH = Path(
    "docs/datasets/no_nube/open-images-v7/validation/ATTRIBUTION.csv"
)

CCSN_FOLDER_TO_CLASS = {
    "Ac": "altocumulus",
    "As": "altostratus",
    "Cb": "cumulonimbus",
    "Cc": "cirrocumulus",
    "Ci": "cirrus",
    "Cs": "cirrostratus",
    "Cu": "cumulus",
    "Ns": "nimbostratus",
    "Sc": "stratocumulus",
    "St": "stratus",
    "no_nube": "no_nube",
}
SOCIETY_FOLDER_TO_CLASS = {
    "altocumulus": "altocumulus",
    "altostratus": "altostratus",
    "cirrocumulus": "cirrocumulus",
    "cirrostratus": "cirrostratus",
    "cirrus": "cirrus",
    "cumulonimbus": "cumulonimbus",
    "cumulus": "cumulus",
    "nimbostratus": "nimbostratus",
    "stratocumulus": "stratocumulus",
    "stratus": "stratus",
}
SPLIT_NAMES = ("train", "calibration", "test")


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    path: str
    class_id: str
    source: str
    author_id: str
    phash64: str


@dataclass(frozen=True)
class SimilarityGroup:
    group_id: str
    image_ids: tuple[str, ...]


class PartitionError(ValueError):
    pass


class FrozenAssignmentConflict(PartitionError):
    pass


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = dict.fromkeys(self.parent, 0)

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _portable_relative_path(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.as_posix().casefold(),
    )


def _validate_class_directories(
    root: Path, expected: set[str], source_name: str
) -> list[Path]:
    if not root.is_dir():
        raise PartitionError(f"No existe el directorio de fuente {source_name}: {root}")
    directories = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )
    unknown = sorted(path.name for path in directories if path.name not in expected)
    missing = sorted(expected - {path.name for path in directories})
    if unknown:
        raise PartitionError(
            f"Carpetas sin equivalencia en {source_name}: {', '.join(unknown)}"
        )
    if missing:
        raise PartitionError(
            f"Faltan carpetas esperadas en {source_name}: {', '.join(missing)}"
        )
    return directories


def _normalized_author(author: str) -> str:
    return " ".join(author.split()).casefold()


def load_or_create_author_ids(
    author_names: Iterable[str], mapping_path: Path
) -> dict[str, str]:
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, str]] = {}
    if mapping_path.exists():
        try:
            document = json.loads(mapping_path.read_text(encoding="utf-8"))
            entries = document["authors"]
            if not isinstance(entries, list):
                raise TypeError("authors debe ser una lista")
            for entry in entries:
                name = entry["name"]
                author_id = entry["id"]
                if not isinstance(name, str) or not isinstance(author_id, str):
                    raise TypeError("nombre e id deben ser cadenas")
                normalized = _normalized_author(name)
                if normalized in existing and existing[normalized]["id"] != author_id:
                    raise ValueError(f"Autor duplicado en tabla: {name}")
                if any(
                    current["id"] == author_id and key != normalized
                    for key, current in existing.items()
                ):
                    raise ValueError(f"ID de autor reutilizado en tabla: {author_id}")
                existing[normalized] = {"name": name, "id": author_id}
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise PartitionError(
                f"Tabla de autores inválida en {mapping_path}: {error}"
            ) from error

    requested = {_normalized_author(name): " ".join(name.split()) for name in author_names}
    if "" in requested:
        raise PartitionError("Open Images contiene un autor vacío.")
    next_number = 1
    for entry in existing.values():
        match = re.fullmatch(r"autor_(\d{3,})", entry["id"])
        if not match:
            raise PartitionError(
                f"ID de autor inválido en {mapping_path}: {entry['id']}"
            )
        next_number = max(next_number, int(match.group(1)) + 1)
    for normalized in sorted(requested):
        if normalized not in existing:
            existing[normalized] = {
                "name": requested[normalized],
                "id": f"autor_{next_number:03d}",
            }
            next_number += 1

    document = {
        "version": 1,
        "authors": sorted(existing.values(), key=lambda entry: entry["id"]),
    }
    temporary_path = mapping_path.with_suffix(mapping_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(mapping_path)
    return {
        normalized: entry["id"]
        for normalized, entry in existing.items()
    }


def _load_open_images_attribution(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise PartitionError(f"Falta la atribución de Open Images: {path}")
    authors: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"ImageID", "Author"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise PartitionError(
                f"La atribución de Open Images debe incluir columnas {sorted(required)}."
            )
        for row in reader:
            image_id = row["ImageID"].strip()
            author = row["Author"].strip()
            if not image_id or not author:
                raise PartitionError("Atribución de Open Images con ID/autor vacío.")
            if image_id in authors:
                raise PartitionError(f"ID duplicado en atribución: {image_id}")
            authors[image_id] = author
    return authors


def _dct_matrix(size: int) -> np.ndarray:
    positions = np.arange(size, dtype=np.float64)
    frequencies = positions[:, None]
    matrix = np.cos(np.pi * (2 * positions + 1) * frequencies / (2 * size))
    matrix[0, :] *= math.sqrt(1 / size)
    matrix[1:, :] *= math.sqrt(2 / size)
    return matrix


_DCT32 = _dct_matrix(32)


def perceptual_hash64(image_path: Path) -> int:
    try:
        with Image.open(image_path) as image:
            oriented = ImageOps.exif_transpose(image)
            grayscale = oriented.convert("L").resize(
                (32, 32), Image.Resampling.LANCZOS
            )
            pixels = np.asarray(grayscale, dtype=np.float64)
    except (OSError, ValueError) as error:
        raise PartitionError(f"No se pudo leer la imagen {image_path}: {error}") from error
    coefficients = _DCT32 @ pixels @ _DCT32.T
    low_frequency = coefficients[:8, :8].ravel()
    median = float(np.median(low_frequency))
    value = 0
    for coefficient in low_frequency:
        value = (value << 1) | int(coefficient >= median)
    return value


def hamming_distance(left: int | str, right: int | str) -> int:
    left_value = int(left, 16) if isinstance(left, str) else left
    right_value = int(right, 16) if isinstance(right, str) else right
    return (left_value ^ right_value).bit_count()


def build_inventory(
    repo_root: Path,
    data_root: Path,
    author_mapping_path: Path,
) -> tuple[list[ImageRecord], dict[str, int]]:
    ccsn_root = data_root / "ccsn" / "images"
    society_root = data_root / "sociedad"
    open_images_root = data_root / "no_nube" / "open-images-v7" / "validation" / "images"
    ccsn_dirs = _validate_class_directories(
        ccsn_root, set(CCSN_FOLDER_TO_CLASS), "CCSN"
    )
    society_dirs = _validate_class_directories(
        society_root, set(SOCIETY_FOLDER_TO_CLASS), "sociedad"
    )
    if not open_images_root.is_dir():
        raise PartitionError(f"No existe el directorio de imágenes: {open_images_root}")

    raw_open_images = _image_files(open_images_root)
    attribution_path = repo_root / OPEN_IMAGES_ATTRIBUTION_PATH
    authors_by_image = _load_open_images_attribution(attribution_path)
    image_ids = {path.stem for path in raw_open_images}
    missing_metadata = sorted(image_ids - authors_by_image.keys())
    orphan_metadata = sorted(authors_by_image.keys() - image_ids)
    if missing_metadata or orphan_metadata:
        raise PartitionError(
            "Conteo/metadatos de Open Images no coincide: "
            f"{len(raw_open_images)} imágenes, {len(authors_by_image)} atribuciones, "
            f"{len(missing_metadata)} sin atribución y {len(orphan_metadata)} atribuciones sin imagen."
        )

    author_ids = load_or_create_author_ids(
        authors_by_image.values(), author_mapping_path
    )
    counts: dict[str, int] = {
        "open_images_disk": len(raw_open_images),
        "open_images_attribution": len(authors_by_image),
    }
    records: list[ImageRecord] = []

    def add_record(path: Path, class_id: str, source: str, author_id: str) -> None:
        relative = _portable_relative_path(path, repo_root)
        image_id = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:20]
        records.append(
            ImageRecord(
                image_id=image_id,
                path=relative,
                class_id=class_id,
                source=source,
                author_id=author_id,
                phash64=f"{perceptual_hash64(path):016x}",
            )
        )

    for directory in ccsn_dirs:
        class_id = CCSN_FOLDER_TO_CLASS[directory.name]
        images = _image_files(directory)
        counts[f"ccsn:{class_id}"] = len(images)
        for path in images:
            add_record(path, class_id, "ccsn", "desconocido")

    for directory in society_dirs:
        class_id = SOCIETY_FOLDER_TO_CLASS[directory.name]
        images = _image_files(directory)
        counts[f"sociedad:{class_id}"] = len(images)
        for path in images:
            add_record(path, class_id, "sociedad", "desconocido")

    for path in raw_open_images:
        author = authors_by_image[path.stem]
        author_id = author_ids[_normalized_author(author)]
        add_record(path, "no_nube", "open_images", author_id)

    counts["sociedad_total"] = sum(
        count for key, count in counts.items() if key.startswith("sociedad:")
    )
    counts["ccsn_total"] = sum(
        count for key, count in counts.items() if key.startswith("ccsn:")
    )
    counts["open_images"] = counts["open_images_disk"]
    if len({record.image_id for record in records}) != len(records):
        raise PartitionError("Se generaron IDs de imagen duplicados.")
    return records, counts


def build_similarity_groups(
    records: Sequence[ImageRecord],
    hamming_threshold: int = PHASH_HAMMING_THRESHOLD,
) -> list[SimilarityGroup]:
    ids = [record.image_id for record in records]
    if len(ids) != len(set(ids)):
        raise PartitionError("Hay image_id duplicados al formar grupos.")
    union_find = UnionFind(ids)
    by_author: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record.author_id != "desconocido":
            by_author[record.author_id].append(record.image_id)
    for author_image_ids in by_author.values():
        for image_id in author_image_ids[1:]:
            union_find.union(author_image_ids[0], image_id)

    for index, left in enumerate(records):
        left_hash = int(left.phash64, 16)
        for right in records[index + 1 :]:
            if hamming_distance(left_hash, int(right.phash64, 16)) <= hamming_threshold:
                union_find.union(left.image_id, right.image_id)

    components: dict[str, list[str]] = defaultdict(list)
    for image_id in ids:
        components[union_find.find(image_id)].append(image_id)
    groups: list[SimilarityGroup] = []
    for image_ids in components.values():
        members = tuple(sorted(image_ids))
        digest = hashlib.sha256("|".join(members).encode("ascii")).hexdigest()[:16]
        groups.append(SimilarityGroup(group_id=f"group_{digest}", image_ids=members))
    return sorted(groups, key=lambda group: group.group_id)


def _group_counts(
    group: SimilarityGroup,
    records_by_id: dict[str, ImageRecord],
    key: str,
) -> Counter[str]:
    return Counter(getattr(records_by_id[image_id], key) for image_id in group.image_ids)


def _integer_split_quotas(total: int) -> dict[str, int]:
    exact = {split: total * SPLIT_RATIOS[split] for split in SPLIT_NAMES}
    quotas = {split: math.floor(exact[split]) for split in SPLIT_NAMES}
    remainder = total - sum(quotas.values())
    priority = sorted(
        SPLIT_NAMES,
        key=lambda split: (-(exact[split] - quotas[split]), SPLIT_NAMES.index(split)),
    )
    for split in priority[:remainder]:
        quotas[split] += 1
    return quotas


def _stratification_signature(
    group: SimilarityGroup, records_by_id: dict[str, ImageRecord]
) -> tuple[object, ...]:
    return (
        len(group.image_ids),
        tuple(sorted(_group_counts(group, records_by_id, "source").items())),
    )


def assign_groups(
    records: Sequence[ImageRecord],
    groups: Sequence[SimilarityGroup],
    seed: int = PARTITION_SEED,
    frozen_assignments: dict[str, str] | None = None,
) -> tuple[dict[str, str], set[str]]:
    records_by_id = {record.image_id: record for record in records}
    class_totals = Counter(record.class_id for record in records)
    image_counts = Counter()
    class_counts = {split: Counter() for split in SPLIT_NAMES}
    group_assignments: dict[str, str] = {}
    fixed_assignments = frozen_assignments or {}

    for group in groups:
        old_splits = {
            fixed_assignments[image_id]
            for image_id in group.image_ids
            if image_id in fixed_assignments
        }
        if len(old_splits) > 1:
            raise FrozenAssignmentConflict(
                f"El grupo {group.group_id} conecta asignaciones congeladas "
                f"incompatibles: {sorted(old_splits)}"
            )
        if old_splits:
            split = next(iter(old_splits))
            if split not in SPLIT_NAMES:
                raise PartitionError(f"Partición congelada inválida: {split}")
            group_assignments[group.group_id] = split
            for image_id in group.image_ids:
                record = records_by_id[image_id]
                image_counts[split] += 1
                class_counts[split][record.class_id] += 1

    unassigned_groups = [
        group for group in groups if group.group_id not in group_assignments
    ]
    groups_large: set[str] = set()
    for group in unassigned_groups:
        class_counts_in_group = _group_counts(group, records_by_id, "class_id")
        if any(
            Fraction(class_count)
            > LARGE_GROUP_EVAL_TARGET_FRACTION
            * Fraction(str(SPLIT_RATIOS["calibration"]))
            * class_totals[class_id]
            for class_id, class_count in class_counts_in_group.items()
        ):
            groups_large.add(group.group_id)

    rng = random.Random(seed)

    def place(group: SimilarityGroup, split: str) -> None:
        group_assignments[group.group_id] = split
        for image_id in group.image_ids:
            record = records_by_id[image_id]
            image_counts[split] += 1
            class_counts[split][record.class_id] += 1

    def objective(split: str, group: SimilarityGroup) -> float:
        group_class_counts = _group_counts(group, records_by_id, "class_id")
        score = 0.0
        for class_id, total in class_totals.items():
            target = total * SPLIT_RATIOS[split]
            denominator = max(target, 1.0)
            previous_error = (class_counts[split][class_id] - target) / denominator
            candidate_error = (
                class_counts[split][class_id] + group_class_counts[class_id] - target
            ) / denominator
            score += candidate_error**2 - previous_error**2
        target_total = len(records) * SPLIT_RATIOS[split]
        denominator = max(target_total, 1.0)
        previous_total_error = (image_counts[split] - target_total) / denominator
        candidate_total_error = (
            image_counts[split] + len(group.image_ids) - target_total
        ) / denominator
        return score + candidate_total_error**2 - previous_total_error**2

    for group in unassigned_groups:
        if group.group_id in groups_large:
            place(group, "train")

    groups_by_signature: dict[tuple[object, ...], list[SimilarityGroup]] = defaultdict(list)
    for group in unassigned_groups:
        if group.group_id not in groups_large:
            groups_by_signature[
                _stratification_signature(group, records_by_id)
            ].append(group)
    for signature in sorted(groups_by_signature, key=repr):
        matching_groups = groups_by_signature[signature]
        quotas = _integer_split_quotas(len(matching_groups))
        order = sorted(matching_groups, key=lambda group: (rng.random(), group.group_id))
        assigned_in_bucket = Counter()
        for group in order:
            eligible_splits = [
                split
                for split in SPLIT_NAMES
                if assigned_in_bucket[split] < quotas[split]
            ]
            scores = {split: objective(split, group) for split in eligible_splits}
            minimum = min(scores.values())
            candidates = [
                split
                for split in eligible_splits
                if math.isclose(scores[split], minimum, abs_tol=1e-12)
            ]
            chosen_split = rng.choice(candidates)
            place(group, chosen_split)
            assigned_in_bucket[chosen_split] += 1

    assignments = {
        image_id: group_assignments[group.group_id]
        for group in groups
        for image_id in group.image_ids
    }
    if set(assignments) != set(records_by_id):
        raise PartitionError("No todas las imágenes recibieron una partición.")
    return assignments, groups_large


def check_future_trap_hash_conflicts(
    frozen_records: Sequence[ImageRecord],
    trap_records: Sequence[ImageRecord],
    threshold: int = PHASH_HAMMING_THRESHOLD,
) -> list[dict[str, str | int]]:
    conflicts: list[dict[str, str | int]] = []
    for trap in trap_records:
        for frozen in frozen_records:
            distance = hamming_distance(trap.phash64, frozen.phash64)
            if distance <= threshold:
                conflicts.append(
                    {
                        "trap_path": trap.path,
                        "frozen_path": frozen.path,
                        "hamming_distance": distance,
                        "frozen_image_id": frozen.image_id,
                    }
                )
    return conflicts


def extend_assignments(
    frozen_records: Sequence[ImageRecord],
    frozen_assignments: dict[str, str],
    new_records: Sequence[ImageRecord],
    seed: int = PARTITION_SEED,
) -> dict[str, str]:
    trap_records = [
        record
        for record in new_records
        if record.source in {"trap", "traps", "set_trampas"}
    ]
    if trap_records:
        conflicts = check_future_trap_hash_conflicts(frozen_records, trap_records)
        if conflicts:
            details = "; ".join(
                f"{item['trap_path']} <-> {item['frozen_path']} "
                f"(Hamming={item['hamming_distance']})"
                for item in conflicts
            )
            raise PartitionError(
                "Se encontraron casi-duplicados de imágenes congeladas; "
                "consultar antes de asignar: "
                f"{details}"
            )
        raise PartitionError(
            "No se encontraron casi-duplicados pHash con las imágenes congeladas. "
            "El set de trampas sigue requiriendo aprobación humana antes de asignar."
        )
    combined = [*frozen_records, *new_records]
    combined_groups = build_similarity_groups(combined)
    assignments, _ = assign_groups(
        combined,
        combined_groups,
        seed=seed,
        frozen_assignments=frozen_assignments,
    )
    for image_id, split in frozen_assignments.items():
        if assignments.get(image_id) != split:
            raise FrozenAssignmentConflict(
                f"La extensión intentó cambiar la partición congelada de {image_id}."
            )
    return assignments


def build_report(
    records: Sequence[ImageRecord],
    groups: Sequence[SimilarityGroup],
    assignments: dict[str, str],
    groups_large: set[str],
    source_counts: dict[str, int],
) -> str:
    records_by_id = {record.image_id: record for record in records}
    class_totals = Counter(record.class_id for record in records)
    class_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    class_split_groups: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    source_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_split_groups: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    class_source_split_counts: dict[tuple[str, str, str], int] = Counter()
    class_source_split_groups: dict[
        tuple[str, str, str], set[str]
    ] = defaultdict(set)
    class_split_group_sizes: dict[tuple[str, str], list[int]] = defaultdict(list)
    split_group_size_distribution: dict[str, Counter[int]] = {
        split: Counter() for split in SPLIT_NAMES
    }
    largest_group_by_class: Counter[str] = Counter()
    for group in groups:
        counts = Counter(records_by_id[image_id].class_id for image_id in group.image_ids)
        for class_id, count in counts.items():
            largest_group_by_class[class_id] = max(
                largest_group_by_class[class_id], count
            )
        split = assignments[group.image_ids[0]]
        split_group_size_distribution[split][len(group.image_ids)] += 1
        for image_id in group.image_ids:
            record = records_by_id[image_id]
            class_split_counts[record.class_id][split] += 1
            class_split_groups[record.class_id][split].add(group.group_id)
            source_split_counts[record.source][split] += 1
            source_split_groups[record.source][split].add(group.group_id)
            class_source_split_counts[(record.class_id, record.source, split)] += 1
            class_source_split_groups[(record.class_id, record.source, split)].add(
                group.group_id
            )
        for class_id, count in counts.items():
            class_split_group_sizes[(class_id, split)].append(count)

    lines = [
        "Informe de partición candidata (aún no congelada)",
        f"Semilla: {PARTITION_SEED}",
        f"Imágenes totales: {len(records)}; grupos de parecido: {len(groups)}",
        (
            "Objetivo 70/15/15; semilla fija; grupo grande si su cantidad para "
            "alguna clase supera LARGE_GROUP_EVAL_TARGET_FRACTION "
            f"({float(LARGE_GROUP_EVAL_TARGET_FRACTION):.6g}) × 15% × total de esa clase."
        ),
        (
            "Fuentes: CCSN no registra autores en la estructura local; se agrupa "
            "solo por pHash (control más débil). Open Images agrupa por autor y pHash. "
            "Sociedad usa autor `desconocido` por imagen y agrupa por pHash."
        ),
        (
            "Open Images: "
            f"{source_counts['open_images_disk']} archivos en disco; "
            f"{source_counts['open_images_attribution']} filas de atribución."
        ),
        "",
        "Por clase (imágenes y porcentaje real por partición; grupos por partición):",
    ]
    for class_id in sorted(class_totals):
        total = class_totals[class_id]
        split_details = []
        for split in SPLIT_NAMES:
            count = class_split_counts[class_id][split]
            ratio = 100 * count / total if total else 0
            group_count = len(class_split_groups[class_id][split])
            split_details.append(f"{split}={count} ({ratio:.1f}%, {group_count} grupos)")
        largest = largest_group_by_class[class_id]
        largest_ratio = 100 * largest / total if total else 0
        warning = " ADVERTENCIA: grupo >30% de la clase" if largest_ratio > 30 else ""
        calibration_count = class_split_counts[class_id]["calibration"]
        calibration_groups = len(class_split_groups[class_id]["calibration"])
        test_count = class_split_counts[class_id]["test"]
        test_groups = len(class_split_groups[class_id]["test"])
        calibration_sizes = sorted(
            class_split_group_sizes[(class_id, "calibration")]
        )
        test_sizes = sorted(class_split_group_sizes[(class_id, "test")])
        training_ratio = class_split_counts[class_id]["train"] / total if total else 0
        training_warning = (
            " ADVERTENCIA: entrenamiento por encima del objetivo 70%"
            if training_ratio > SPLIT_RATIOS["train"]
            else ""
        )
        sample_warning = (
            " ADVERTENCIA: menos de 30 imágenes de calibración"
            if calibration_count < CALIBRATION_WARNING_MIN_IMAGES
            else ""
        )
        lines.append(
            f"- {class_id} (N={total}): {'; '.join(split_details)}; "
            f"grupo máximo={largest} ({largest_ratio:.1f}%). "
            f"calibración={calibration_count} imágenes/{calibration_groups} grupos; "
            f"prueba={test_count} imágenes/{test_groups} grupos; "
            f"tamaños de grupos calibración={calibration_sizes}, prueba={test_sizes}."
            f"{warning}{sample_warning}{training_warning}"
        )
    lines.extend(
        [
            "",
            "Por fuente (imágenes y grupos por partición):",
        ]
    )
    for source in sorted(source_split_counts):
        lines.append(
            f"- {source}: "
            + "; ".join(
                f"{split}={source_split_counts[source][split]} imágenes/"
                f"{len(source_split_groups[source][split])} grupos"
                for split in SPLIT_NAMES
            )
        )
    lines.extend(
        [
            "",
            "Distribución de tamaños de grupo por partición (tamaño: grupos):",
        ]
    )
    all_group_sizes = sorted(
        {
            size
            for distribution in split_group_size_distribution.values()
            for size in distribution
        }
    )
    for split in SPLIT_NAMES:
        distribution = ", ".join(
            f"{size}: {split_group_size_distribution[split][size]}"
            for size in all_group_sizes
        )
        lines.append(f"- {split}: {distribution}")
    lines.extend(
        [
            "",
            "Desglose por clase, partición y fuente (imágenes/grupos):",
        ]
    )
    sources = sorted(source_split_counts)
    for class_id in sorted(class_totals):
        for split in SPLIT_NAMES:
            source_details = [
                f"{source}={class_source_split_counts[(class_id, source, split)]}/"
                f"{len(class_source_split_groups[(class_id, source, split)])}"
                for source in sources
            ]
            lines.append(f"- {class_id}, {split}: " + "; ".join(source_details))
    lines.extend(
        [
            "",
            f"Grupos grandes enviados a entrenamiento: {len(groups_large)}.",
            (
                "El informe describe una propuesta reproducible; no altera ni corrige "
                "automáticamente proporciones o clases con muestras pequeñas."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def candidate_document(
    records: Sequence[ImageRecord],
    groups: Sequence[SimilarityGroup],
    assignments: dict[str, str],
    groups_large: set[str],
    source_counts: dict[str, int],
) -> dict[str, object]:
    group_by_image = {
        image_id: group.group_id
        for group in groups
        for image_id in group.image_ids
    }
    return {
        "status": "candidate_requires_human_approval",
        "seed": PARTITION_SEED,
        "split_ratios": SPLIT_RATIOS,
        "phash_hamming_threshold": PHASH_HAMMING_THRESHOLD,
        "large_group_eval_target_fraction": float(
            LARGE_GROUP_EVAL_TARGET_FRACTION
        ),
        "large_group_ids": sorted(groups_large),
        "source_counts": source_counts,
        "images": [
            {
                **asdict(record),
                "group_id": group_by_image[record.image_id],
                "split": assignments[record.image_id],
            }
            for record in sorted(records, key=lambda item: item.path)
        ],
    }


def versioned_manifest_rows(
    records: Sequence[ImageRecord], assignments: dict[str, str]
) -> list[dict[str, str]]:
    record_ids = {record.image_id for record in records}
    if not records or len(record_ids) != len(records) or set(assignments) != record_ids:
        raise PartitionError("Las asignaciones no coinciden con el inventario.")

    rows: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    valid_class_ids = set(CCSN_FOLDER_TO_CLASS.values())
    for record in sorted(records, key=lambda item: item.path):
        path = PurePosixPath(record.path)
        if (
            not record.path
            or "\\" in record.path
            or re.match(r"^[A-Za-z]:", record.path)
            or path.is_absolute()
            or ".." in path.parts
        ):
            raise PartitionError(
                f"La ruta del manifiesto debe ser relativa y segura: {record.path}"
            )
        if record.path in seen_paths:
            raise PartitionError(f"Ruta duplicada en el manifiesto: {record.path}")
        seen_paths.add(record.path)
        if record.source not in {"ccsn", "open_images", "sociedad"}:
            raise PartitionError(f"Fuente inválida en el manifiesto: {record.source}")
        if record.class_id not in valid_class_ids:
            raise PartitionError(f"Clase inválida en el manifiesto: {record.class_id}")
        if not re.fullmatch(r"(?:desconocido|autor_[0-9]{3,})", record.author_id):
            raise PartitionError(f"ID de autor inválido: {record.author_id}")
        if not re.fullmatch(r"[0-9a-f]{16}", record.phash64):
            raise PartitionError(f"pHash64 inválido para {record.path}")
        split = assignments[record.image_id]
        if split not in SPLIT_NAMES:
            raise PartitionError(f"Partición inválida para {record.path}: {split}")
        rows.append(
            {
                "path": record.path,
                "class_id": record.class_id,
                "source": record.source,
                "author_id": record.author_id,
                "phash64": record.phash64,
                "split": split,
            }
        )
    return rows
