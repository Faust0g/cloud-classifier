from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from cloud_classifier.partitioning import (
    ImageRecord,
    SimilarityGroup,
    build_similarity_groups,
)
from cloud_classifier.preprocessing import load_model_contract, preprocess_image


MODEL_NAME = "mobilenetv3_large_100"
PARTITION_MANIFEST = Path("manifests/partitions-v1.json")
VALIDATION_MANIFEST = Path("manifests/validation-v2.json")
EXCLUSION_MANIFEST = Path("manifests/exclusions-v1.json")
VALIDATION_FRACTION = 0.10
VALIDATION_ALGORITHM = "grouped-greedy-class-and-size-stratification-v2"
GROUP_SIZE_STRATIFICATION_WEIGHT = 5.0


class TrainingDataError(ValueError):
    pass


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 3e-4
    batch_size: int = 32
    max_epochs: int = 20
    patience: int = 4
    seed: int = 42
    minimum_accuracy_target: float | None = None

    def validate(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("La tasa de aprendizaje debe ser positiva.")
        if self.batch_size < 1:
            raise ValueError("El tamaño de lote debe ser al menos 1.")
        if self.max_epochs < 1:
            raise ValueError("El máximo de épocas debe ser al menos 1.")
        if self.patience < 1:
            raise ValueError("La paciencia debe ser al menos 1.")
        if self.minimum_accuracy_target is not None:
            if not math.isfinite(self.minimum_accuracy_target) or not (
                0 <= self.minimum_accuracy_target <= 1
            ):
                raise ValueError("El objetivo de exactitud debe estar entre 0 y 1.")


@dataclass
class EarlyStopping:
    patience: int
    best_epoch: int = 0
    best_macro_f1: float = -1.0
    best_validation_loss: float = math.inf
    stale_epochs: int = 0

    def observe(self, epoch: int, macro_f1: float, validation_loss: float) -> tuple[bool, bool]:
        tied_score = math.isclose(macro_f1, self.best_macro_f1, abs_tol=1e-12)
        improved = (
            self.best_epoch == 0
            or macro_f1 > self.best_macro_f1
            or (tied_score and validation_loss < self.best_validation_loss)
        )
        if improved:
            self.best_epoch = epoch
            self.best_macro_f1 = macro_f1
            self.best_validation_loss = validation_loss
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return improved, self.stale_epochs >= self.patience


def class_order_from_contract(contract: dict[str, Any] | None = None) -> tuple[str, ...]:
    model_contract = load_model_contract() if contract is None else contract
    classes = model_contract.get("classes")
    if not isinstance(classes, list):
        raise TrainingDataError("El contrato no define una lista de clases.")
    class_ids = tuple(
        item.get("id") for item in classes if isinstance(item, dict)
    )
    if (
        len(class_ids) != len(classes)
        or not class_ids
        or len(class_ids) != 11
        or any(not isinstance(class_id, str) or not class_id for class_id in class_ids)
        or len(class_ids) != len(set(class_ids))
    ):
        raise TrainingDataError(
            "El contrato debe definir once IDs de clase únicos y no vacíos."
        )
    return class_ids


def read_training_rows(partition_path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        raw_manifest = partition_path.read_bytes()
        rows = json.loads(raw_manifest)
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingDataError(
            f"No se pudo leer el manifiesto de particiones: {partition_path}"
        ) from error
    if not isinstance(rows, list):
        raise TrainingDataError("El manifiesto de particiones debe ser una lista.")

    training_rows = []
    for row in rows:
        if not isinstance(row, dict):
            raise TrainingDataError("El manifiesto contiene una fila inválida.")
        if row.get("split") == "train":
            training_rows.append(row)
    if not training_rows:
        raise TrainingDataError("El manifiesto no contiene filas de entrenamiento.")
    return training_rows, hashlib.sha256(raw_manifest).hexdigest()


def read_exclusion_manifest(
    exclusion_path: Path, partition_sha256: str, class_ids: Sequence[str]
) -> tuple[list[dict[str, Any]], str]:
    try:
        raw_manifest = exclusion_path.read_bytes()
        manifest = json.loads(raw_manifest)
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingDataError(
            f"No se pudo leer el manifiesto de exclusiones: {exclusion_path}"
        ) from error
    if not isinstance(manifest, dict) or manifest.get("exclusions_version") != 1:
        raise TrainingDataError("La versión del manifiesto de exclusiones no es válida.")
    if manifest.get("partition_sha256") != partition_sha256:
        raise TrainingDataError(
            "El manifiesto de exclusiones no corresponde a las particiones congeladas."
        )
    groups = manifest.get("groups")
    if not isinstance(groups, list):
        raise TrainingDataError("El manifiesto de exclusiones debe contener grupos.")

    allowed_classes = set(class_ids)
    exclusion_rows: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    seen_paths: set[str] = set()
    for group in groups:
        if (
            not isinstance(group, dict)
            or not isinstance(group.get("group_id"), str)
            or not group["group_id"]
            or group.get("reason") != "misma foto, etiquetas contradictorias"
            or group["group_id"] in seen_groups
        ):
            raise TrainingDataError("El manifiesto contiene un grupo de exclusión inválido.")
        seen_groups.add(group["group_id"])
        images = group.get("images")
        if not isinstance(images, list) or len(images) < 2:
            raise TrainingDataError(
                f"El grupo de exclusión {group['group_id']} no contiene suficientes imágenes."
            )
        group_classes: set[str] = set()
        for image in images:
            if not isinstance(image, dict):
                raise TrainingDataError("Una imagen excluida tiene formato inválido.")
            path = image.get("path")
            class_id = image.get("class_id")
            source = image.get("source")
            split = image.get("split")
            posix_path = PurePosixPath(path) if isinstance(path, str) else None
            if (
                posix_path is None
                or not path
                or posix_path.is_absolute()
                or ".." in posix_path.parts
                or "\\" in path
                or ":" in path
                or not isinstance(class_id, str)
                or class_id not in allowed_classes
                or not isinstance(source, str)
                or not source
                or not isinstance(split, str)
                or split not in {"train", "calibration", "test"}
                or path in seen_paths
            ):
                raise TrainingDataError("Una imagen excluida tiene campos inválidos.")
            seen_paths.add(path)
            group_classes.add(class_id)
            exclusion_rows.append(
                {
                    "group_id": group["group_id"],
                    "path": path,
                    "class_id": class_id,
                    "source": source,
                    "split": split,
                }
            )
        if len(group_classes) < 2:
            raise TrainingDataError(
                f"El grupo de exclusión {group['group_id']} no tiene etiquetas contradictorias."
            )
    if not groups:
        raise TrainingDataError("El manifiesto de exclusiones no contiene grupos.")
    return exclusion_rows, hashlib.sha256(raw_manifest).hexdigest()


def filter_excluded_training_rows(
    rows: Sequence[dict[str, Any]],
    exclusion_rows: Sequence[dict[str, Any]],
    class_ids: Sequence[str],
) -> list[dict[str, Any]]:
    rows_by_path = {row.get("path"): row for row in rows}
    excluded_training = {
        exclusion["path"]: exclusion
        for exclusion in exclusion_rows
        if exclusion["split"] == "train"
    }
    for path, exclusion in excluded_training.items():
        row = rows_by_path.get(path)
        if (
            row is None
            or row.get("split") != "train"
            or row.get("class_id") != exclusion["class_id"]
            or row.get("source") != exclusion["source"]
        ):
            raise TrainingDataError(
                f"La exclusión no coincide con la fila de entrenamiento: {path}"
            )

    records = _records_from_training_rows(rows, class_ids)
    records_by_id = {record.image_id: record for record in records}
    group_paths_by_id: dict[str, set[str]] = {}
    for group in build_similarity_groups(records):
        group_paths = {
            records_by_id[image_id].path for image_id in group.image_ids
        }
        group_paths_by_id[group.group_id] = group_paths
        excluded_paths = group_paths.intersection(excluded_training)
        if excluded_paths and excluded_paths != group_paths:
            raise TrainingDataError(
                f"La exclusión divide el grupo de parecido {group.group_id}."
            )
    train_exclusion_groups: dict[str, set[str]] = {}
    for path, exclusion in excluded_training.items():
        train_exclusion_groups.setdefault(exclusion["group_id"], set()).add(path)
    for group_id, paths in train_exclusion_groups.items():
        if group_paths_by_id.get(group_id) != paths:
            raise TrainingDataError(
                f"Las rutas no coinciden con el grupo de exclusión {group_id}."
            )
    return [row for row in rows if row["path"] not in excluded_training]


def prepare_training_validation_rows(
    rows: Sequence[dict[str, Any]],
    validation_paths: set[str],
    exclusion_rows: Sequence[dict[str, Any]],
    class_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    included_rows = filter_excluded_training_rows(rows, exclusion_rows, class_ids)
    training_rows = [
        row for row in included_rows if row["path"] not in validation_paths
    ]
    validation_rows = [
        row for row in included_rows if row["path"] in validation_paths
    ]
    if not training_rows or not validation_rows:
        raise TrainingDataError("La división dejó vacío entrenamiento o validación.")
    return training_rows, validation_rows


def _records_from_training_rows(
    rows: Sequence[dict[str, Any]], class_ids: Sequence[str]
) -> list[ImageRecord]:
    allowed_classes = set(class_ids)
    records = []
    seen_paths: set[str] = set()
    for row in rows:
        path = row.get("path")
        class_id = row.get("class_id")
        source = row.get("source")
        author_id = row.get("author_id")
        phash64 = row.get("phash64")
        if (
            not isinstance(path, str)
            or not path
            or not isinstance(class_id, str)
            or not isinstance(source, str)
            or not isinstance(author_id, str)
            or not isinstance(phash64, str)
        ):
            raise TrainingDataError("Una fila de entrenamiento tiene campos inválidos.")
        if class_id not in allowed_classes:
            raise TrainingDataError(
                f"Clase fuera del orden definido en el contrato: {class_id}"
            )
        if path in seen_paths:
            raise TrainingDataError(f"Ruta duplicada en entrenamiento: {path}")
        try:
            hash_value = int(phash64, 16)
        except ValueError as error:
            raise TrainingDataError(f"pHash inválido para {path}.") from error
        if len(phash64) != 16 or hash_value < 0:
            raise TrainingDataError(f"pHash inválido para {path}.")
        seen_paths.add(path)
        image_id = hashlib.sha256(path.encode("utf-8")).hexdigest()[:24]
        records.append(
            ImageRecord(
                image_id=image_id,
                path=path,
                class_id=class_id,
                source=source,
                author_id=author_id,
                phash64=phash64,
            )
        )
    return records


def _group_class_counts(
    group: SimilarityGroup, records_by_id: dict[str, ImageRecord]
) -> Counter[str]:
    return Counter(records_by_id[image_id].class_id for image_id in group.image_ids)


def _group_size_bucket(size: int) -> str:
    return str(size) if size <= 4 else "5+"


def _validation_group_summary(
    groups: Sequence[SimilarityGroup],
    records_by_id: dict[str, ImageRecord],
    validation_group_ids: set[str],
    class_ids: Sequence[str],
) -> dict[str, Any]:
    size_counts = {
        "validation": Counter(),
        "training_remainder": Counter(),
    }
    class_group_counts = {
        "validation": Counter(),
        "training_remainder": Counter(),
    }
    for group in groups:
        split = (
            "validation"
            if group.group_id in validation_group_ids
            else "training_remainder"
        )
        size_counts[split][_group_size_bucket(len(group.image_ids))] += 1
        for class_id in _group_class_counts(group, records_by_id):
            class_group_counts[split][class_id] += 1

    size_buckets = ("1", "2", "3", "4", "5+")
    return {
        "group_size_counts": {
            split: {bucket: counts.get(bucket, 0) for bucket in size_buckets}
            for split, counts in size_counts.items()
        },
        "class_group_counts": {
            split: {class_id: counts.get(class_id, 0) for class_id in class_ids}
            for split, counts in class_group_counts.items()
        },
    }


def _effective_validation_group_summary(
    training_rows: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
    class_ids: Sequence[str],
) -> dict[str, Any]:
    records = _records_from_training_rows(training_rows, class_ids)
    records_by_id = {record.image_id: record for record in records}
    validation_image_ids = {
        hashlib.sha256(row["path"].encode("utf-8")).hexdigest()[:24]
        for row in validation_rows
    }
    groups = build_similarity_groups(records)
    validation_group_ids = {
        group.group_id
        for group in groups
        if validation_image_ids.intersection(group.image_ids)
    }
    return _validation_group_summary(
        groups, records_by_id, validation_group_ids, class_ids
    )


def _validation_objective(
    selected: Counter[str],
    totals: Counter[str],
    selected_group_sizes: Counter[str],
    total_group_sizes: Counter[str],
    selected_class_groups: Counter[str],
    total_class_groups: Counter[str],
    selected_total: int,
    validation_fraction: float,
) -> float:
    objective = 0.0
    for class_id, total in totals.items():
        target = total * validation_fraction
        objective += ((selected[class_id] - target) / max(target, 1.0)) ** 2
    target_total = sum(totals.values()) * validation_fraction
    objective += ((selected_total - target_total) / max(target_total, 1.0)) ** 2
    for bucket, total in total_group_sizes.items():
        target = total * validation_fraction
        objective += (
            (selected_group_sizes[bucket] - target) / max(target, 1.0)
        ) ** 2 * GROUP_SIZE_STRATIFICATION_WEIGHT
    for class_id, total in total_class_groups.items():
        target = total * validation_fraction
        objective += (
            (selected_class_groups[class_id] - target) / max(target, 1.0)
        ) ** 2
    return objective


def create_validation_assignment(
    rows: Sequence[dict[str, Any]],
    partition_sha256: str,
    class_ids: Sequence[str],
    seed: int,
    validation_fraction: float = VALIDATION_FRACTION,
) -> dict[str, Any]:
    if not 0 < validation_fraction < 1:
        raise ValueError("La proporción de validación debe estar entre 0 y 1.")
    records = _records_from_training_rows(rows, class_ids)
    groups = build_similarity_groups(records)
    if len(groups) < 2:
        raise TrainingDataError(
            "No hay al menos dos grupos de parecido para separar entrenamiento y validación."
        )

    records_by_id = {record.image_id: record for record in records}
    counts_by_group = {
        group.group_id: _group_class_counts(group, records_by_id) for group in groups
    }
    totals = Counter(record.class_id for record in records)
    total_group_sizes = Counter(_group_size_bucket(len(group.image_ids)) for group in groups)
    total_class_groups = Counter(
        class_id
        for group in groups
        for class_id in counts_by_group[group.group_id]
    )
    rng = random.Random(seed)
    tie_order = list(groups)
    rng.shuffle(tie_order)
    group_order = {group.group_id: index for index, group in enumerate(tie_order)}
    remaining = {group.group_id: group for group in groups}
    selected_groups: list[SimilarityGroup] = []
    selected_counts: Counter[str] = Counter()
    selected_group_sizes: Counter[str] = Counter()
    selected_class_groups: Counter[str] = Counter()
    selected_total = 0
    current_objective = _validation_objective(
        selected_counts,
        totals,
        selected_group_sizes,
        total_group_sizes,
        selected_class_groups,
        total_class_groups,
        selected_total,
        validation_fraction,
    )

    while remaining:
        candidates = []
        for group in remaining.values():
            group_counts = counts_by_group[group.group_id]
            candidate_counts = selected_counts + group_counts
            bucket = _group_size_bucket(len(group.image_ids))
            candidate_group_sizes = selected_group_sizes + Counter({bucket: 1})
            candidate_class_groups = selected_class_groups + Counter(
                {class_id: 1 for class_id in group_counts}
            )
            candidate_total = selected_total + len(group.image_ids)
            objective = _validation_objective(
                candidate_counts,
                totals,
                candidate_group_sizes,
                total_group_sizes,
                candidate_class_groups,
                total_class_groups,
                candidate_total,
                validation_fraction,
            )
            if objective < current_objective:
                candidates.append(
                    (
                        objective,
                        group_order[group.group_id],
                        group,
                        candidate_counts,
                        candidate_group_sizes,
                        candidate_class_groups,
                        candidate_total,
                    )
                )
        if not candidates:
            break
        (
            objective,
            _,
            chosen,
            selected_counts,
            selected_group_sizes,
            selected_class_groups,
            selected_total,
        ) = min(
            candidates, key=lambda item: (item[0], item[1])
        )
        selected_groups.append(chosen)
        del remaining[chosen.group_id]
        current_objective = objective

    if not selected_groups or len(selected_groups) == len(groups):
        raise TrainingDataError(
            "La asignación no puede dejar imágenes tanto para entrenamiento como para validación."
        )
    selected_paths = sorted(
        records_by_id[image_id].path
        for group in selected_groups
        for image_id in group.image_ids
    )
    validation_group_ids = {group.group_id for group in selected_groups}
    return {
        "validation_version": 1,
        "algorithm": VALIDATION_ALGORITHM,
        "partition_sha256": partition_sha256,
        "seed": seed,
        "validation_fraction_target": validation_fraction,
        "validation_paths": selected_paths,
        "groups": [
            {
                "group_id": group.group_id,
                "paths": sorted(records_by_id[image_id].path for image_id in group.image_ids),
            }
            for group in sorted(selected_groups, key=lambda item: item.group_id)
        ],
        **_validation_group_summary(
            groups, records_by_id, validation_group_ids, class_ids
        ),
    }


def load_or_create_validation_assignment(
    validation_path: Path,
    rows: Sequence[dict[str, Any]],
    partition_sha256: str,
    class_ids: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    if validation_path.exists():
        try:
            assignment = json.loads(validation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TrainingDataError(
                f"No se pudo leer la asignación de validación: {validation_path}"
            ) from error
    else:
        assignment = create_validation_assignment(
            rows, partition_sha256, class_ids, seed
        )
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        with validation_path.open("x", encoding="utf-8", newline="\n") as output:
            json.dump(assignment, output, ensure_ascii=False, indent=2)
            output.write("\n")

    _validate_validation_assignment(
        assignment, rows, partition_sha256, class_ids, seed
    )
    return assignment


def create_validation_v2_assignment(
    rows: Sequence[dict[str, Any]],
    partition_sha256: str,
    exclusion_sha256: str,
    class_ids: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    if seed != 42:
        raise TrainingDataError("validation-v2 requiere la semilla fija 42.")
    if not exclusion_sha256:
        raise TrainingDataError("validation-v2 requiere el hash de exclusiones.")
    if any(row.get("split") != "train" for row in rows):
        raise TrainingDataError("validation-v2 solo admite filas del split train.")
    assignment = create_validation_assignment(
        rows, partition_sha256, class_ids, seed
    )
    assignment["validation_version"] = 2
    assignment["exclusion_sha256"] = exclusion_sha256
    return assignment


def load_validation_v2_assignment(
    validation_path: Path,
    rows: Sequence[dict[str, Any]],
    partition_sha256: str,
    exclusion_sha256: str,
    class_ids: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    if seed != 42:
        raise TrainingDataError("validation-v2 requiere la semilla fija 42.")
    try:
        assignment = json.loads(validation_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise TrainingDataError(
            f"Falta {validation_path}; validation-v2 debe aprobarse y congelarse "
            "antes de entrenar."
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingDataError(
            f"No se pudo leer validation-v2: {validation_path}"
        ) from error
    if not isinstance(assignment, dict):
        raise TrainingDataError("validation-v2 debe ser un objeto JSON.")
    if assignment.get("validation_version") == 1:
        raise TrainingDataError(
            "Se requiere validation-v2; validation-v1 es solo historial "
            "y no se puede usar para entrenar."
        )
    _validate_validation_assignment(
        assignment,
        rows,
        partition_sha256,
        class_ids,
        seed,
        expected_version=2,
        exclusion_sha256=exclusion_sha256,
    )
    return assignment


def _validate_validation_assignment(
    assignment: Any,
    rows: Sequence[dict[str, Any]],
    partition_sha256: str,
    class_ids: Sequence[str],
    seed: int,
    *,
    expected_version: int = 1,
    exclusion_sha256: str | None = None,
) -> None:
    if (
        not isinstance(assignment, dict)
        or assignment.get("validation_version") != expected_version
    ):
        raise TrainingDataError("La versión de la asignación de validación no es válida.")
    if assignment.get("partition_sha256") != partition_sha256:
        raise TrainingDataError(
            "La asignación de validación no corresponde al manifiesto congelado actual."
        )
    if (
        expected_version == 2
        and assignment.get("exclusion_sha256") != exclusion_sha256
    ):
        raise TrainingDataError(
            "validation-v2 no corresponde al manifiesto de exclusiones actual."
        )
    if assignment.get("seed") != seed:
        raise TrainingDataError(
            "La semilla no coincide con la asignación de validación versionada."
        )
    if assignment.get("algorithm") != VALIDATION_ALGORITHM:
        raise TrainingDataError("El algoritmo de validación versionado no es compatible.")
    validation_fraction = assignment.get("validation_fraction_target")
    if (
        not isinstance(validation_fraction, (int, float))
        or not math.isclose(
            validation_fraction, VALIDATION_FRACTION, abs_tol=1e-12
        )
    ):
        raise TrainingDataError("La proporción de validación versionada no es compatible.")

    records = _records_from_training_rows(rows, class_ids)
    records_by_path = {record.path: record for record in records}
    records_by_id = {record.image_id: record for record in records}
    validation_paths = assignment.get("validation_paths")
    if (
        not isinstance(validation_paths, list)
        or not validation_paths
        or any(not isinstance(path, str) for path in validation_paths)
        or len(validation_paths) != len(set(validation_paths))
        or not set(validation_paths).issubset(records_by_path)
    ):
        raise TrainingDataError(
            "La asignación contiene rutas duplicadas o que no pertenecen a entrenamiento."
        )

    groups = build_similarity_groups(records)
    selected = set(validation_paths)
    expected_groups = []
    for group in groups:
        member_paths = {records_by_id[record_id].path for record_id in group.image_ids}
        if selected.intersection(member_paths):
            if not member_paths.issubset(selected):
                raise TrainingDataError(
                    "La asignación versionada divide un grupo de parecido."
                )
            expected_groups.append(
                {
                    "group_id": group.group_id,
                    "paths": sorted(member_paths),
                }
            )
    if assignment.get("groups") != expected_groups:
        raise TrainingDataError(
            "Los grupos de la asignación no coinciden con las rutas de validación."
        )
    expected_summary = _validation_group_summary(
        groups,
        records_by_id,
        {group["group_id"] for group in expected_groups},
        class_ids,
    )
    if (
        assignment.get("group_size_counts") != expected_summary["group_size_counts"]
        or assignment.get("class_group_counts")
        != expected_summary["class_group_counts"]
    ):
        raise TrainingDataError(
            "Los resúmenes de tamaños o grupos por clase no coinciden con la asignación."
        )


def _resolve_image_path(root: Path, relative_path: str) -> Path:
    image_path = (root / Path(relative_path)).resolve()
    try:
        image_path.relative_to(root.resolve())
    except ValueError as error:
        raise TrainingDataError(f"La ruta de imagen sale de la raíz del proyecto: {relative_path}") from error
    return image_path


def _augment_training_image(image: Image.Image, rng: Any) -> Image.Image:
    oriented = ImageOps.exif_transpose(image)
    try:
        rgb = oriented.convert("RGB")
    finally:
        if oriented is not image:
            oriented.close()
    width, height = rgb.size
    scale = rng.uniform(0.8, 1.0)
    crop_width = max(1, round(width * scale))
    crop_height = max(1, round(height * scale))
    left = rng.randint(0, width - crop_width)
    top = rng.randint(0, height - crop_height)
    cropped = rgb.crop((left, top, left + crop_width, top + crop_height))
    rgb.close()

    if rng.random() < 0.5:
        flipped = ImageOps.mirror(cropped)
        cropped.close()
        cropped = flipped
    brightness = ImageEnhance.Brightness(cropped).enhance(rng.uniform(0.9, 1.1))
    cropped.close()
    contrasted = ImageEnhance.Contrast(brightness).enhance(rng.uniform(0.9, 1.1))
    brightness.close()
    return contrasted


def _make_dataset(
    rows: Sequence[dict[str, Any]],
    class_ids: Sequence[str],
    root: Path,
    augment: bool,
):
    import torch
    from torch.utils.data import Dataset

    class TrainingImageDataset(Dataset):
        def __init__(self) -> None:
            self.rows = list(rows)
            self.class_to_index = {class_id: index for index, class_id in enumerate(class_ids)}

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            row = self.rows[index]
            tensor_array = _load_preprocessed_image(row, root, augment)
            return (
                torch.from_numpy(tensor_array.copy()),
                self.class_to_index[row["class_id"]],
                row["source"],
            )

    return TrainingImageDataset()


def _load_preprocessed_image(
    row: dict[str, Any], root: Path, augment: bool
) -> np.ndarray:
    image_path = _resolve_image_path(root, row["path"])
    with Image.open(image_path) as image:
        if augment:
            augmented = _augment_training_image(image, random)
            try:
                return preprocess_image(augmented)[0]
            finally:
                augmented.close()
        return preprocess_image(image)[0]


def _seed_everything(seed: int):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return torch


def _verify_pretrained_normalization(
    pretrained_config: Any, contract: dict[str, Any]
) -> None:
    if not isinstance(pretrained_config, dict):
        raise TrainingDataError("timm no expone la configuración del modelo preentrenado.")
    normalization = contract["input"]["normalization"]
    for field in ("mean", "std"):
        actual = pretrained_config.get(field)
        expected = normalization.get(field)
        if (
            not isinstance(actual, (list, tuple))
            or not isinstance(expected, list)
            or len(actual) != len(expected)
            or any(
                not math.isclose(float(actual_value), float(expected_value), abs_tol=1e-7)
                for actual_value, expected_value in zip(actual, expected)
            )
        ):
            raise TrainingDataError(
                f"El {field} del modelo timm no coincide con el contrato compartido: "
                f"{actual!r} != {expected!r}."
            )


def _top_k_accuracy(logits: Any, labels: Any, k: int) -> int:
    predictions = logits.topk(min(k, logits.shape[1]), dim=1).indices
    return int(predictions.eq(labels.view(-1, 1)).any(dim=1).sum().item())


def _macro_f1(labels: Sequence[int], predictions: Sequence[int], class_count: int) -> float:
    true_values = np.asarray(labels, dtype=np.int64)
    predicted_values = np.asarray(predictions, dtype=np.int64)
    scores = []
    for class_index in range(class_count):
        true_positive = int(np.count_nonzero((true_values == class_index) & (predicted_values == class_index)))
        false_positive = int(np.count_nonzero((true_values != class_index) & (predicted_values == class_index)))
        false_negative = int(np.count_nonzero((true_values == class_index) & (predicted_values != class_index)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def _run_epoch(model: Any, loader: Any, criterion: Any, optimizer: Any | None, device: Any, class_count: int):
    import torch

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_top2 = 0
    total_count = 0
    labels_seen: list[int] = []
    predictions_seen: list[int] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels, _sources in loader:
            images = images.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int(logits.argmax(dim=1).eq(labels).sum().item())
            total_top2 += _top_k_accuracy(logits, labels, 2)
            total_count += batch_size
            predicted = logits.argmax(dim=1)
            labels_seen.extend(labels.detach().cpu().tolist())
            predictions_seen.extend(predicted.detach().cpu().tolist())
    if total_count == 0:
        raise TrainingDataError("Un conjunto de entrenamiento/validación quedó vacío.")
    return {
        "loss": total_loss / total_count,
        "accuracy": total_correct / total_count,
        "top2_accuracy": total_top2 / total_count,
        "macro_f1": _macro_f1(labels_seen, predictions_seen, class_count),
        "labels": labels_seen,
        "predictions": predictions_seen,
    }


def _validation_breakdown(
    rows: Sequence[dict[str, Any]],
    labels: Sequence[int],
    predictions: Sequence[int],
    class_ids: Sequence[str],
) -> dict[str, Any]:
    class_metrics = {}
    source_totals: dict[str, list[int]] = {}
    for row, label, prediction in zip(rows, labels, predictions):
        source = row["source"]
        counts = source_totals.setdefault(source, [0, 0])
        counts[1] += 1
        counts[0] += int(label == prediction)
    for class_index, class_id in enumerate(class_ids):
        matching = [
            (label, prediction)
            for label, prediction in zip(labels, predictions)
            if label == class_index
        ]
        correct = sum(label == prediction for label, prediction in matching)
        class_metrics[class_id] = {
            "accuracy": correct / len(matching) if matching else None,
            "count": len(matching),
        }
    return {
        "by_class": class_metrics,
        "by_source": {
            source: {
                "accuracy": correct / count,
                "count": count,
            }
            for source, (correct, count) in sorted(source_totals.items())
        },
    }


def _write_curves(path: Path, history: Sequence[dict[str, Any]]) -> None:
    fields = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "train_top2_accuracy",
        "validation_loss",
        "validation_accuracy",
        "validation_top2_accuracy",
        "validation_macro_f1",
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: entry[field] for field in fields} for entry in history)


def train(
    root: Path,
    partition_manifest: Path,
    validation_manifest: Path,
    run_root: Path,
    config: TrainingConfig,
    exclusion_manifest: Path = EXCLUSION_MANIFEST,
    device_name: str | None = None,
) -> Path:
    config.validate()
    contract = load_model_contract()
    class_ids = class_order_from_contract(contract)
    rows, partition_sha256 = read_training_rows(partition_manifest)
    exclusion_rows, exclusion_sha256 = read_exclusion_manifest(
        exclusion_manifest, partition_sha256, class_ids
    )
    included_rows = filter_excluded_training_rows(
        rows, exclusion_rows, class_ids
    )
    assignment = load_validation_v2_assignment(
        validation_manifest,
        included_rows,
        partition_sha256,
        exclusion_sha256,
        class_ids,
        config.seed,
    )
    validation_paths = set(assignment["validation_paths"])
    training_rows = [
        row for row in included_rows if row["path"] not in validation_paths
    ]
    validation_rows = [
        row for row in included_rows if row["path"] in validation_paths
    ]
    if not training_rows or not validation_rows:
        raise TrainingDataError("La división dejó vacío entrenamiento o validación.")
    effective_validation_summary = _effective_validation_group_summary(
        training_rows + validation_rows, validation_rows, class_ids
    )

    torch = _seed_everything(config.seed)
    import timm
    from torch.utils.data import DataLoader

    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = timm.create_model(
        MODEL_NAME, pretrained=True, num_classes=len(class_ids)
    )
    pretrained_data_config = timm.data.resolve_model_data_config(model)
    _verify_pretrained_normalization(pretrained_data_config, contract)
    model.to(device)

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    training_loader = DataLoader(
        _make_dataset(training_rows, class_ids, root, augment=True),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        _make_dataset(validation_rows, class_ids, root, augment=False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_directory = run_root / f"{timestamp}-seed-{config.seed}"
    run_directory.mkdir(parents=True, exist_ok=False)
    best_path = run_directory / "best.pt"
    last_path = run_directory / "last.pt"
    history: list[dict[str, Any]] = []
    early_stopping = EarlyStopping(patience=config.patience)

    run_metadata = {
        "model": MODEL_NAME,
        "class_order": list(class_ids),
        "contract_version": contract["contract_version"],
        "preprocessing": contract["input"],
        "pretrained_data_config": pretrained_data_config,
        "augmentation": {
            "training_only": True,
            "random_crop_scale": [0.8, 1.0],
            "horizontal_flip_probability": 0.5,
            "brightness_factor": [0.9, 1.1],
            "contrast_factor": [0.9, 1.1],
            "vertical_flip": False,
        },
        "config": asdict(config),
        "partition": {
            "path": partition_manifest.as_posix(),
            "sha256": partition_sha256,
        },
        "exclusion": {
            "path": exclusion_manifest.as_posix(),
            "sha256": exclusion_sha256,
            "groups": len({row["group_id"] for row in exclusion_rows}),
            "images": len(exclusion_rows),
        },
        "validation": {
            "path": validation_manifest.as_posix(),
            "sha256": hashlib.sha256(validation_manifest.read_bytes()).hexdigest(),
            "version": assignment["validation_version"],
            "algorithm": assignment["algorithm"],
            "fraction_target": assignment["validation_fraction_target"],
            "group_size_counts": effective_validation_summary["group_size_counts"],
            "class_group_counts": effective_validation_summary["class_group_counts"],
            "assignment_group_size_counts": assignment["group_size_counts"],
            "assignment_class_group_counts": assignment["class_group_counts"],
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "pillow": Image.__version__,
            "torch": torch.__version__,
            "timm": timm.__version__,
            "device": str(device),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
    }
    _write_json(run_directory / "run.json", run_metadata)

    for epoch in range(1, config.max_epochs + 1):
        train_metrics = _run_epoch(
            model, training_loader, criterion, optimizer, device, len(class_ids)
        )
        validation_metrics = _run_epoch(
            model, validation_loader, criterion, None, device, len(class_ids)
        )
        entry = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_top2_accuracy": train_metrics["top2_accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_top2_accuracy": validation_metrics["top2_accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
        }
        history.append(entry)
        improved, should_stop = early_stopping.observe(
            epoch, entry["validation_macro_f1"], entry["validation_loss"]
        )
        if improved:
            torch.save(model.state_dict(), best_path)
        torch.save(model.state_dict(), last_path)
        _write_curves(run_directory / "curves.csv", history)
        print(json.dumps(entry, ensure_ascii=False))
        if should_stop:
            break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    best_validation_metrics = _run_epoch(
        model, validation_loader, criterion, None, device, len(class_ids)
    )
    breakdown = _validation_breakdown(
        validation_rows,
        best_validation_metrics["labels"],
        best_validation_metrics["predictions"],
        class_ids,
    )
    final_metrics = {
        "best_epoch": early_stopping.best_epoch,
        "best_validation_macro_f1": early_stopping.best_macro_f1,
        "best_validation_loss": early_stopping.best_validation_loss,
        "last_epoch": history[-1]["epoch"],
        "best_validation_accuracy": best_validation_metrics["accuracy"],
        "best_validation_top2_accuracy": best_validation_metrics["top2_accuracy"],
        "last_validation_accuracy": history[-1]["validation_accuracy"],
        "last_validation_top2_accuracy": history[-1]["validation_top2_accuracy"],
        "minimum_accuracy_target": config.minimum_accuracy_target,
        "minimum_accuracy_target_reached": (
            None
            if config.minimum_accuracy_target is None
            else best_validation_metrics["accuracy"] >= config.minimum_accuracy_target
        ),
        **breakdown,
    }
    run_metadata["selection"] = {
        "metric": "validation_macro_f1",
        "tie_breaker": "validation_loss",
        "best_epoch": early_stopping.best_epoch,
        "best_macro_f1": early_stopping.best_macro_f1,
        "best_validation_loss": early_stopping.best_validation_loss,
        "last_epoch": history[-1]["epoch"],
    }
    _write_json(run_directory / "run.json", run_metadata)
    _write_json(run_directory / "validation-metrics.json", final_metrics)
    return run_directory


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena el clasificador usando únicamente el split train congelado."
    )
    parser.add_argument("--partition-manifest", type=Path, default=PARTITION_MANIFEST)
    parser.add_argument("--validation-manifest", type=Path, default=VALIDATION_MANIFEST)
    parser.add_argument("--exclusion-manifest", type=Path, default=EXCLUSION_MANIFEST)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-accuracy-target", type=float)
    parser.add_argument("--device")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    config = TrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
        minimum_accuracy_target=args.minimum_accuracy_target,
    )
    run_directory = train(
        root=args.root,
        partition_manifest=args.root / args.partition_manifest,
        validation_manifest=args.root / args.validation_manifest,
        run_root=args.root / args.run_root,
        config=config,
        exclusion_manifest=args.root / args.exclusion_manifest,
        device_name=args.device,
    )
    print(f"Corrida guardada en {run_directory}")
