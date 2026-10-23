from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from cloud_classifier.partitioning import build_similarity_groups
from cloud_classifier.preprocessing import load_model_contract
from cloud_classifier.training import (
    TrainingDataError,
    _records_from_training_rows,
    _validate_validation_assignment,
    class_order_from_contract,
    create_validation_v2_assignment,
    filter_excluded_training_rows,
    read_exclusion_manifest,
    read_training_rows,
)


PARTITION_PATH = Path("manifests/partitions-v1.json")
VALIDATION_V1_PATH = Path("manifests/validation-v1.json")
EXCLUSION_PATH = Path("manifests/exclusions-v1.json")
OUTPUT_DIRECTORY = Path("data/audits/label-conflicts")
CANDIDATE_PATH = OUTPUT_DIRECTORY / "validation-v2-candidate.json"
REPORT_PATH = OUTPUT_DIRECTORY / "validation-v2-report.md"
SEED = 42


def _class_group_counts(
    rows: list[dict[str, Any]], selected_paths: set[str], class_ids: tuple[str, ...]
) -> tuple[Counter[str], Counter[str]]:
    records = _records_from_training_rows(rows, class_ids)
    records_by_id = {record.image_id: record for record in records}
    class_by_path = {row["path"]: row["class_id"] for row in rows}
    groups = build_similarity_groups(records)
    image_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    for group in groups:
        member_paths = {
            records_by_id[image_id].path for image_id in group.image_ids
        }
        selected = member_paths.intersection(selected_paths)
        if selected and selected != member_paths:
            raise TrainingDataError(
                f"La asignación divide el grupo de parecido {group.group_id}."
            )
        if selected:
            for path in member_paths:
                image_counts[class_by_path[path]] += 1
            for class_id in {
                records_by_id[image_id].class_id for image_id in group.image_ids
            }:
                group_counts[class_id] += 1
    return image_counts, group_counts


def _group_counts_by_size(
    rows: list[dict[str, Any]], selected_paths: set[str], class_ids: tuple[str, ...]
) -> tuple[Counter[str], Counter[str]]:
    records = _records_from_training_rows(rows, class_ids)
    records_by_id = {record.image_id: record for record in records}
    validation_sizes: Counter[str] = Counter()
    remainder_sizes: Counter[str] = Counter()
    for group in build_similarity_groups(records):
        member_paths = {
            records_by_id[image_id].path for image_id in group.image_ids
        }
        selected = member_paths.intersection(selected_paths)
        if selected and selected != member_paths:
            raise TrainingDataError(
                f"La asignación divide el grupo de parecido {group.group_id}."
            )
        bucket = str(len(member_paths)) if len(member_paths) <= 4 else "5+"
        (validation_sizes if selected else remainder_sizes)[bucket] += 1
    return validation_sizes, remainder_sizes


def _multiimage_fraction(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return sum(count for size, count in counts.items() if size != "1") / total


def build_validation_v2_report(
    partition_path: Path = PARTITION_PATH,
    validation_v1_path: Path = VALIDATION_V1_PATH,
    exclusion_path: Path = EXCLUSION_PATH,
) -> tuple[dict[str, Any], str]:
    train_rows, partition_sha256 = read_training_rows(partition_path)
    classes = class_order_from_contract(load_model_contract())
    exclusion_rows, exclusion_sha256 = read_exclusion_manifest(
        exclusion_path, partition_sha256, classes
    )
    all_partition_rows = json.loads(partition_path.read_text(encoding="utf-8"))
    rows_by_path = {row["path"]: row for row in all_partition_rows}
    for excluded in exclusion_rows:
        row = rows_by_path.get(excluded["path"])
        if (
            row is None
            or row.get("class_id") != excluded["class_id"]
            or row.get("source") != excluded["source"]
            or row.get("split") != excluded["split"]
        ):
            raise TrainingDataError(
                f"La exclusión no coincide con la partición: {excluded['path']}"
            )

    included_rows = filter_excluded_training_rows(
        train_rows, exclusion_rows, classes
    )
    if len(train_rows) != 2125 or len(included_rows) != 1944:
        raise TrainingDataError(
            "El soporte de train no coincide con la versión revisada "
            f"(antes {len(train_rows)}, después {len(included_rows)})."
        )
    candidate = create_validation_v2_assignment(
        included_rows, partition_sha256, exclusion_sha256, classes, SEED
    )

    try:
        validation_v1 = json.loads(validation_v1_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingDataError(
            f"No se pudo leer validation-v1: {validation_v1_path}"
        ) from error
    if (
        not isinstance(validation_v1, dict)
        or validation_v1.get("partition_sha256") != partition_sha256
    ):
        raise TrainingDataError(
            "validation-v1 no corresponde a partitions-v1.json."
        )
    _validate_validation_assignment(
        validation_v1,
        train_rows,
        partition_sha256,
        classes,
        SEED,
    )
    validation_v1_paths = set(validation_v1.get("validation_paths", []))
    if not validation_v1_paths.issubset({row["path"] for row in train_rows}):
        raise TrainingDataError("validation-v1 contiene rutas fuera de train.")
    validation_v2_paths = set(candidate["validation_paths"])
    excluded_train_paths = {
        row["path"] for row in exclusion_rows if row["split"] == "train"
    }
    if validation_v2_paths & excluded_train_paths:
        raise TrainingDataError("validation-v2 incluye rutas excluidas.")
    if not validation_v2_paths.issubset({row["path"] for row in included_rows}):
        raise TrainingDataError("validation-v2 contiene rutas fuera de train incluido.")
    if len(validation_v2_paths) != len(candidate["validation_paths"]):
        raise TrainingDataError("validation-v2 contiene rutas duplicadas.")

    v1_images, v1_groups = _class_group_counts(
        train_rows, validation_v1_paths, classes
    )
    remaining_v1_paths = validation_v1_paths - excluded_train_paths
    filtered_v1_images, filtered_v1_groups = _class_group_counts(
        included_rows, remaining_v1_paths, classes
    )
    v2_images, v2_groups = _class_group_counts(
        included_rows, validation_v2_paths, classes
    )

    v1_val_sizes, v1_remainder_sizes = _group_counts_by_size(
        train_rows, validation_v1_paths, classes
    )
    filtered_v1_val_sizes, filtered_v1_remainder_sizes = _group_counts_by_size(
        included_rows, remaining_v1_paths, classes
    )
    v2_val_sizes, v2_remainder_sizes = _group_counts_by_size(
        included_rows, validation_v2_paths, classes
    )

    grouped_rows = [
        {
            "class_id": class_id,
            "v1_images": v1_images[class_id],
            "v1_groups": v1_groups[class_id],
            "v1_after_exclusions_images": filtered_v1_images[class_id],
            "v1_after_exclusions_groups": filtered_v1_groups[class_id],
            "v2_images": v2_images[class_id],
            "v2_groups": v2_groups[class_id],
        }
        for class_id in classes
    ]
    low_group_support = [
        item["class_id"] for item in grouped_rows if item["v2_groups"] < 8
    ]
    for name, path_set, rows in (
        ("validation-v1", validation_v1_paths, train_rows),
        ("validation-v2", validation_v2_paths, included_rows),
    ):
        record_map = {row["path"]: row for row in rows}
        if any(record_map[path]["split"] != "train" for path in path_set):
            raise TrainingDataError(f"{name} contiene filas no train.")

    report_lines = [
        "# Informe candidato de validation-v2",
        "",
        "**Pendiente de aprobación explícita; `manifests/validation-v2.json` "
        "todavía no fue creado ni congelado.**",
        "",
        "- Generado con el mismo algoritmo agrupado y estratificado que v1, "
        "semilla fija 42 y objetivo del 10%; no se usaron datos de modelos, "
        "métricas ni predicciones.",
        f"- Filas train antes/después de exclusiones: {len(train_rows)} / "
        f"{len(included_rows)}; filas seleccionadas para validación-v2: "
        f"{len(validation_v2_paths)}.",
        f"- Hash de partitions-v1: `{partition_sha256}`.",
        f"- Hash de exclusions-v1: `{exclusion_sha256}`.",
        "",
        "## Imágenes y grupos por clase",
        "",
        "| Clase | v1 imágenes | v1 grupos | v1 tras exclusiones imágenes | "
        "v1 tras exclusiones grupos | v2 imágenes | v2 grupos |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    report_lines.extend(
        f"| {item['class_id']} | {item['v1_images']} | {item['v1_groups']} | "
        f"{item['v1_after_exclusions_images']} | "
        f"{item['v1_after_exclusions_groups']} | {item['v2_images']} | "
        f"{item['v2_groups']} |"
        for item in grouped_rows
    )
    report_lines.extend(
        [
            "",
            f"Cirrus: v1 {v1_images['cirrus']} imágenes / {v1_groups['cirrus']} "
            f"grupos; v1 filtrada {filtered_v1_images['cirrus']} imágenes / "
            f"{filtered_v1_groups['cirrus']} grupos; v2 {v2_images['cirrus']} "
            f"imágenes / {v2_groups['cirrus']} grupos.",
            "",
            "## Proporción de grupos multiimagen",
            "",
            "| Asignación | Validación multiimagen/grupos | % validación | "
            "Resto train multiimagen/grupos | % resto train |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, val_sizes, remainder_sizes in (
        ("v1", v1_val_sizes, v1_remainder_sizes),
        (
            "v1 después de exclusiones",
            filtered_v1_val_sizes,
            filtered_v1_remainder_sizes,
        ),
        ("v2 candidato", v2_val_sizes, v2_remainder_sizes),
    ):
        val_multi = sum(count for size, count in val_sizes.items() if size != "1")
        remainder_multi = sum(
            count for size, count in remainder_sizes.items() if size != "1"
        )
        report_lines.append(
            f"| {label} | {val_multi}/{sum(val_sizes.values())} | "
            f"{_multiimage_fraction(val_sizes):.2%} | "
            f"{remainder_multi}/{sum(remainder_sizes.values())} | "
            f"{_multiimage_fraction(remainder_sizes):.2%} |"
        )

    report_lines.extend(
        [
            "",
            "## Integridad y soporte mínimo",
            "",
            "- Ningún grupo de parecido está partido entre validación y resto de train.",
            "- validation-v2 no contiene rutas excluidas.",
            "- Todas las rutas seleccionadas pertenecen a `split=train`; no se "
            "incluyeron filas de calibration o test.",
            f"- Grupos multiimagen: v2 validación {sum(count for size, count in v2_val_sizes.items() if size != '1')}; "
            f"resto train {sum(count for size, count in v2_remainder_sizes.items() if size != '1')}.",
            (
                "AVISO: menos de 8 grupos en validation-v2 para: "
                + ", ".join(low_group_support)
                if low_group_support
                else "Todas las clases alcanzan al menos 8 grupos en validation-v2."
            ),
            "",
            "La asignación candidata reproducible se guardó fuera de `manifests/` "
            "para revisión. No se modifica `validation-v1.json`.",
            "",
        ]
    )
    return candidate, "\n".join(report_lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera un candidato e informe de validation-v2 sin congelarlo."
    )
    parser.add_argument("--partition-manifest", type=Path, default=PARTITION_PATH)
    parser.add_argument("--validation-v1", type=Path, default=VALIDATION_V1_PATH)
    parser.add_argument("--exclusion-manifest", type=Path, default=EXCLUSION_PATH)
    parser.add_argument("--candidate-output", type=Path, default=CANDIDATE_PATH)
    parser.add_argument("--report-output", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    candidate, report = build_validation_v2_report(
        args.partition_manifest, args.validation_v1, args.exclusion_manifest
    )
    args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_output.write_text(
        json.dumps(candidate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.report_output.write_text(report, encoding="utf-8")
    print(f"Candidato local: {args.candidate_output}")
    print(f"Informe: {args.report_output}")


if __name__ == "__main__":
    main()
