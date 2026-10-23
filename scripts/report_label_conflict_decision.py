from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any

from cloud_classifier.partitioning import ImageRecord, build_similarity_groups


PARTITION_PATH = Path("manifests/partitions-v1.json")
VALIDATION_PATH = Path("manifests/validation-v1.json")
AUDIT_DIRECTORY = Path("data/audits/label-conflicts")
REPORT_PATH = AUDIT_DIRECTORY / "impact-report.md"
PARTITION_SHA256 = "0ebcf4d3c72ce3344cb51a4024ff5e7333dc9fcfe7e6c5e6bd0c7d5c53c0eae6"
PRESERVED_GROUP_IDS = {
    "group_3279ca96570ee968",
    "group_a5a19a1698ce22b8",
}
SPLITS = ("train", "calibration", "test")
SOURCES = ("ccsn", "open_images", "sociedad")
GROUP_SIZE_BUCKETS = ("1", "2", "3", "4", "5+")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"No se pudo leer JSON válido desde {path}.") from error


def _format_counts_table(
    rows: list[tuple[str, str, int, int, int, int]],
) -> list[str]:
    lines = [
        "| Clase | Partición | Imágenes antes | Imágenes después | "
        "Grupos antes | Grupos después |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {class_id} | {split} | {images_before} | {images_after} | "
        f"{groups_before} | {groups_after} |"
        for class_id, split, images_before, images_after, groups_before, groups_after in rows
    )
    return lines


def _group_contrast_minimum(group: dict[str, Any]) -> float:
    return min(image["luminance_std"] for image in group["images"])


def _complete_group_size_counts(counts: Counter[str]) -> dict[str, int]:
    return {bucket: counts.get(bucket, 0) for bucket in GROUP_SIZE_BUCKETS}


def _multiimage_percentage(counts: dict[str, int]) -> str:
    total = sum(counts.values())
    multiimage = sum(value for bucket, value in counts.items() if bucket != "1")
    return f"{multiimage / total:.2%}" if total else "0.00%"


def _evidence_summary(group: dict[str, Any]) -> tuple[int, float, float, float, float]:
    pairs = group["pairs"]
    return (
        min(pair["hamming_distance"] for pair in pairs),
        min(pair["ssim"] for pair in pairs),
        max(pair["ssim"] for pair in pairs),
        min(pair["mae_rgb_normalized"] for pair in pairs),
        max(pair["mae_rgb_normalized"] for pair in pairs),
    )


def build_impact_report(
    partition_path: Path = PARTITION_PATH,
    validation_path: Path = VALIDATION_PATH,
    audit_directory: Path = AUDIT_DIRECTORY,
) -> str:
    raw_partition = partition_path.read_bytes()
    partition_sha256 = hashlib.sha256(raw_partition).hexdigest()
    if partition_sha256 != PARTITION_SHA256:
        raise ValueError("El SHA-256 no coincide con el manifiesto congelado esperado.")
    rows = json.loads(raw_partition)
    if not isinstance(rows, list):
        raise ValueError("El manifiesto de particiones debe ser una lista.")
    rows_by_path = {row["path"]: row for row in rows}
    if len(rows_by_path) != len(rows):
        raise ValueError("El manifiesto de particiones contiene rutas duplicadas.")

    audit_groups = _read_json(audit_directory / "groups.json")
    if not isinstance(audit_groups, list) or len(audit_groups) != 130:
        raise ValueError("La auditoría no contiene los 130 grupos revisados.")
    groups_by_id = {group["group_id"]: group for group in audit_groups}
    if len(groups_by_id) != len(audit_groups):
        raise ValueError("La auditoría contiene IDs de grupo duplicados.")
    if not PRESERVED_GROUP_IDS.issubset(groups_by_id):
        raise ValueError("Falta uno de los dos grupos que deben conservarse.")
    excluded_group_ids = set(groups_by_id) - PRESERVED_GROUP_IDS
    if len(excluded_group_ids) != 128:
        raise ValueError("La decisión visual no produce exactamente 128 exclusiones.")

    for group in audit_groups:
        for image in group["images"]:
            row = rows_by_path.get(image["path"])
            if (
                row is None
                or row["class_id"] != image["class_id"]
                or row["source"] != image["source"]
                or row["split"] != image["split"]
            ):
                raise ValueError(
                    f"La auditoría y el manifiesto no coinciden: {image['path']}"
                )

    validation = _read_json(validation_path)
    if validation.get("partition_sha256") != partition_sha256:
        raise ValueError("La validación no corresponde al manifiesto congelado.")
    validation_paths = set(validation["validation_paths"])
    if not validation_paths.issubset(
        {row["path"] for row in rows if row["split"] == "train"}
    ):
        raise ValueError("La validación contiene rutas fuera del split train.")

    records = [
        ImageRecord(
            image_id=hashlib.sha256(row["path"].encode("utf-8")).hexdigest()[:24],
            path=row["path"],
            class_id=row["class_id"],
            source=row["source"],
            author_id=row["author_id"],
            phash64=row["phash64"],
        )
        for row in rows
    ]
    records_by_id = {record.image_id: record for record in records}
    groups = build_similarity_groups(records)
    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    group_paths: dict[str, set[str]] = {}
    for group in groups:
        members = [records_by_id[image_id] for image_id in group.image_ids]
        member_rows = [rows_by_path[member.path] for member in members]
        if len({row["split"] for row in member_rows}) != 1:
            raise ValueError(f"El grupo {group.group_id} cruza particiones.")
        grouped_rows[group.group_id] = member_rows
        group_paths[group.group_id] = {row["path"] for row in member_rows}

    audit_ids = set(groups_by_id)
    calculated_mixed_ids = {
        group_id
        for group_id, members in grouped_rows.items()
        if len({row["class_id"] for row in members}) > 1
    }
    if audit_ids != calculated_mixed_ids:
        raise ValueError("Los grupos de auditoría no coinciden con el manifiesto.")
    if not validation_paths:
        raise ValueError("La asignación de validación no contiene rutas.")
    for group_id, paths in group_paths.items():
        selected = paths.intersection(validation_paths)
        if selected and selected != paths:
            raise ValueError(f"La validación divide el grupo {group_id}.")

    excluded_paths = {
        image["path"]
        for group_id in excluded_group_ids
        for image in groups_by_id[group_id]["images"]
    }
    image_counts_before: Counter[tuple[str, str, str]] = Counter()
    image_counts_after: Counter[tuple[str, str, str]] = Counter()
    group_counts_before: Counter[tuple[str, str, str]] = Counter()
    group_counts_after: Counter[tuple[str, str, str]] = Counter()
    class_group_totals_before: Counter[tuple[str, str]] = Counter()
    class_group_totals_after: Counter[tuple[str, str]] = Counter()

    for row in rows:
        key = (row["class_id"], row["split"], row["source"])
        image_counts_before[key] += 1
        if row["path"] not in excluded_paths:
            image_counts_after[key] += 1

    for group_id, members in grouped_rows.items():
        split = members[0]["split"]
        for class_id in {row["class_id"] for row in members}:
            class_group_totals_before[(class_id, split)] += 1
            if group_id not in excluded_group_ids:
                class_group_totals_after[(class_id, split)] += 1
            sources_for_class = {row["source"] for row in members if row["class_id"] == class_id}
            for source in sources_for_class:
                key = (class_id, split, source)
                group_counts_before[key] += 1
                if group_id not in excluded_group_ids:
                    group_counts_after[key] += 1

    class_ids = sorted({row["class_id"] for row in rows})
    class_split_rows = []
    for class_id in class_ids:
        for split in SPLITS:
            images_before = sum(
                image_counts_before[(class_id, split, source)] for source in SOURCES
            )
            images_after = sum(
                image_counts_after[(class_id, split, source)] for source in SOURCES
            )
            groups_before = class_group_totals_before[(class_id, split)]
            groups_after = class_group_totals_after[(class_id, split)]
            class_split_rows.append(
                (class_id, split, images_before, images_after, groups_before, groups_after)
            )

    source_split_rows = []
    for source in SOURCES:
        for split in SPLITS:
            before = sum(
                1 for row in rows if row["source"] == source and row["split"] == split
            )
            after = sum(
                1
                for row in rows
                if row["source"] == source
                and row["split"] == split
                and row["path"] not in excluded_paths
            )
            groups_before = sum(
                1
                for group_id, members in grouped_rows.items()
                if members[0]["split"] == split
                and any(row["source"] == source for row in members)
            )
            groups_after = sum(
                1
                for group_id, members in grouped_rows.items()
                if group_id not in excluded_group_ids
                and members[0]["split"] == split
                and any(row["source"] == source for row in members)
            )
            source_split_rows.append(
                (source, split, before, after, groups_before, groups_after)
            )

    class_split_source_rows = [
        (
            class_id,
            split,
            source,
            image_counts_before[(class_id, split, source)],
            image_counts_after[(class_id, split, source)],
            group_counts_before[(class_id, split, source)],
            group_counts_after[(class_id, split, source)],
        )
        for class_id in class_ids
        for split in SPLITS
        for source in SOURCES
    ]

    val_groups_before = [
        group_id for group_id, paths in group_paths.items() if paths.issubset(validation_paths)
    ]
    val_groups_after = [
        group_id for group_id in val_groups_before if group_id not in excluded_group_ids
    ]
    validation_class_counts: dict[str, Counter[str]] = {
        "before": Counter(),
        "after": Counter(),
    }
    validation_image_counts: dict[str, Counter[str]] = {
        "before": Counter(),
        "after": Counter(),
    }
    for group_id in val_groups_before:
        members = grouped_rows[group_id]
        for class_id in {row["class_id"] for row in members}:
            validation_class_counts["before"][class_id] += 1
        for row in members:
            validation_image_counts["before"][row["class_id"]] += 1
        if group_id not in excluded_group_ids:
            for class_id in {row["class_id"] for row in members}:
                validation_class_counts["after"][class_id] += 1
            for row in members:
                validation_image_counts["after"][row["class_id"]] += 1

    source_excluded = Counter(
        row["source"] for row in rows if row["path"] in excluded_paths
    )
    split_excluded = Counter(
        row["split"] for row in rows if row["path"] in excluded_paths
    )
    class_pair_counts: Counter[tuple[str, str]] = Counter()
    for group_id in excluded_group_ids:
        classes = sorted({row["class_id"] for row in grouped_rows[group_id]})
        class_pair_counts.update(itertools.combinations(classes, 2))

    current_sizes = validation["group_size_counts"]
    train_size_before = current_sizes["training_remainder"]
    validation_size_before = current_sizes["validation"]
    excluded_val_groups = set(val_groups_before).intersection(excluded_group_ids)
    train_group_sizes_after: Counter[str] = Counter()
    validation_group_sizes_after: Counter[str] = Counter()
    for group_id, members in grouped_rows.items():
        if members[0]["split"] != "train" or group_id in excluded_group_ids:
            continue
        bucket = str(len(members)) if len(members) <= 4 else "5+"
        destination = (
            validation_group_sizes_after
            if group_paths[group_id].issubset(validation_paths)
            else train_group_sizes_after
        )
        destination[bucket] += 1
    validation_group_sizes_after_complete = _complete_group_size_counts(
        validation_group_sizes_after
    )
    train_group_sizes_after_complete = _complete_group_size_counts(
        train_group_sizes_after
    )

    evidence_lines = [
        "| Grupo excluido | Split | Fuente | Clases | Imágenes | H mín. | "
        "SSIM mín.–máx. | MAE mín.–máx. | Contraste mín. |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for group_id in sorted(excluded_group_ids):
        group = groups_by_id[group_id]
        hamming_min, ssim_min, ssim_max, mae_min, mae_max = _evidence_summary(group)
        evidence_lines.append(
            f"| `{group_id}` | {', '.join(group['splits'])} | "
            f"{', '.join(group['sources'])} | {', '.join(group['classes'])} | "
            f"{group['image_count']} | {hamming_min} | {ssim_min:.6f}–{ssim_max:.6f} | "
            f"{mae_min:.6f}–{mae_max:.6f} | {_group_contrast_minimum(group):.6f} |"
        )

    lines = [
        "# Informe candidato de impacto de exclusiones",
        "",
        "**Estado: pendiente de aprobación explícita; no se congelaron manifiestos.**",
        "Las 128 exclusiones se basan únicamente en la revisión visual del usuario. "
        "SSIM, MAE, Hamming y contraste se conservan como evidencia y no deciden "
        "la clasificación.",
        "",
        f"- SHA-256 de `partitions-v1.json`: `{partition_sha256}`.",
        f"- Grupos multiclase revisados: {len(groups_by_id)}; confirmados como duplicados: "
        f"{len(excluded_group_ids)}; conservados como fotos distintas: 2.",
        f"- Imágenes excluidas: {len(excluded_paths)}; por split: "
        f"`{dict(split_excluded)}`; por fuente: `{dict(source_excluded)}`.",
        f"- Grupos de 3+ imágenes excluidos: "
        f"{sum(groups_by_id[group_id]['image_count'] >= 3 for group_id in excluded_group_ids)} "
        f"({sum(groups_by_id[group_id]['image_count'] for group_id in excluded_group_ids if groups_by_id[group_id]['image_count'] >= 3)} imágenes).",
        "- CCSN y sociedad contienen copias de fotos bajo etiquetas distintas; "
        "Open Images no aporta grupos excluidos.",
        "- Los valores completos por par permanecen en `pair-metrics.csv`; la tabla "
        "siguiente resume la evidencia de cada grupo excluido.",
        "",
        "## Grupos preservados tras la revisión visual",
        "",
        "| Grupo | Estado previo | Split | Clases | H mínimo | SSIM | MAE | Contraste mínimo |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for group_id in sorted(PRESERVED_GROUP_IDS):
        group = groups_by_id[group_id]
        pair = group["pairs"][0]
        review_status = {
            "doubtful_pair_visual_review": "dudoso",
            "review_low_contrast": "bajo contraste",
        }.get(pair["review_status"], pair["review_status"])
        contrast = min(
            image["luminance_std"] for image in group["images"]
        )
        lines.append(
            f"| `{group_id}` | {review_status} | "
            f"{', '.join(group['splits'])} | {', '.join(group['classes'])} | "
            f"{group['minimum_hamming_distance']} | {pair['ssim']:.6f} | "
            f"{pair['mae_rgb_normalized']:.6f} | {contrast:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Imágenes y grupos por clase y partición",
            "",
            *_format_counts_table(class_split_rows),
            "",
            "## Imágenes y grupos por fuente y partición",
            "",
            "| Fuente | Partición | Imágenes antes | Imágenes después | "
            "Grupos antes | Grupos después |",
            "|---|---|---:|---:|---:|---:|",
            *(
                f"| {source} | {split} | {images_before} | {images_after} | "
                f"{groups_before} | {groups_after} |"
                for source, split, images_before, images_after, groups_before, groups_after
                in source_split_rows
            ),
            "",
            "## Desglose conjunto por clase, partición y fuente",
            "",
            "| Clase | Partición | Fuente | Imágenes antes | Imágenes después | "
            "Grupos antes | Grupos después |",
            "|---|---|---|---:|---:|---:|---:|",
            *(
                f"| {class_id} | {split} | {source} | {images_before} | "
                f"{images_after} | {groups_before} | {groups_after} |"
                for (
                    class_id,
                    split,
                    source,
                    images_before,
                    images_after,
                    groups_before,
                    groups_after,
                ) in class_split_source_rows
            ),
            "",
            "## Grupos por clase en validation-v1",
            "",
            "| Clase | Grupos antes | Grupos después | Imágenes antes | Imágenes después |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| {class_id} | {validation_class_counts['before'][class_id]} | "
        f"{validation_class_counts['after'][class_id]} | "
        f"{validation_image_counts['before'][class_id]} | "
        f"{validation_image_counts['after'][class_id]} |"
        for class_id in class_ids
    )
    lines.extend(
        [
            "",
            f"Validation-v1 pasaría de {len(validation_paths)} imágenes y "
            f"{len(val_groups_before)} grupos a "
            f"{len(validation_paths - excluded_paths)} imágenes y "
            f"{len(val_groups_after)} grupos. No se modifica `validation-v1.json`.",
            f"Grupos perdidos en validation-v1: {len(excluded_paths.intersection(validation_paths))} "
            f"imágenes en {len(excluded_val_groups)} grupos.",
            "",
            "### Distribución de tamaños de grupo tras filtrar",
            "",
            "| Zona | Antes | Después |",
            "|---|---|---|",
            f"| Validación | `{validation_size_before}` | "
            f"`{validation_group_sizes_after_complete}` |",
            f"| Resto de train | `{train_size_before}` | "
            f"`{train_group_sizes_after_complete}` |",
            "",
            "La validación no parte grupos. La proporción multiimagen cae de "
            f"{sum(value for key, value in validation_size_before.items() if key != '1')}/"
            f"{sum(validation_size_before.values())} "
            f"({_multiimage_percentage(validation_size_before)}) a "
            f"{sum(value for key, value in validation_group_sizes_after.items() if key != '1')}/"
            f"{sum(validation_group_sizes_after.values())} "
            f"({_multiimage_percentage(validation_group_sizes_after_complete)}); "
            "el resto de train pasa de "
            f"{sum(value for key, value in train_size_before.items() if key != '1')}/"
            f"{sum(train_size_before.values())} "
            f"({_multiimage_percentage(train_size_before)}) a "
            f"{sum(value for key, value in train_group_sizes_after.items() if key != '1')}/"
            f"{sum(train_group_sizes_after.values())} "
            f"({_multiimage_percentage(train_group_sizes_after_complete)}). "
            "No se rebalancea: los grupos "
            "multiimagen restantes contienen una sola clase.",
            "",
            "## Soporte de calibration y test",
            "",
            "| Clase | Calibration antes | Calibration después | Test antes | Test después | Aviso |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for class_id in class_ids:
        calibration_before = sum(
            image_counts_before[(class_id, "calibration", source)] for source in SOURCES
        )
        calibration_after = sum(
            image_counts_after[(class_id, "calibration", source)] for source in SOURCES
        )
        test_before = sum(
            image_counts_before[(class_id, "test", source)] for source in SOURCES
        )
        test_after = sum(
            image_counts_after[(class_id, "test", source)] for source in SOURCES
        )
        warnings = []
        if calibration_after < 30:
            warnings.append("calibration <30")
        if test_after < 30:
            warnings.append("test <30")
        lines.append(
            f"| {class_id} | {calibration_before} | {calibration_after} | "
            f"{test_before} | {test_after} | {', '.join(warnings)} |"
        )

    lines.extend(
        [
            "",
            "## Set descriptivo candidato de doble etiqueta",
            "",
            f"- Fotos únicas: {len(excluded_group_ids)}.",
            f"- Pares de clases distintos: {len(class_pair_counts)}.",
            "- Pares más frecuentes:",
        ]
    )
    lines.extend(
        f"  - {left}–{right}: {count}"
        for (left, right), count in class_pair_counts.most_common(10)
    )
    lines.extend(
        [
            "",
            "## Evidencia por grupo excluido",
            "",
            "SSIM/MAE/Hamming/contraste se muestran como evidencia; la decisión "
            "de excluir proviene de la revisión visual del usuario.",
            "",
            *evidence_lines,
            "",
            "## Sesgo y limitaciones",
            "",
            "Al excluir fotos ambiguas, el conjunto restante favorece ejemplos más "
            "claros de etiquetar. Por eso las métricas de calibration y test pueden "
            "ser algo optimistas frente a fotos reales, que pueden mostrar dos capas "
            "de nubes; el set de doble etiqueta solo compensa parcialmente este sesgo. "
            "El pHash con Hamming ≤ 8 tampoco garantiza detectar copias recortadas o "
            "con otra escala.",
            "",
            "No se abrieron imágenes de calibration o test para este informe; solo "
            "se consultaron sus rutas y etiquetas del manifiesto y la evidencia de "
            "comparación ya calculada.",
            "",
            "Los dos grupos preservados son falsos positivos del pHash según la "
            "revisión visual: `group_3279ca96570ee968` (dudoso) y "
            "`group_a5a19a1698ce22b8` (bajo contraste).",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera el informe candidato sin congelar manifiestos."
    )
    parser.add_argument("--partition-manifest", type=Path, default=PARTITION_PATH)
    parser.add_argument("--validation-manifest", type=Path, default=VALIDATION_PATH)
    parser.add_argument("--audit-directory", type=Path, default=AUDIT_DIRECTORY)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    report = build_impact_report(
        args.partition_manifest, args.validation_manifest, args.audit_directory
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Informe candidato guardado en {args.output}")


if __name__ == "__main__":
    main()
