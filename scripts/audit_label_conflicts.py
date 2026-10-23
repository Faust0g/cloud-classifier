from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from cloud_classifier.partitioning import (
    ImageRecord,
    build_similarity_groups,
    hamming_distance,
)


DEFAULT_MANIFEST = Path("manifests/partitions-v1.json")
DEFAULT_OUTPUT = Path("data/audits/label-conflicts")
IMAGE_SIZE = 256
SSIM_SIGMA = 1.5
SSIM_THRESHOLD = 0.98
MAE_THRESHOLD = 0.03
LOW_CONTRAST_THRESHOLD = 0.10


def _gaussian_blur(array: np.ndarray) -> np.ndarray:
    radius = int(3.5 * SSIM_SIGMA + 0.5)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets * offsets) / (2 * SSIM_SIGMA**2))
    kernel /= kernel.sum()
    result = np.asarray(array, dtype=np.float64)
    padded = np.pad(result, ((radius, radius), (0, 0), (0, 0)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, len(kernel), axis=0
    )
    result = np.einsum("hwck,k->hwc", windows, kernel, optimize=True)
    padded = np.pad(result, ((0, 0), (radius, radius), (0, 0)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, len(kernel), axis=1
    )
    return np.einsum("hwck,k->hwc", windows, kernel, optimize=True)


def _load_comparison_image(root: Path, relative_path: str) -> Image.Image:
    with Image.open(root / Path(relative_path)) as source:
        oriented = ImageOps.exif_transpose(source)
        try:
            rgb = oriented.convert("RGB")
        finally:
            if oriented is not source:
                oriented.close()
    rgb.thumbnail((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
    canvas.paste(
        rgb,
        ((IMAGE_SIZE - rgb.width) // 2, (IMAGE_SIZE - rgb.height) // 2),
    )
    rgb.close()
    return canvas


def _image_statistics(image: Image.Image) -> dict[str, Any]:
    rgb = np.asarray(image, dtype=np.float64) / 255.0
    gray = np.asarray(ImageOps.grayscale(image), dtype=np.float64) / 255.0
    mean = _gaussian_blur(rgb)
    variance = np.maximum(_gaussian_blur(rgb * rgb) - mean * mean, 0.0)
    return {
        "array": rgb,
        "mean": mean,
        "variance": variance,
        "luminance_std": float(gray.std()),
    }


def _compare_images(
    left: dict[str, Any], right: dict[str, Any]
) -> tuple[float, float]:
    left_rgb = left["array"]
    right_rgb = right["array"]
    mae = float(np.mean(np.abs(left_rgb - right_rgb)))
    covariance = _gaussian_blur(left_rgb * right_rgb) - left["mean"] * right["mean"]
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (
        (2 * left["mean"] * right["mean"] + c1) * (2 * covariance + c2)
    )
    denominator = (
        (left["mean"] ** 2 + right["mean"] ** 2 + c1)
        * (left["variance"] + right["variance"] + c2)
    )
    return float(np.mean(numerator / denominator)), mae


def _review_status(
    group_size: int, low_contrast: bool, meets_initial_cut: bool
) -> str:
    if group_size >= 3:
        return "review_mixed_group_3plus"
    if low_contrast:
        return "review_low_contrast"
    if meets_initial_cut:
        return "candidate_same_image_visual_confirmation_required"
    return "doubtful_pair_visual_review"


def _pair_rows_for_group(
    group_id: str,
    members: list[ImageRecord],
    rows_by_path: dict[str, dict[str, Any]],
    images: dict[str, Image.Image],
    statistics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    pairs = []
    for left, right in itertools.combinations(members, 2):
        left_stats = statistics[left.path]
        right_stats = statistics[right.path]
        ssim, mae = _compare_images(left_stats, right_stats)
        meets_cut = ssim >= SSIM_THRESHOLD and mae <= MAE_THRESHOLD
        low_contrast = (
            min(left_stats["luminance_std"], right_stats["luminance_std"])
            < LOW_CONTRAST_THRESHOLD
        )
        pairs.append(
            {
                "group_id": group_id,
                "group_size": len(members),
                "left_path": left.path,
                "left_class": left.class_id,
                "left_source": left.source,
                "left_split": rows_by_path[left.path]["split"],
                "left_luminance_std": left_stats["luminance_std"],
                "right_path": right.path,
                "right_class": right.class_id,
                "right_source": right.source,
                "right_split": rows_by_path[right.path]["split"],
                "right_luminance_std": right_stats["luminance_std"],
                "hamming_distance": hamming_distance(
                    int(left.phash64, 16), int(right.phash64, 16)
                ),
                "ssim": ssim,
                "mae_rgb_normalized": mae,
                "meets_initial_cut": meets_cut,
                "low_contrast_flag": low_contrast,
                "review_status": _review_status(
                    len(members), low_contrast, meets_cut
                ),
            }
        )
    return pairs


def _write_contact_sheet(
    path: Path,
    group_id: str,
    members: list[ImageRecord],
    rows_by_path: dict[str, dict[str, Any]],
    images: dict[str, Image.Image],
    statistics: dict[str, dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> None:
    columns = min(3, len(members))
    tile_width = 320
    tile_height = 350
    image_rows = math.ceil(len(members) / columns)
    pair_height = 32 + len(pairs) * 22
    sheet = Image.new(
        "RGB",
        (columns * tile_width, 48 + image_rows * tile_height + pair_height),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    classes = sorted({member.class_id for member in members})
    draw.text(
        (12, 12),
        f"{group_id} | {len(members)} images | classes: {', '.join(classes)}",
        fill="black",
        font=font,
    )
    for index, member in enumerate(members):
        column = index % columns
        row_index = index // columns
        x = column * tile_width + (tile_width - IMAGE_SIZE) // 2
        y = 48 + row_index * tile_height
        sheet.paste(images[member.path], (x, y))
        caption_x = column * tile_width + 8
        draw.text(
            (caption_x, y + IMAGE_SIZE + 3),
            f"#{index + 1} {member.class_id} | {member.source} | "
            f"{rows_by_path[member.path]['split']}",
            fill="black",
            font=font,
        )
        draw.text(
            (caption_x, y + IMAGE_SIZE + 18),
            f"luminance std={statistics[member.path]['luminance_std']:.5f}",
            fill="black",
            font=font,
        )
        display_path = member.path
        if len(display_path) > 49:
            display_path = "..." + display_path[-46:]
        draw.text(
            (caption_x, y + IMAGE_SIZE + 33),
            display_path,
            fill="black",
            font=font,
        )

    metrics_y = 48 + image_rows * tile_height + 4
    draw.text(
        (12, metrics_y),
        f"SSIM >= {SSIM_THRESHOLD:.2f}; MAE <= {MAE_THRESHOLD:.2f}; "
        f"low contrast if min std < {LOW_CONTRAST_THRESHOLD:.2f}",
        fill="black",
        font=font,
    )
    member_index = {member.path: index + 1 for index, member in enumerate(members)}
    for pair_index, pair in enumerate(pairs):
        line = (
            f"#{member_index[pair['left_path']]}-#{member_index[pair['right_path']]} "
            f"H={pair['hamming_distance']} SSIM={pair['ssim']:.5f} "
            f"MAE={pair['mae_rgb_normalized']:.5f} "
            f"std=({pair['left_luminance_std']:.5f},"
            f"{pair['right_luminance_std']:.5f}) {pair['review_status']}"
        )
        draw.text(
            (12, metrics_y + 20 + pair_index * 20),
            line,
            fill="black",
            font=font,
        )
    sheet.save(path)
    sheet.close()


def _summarize(groups: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    image_counts = {
        "source": Counter(),
        "split": Counter(),
        "class": Counter(),
    }
    group_counts = {
        "source": Counter(),
        "split": Counter(),
        "class": Counter(),
    }
    for group in groups:
        for source in group["sources"]:
            group_counts["source"][source] += 1
        for split in group["splits"]:
            group_counts["split"][split] += 1
        for class_id in group["classes"]:
            group_counts["class"][class_id] += 1
        for image in group["images"]:
            image_counts["source"][image["source"]] += 1
            image_counts["split"][image["split"]] += 1
            image_counts["class"][image["class_id"]] += 1
    return {
        "image_counts": image_counts,
        "group_counts": group_counts,
    }


def _write_report(
    output: Path,
    manifest_hash: str,
    rows: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> None:
    summary = _summarize(groups)
    status_counts = Counter(pair["review_status"] for pair in pairs)
    low_contrast_count = sum(pair["low_contrast_flag"] for pair in pairs)
    candidates = [
        pair
        for pair in pairs
        if pair["group_size"] == 2
        and pair["meets_initial_cut"]
        and not pair["low_contrast_flag"]
    ]
    multi_image_groups = [group for group in groups if group["image_count"] >= 3]
    candidate_paths = {
        path
        for pair in candidates
        for path in (pair["left_path"], pair["right_path"])
    }
    current_support = Counter(
        (row["class_id"], row["split"]) for row in rows
    )
    candidate_removed = Counter(
        (row["class_id"], row["split"])
        for row in rows
        if row["path"] in candidate_paths
    )

    lines = [
        "# Auditoría de grupos con etiquetas múltiples",
        "",
        "Solo comparación de imágenes: no se calcularon métricas del modelo ni se "
        "consultaron predicciones.",
        f"Manifiesto SHA-256: `{manifest_hash}`.",
        f"Filas: {len(rows)}; grupos: {len(groups)}; imágenes en grupos multiclase: "
        f"{sum(group['image_count'] for group in groups)}; comparaciones: {len(pairs)}.",
        f"Corte exploratorio, no decisión de exclusión: SSIM >= {SSIM_THRESHOLD:.2f} "
        f"y MAE RGB <= {MAE_THRESHOLD:.2f}. Bajo contraste: desvío estándar de "
        f"luminancia < {LOW_CONTRAST_THRESHOLD:.2f}. Las parejas de grupos 3+ "
        "separadas requieren revisión y no implican excluir el grupo entero.",
        "Método: aplicar EXIF, convertir a RGB, reducir con Lanczos conservando "
        "proporción y centrar en lienzo negro 256x256; SSIM local RGB con ventana "
        "Gaussiana sigma=1.5 y soporte de 11x11; MAE es promedio absoluto RGB "
        "normalizado; contraste es la desviación estándar de luminancia en [0,1].",
        "",
        "## Recuento de imágenes y grupos afectados",
        "",
        f"- Imágenes por fuente: `{dict(summary['image_counts']['source'])}`.",
        f"- Grupos por fuente: `{dict(summary['group_counts']['source'])}`.",
        f"- Imágenes por partición: `{dict(summary['image_counts']['split'])}`.",
        f"- Grupos que tocan cada partición: `{dict(summary['group_counts']['split'])}`.",
        f"- Imágenes por clase: `{dict(summary['image_counts']['class'])}`.",
        f"- Grupos que contienen cada clase: `{dict(summary['group_counts']['class'])}`.",
        f"- Estados de pares: `{dict(status_counts)}`; flags de bajo contraste: "
        f"{low_contrast_count}; grupos 3+: {len(multi_image_groups)}.",
        "",
        "## Soporte actual y escenario candidato (no aprobado)",
        "",
        "El escenario solo resta las imágenes de pares binarios que pasan ambos "
        "umbrales y no activan bajo contraste. No es una propuesta congelada; "
        "requiere confirmación visual y aprobación posterior.",
        "",
        "| Clase | Partición | Antes | Restaría el candidato | Después hipotético | Aviso <30 cal/test |",
        "|---|---|---:|---:|---:|---|",
    ]
    for class_id in sorted({row["class_id"] for row in rows}):
        for split in ("train", "calibration", "test"):
            before = current_support[(class_id, split)]
            removed = candidate_removed[(class_id, split)]
            after = before - removed
            warning = "SÍ" if split in {"calibration", "test"} and after < 30 else ""
            lines.append(
                f"| {class_id} | {split} | {before} | {removed} | {after} | {warning} |"
            )
    lines.extend(
        [
            "",
            "## Grupos multiclase",
            "",
            "| Grupo | N | Particiones | Fuentes | Clases | Hamming mínimo | Pares | Hoja de contacto |",
            "|---|---:|---|---|---|---:|---:|---|",
        ]
    )
    for group in groups:
        lines.append(
            f"| `{group['group_id']}` | {group['image_count']} | "
            f"{', '.join(group['splits'])} | {', '.join(group['sources'])} | "
            f"{', '.join(group['classes'])} | {group['minimum_hamming_distance']} | "
            f"{len(group['pairs'])} | "
            f"`contact-sheets/{group['group_id']}.png` |"
        )
    lines.extend(["", "## Imágenes por grupo", ""])
    for group in groups:
        lines.append(
            f"### {group['group_id']} — {group['image_count']} imágenes; "
            f"{', '.join(group['splits'])}; {', '.join(group['sources'])}; "
            f"Hamming mínimo {group['minimum_hamming_distance']}"
        )
        for image in group["images"]:
            lines.append(
                f"- `{image['path']}` — **{image['class_id']}**; "
                f"{image['source']}; {image['split']}; "
                f"σ luminancia={image['luminance_std']:.5f}."
            )
        lines.append("")
    lines.extend(
        [
            "## Comparaciones por pares",
            "",
            "Orden global: SSIM descendente, MAE ascendente. `pair-metrics.csv` "
            "contiene también los valores completos, banderas y estado de revisión.",
            "",
            "| Grupo | Par (clase: ruta) | Hamming | SSIM | MAE RGB | σ izq./der. | Estado |",
            "|---|---|---:|---:|---:|---|---|",
        ]
    )
    for pair in pairs:
        lines.append(
            f"| `{pair['group_id']}` | {pair['left_class']}: "
            f"`{pair['left_path']}` ↔ {pair['right_class']}: "
            f"`{pair['right_path']}` | {pair['hamming_distance']} | "
            f"{pair['ssim']:.6f} | {pair['mae_rgb_normalized']:.6f} | "
            f"{pair['left_luminance_std']:.5f} / "
            f"{pair['right_luminance_std']:.5f} | {pair['review_status']} |"
        )
    lines.extend(
        [
            "",
            "## Pendiente de aprobación",
            "",
            "- No se excluyó ninguna imagen y no se creó `exclusions-v1.json`.",
            "- Los pares binarios que pasan el corte son candidatos, no decisiones.",
            "- Los pares de bajo contraste y todos los pares de grupos 3+ requieren "
            "revisión visual.",
            "- Para un par confirmado se excluirían ambas copias; en grupos 3+ solo "
            "las imágenes confirmadas, nunca el grupo completo por defecto.",
        ]
    )
    (output / "audit-report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def audit(
    manifest_path: Path = DEFAULT_MANIFEST,
    output_directory: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    root = Path.cwd()
    manifest_path = manifest_path if manifest_path.is_absolute() else root / manifest_path
    output_directory = (
        output_directory
        if output_directory.is_absolute()
        else root / output_directory
    )
    raw_manifest = manifest_path.read_bytes()
    rows = json.loads(raw_manifest)
    if not isinstance(rows, list):
        raise ValueError("El manifiesto debe contener una lista de filas.")
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
    rows_by_path = {row["path"]: row for row in rows}
    groups = build_similarity_groups(records)
    mixed_groups = [
        group
        for group in groups
        if len({records_by_id[image_id].class_id for image_id in group.image_ids}) > 1
    ]
    output_directory.mkdir(parents=True, exist_ok=True)
    sheets_directory = output_directory / "contact-sheets"
    sheets_directory.mkdir(parents=True, exist_ok=True)
    images: dict[str, Image.Image] = {}
    statistics: dict[str, dict[str, Any]] = {}
    group_reports: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    for group in mixed_groups:
        members = sorted(
            (records_by_id[image_id] for image_id in group.image_ids),
            key=lambda record: record.path,
        )
        for member in members:
            if member.path not in images:
                images[member.path] = _load_comparison_image(root, member.path)
                statistics[member.path] = _image_statistics(images[member.path])
        pairs = _pair_rows_for_group(
            group.group_id, members, rows_by_path, images, statistics
        )
        pair_rows.extend(pairs)
        group_reports.append(
            {
                "group_id": group.group_id,
                "image_count": len(members),
                "classes": sorted({member.class_id for member in members}),
                "sources": sorted({member.source for member in members}),
                "splits": sorted(
                    {rows_by_path[member.path]["split"] for member in members}
                ),
                "minimum_hamming_distance": min(
                    pair["hamming_distance"] for pair in pairs
                ),
                "images": [
                    {
                        "path": member.path,
                        "class_id": member.class_id,
                        "source": member.source,
                        "split": rows_by_path[member.path]["split"],
                        "luminance_std": statistics[member.path]["luminance_std"],
                    }
                    for member in members
                ],
                "pairs": pairs,
            }
        )
        _write_contact_sheet(
            sheets_directory / f"{group.group_id}.png",
            group.group_id,
            members,
            rows_by_path,
            images,
            statistics,
            pairs,
        )

    for image in images.values():
        image.close()

    pair_rows.sort(
        key=lambda pair: (
            -pair["ssim"],
            pair["mae_rgb_normalized"],
            pair["group_id"],
            pair["left_path"],
            pair["right_path"],
        )
    )
    with (output_directory / "pair-metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as target:
        writer = csv.DictWriter(target, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    (output_directory / "groups.json").write_text(
        json.dumps(group_reports, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_directory,
        hashlib.sha256(raw_manifest).hexdigest(),
        rows,
        group_reports,
        pair_rows,
    )
    status_counts = Counter(pair["review_status"] for pair in pair_rows)
    return {
        "report": output_directory / "audit-report.md",
        "pairs_csv": output_directory / "pair-metrics.csv",
        "groups_json": output_directory / "groups.json",
        "contact_sheets": len(group_reports),
        "groups": len(group_reports),
        "images": sum(group["image_count"] for group in group_reports),
        "pairs": len(pair_rows),
        "status_counts": dict(status_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara imágenes de grupos multiclase sin evaluar el modelo."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(args.manifest, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
