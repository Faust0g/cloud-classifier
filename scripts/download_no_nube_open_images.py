"""Download a bounded, human-verified Open Images subset for the no_nube class.

The selected split's image-level annotation CSV is streamed to find candidates;
only the bounded, filtered image subset is downloaded and retained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import io
import os
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


BASE_URL = "https://storage.googleapis.com/openimages"
IMAGE_BASE_URL = "https://open-images-dataset.s3.amazonaws.com"
CC_BY_2_LICENSE = "https://creativecommons.org/licenses/by/2.0/"
HUMAN_SOURCES = {"verification", "crowdsource-verification"}
EXCLUDED_LABEL_NAMES = (
    "Sky",
    "Cloud",
    "Atmosphere",
    "Weather",
    "Fog",
    "Haze",
    "Mist",
    "Storm",
    "Thunderstorm",
    "Rain",
    "Sunset",
    "Horizon",
    "Aircraft",
    "Airplane",
    "Airship",
    "Biplane",
    "Blimp",
    "Bird",
    "Cargo aircraft",
    "Drone",
    "Experimental aircraft",
    "Fighter aircraft",
    "Fixed-wing aircraft",
    "Floatplane",
    "Glider",
    "Ground attack aircraft",
    "Helicopter",
    "Hot air balloon",
    "Jet aircraft",
    "Light aircraft",
    "Military aircraft",
    "Military helicopter",
    "Military transport aircraft",
    "Monoplane",
    "Motor glider",
    "Narrow-body aircraft",
    "Powered hang glider",
    "Propeller-driven aircraft",
    "Radio-controlled aircraft",
    "Radio-controlled helicopter",
    "Rigid airship",
    "Rocket-powered aircraft",
    "Seaplane",
    "Spaceplane",
    "Supersonic aircraft",
    "Toy airplane",
    "Trijet",
    "Triplane",
    "Twinjet",
    "Wide-body aircraft",
)
DEFAULT_SPLIT = "validation"
SPLITS = {
    "train": (
        "v7/oidv7-train-annotations-human-imagelabels.csv",
        "v6/oidv6-train-images-with-labels-with-rotation.csv",
    ),
    "validation": (
        "v7/oidv7-val-annotations-human-imagelabels.csv",
        "2018_04/validation/validation-images-with-rotation.csv",
    ),
    "test": (
        "v7/oidv7-test-annotations-human-imagelabels.csv",
        "2018_04/test/test-images-with-rotation.csv",
    ),
}
CLASS_DESCRIPTIONS_URL = f"{BASE_URL}/v7/oidv7-class-descriptions.csv"
DATASET_CITATION = (
    "Kuznetsova, A., et al. (2020). The Open Images Dataset V4: Unified image "
    "classification, object detection, and visual relationship detection at scale. "
    "International Journal of Computer Vision. "
    "https://doi.org/10.1007/s11263-020-01316-z"
)
DATASET_PAGE = "https://storage.googleapis.com/openimages/web/index.html"


def _open_csv(url: str) -> io.TextIOWrapper:
    request = Request(url, headers={"User-Agent": "cloud-classifier-no-nube/1.0"})
    try:
        response = urlopen(request, timeout=60)
    except HTTPError as error:
        raise RuntimeError(
            f"Error HTTP {error.code} al descargar el CSV {url}: {error.reason}"
        ) from error
    return io.TextIOWrapper(response, encoding="utf-8-sig", newline="")


def _require_columns(reader: csv.DictReader, columns: set[str], url: str) -> None:
    actual = set(reader.fieldnames or ())
    missing = columns - actual
    if missing:
        raise ValueError(f"CSV oficial sin columnas {sorted(missing)}: {url}")


def _resolve_excluded_label_ids(
    excluded_names: tuple[str, ...], class_names: dict[str, str]
) -> set[str]:
    labels_by_name: dict[str, list[str]] = {}
    for label_id, display_name in class_names.items():
        labels_by_name.setdefault(display_name, []).append(label_id)

    unresolved = [
        name
        for name in excluded_names
        if len(labels_by_name.get(name, [])) != 1
    ]
    if unresolved:
        raise ValueError(
            "No se pudieron resolver de forma única estas etiquetas de exclusión "
            f"en Open Images V7: {', '.join(unresolved)}"
        )
    return {labels_by_name[name][0] for name in excluded_names}


def _load_class_names() -> dict[str, str]:
    with closing(_open_csv(CLASS_DESCRIPTIONS_URL)) as source:
        reader = csv.DictReader(source)
        _require_columns(reader, {"LabelName", "DisplayName"}, CLASS_DESCRIPTIONS_URL)
        names = {row["LabelName"]: row["DisplayName"] for row in reader}

    _resolve_excluded_label_ids(EXCLUDED_LABEL_NAMES, names)
    return names


def _merge_annotation_batch(
    database: sqlite3.Connection,
    batch: list[tuple[str, int, int, str | None]],
) -> None:
    database.executemany(
        """
        INSERT INTO image_labels (image_id, has_excluded_label, has_other_label, other_label)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(image_id) DO UPDATE SET
            has_excluded_label = MAX(image_labels.has_excluded_label, excluded.has_excluded_label),
            has_other_label = MAX(image_labels.has_other_label, excluded.has_other_label),
            other_label = COALESCE(image_labels.other_label, excluded.other_label)
        """,
        batch,
    )


def _collect_candidates(
    annotation_csv: io.TextIOBase,
    class_names: dict[str, str],
    limit: int,
    seed: int,
) -> list[tuple[str, str]]:
    reader = csv.DictReader(annotation_csv)
    _require_columns(
        reader, {"ImageID", "Source", "LabelName", "Confidence"}, "anotaciones"
    )
    excluded_label_ids = _resolve_excluded_label_ids(EXCLUDED_LABEL_NAMES, class_names)

    with tempfile.TemporaryDirectory() as temporary_directory:
        database_path = Path(temporary_directory) / "annotations.sqlite3"
        with closing(sqlite3.connect(database_path)) as database:
            database.execute(
                """
                CREATE TABLE image_labels (
                    image_id TEXT PRIMARY KEY,
                    has_excluded_label INTEGER NOT NULL DEFAULT 0,
                    has_other_label INTEGER NOT NULL DEFAULT 0,
                    other_label TEXT
                )
                """
            )
            batch: list[tuple[str, int, int, str | None]] = []
            for row in reader:
                if row["Source"] not in HUMAN_SOURCES or row["Confidence"] != "1":
                    continue
                label_id = row["LabelName"]
                if label_id not in class_names:
                    raise ValueError(f"Etiqueta desconocida en anotaciones: {label_id}")

                excluded = int(label_id in excluded_label_ids)
                batch.append(
                    (
                        row["ImageID"],
                        excluded,
                        int(not excluded),
                        None if excluded else label_id,
                    )
                )
                if len(batch) >= 10_000:
                    _merge_annotation_batch(database, batch)
                    batch.clear()
            if batch:
                _merge_annotation_batch(database, batch)
            database.commit()

            # Stable hash ranking gives a reproducible sample without favoring CSV order.
            selection: list[tuple[int, str, str]] = []
            query = database.execute(
                """
                SELECT image_id, other_label
                FROM image_labels
                WHERE has_other_label = 1 AND has_excluded_label = 0
                """
            )
            for image_id, label_id in query:
                digest = hashlib.sha256(f"{seed}:{image_id}".encode()).digest()
                rank = int.from_bytes(digest[:8], "big")
                item = (-rank, image_id, label_id)
                if len(selection) < limit:
                    heapq.heappush(selection, item)
                elif item > selection[0]:
                    heapq.heapreplace(selection, item)

            selection.sort(key=lambda item: (-item[0], item[1]))
            return [(image_id, label_id) for _, image_id, label_id in selection]


def _load_image_metadata(
    metadata_csv: io.TextIOBase, requested_ids: set[str], url: str
) -> dict[str, dict[str, str]]:
    reader = csv.DictReader(metadata_csv)
    required = {
        "ImageID",
        "OriginalURL",
        "OriginalLandingURL",
        "License",
        "Author",
        "Title",
    }
    _require_columns(reader, required, url)
    result: dict[str, dict[str, str]] = {}
    for row in reader:
        image_id = row["ImageID"]
        if (
            image_id in requested_ids
            and row["License"].rstrip("/") + "/" == CC_BY_2_LICENSE
            and row["Author"].strip()
            and row["OriginalLandingURL"].strip()
        ):
            result[image_id] = row
            if len(result) == len(requested_ids):
                break
    return result


def _download_image(image_id: str, split: str, destination: Path) -> None:
    url = f"{IMAGE_BASE_URL}/{split}/{image_id}.jpg"
    temporary_path = destination.with_suffix(".jpg.part")
    request = Request(url, headers={"User-Agent": "cloud-classifier-no-nube/1.0"})
    try:
        with urlopen(request, timeout=60) as response, temporary_path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        with temporary_path.open("rb") as image:
            if image.read(3) != b"\xff\xd8\xff":
                raise ValueError(f"El archivo descargado no parece JPEG: {image_id}")
        os.replace(temporary_path, destination)
    except HTTPError as error:
        raise RuntimeError(
            f"Error HTTP {error.code} al descargar la imagen {image_id} desde {url}: "
            f"{error.reason}"
        ) from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_text_atomically(path: Path, contents: str) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(contents, encoding="utf-8", newline="")
    os.replace(temporary_path, path)


def _write_attributions(
    path: Path,
    selected: list[tuple[str, str]],
    metadata: dict[str, dict[str, str]],
    class_names: dict[str, str],
) -> None:
    temporary_path = path.with_suffix(".csv.tmp")
    fields = [
        "ImageID",
        "Author",
        "Title",
        "OriginalLandingURL",
        "OriginalURL",
        "License",
        "VerifiedHumanLabel",
        "VerifiedHumanLabelName",
        "Changes",
    ]
    with temporary_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for image_id, label_id in selected:
            if image_id not in metadata:
                continue
            row = metadata[image_id]
            writer.writerow(
                {
                    "ImageID": image_id,
                    "Author": row["Author"],
                    "Title": row["Title"],
                    "OriginalLandingURL": row["OriginalLandingURL"],
                    "OriginalURL": row["OriginalURL"],
                    "License": row["License"],
                    "VerifiedHumanLabel": label_id,
                    "VerifiedHumanLabelName": class_names[label_id],
                    "Changes": "No changes; original image",
                }
            )
    os.replace(temporary_path, path)


def download_subset(
    split: str, limit: int, seed: int, output_dir: Path, metadata_dir: Path
) -> tuple[int, int]:
    annotation_path, metadata_path = SPLITS[split]
    annotation_url = f"{BASE_URL}/{annotation_path}"
    metadata_url = f"{BASE_URL}/{metadata_path}"
    class_names = _load_class_names()

    print(
        f"Analizando etiquetas humanas de Open Images V7 ({split}); "
        "no se descarga el conjunto completo de imágenes."
    )
    with closing(_open_csv(annotation_url)) as annotation_csv:
        candidates = _collect_candidates(annotation_csv, class_names, limit, seed)
    if not candidates:
        raise RuntimeError("No se encontraron imágenes que cumplan los filtros.")

    requested_ids = {image_id for image_id, _ in candidates}
    with closing(_open_csv(metadata_url)) as metadata_csv:
        metadata = _load_image_metadata(metadata_csv, requested_ids, metadata_url)
    selected = [(image_id, label) for image_id, label in candidates if image_id in metadata]
    if not selected:
        raise RuntimeError(
            "Ninguna imagen seleccionada tiene metadatos CC BY 2.0; "
            "no se descargaron imágenes."
        )

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for index, (image_id, _) in enumerate(selected, start=1):
        destination = images_dir / f"{image_id}.jpg"
        if not destination.exists():
            _download_image(image_id, split, destination)
        print(f"\rDescargando imágenes seleccionadas: {index}/{len(selected)}", end="")
    print()

    metadata_dir.mkdir(parents=True, exist_ok=True)
    _write_attributions(metadata_dir / "ATTRIBUTION.csv", selected, metadata, class_names)
    citation = (
        f"Cita del dataset:\n{DATASET_CITATION}\n\n"
        f"Versión y página oficial de Open Images: {DATASET_PAGE}\n"
        f"Anotaciones de imágenes: {annotation_url}\n\n"
        "Las imágenes de ATTRIBUTION.csv están identificadas como CC BY 2.0 "
        f"({CC_BY_2_LICENSE}). Consulte la atribución individual por autor, título "
        "y enlace a la página original. Las imágenes descargadas no se modifican; "
        "si se realizan cambios posteriormente, actualice el campo Changes.\n"
    )
    _write_text_atomically(metadata_dir / "CITATION.txt", citation)
    return len(selected), len(candidates) - len(selected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Descarga un subconjunto acotado de imágenes de Open Images V7 para la clase no_nube."
        ),
        epilog=(
            "El filtrado recorre en streaming los CSV completos de anotaciones y metadatos "
            "del split; solo descarga las imágenes seleccionadas."
        ),
    )
    parser.add_argument(
        "--split",
        choices=SPLITS,
        default=DEFAULT_SPLIT,
        help=f"Split de Open Images (default: {DEFAULT_SPLIT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Máximo de imágenes a descargar (default: 1000)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla de selección reproducible")
    parser.add_argument("--output", type=Path, help="Directorio raíz destino de las imágenes")
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="Directorio versionable para ATTRIBUTION.csv y CITATION.txt",
    )
    arguments = parser.parse_args(argv)
    if arguments.limit <= 0:
        parser.error("--limit debe ser mayor que cero")

    output_dir = arguments.output or (
        Path("data") / "no_nube" / "open-images-v7" / arguments.split
    )
    metadata_dir = arguments.metadata_output or (
        Path("docs") / "datasets" / "no_nube" / "open-images-v7" / arguments.split
    )
    try:
        downloaded, skipped = download_subset(
            arguments.split, arguments.limit, arguments.seed, output_dir, metadata_dir
        )
    except (OSError, ValueError, RuntimeError, HTTPError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(
        f"Descargadas {downloaded} imágenes para no_nube; "
        f"{skipped} candidatas omitidas por licencia o metadatos incompletos."
    )
    print(f"Metadatos de atribución y cita: {metadata_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
