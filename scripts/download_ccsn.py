"""Download and prepare the CC0 CCSN dataset."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

CCSN_FILE_URL = "https://dataverse.harvard.edu/api/access/datafile/3569338"
CCSN_FILE_MD5 = "0787f50947a3f65630a7463206316fd3"
EXPECTED_IMAGE_COUNT = 2543
CLASS_MAPPING = {
    "Ac": "Ac",
    "As": "As",
    "Cb": "Cb",
    "Cc": "Cc",
    "Ci": "Ci",
    "Cs": "Cs",
    "Ct": "no_nube",
    "Cu": "Cu",
    "Ns": "Ns",
    "Sc": "Sc",
    "St": "St",
}


def _download_archive(destination: Path) -> None:
    request = Request(
        CCSN_FILE_URL,
        headers={"User-Agent": "cloud-classifier-ccsn/1.0"},
    )
    digest = hashlib.md5(usedforsecurity=False)
    try:
        with urlopen(request, timeout=120) as response, destination.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
    except HTTPError as error:
        raise RuntimeError(
            f"Error HTTP {error.code} al descargar CCSN: {error.reason}"
        ) from error

    if digest.hexdigest() != CCSN_FILE_MD5:
        raise ValueError(
            "El checksum MD5 del archivo CCSN no coincide con el publicado por Dataverse."
        )


def _extract_images(archive_path: Path, output_dir: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with zipfile.ZipFile(archive_path) as archive:
        image_entries: list[tuple[zipfile.ZipInfo, str]] = []
        for entry in archive.infolist():
            source_path = PurePosixPath(entry.filename)
            if entry.is_dir() or source_path.parts[:1] != ("CCSN_v2",):
                continue
            if (
                len(source_path.parts) != 3
                or source_path.name.startswith("._")
                or source_path.suffix.lower() not in {".jpg", ".jpeg"}
            ):
                continue

            source_class = source_path.parts[1]
            if source_class not in CLASS_MAPPING:
                continue
            image_entries.append((entry, source_class))
            counts[source_class] += 1

        if sum(counts.values()) != EXPECTED_IMAGE_COUNT:
            raise ValueError(
                "El archivo CCSN no contiene la cantidad esperada de imágenes: "
                f"{sum(counts.values())} en lugar de {EXPECTED_IMAGE_COUNT}."
            )
        if set(counts) != set(CLASS_MAPPING):
            raise ValueError(
                "El archivo CCSN no contiene todas las clases esperadas: "
                f"{', '.join(sorted(set(CLASS_MAPPING) - set(counts)))}."
            )

        for entry, source_class in image_entries:
            source_path = PurePosixPath(entry.filename)
            target_class = CLASS_MAPPING[source_class]
            destination = output_dir / target_class / source_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = destination.with_suffix(destination.suffix + ".part")
            try:
                with archive.open(entry) as source, temporary_path.open("wb") as output:
                    shutil.copyfileobj(source, output)
                with Image.open(temporary_path) as image:
                    if image.format != "JPEG":
                        raise ValueError(
                            f"La imagen {entry.filename} no es un JPEG válido."
                        )
                    image.verify()
                os.replace(temporary_path, destination)
            finally:
                temporary_path.unlink(missing_ok=True)
    return counts


def prepare_dataset(output_dir: Path) -> Counter[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cloud-classifier-ccsn-") as temporary:
        temporary_path = Path(temporary)
        archive_path = temporary_path / "CCSN.zip"
        _download_archive(archive_path)
        return _extract_images(archive_path, output_dir)


def main() -> int:
    output_dir = Path("data") / "ccsn" / "images"
    try:
        counts = prepare_dataset(output_dir)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"CCSN preparado en {output_dir.resolve()}")
    for source_class, count in sorted(counts.items()):
        target_class = CLASS_MAPPING[source_class]
        print(f"{source_class} -> {target_class}: {count} imágenes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
