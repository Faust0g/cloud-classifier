from __future__ import annotations

import argparse
import json
from pathlib import Path

from cloud_classifier.partitioning import (
    ImageRecord,
    PartitionError,
    versioned_manifest_rows,
)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Congela el manifiesto de particiones aprobado."
    )
    parser.add_argument(
        "--approved",
        action="store_true",
        help="Confirma la aprobación humana explícita del informe candidato.",
    )
    args = parser.parse_args()
    if not args.approved:
        parser.error("Se requiere --approved tras la aprobación humana del informe.")

    candidate_path = repo_root / "data" / "partition-candidate.json"
    output_path = repo_root / "manifests" / "partitions-v1.json"
    if output_path.exists():
        parser.error(
            f"El manifiesto ya existe y no se sobrescribe: {output_path}"
        )

    try:
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        if not isinstance(candidate, dict) or candidate.get(
            "status"
        ) != "candidate_requires_human_approval":
            raise PartitionError("El archivo no es un manifiesto candidato aprobado.")
        candidate_images = candidate["images"]
        if not isinstance(candidate_images, list):
            raise PartitionError("La lista de imágenes candidatas no es válida.")
        records: list[ImageRecord] = []
        assignments: dict[str, str] = {}
        for row in candidate_images:
            if not isinstance(row, dict):
                raise PartitionError("Una fila del candidato no es un objeto.")
            image_id = row.get("image_id")
            values = {
                key: row.get(key)
                for key in ("path", "class_id", "source", "author_id", "phash64")
            }
            split = row.get("split")
            if not isinstance(image_id, str) or not all(
                isinstance(value, str) for value in values.values()
            ):
                raise PartitionError("Una fila del candidato tiene campos inválidos.")
            if not isinstance(split, str) or image_id in assignments:
                raise PartitionError("ID duplicado o partición inválida en candidato.")
            records.append(ImageRecord(image_id=image_id, **values))
            assignments[image_id] = split
        rows = versioned_manifest_rows(records, assignments)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Error al congelar el manifiesto: {error}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    print(f"Manifiesto congelado: {output_path}")
    print(f"Filas: {len(rows)}; campos por fila: {len(rows[0]) if rows else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
