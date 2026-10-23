from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PARTITION_PATH = Path("manifests/partitions-v1.json")
AUDIT_DIRECTORY = Path("data/audits/label-conflicts")
EXCLUSIONS_PATH = Path("manifests/exclusions-v1.json")
DOUBLE_LABEL_PATH = Path("manifests/double-label-v1.json")
PARTITION_SHA256 = "0ebcf4d3c72ce3344cb51a4024ff5e7333dc9fcfe7e6c5e6bd0c7d5c53c0eae6"
PRESERVED_GROUP_IDS = {
    "group_3279ca96570ee968",
    "group_a5a19a1698ce22b8",
}
EXCLUSION_REASON = "misma foto, etiquetas contradictorias"


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"No se pudo leer JSON válido desde {path}.") from error


def _pair_evidence(pair: dict[str, Any]) -> dict[str, Any]:
    return {
        "left_path": pair["left_path"],
        "right_path": pair["right_path"],
        "hamming_distance": pair["hamming_distance"],
        "ssim": pair["ssim"],
        "mae_rgb_normalized": pair["mae_rgb_normalized"],
        "left_luminance_std": pair["left_luminance_std"],
        "right_luminance_std": pair["right_luminance_std"],
    }


def build_manifests(
    partition_path: Path = PARTITION_PATH,
    audit_directory: Path = AUDIT_DIRECTORY,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_partition = partition_path.read_bytes()
    partition_sha256 = hashlib.sha256(raw_partition).hexdigest()
    if partition_sha256 != PARTITION_SHA256:
        raise ValueError("El SHA-256 no coincide con el manifiesto congelado esperado.")
    partition_rows = json.loads(raw_partition)
    if not isinstance(partition_rows, list):
        raise ValueError("El manifiesto de particiones debe ser una lista.")
    rows_by_path = {row["path"]: row for row in partition_rows}
    if len(rows_by_path) != len(partition_rows):
        raise ValueError("El manifiesto de particiones contiene rutas duplicadas.")

    groups = _load_json(audit_directory / "groups.json")
    if not isinstance(groups, list) or len(groups) != 130:
        raise ValueError("La auditoría no contiene los 130 grupos revisados.")
    groups_by_id = {group["group_id"]: group for group in groups}
    if len(groups_by_id) != 130 or not PRESERVED_GROUP_IDS.issubset(groups_by_id):
        raise ValueError("Los IDs de grupo no coinciden con la revisión visual aprobada.")
    duplicate_groups = [
        groups_by_id[group_id]
        for group_id in sorted(set(groups_by_id) - PRESERVED_GROUP_IDS)
    ]
    if len(duplicate_groups) != 128:
        raise ValueError("Se esperaban 128 grupos confirmados como duplicados.")

    exclusion_groups = []
    double_label_photos = []
    seen_paths: set[str] = set()
    for group in duplicate_groups:
        images = sorted(group["images"], key=lambda image: image["path"])
        classes = sorted({image["class_id"] for image in images})
        if len(classes) < 2:
            raise ValueError(
                f"El grupo confirmado {group['group_id']} no tiene varias clases."
            )
        for image in images:
            path = image["path"]
            row = rows_by_path.get(path)
            if (
                row is None
                or row["class_id"] != image["class_id"]
                or row["source"] != image["source"]
                or row["split"] != image["split"]
                or path in seen_paths
            ):
                raise ValueError(
                    f"La auditoría y el manifiesto no coinciden: {path}"
                )
            seen_paths.add(path)

        pairs = sorted(
            (_pair_evidence(pair) for pair in group["pairs"]),
            key=lambda pair: (pair["left_path"], pair["right_path"]),
        )
        if len(pairs) != len(images) * (len(images) - 1) // 2:
            raise ValueError(f"Falta evidencia de pares en {group['group_id']}.")
        exclusion_groups.append(
            {
                "group_id": group["group_id"],
                "reason": EXCLUSION_REASON,
                "images": [
                    {
                        "path": image["path"],
                        "class_id": image["class_id"],
                        "source": image["source"],
                        "split": image["split"],
                    }
                    for image in images
                ],
                "evidence": {
                    "minimum_hamming_distance": group["minimum_hamming_distance"],
                    "pairs": pairs,
                },
            }
        )
        double_label_photos.append(
            {
                "group_id": group["group_id"],
                "classes": classes,
                "representative_path": images[0]["path"],
                "copies": [
                    {
                        "path": image["path"],
                        "class_id": image["class_id"],
                        "source": image["source"],
                        "split": image["split"],
                    }
                    for image in images
                ],
            }
        )

    if len(seen_paths) != 271:
        raise ValueError(f"Se esperaban 271 rutas excluidas, se hallaron {len(seen_paths)}.")

    exclusions = {
        "exclusions_version": 1,
        "partition_sha256": partition_sha256,
        "groups": exclusion_groups,
    }
    double_labels = {
        "double_label_version": 1,
        "partition_sha256": partition_sha256,
        "photos": double_label_photos,
    }
    return exclusions, double_labels


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Congela los manifiestos aprobados de exclusión y doble etiqueta."
    )
    parser.add_argument("--partition-manifest", type=Path, default=PARTITION_PATH)
    parser.add_argument("--audit-directory", type=Path, default=AUDIT_DIRECTORY)
    parser.add_argument("--exclusions-output", type=Path, default=EXCLUSIONS_PATH)
    parser.add_argument("--double-label-output", type=Path, default=DOUBLE_LABEL_PATH)
    args = parser.parse_args()
    if args.exclusions_output.exists() or args.double_label_output.exists():
        raise FileExistsError("No se sobrescriben manifiestos versionados existentes.")
    exclusions, double_labels = build_manifests(
        args.partition_manifest, args.audit_directory
    )
    args.exclusions_output.parent.mkdir(parents=True, exist_ok=True)
    args.double_label_output.parent.mkdir(parents=True, exist_ok=True)
    _write_new_json(args.exclusions_output, exclusions)
    _write_new_json(args.double_label_output, double_labels)
    print(
        "Manifiestos congelados: "
        f"{len(exclusions['groups'])} grupos, "
        f"{sum(len(group['images']) for group in exclusions['groups'])} rutas excluidas, "
        f"{len(double_labels['photos'])} fotos únicas de doble etiqueta."
    )


if __name__ == "__main__":
    main()
