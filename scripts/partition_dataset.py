from __future__ import annotations

import argparse
import json
from pathlib import Path

from cloud_classifier.partitioning import (
    PARTITION_SEED,
    PartitionError,
    assign_groups,
    build_inventory,
    build_report,
    build_similarity_groups,
    candidate_document,
)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Genera un informe y manifiesto candidato de particiones."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=repo_root / "data",
        help="Directorio de datos locales ignorado por Git.",
    )
    parser.add_argument(
        "--candidate-out",
        type=Path,
        default=repo_root / "data" / "partition-candidate.json",
        help="Ruta ignorada donde guardar el manifiesto candidato (no congelado).",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=repo_root / "data" / "partition-report.txt",
        help="Ruta ignorada donde guardar el informe para revisión humana.",
    )
    args = parser.parse_args()
    mapping_path = args.data_root / "author_ids.json"

    try:
        records, source_counts = build_inventory(
            repo_root.resolve(), args.data_root.resolve(), mapping_path.resolve()
        )
        groups = build_similarity_groups(records)
        assignments, groups_large = assign_groups(
            records, groups, seed=PARTITION_SEED
        )
        report = build_report(
            records, groups, assignments, groups_large, source_counts
        )
        document = candidate_document(
            records, groups, assignments, groups_large, source_counts
        )
    except (OSError, PartitionError, ValueError) as error:
        parser.exit(2, f"Error al particionar los datos: {error}\n")

    args.candidate_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = args.candidate_out.with_suffix(
        args.candidate_out.suffix + ".tmp"
    )
    temporary_report = args.report_out.with_suffix(args.report_out.suffix + ".tmp")
    temporary_manifest.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_report.write_text(report, encoding="utf-8")
    temporary_manifest.replace(args.candidate_out)
    temporary_report.replace(args.report_out)
    print(report, end="")
    print(
        f"\nCandidato local (no congelado): {args.candidate_out}\n"
        f"Informe local: {args.report_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
