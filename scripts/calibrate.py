from __future__ import annotations

import argparse
import json
from pathlib import Path

from cloud_classifier.calibration import (
    CALIBRATION_MANIFEST,
    DEFAULT_MODEL,
    EXCLUSION_MANIFEST,
    PARTITION_MANIFEST,
    TARGET_PRECISION,
    calibrate_dataset,
    calibration_artifact,
    calibration_report,
    write_json_exclusive,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ajusta temperatura y umbrales con la partición de calibración."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-precision",
        type=float,
        default=TARGET_PRECISION,
        help="Precisión mínima para resultados firmes y pares (default: 0.85).",
    )
    parser.add_argument("--partition", type=Path, default=PARTITION_MANIFEST)
    parser.add_argument("--exclusions", type=Path, default=EXCLUSION_MANIFEST)
    parser.add_argument("--output", type=Path, default=CALIBRATION_MANIFEST)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size debe ser al menos 1.")
    if args.seed != 42:
        raise ValueError("La calibración congelada requiere la semilla 42.")
    if not 0 <= args.target_precision <= 1:
        raise ValueError("--target-precision debe estar entre 0 y 1.")
    root = args.root.resolve()
    model_path = args.model if args.model.is_absolute() else root / args.model
    partition_path = (
        args.partition if args.partition.is_absolute() else root / args.partition
    )
    exclusion_path = (
        args.exclusions if args.exclusions.is_absolute() else root / args.exclusions
    )
    output_path = args.output if args.output.is_absolute() else root / args.output
    if args.report is None:
        report_name = (
            "calibration-report.md"
            if output_path.name == CALIBRATION_MANIFEST.name
            else f"{output_path.stem}-report.md"
        )
        report_path = root / "runs" / report_name
    else:
        report_path = args.report if args.report.is_absolute() else root / args.report
    if output_path.resolve() == report_path.resolve():
        raise ValueError("El artefacto y el informe requieren rutas distintas.")
    if output_path.exists() or report_path.exists():
        raise FileExistsError(
            "La calibración ya tiene una salida; no se sobrescribe ningún artefacto."
        )
    results, partition_hash, exclusion_hash, model_hash, count = calibrate_dataset(
        root,
        partition_path,
        exclusion_path,
        model_path,
        args.device,
        args.batch_size,
        args.seed,
        args.target_precision,
    )
    artifact = calibration_artifact(
        results, partition_hash, exclusion_hash, model_hash, args.seed
    )
    write_json_exclusive(output_path, artifact)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8", newline="\n") as report:
        report.write(calibration_report(results, count))
    print(json.dumps(artifact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
