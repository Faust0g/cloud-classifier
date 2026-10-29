from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from cloud_classifier.calibration import (
    CALIBRATION_MANIFEST,
    DEFAULT_MODEL,
    EXCLUSION_MANIFEST,
    PARTITION_MANIFEST,
    _evaluate_rows,
    _load_calibration,
    acquire_final_evaluation_lock,
    evaluate_double_label_photos,
    format_final_report,
    load_model,
    predict_rows,
    read_double_label_photos,
    read_own_photo_rows,
    read_split_rows,
)
from cloud_classifier.preprocessing import load_model_contract
from cloud_classifier.training import class_order_from_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta una sola vez la evaluación final del modelo."
    )
    parser.add_argument("--confirm-final", action="store_true")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--own-manifest", type=Path, required=True)
    parser.add_argument("--partition", type=Path, default=PARTITION_MANIFEST)
    parser.add_argument("--exclusions", type=Path, default=EXCLUSION_MANIFEST)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_MANIFEST)
    parser.add_argument("--lock", type=Path, default=Path("runs/evaluate-final.lock"))
    parser.add_argument(
        "--report", type=Path, default=Path("runs/evaluate-final-report.md")
    )
    return parser.parse_args()


def _from_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size debe ser al menos 1.")
    root = args.root.resolve()
    calibration_path = _from_root(root, args.calibration)
    lock_path = _from_root(root, args.lock)
    acquire_final_evaluation_lock(lock_path, calibration_path, args.confirm_final)

    partition_path = _from_root(root, args.partition)
    exclusion_path = _from_root(root, args.exclusions)
    model_path = _from_root(root, args.model)
    contract = load_model_contract()
    class_ids = class_order_from_contract(contract)
    test_rows, partition_hash, exclusion_hash = read_split_rows(
        partition_path, exclusion_path, "test", class_ids
    )
    if not model_path.is_file():
        raise FileNotFoundError(f"No existe el checkpoint indicado: {model_path}")
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    calibration = _load_calibration(
        calibration_path, partition_hash, exclusion_hash, model_hash
    )
    own_rows = read_own_photo_rows(args.own_manifest, root, class_ids)
    model, torch, device = load_model(model_path, len(class_ids), args.device)

    test_logits = predict_rows(
        test_rows, model, torch, device, root, class_ids, args.batch_size
    )
    own_logits = predict_rows(
        own_rows, model, torch, device, root, class_ids, args.batch_size
    )
    test_results = _evaluate_rows(test_rows, test_logits, class_ids, calibration)
    own_results = _evaluate_rows(own_rows, own_logits, class_ids, calibration)

    double_label_photos = read_double_label_photos(
        _from_root(root, Path("manifests/double-label-v1.json")), partition_hash
    )
    double_label_results = evaluate_double_label_photos(
        double_label_photos,
        model,
        torch,
        device,
        root,
        class_ids,
        calibration,
        args.batch_size,
    )
    report = format_final_report(
        test_results, own_results, double_label_results
    )
    report_path = _from_root(root, args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(report)
    print(f"Informe final escrito en {report_path}")


if __name__ == "__main__":
    main()
