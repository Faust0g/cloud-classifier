from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import numpy as np

from cloud_classifier.prediction import classify_logits
from cloud_classifier.preprocessing import load_model_contract, preprocess_image
from cloud_classifier.training import TrainingDataError, class_order_from_contract


PARTITION_MANIFEST = Path("manifests/partitions-v1.json")
EXCLUSION_MANIFEST = Path("manifests/exclusions-v1.json")
CALIBRATION_MANIFEST = Path("manifests/calibration-v1.json")
DEFAULT_MODEL = Path("runs/20261029T220258Z-seed-42/best.pt")
TARGET_PRECISION = 0.85
PAIR_TOLERANCE = 0.08
MIN_PAIR_BIN_COUNT = 30


def _read_json_bytes(path: Path, description: str) -> tuple[Any, str]:
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingDataError(f"No se pudo leer {description}: {path}") from error
    return document, hashlib.sha256(raw).hexdigest()


def read_split_rows(
    partition_path: Path,
    exclusion_path: Path,
    split: str,
    class_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], str, str]:
    if split not in {"calibration", "test"}:
        raise ValueError("Solo se admiten las particiones calibration y test.")
    document, partition_hash = _read_json_bytes(
        partition_path, "el manifiesto de particiones"
    )
    if not isinstance(document, list):
        raise TrainingDataError("El manifiesto de particiones debe ser una lista.")
    exclusion_document, exclusion_hash = _read_json_bytes(
        exclusion_path, "el manifiesto de exclusiones"
    )
    if (
        not isinstance(exclusion_document, dict)
        or exclusion_document.get("exclusions_version") != 1
        or exclusion_document.get("partition_sha256") != partition_hash
        or not isinstance(exclusion_document.get("groups"), list)
    ):
        raise TrainingDataError(
            "El manifiesto de exclusiones no corresponde a las particiones."
        )

    excluded: dict[str, dict[str, str]] = {}
    for group in exclusion_document["groups"]:
        if not isinstance(group, dict) or not isinstance(group.get("images"), list):
            raise TrainingDataError("El manifiesto de exclusiones contiene un grupo inválido.")
        for image in group["images"]:
            if not isinstance(image, dict) or image.get("split") != split:
                continue
            path = image.get("path")
            class_id = image.get("class_id")
            source = image.get("source")
            if (
                not isinstance(path, str)
                or not path
                or not isinstance(class_id, str)
                or class_id not in class_ids
                or not isinstance(source, str)
                or not source
                or path in excluded
            ):
                raise TrainingDataError("Una exclusión de la partición es inválida.")
            excluded[path] = {"class_id": class_id, "source": source}

    selected_rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for row in document:
        if not isinstance(row, dict):
            raise TrainingDataError("El manifiesto contiene una fila inválida.")
        if row.get("split") != split:
            continue
        path = row.get("path")
        class_id = row.get("class_id")
        source = row.get("source")
        posix_path = PurePosixPath(path) if isinstance(path, str) else None
        if (
            posix_path is None
            or not path
            or posix_path.is_absolute()
            or ".." in posix_path.parts
            or "\\" in path
            or ":" in path
            or not isinstance(class_id, str)
            or class_id not in class_ids
            or not isinstance(source, str)
            or not source
            or path in seen_paths
        ):
            raise TrainingDataError(f"Fila {split} inválida.")
        seen_paths.add(path)
        exclusion = excluded.get(path)
        if exclusion is not None:
            if exclusion != {"class_id": class_id, "source": source}:
                raise TrainingDataError(f"La exclusión no coincide con la fila: {path}")
            continue
        selected_rows.append(row)

    if not selected_rows:
        raise TrainingDataError(f"No hay filas {split} después de exclusiones.")
    if set(excluded) - seen_paths:
        raise TrainingDataError("Hay exclusiones que no coinciden con la partición.")
    return selected_rows, partition_hash, exclusion_hash


def _negative_log_likelihood(
    logits: np.ndarray, labels: np.ndarray, temperature: float
) -> float:
    scaled = logits / temperature
    maxima = np.max(scaled, axis=1)
    log_sum_exp = maxima + np.log(np.exp(scaled - maxima[:, None]).sum(axis=1))
    return float(np.mean(log_sum_exp - scaled[np.arange(len(labels)), labels]))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if (
        values.ndim != 2
        or targets.ndim != 1
        or len(values) != len(targets)
        or values.shape[1] < 2
        or len(targets) == 0
        or np.any(targets < 0)
        or np.any(targets >= values.shape[1])
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("Logits y etiquetas tienen dimensiones incompatibles.")

    def objective(log_temperature: float) -> float:
        return _negative_log_likelihood(
            values, targets, math.exp(log_temperature)
        )

    lower, upper = -20.0, 20.0
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    left_value = objective(left)
    right_value = objective(right)
    for _ in range(120):
        if left_value <= right_value:
            upper, right, right_value = right, left, left_value
            left = upper - ratio * (upper - lower)
            left_value = objective(left)
        else:
            lower, left, left_value = left, right, right_value
            right = lower + ratio * (upper - lower)
            right_value = objective(right)

    candidates = (lower, upper, left, right, 0.0)
    best_log_temperature = min(candidates, key=objective)
    temperature = math.exp(best_log_temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("El ajuste produjo una temperatura no válida.")
    return temperature


def _softmax_rows(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / np.sum(values, axis=1, keepdims=True)


def _smallest_threshold(
    scores: np.ndarray,
    correct: np.ndarray,
    target_precision: float = TARGET_PRECISION,
) -> float | None:
    if not math.isfinite(target_precision) or not 0 <= target_precision <= 1:
        raise ValueError("El objetivo de precisión debe estar entre 0 y 1.")
    for threshold in np.unique(scores):
        eligible = scores >= threshold
        if int(np.sum(correct[eligible])) / int(np.sum(eligible)) >= target_precision:
            return float(threshold)
    return None


def _pair_bins(
    top2_probability: np.ndarray,
    correct_top2: np.ndarray,
    pair_threshold: float | None,
    eligible: np.ndarray,
) -> tuple[list[dict[str, Any]], bool, bool]:
    if pair_threshold is None:
        return [], False, True
    eligible_indices = np.flatnonzero(
        eligible & (top2_probability >= pair_threshold)
    )
    if len(eligible_indices) == 0:
        return [], False, True

    lower = pair_threshold
    initial = [
        {
            "lower": lower + (1 - lower) * index / 3,
            "upper": lower + (1 - lower) * (index + 1) / 3,
            "indices": [],
            "members": [index + 1],
        }
        for index in range(3)
    ]
    for sample_index in eligible_indices:
        value = top2_probability[sample_index]
        bucket_index = min(2, int((value - lower) / (1 - lower) * 3)) if lower < 1 else 0
        initial[bucket_index]["indices"].append(int(sample_index))
    bins = initial

    while len(bins) > 1:
        undersized = [
            (len(bucket["indices"]), index)
            for index, bucket in enumerate(bins)
            if len(bucket["indices"]) < MIN_PAIR_BIN_COUNT
        ]
        if not undersized:
            break
        _, index = min(undersized, key=lambda item: (item[0], item[1]))
        if index == 0:
            neighbor = 1
        elif index == len(bins) - 1:
            neighbor = index - 1
        else:
            left_size = len(bins[index - 1]["indices"])
            right_size = len(bins[index + 1]["indices"])
            neighbor = index - 1 if left_size >= right_size else index + 1
        first = min(index, neighbor)
        second = max(index, neighbor)
        bins[first] = {
            "lower": bins[first]["lower"],
            "upper": bins[second]["upper"],
            "indices": bins[first]["indices"] + bins[second]["indices"],
            "members": bins[first]["members"] + bins[second]["members"],
        }
        del bins[second]

    inconclusive = len(bins) == 1 and len(bins[0]["indices"]) < MIN_PAIR_BIN_COUNT
    output: list[dict[str, Any]] = []
    all_calibrated = not inconclusive and bool(bins)
    for bucket in bins:
        indices = bucket["indices"]
        count = len(indices)
        mean_confidence = (
            float(np.mean(top2_probability[indices])) if count else None
        )
        observed_accuracy = (
            float(np.mean(correct_top2[indices])) if count else None
        )
        difference = (
            abs(mean_confidence - observed_accuracy)
            if mean_confidence is not None and observed_accuracy is not None
            else None
        )
        calibrated = difference is not None and difference <= PAIR_TOLERANCE
        all_calibrated = all_calibrated and calibrated
        output.append(
            {
                "range": [float(bucket["lower"]), float(bucket["upper"])],
                "initial_bins": bucket["members"],
                "count": count,
                "mean_top2_probability": mean_confidence,
                "observed_top2_accuracy": observed_accuracy,
                "absolute_difference": difference,
                "calibrated": calibrated,
            }
        )
    return output, all_calibrated, inconclusive


def calibrate_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    class_ids: Sequence[str],
    target_precision: float = TARGET_PRECISION,
) -> dict[str, Any]:
    values = np.asarray(logits, dtype=np.float64)
    actual = np.asarray(labels, dtype=np.int64)
    if (
        values.ndim != 2
        or len(values) != len(actual)
        or values.shape[1] < 2
        or values.shape[1] != len(class_ids)
        or len(actual) == 0
        or np.any(actual < 0)
        or np.any(actual >= values.shape[1])
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("Logits o etiquetas de calibración inválidos.")
    if not math.isfinite(target_precision) or not 0 <= target_precision <= 1:
        raise ValueError("El objetivo de precisión debe estar entre 0 y 1.")

    temperature = fit_temperature(values, actual)
    probabilities = _softmax_rows(values, temperature)
    order = np.argsort(-probabilities, axis=1, kind="stable")[:, :2]
    top1_confidence = probabilities[np.arange(len(actual)), order[:, 0]]
    top2_probability = probabilities[np.arange(len(actual))[:, None], order].sum(axis=1)
    top1_correct = order[:, 0] == actual
    top2_correct = np.any(order == actual[:, None], axis=1)

    if "no_nube" not in class_ids:
        raise ValueError("El contrato debe incluir la clase no_nube.")
    no_cloud_index = class_ids.index("no_nube")
    firm_candidates = order[:, 0] != no_cloud_index
    firm_threshold = _smallest_threshold(
        top1_confidence[firm_candidates],
        top1_correct[firm_candidates],
        target_precision,
    ) if np.any(firm_candidates) else None
    is_firm = (
        firm_candidates
        & (top1_confidence >= firm_threshold)
        if firm_threshold is not None
        else np.zeros(len(actual), dtype=bool)
    )
    pair_eligible = ~is_firm & (order[:, 0] != no_cloud_index)
    pair_threshold = _smallest_threshold(
        top2_probability[pair_eligible],
        top2_correct[pair_eligible],
        target_precision,
    ) if np.any(pair_eligible) else None

    pair_bins, pair_enabled, pair_inconclusive = _pair_bins(
        top2_probability,
        top2_correct,
        pair_threshold,
        pair_eligible,
    )
    decisions = [
        classify_logits(
            values[index],
            temperature,
            firm_threshold,
            pair_threshold,
            class_ids,
        ).decision
        for index in range(len(actual))
    ]
    counts = Counter(decisions)
    nll_before = _negative_log_likelihood(values, actual, 1.0)
    nll_after = _negative_log_likelihood(values, actual, temperature)
    confidence_bins = _confidence_reliability_bins(
        top1_confidence, top1_correct
    )
    return {
        "temperature": temperature,
        "target_precision": target_precision,
        "firm_threshold": firm_threshold,
        "pair_threshold": pair_threshold,
        "percentage_pair_enabled": pair_enabled,
        "pair_bins": pair_bins,
        "pair_calibration_inconclusive": pair_inconclusive,
        "nll_before": nll_before,
        "nll_after": nll_after,
        "confidence_bins": confidence_bins,
        "decision_counts": {
            decision: counts.get(decision, 0)
            for decision in ("firme", "par", "repetir", "no_nube")
        },
        "top1_confidence": top1_confidence,
        "top2_probability": top2_probability,
        "top1_correct": top1_correct,
        "top2_correct": top2_correct,
    }


def _confidence_reliability_bins(
    confidence: np.ndarray, correct: np.ndarray
) -> list[dict[str, Any]]:
    output = []
    for index in range(10):
        lower = index / 10
        upper = (index + 1) / 10
        selected = (confidence >= lower) & (
            confidence <= upper if index == 9 else confidence < upper
        )
        count = int(np.sum(selected))
        if count == 0:
            continue
        mean_confidence = float(np.mean(confidence[selected]))
        accuracy = float(np.mean(correct[selected]))
        output.append(
            {
                "range": [lower, upper],
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "absolute_difference": abs(mean_confidence - accuracy),
            }
        )
    return output


def load_model(
    weights_path: Path,
    class_count: int,
    device_name: str | None,
) -> tuple[Any, Any, Any]:
    import timm
    import torch

    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = timm.create_model(
        "mobilenetv3_large_100", pretrained=False, num_classes=class_count
    )
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, torch, device


def predict_rows(
    rows: Sequence[dict[str, Any]],
    model: Any,
    torch: Any,
    device: Any,
    root: Path,
    class_ids: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("El tamaño de lote debe ser al menos 1.")
    class_to_index = {class_id: index for index, class_id in enumerate(class_ids)}
    outputs = []
    for offset in range(0, len(rows), batch_size):
        batch_rows = rows[offset : offset + batch_size]
        tensors = []
        for row in batch_rows:
            raw_path = Path(row["path"])
            if raw_path.is_absolute():
                image_path = raw_path
            else:
                relative = PurePosixPath(row["path"])
                if ".." in relative.parts or "\\" in row["path"]:
                    raise TrainingDataError(f"Ruta de imagen inválida: {row['path']}")
                image_path = root.joinpath(*relative.parts)
            tensors.append(preprocess_image(image_path)[0])
        batch = torch.from_numpy(np.stack(tensors)).to(device)
        with torch.inference_mode():
            outputs.append(model(batch).detach().cpu().numpy())
    if not outputs:
        raise TrainingDataError("No hay imágenes para evaluar.")
    result = np.concatenate(outputs, axis=0)
    if result.shape != (len(rows), len(class_ids)):
        raise TrainingDataError("El modelo devolvió logits de dimensiones inesperadas.")
    for row in rows:
        if row["class_id"] not in class_to_index:
            raise TrainingDataError(f"Clase fuera del contrato: {row['class_id']}")
    return result


def calibration_artifact(
    results: dict[str, Any],
    partition_hash: str,
    exclusion_hash: str,
    model_hash: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "calibration_version": 1,
        "T": results["temperature"],
        "t_firme": results["firm_threshold"],
        "s_par": results["pair_threshold"],
        "porcentaje_par_habilitado": results["percentage_pair_enabled"],
        "objetivo_precision": results["target_precision"],
        "tolerancia_suma_top2": PAIR_TOLERANCE,
        "minimo_fotos_por_cajon": MIN_PAIR_BIN_COUNT,
        "partition_sha256": partition_hash,
        "exclusions_sha256": exclusion_hash,
        "best_pt_sha256": model_hash,
        "seed": seed,
    }


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"El archivo ya existe y no se sobrescribe: {path}") from error


def calibration_report(results: dict[str, Any], count: int) -> str:
    def threshold_text(value: float | None) -> str:
        return "no alcanzado" if value is None else f"{value:.6f}"

    decision_counts = results["decision_counts"]
    lines = [
        "# Informe de calibración",
        "",
        f"- Fotos de calibración incluidas: {count}",
        f"- Temperatura T: {results['temperature']:.8f}",
        f"- NLL antes / después: {results['nll_before']:.6f} / {results['nll_after']:.6f}",
        f"- t_firme: {threshold_text(results['firm_threshold'])}",
        f"- s_par: {threshold_text(results['pair_threshold'])}",
        f"- Porcentaje top-2 habilitado: {str(results['percentage_pair_enabled']).lower()}",
        "",
        "## Decisiones",
        "",
        "| Resultado | Fotos | Proporción |",
        "|---|---:|---:|",
    ]
    for decision in ("firme", "par", "repetir", "no_nube"):
        amount = decision_counts[decision]
        lines.append(f"| {decision} | {amount} | {amount / count:.1%} |")
    lines.extend(["", "## Calibración de la suma top-2", ""])
    if not results["pair_bins"]:
        lines.append("No se pudo formar ningún cajón elegible; porcentaje deshabilitado.")
    else:
        lines.extend(
            [
                "| Rango | Cajones iniciales | Fotos | Suma media | Exactitud top-2 | Diferencia |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for bucket in results["pair_bins"]:
            lines.append(
                f"| {bucket['range'][0]:.3f}–{bucket['range'][1]:.3f} "
                f"| {', '.join(map(str, bucket['initial_bins']))} "
                f"| {bucket['count']} | {bucket['mean_top2_probability']:.3f} "
                f"| {bucket['observed_top2_accuracy']:.3f} "
                f"| {bucket['absolute_difference']:.3f} |"
            )
        if results["pair_calibration_inconclusive"]:
            lines.append("")
            lines.append(
                "Muestra insuficiente incluso después de fusionar cajones; "
                "la verificación es inconclusa y el porcentaje queda deshabilitado."
            )
    lines.extend(
        [
            "",
            "## Confiabilidad de la confianza antes y después de T",
            "",
            "| Rango | Fotos antes | Media antes | Exactitud antes | Fotos después | Media después | Exactitud después |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    before = results["confidence_bins_before"]
    after = results["confidence_bins"]
    for bucket in sorted(
        set(tuple(item["range"]) for item in before)
        | set(tuple(item["range"]) for item in after)
    ):
        before_item = next((item for item in before if tuple(item["range"]) == bucket), None)
        after_item = next((item for item in after if tuple(item["range"]) == bucket), None)
        before_mean = (
            f"{before_item['mean_confidence']:.3f}" if before_item else "—"
        )
        before_accuracy = f"{before_item['accuracy']:.3f}" if before_item else "—"
        after_mean = f"{after_item['mean_confidence']:.3f}" if after_item else "—"
        after_accuracy = f"{after_item['accuracy']:.3f}" if after_item else "—"
        lines.append(
            f"| {bucket[0]:.1f}–{bucket[1]:.1f} "
            f"| {before_item['count'] if before_item else 0} "
            f"| {before_mean} | {before_accuracy} "
            f"| {after_item['count'] if after_item else 0} "
            f"| {after_mean} | {after_accuracy} |"
        )
    return "\n".join(lines) + "\n"


def calibrate_dataset(
    root: Path,
    partition_path: Path,
    exclusion_path: Path,
    weights_path: Path,
    device_name: str | None,
    batch_size: int,
    seed: int,
    target_precision: float = TARGET_PRECISION,
) -> tuple[dict[str, Any], str, str, str, int]:
    contract = load_model_contract()
    class_ids = class_order_from_contract(contract)
    rows, partition_hash, exclusion_hash = read_split_rows(
        partition_path, exclusion_path, "calibration", class_ids
    )
    if not weights_path.is_file():
        raise FileNotFoundError(f"No existe el checkpoint indicado: {weights_path}")
    model_hash = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    model, torch, device = load_model(weights_path, len(class_ids), device_name)
    logits = predict_rows(rows, model, torch, device, root, class_ids, batch_size)
    labels = np.asarray([class_ids.index(row["class_id"]) for row in rows])
    results = calibrate_logits(logits, labels, class_ids, target_precision)
    results["confidence_bins_before"] = _confidence_reliability_bins(
        _softmax_rows(logits, 1.0).max(axis=1),
        np.argmax(logits, axis=1) == labels,
    )
    return (
        results,
        partition_hash,
        exclusion_hash,
        model_hash,
        len(rows),
    )


def wilson_interval(successes: int, count: int) -> tuple[float, float] | None:
    if count == 0:
        return None
    z = 1.959963984540054
    proportion = successes / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / count + z * z / (4 * count * count)
        )
        / denominator
    )
    return center - margin, center + margin


def acquire_final_evaluation_lock(
    lock_path: Path, calibration_path: Path, confirm_final: bool
) -> None:
    if not confirm_final:
        raise ValueError("La evaluación requiere --confirm-final.")
    if not calibration_path.is_file():
        raise FileNotFoundError(
            f"Falta el artefacto de calibración: {calibration_path}"
        )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_document = {
        "locked_at": datetime.now(UTC).isoformat(),
        "calibration_sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
    }
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
        json.dump(lock_document, output, indent=2)
        output.write("\n")


def _load_calibration(
    path: Path,
    partition_hash: str,
    exclusion_hash: str,
    model_hash: str,
) -> dict[str, Any]:
    value, _ = _read_json_bytes(path, "el artefacto de calibración")
    if (
        not isinstance(value, dict)
        or value.get("calibration_version") != 1
        or value.get("partition_sha256") != partition_hash
        or value.get("exclusions_sha256") != exclusion_hash
        or value.get("best_pt_sha256") != model_hash
    ):
        raise TrainingDataError(
            "El artefacto de calibración no corresponde a los datos y modelo."
        )
    if (
        not isinstance(value.get("T"), (int, float))
        or not math.isfinite(value["T"])
        or value["T"] <= 0
    ):
        raise TrainingDataError("El artefacto contiene una temperatura inválida.")
    return value


def _evaluate_rows(
    rows: Sequence[dict[str, Any]],
    logits: np.ndarray,
    class_ids: Sequence[str],
    calibration: dict[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for row, row_logits in zip(rows, logits, strict=True):
        predicted = classify_logits(
            row_logits,
            float(calibration["T"]),
            calibration["t_firme"],
            calibration["s_par"],
            class_ids,
        )
        true_class = row["class_id"]
        output.append(
            {
                **row,
                "decision": predicted.decision,
                "top_classes": predicted.classes,
                "top1_correct": predicted.classes[0] == true_class,
                "top2_correct": true_class in predicted.classes,
            }
        )
    return output


def _wilson_text(successes: int, total: int) -> str:
    interval = wilson_interval(successes, total)
    if interval is None:
        return "n/a"
    return f"{successes / total:.1%} ({interval[0]:.1%}–{interval[1]:.1%})"


def _evaluation_section(title: str, rows: Sequence[dict[str, Any]]) -> list[str]:
    total = len(rows)
    correct_top1 = sum(bool(row["top1_correct"]) for row in rows)
    correct_top2 = sum(bool(row["top2_correct"]) for row in rows)
    decisions = Counter(row["decision"] for row in rows)
    firm_rows = [row for row in rows if row["decision"] == "firme"]
    correct_firm = sum(bool(row["top1_correct"]) for row in firm_rows)
    pair_rows = [row for row in rows if row["decision"] == "par"]
    retry_pool = [row for row in rows if row["decision"] in {"par", "repetir"}]
    lines = [
        f"## {title}",
        "",
        f"- Fotos: {total}",
    ]
    if title == "Fotos propias":
        lines.append(f"- Sesiones distintas: {len({row['session'] for row in rows})}")
    lines.extend(
        [
            f"- Exactitud top-1: {_wilson_text(correct_top1, total)}",
            f"- Exactitud top-2: {_wilson_text(correct_top2, total)}",
            f"- Precisión de resultados firmes: {_wilson_text(correct_firm, len(firm_rows))}",
            f"- Cobertura del par: {_wilson_text(len(pair_rows), total)}; entre no firmes nube: {_wilson_text(len(pair_rows), len(retry_pool))}",
            "",
            "| Resultado | Cantidad | Proporción (IC Wilson 95%) |",
            "|---|---:|---:|",
        ]
    )
    for decision in ("firme", "par", "repetir", "no_nube"):
        count = decisions.get(decision, 0)
        lines.append(f"| {decision} | {count} | {_wilson_text(count, total)} |")

    lines.extend(
        [
            "",
            "### Por clase real",
            "",
            "| Clase | Cantidad | Top-1 correctas |",
            "|---|---:|---:|",
        ]
    )
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[row["class_id"]].append(row)
    for class_id in sorted(by_class):
        group = by_class[class_id]
        lines.append(
            f"| {class_id} | {len(group)} | "
            f"{sum(bool(row['top1_correct']) for row in group)} |"
        )

    lines.extend(
        [
            "",
            "### Por fuente",
            "",
            "| Fuente | Cantidad | Top-1 | Top-2 | Firmes | Pares | Repetir | No nube | Precisión firmes | Cobertura par |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    for source in sorted(by_source):
        group = by_source[source]
        lines.append(
            f"| {source} | {len(group)} | "
            f"{sum(bool(row['top1_correct']) for row in group)} "
            f"({sum(bool(row['top1_correct']) for row in group) / len(group):.1%}) | "
            f"{sum(bool(row['top2_correct']) for row in group)} "
            f"({sum(bool(row['top2_correct']) for row in group) / len(group):.1%}) | "
            f"{sum(row['decision'] == 'firme' for row in group)} | "
            f"{sum(row['decision'] == 'par' for row in group)} | "
            f"{sum(row['decision'] == 'repetir' for row in group)} | "
            f"{sum(row['decision'] == 'no_nube' for row in group)} | "
            f"{_wilson_text(sum(bool(row['top1_correct']) for row in group if row['decision'] == 'firme'), sum(row['decision'] == 'firme' for row in group))} | "
            f"{sum(row['decision'] == 'par' for row in group) / len(group):.1%} |"
        )
    return lines


def read_own_photo_rows(
    csv_path: Path, repository_root: Path, class_ids: Sequence[str]
) -> list[dict[str, Any]]:
    manifest_path = csv_path.resolve(strict=True)
    if manifest_path.is_relative_to(repository_root.resolve()):
        raise ValueError("El manifiesto CSV de fotos propias debe estar fuera de Git.")
    rows = []
    with manifest_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"ruta", "clase", "sesion"}.issubset(reader.fieldnames or ()):
            raise ValueError("El CSV requiere las columnas ruta, clase y sesion.")
        for index, item in enumerate(reader, start=2):
            route = (item.get("ruta") or "").strip()
            class_id = (item.get("clase") or "").strip()
            session = (item.get("sesion") or "").strip()
            if not route or class_id not in class_ids or not session:
                raise ValueError(f"Fila {index} inválida en el CSV de fotos propias.")
            image_path = Path(route)
            if not image_path.is_absolute():
                image_path = manifest_path.parent / image_path
            if not image_path.is_file():
                raise FileNotFoundError(f"No existe la foto propia: {image_path}")
            rows.append(
                {
                    "path": str(image_path.resolve()),
                    "class_id": class_id,
                    "source": "propias",
                    "session": session,
                }
            )
    if not rows:
        raise ValueError("El CSV de fotos propias no contiene fotos.")
    return rows


def read_double_label_photos(
    path: Path, partition_hash: str
) -> list[dict[str, Any]]:
    document, _ = _read_json_bytes(path, "el set de doble etiqueta")
    if (
        not isinstance(document, dict)
        or document.get("double_label_version") != 1
        or document.get("partition_sha256") != partition_hash
        or not isinstance(document.get("photos"), list)
    ):
        raise TrainingDataError("El set de doble etiqueta no corresponde a la partición.")
    return document["photos"]


def evaluate_double_label_photos(
    photos: Sequence[dict[str, Any]],
    model: Any,
    torch: Any,
    device: Any,
    root: Path,
    class_ids: Sequence[str],
    calibration: dict[str, Any],
    batch_size: int,
) -> list[dict[str, Any]]:
    rows = []
    for photo in photos:
        if not isinstance(photo, dict) or not isinstance(photo.get("representative_path"), str):
            raise TrainingDataError("Una foto de doble etiqueta no tiene ruta representativa.")
        rows.append(
            {
                "path": photo["representative_path"],
                "class_id": photo["classes"][0],
                "source": "doble-etiqueta",
            }
        )
    logits = predict_rows(rows, model, torch, device, root, class_ids, batch_size)
    results = []
    for photo, row_logits in zip(photos, logits, strict=True):
        prediction = classify_logits(
            row_logits,
            float(calibration["T"]),
            calibration["t_firme"],
            calibration["s_par"],
            class_ids,
        )
        results.append(
            {
                "group_id": photo["group_id"],
                "decision": prediction.decision,
                "top_classes": prediction.classes,
            }
        )
    return results


def format_final_report(
    test_results: Sequence[dict[str, Any]],
    own_results: Sequence[dict[str, Any]],
    double_label_results: Sequence[dict[str, Any]],
) -> str:
    lines = [
        "# Evaluación final",
        "",
        "La prueba final se ejecutó una única vez con el artefacto de calibración existente.",
        "",
        *_evaluation_section("Test", test_results),
        "",
        *_evaluation_section("Fotos propias", own_results),
        "",
        "## Doble etiqueta (descriptivo; excluido de las métricas)",
        "",
        "| Grupo | Resultado | Clases top-1 / top-2 |",
        "|---|---|---|",
    ]
    for result in double_label_results:
        lines.append(
            f"| {result['group_id']} | {result['decision']} "
            f"| {', '.join(result['top_classes'])} |"
        )
    return "\n".join(lines) + "\n"
