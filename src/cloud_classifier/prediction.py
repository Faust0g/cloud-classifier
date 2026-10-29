from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np

PredictionStatus = Literal["accepted", "no_cloud", "low_confidence"]
LogitDecision = Literal["firme", "par", "repetir", "no_nube"]


@dataclass(frozen=True)
class LogitClassification:
    decision: LogitDecision
    classes: tuple[str, ...]
    probabilities: tuple[float, ...]
    confidence: float
    top2_probability: float


def classify_logits(
    logits: Sequence[float] | np.ndarray,
    temperature: float,
    firm_threshold: float | None,
    pair_threshold: float | None,
    class_ids: Sequence[str],
) -> LogitClassification:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(class_ids) or len(values) < 2:
        raise ValueError("Los logits deben tener un valor por clase y al menos dos clases.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Los logits deben ser finitos.")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("La temperatura debe ser positiva y finita.")
    if len(class_ids) != len(set(class_ids)) or any(not item for item in class_ids):
        raise ValueError("Los IDs de clase deben ser únicos y no vacíos.")
    for name, threshold in (
        ("firme", firm_threshold),
        ("par", pair_threshold),
    ):
        if threshold is not None and (
            not math.isfinite(threshold) or not 0 <= threshold <= 1
        ):
            raise ValueError(f"El umbral {name} debe estar entre 0 y 1.")

    scaled = values / temperature
    probabilities_array = np.exp(scaled - np.max(scaled))
    probabilities_array /= probabilities_array.sum()
    order = np.argsort(-probabilities_array, kind="stable")[:2]
    classes = tuple(class_ids[int(index)] for index in order)
    confidence = float(probabilities_array[order[0]])
    top2_probability = float(probabilities_array[order].sum())

    if classes[0] == "no_nube":
        decision: LogitDecision = "no_nube"
    elif firm_threshold is not None and confidence >= firm_threshold:
        decision = "firme"
    elif pair_threshold is not None and top2_probability >= pair_threshold:
        decision = "par"
    else:
        decision = "repetir"

    return LogitClassification(
        decision=decision,
        classes=classes,
        probabilities=tuple(float(probabilities_array[index]) for index in order),
        confidence=confidence,
        top2_probability=top2_probability,
    )


@dataclass(frozen=True)
class ClassPrediction:
    class_id: str
    common_name: str
    scientific_name: str | None
    confidence: float

    def __post_init__(self) -> None:
        if not self.class_id:
            raise ValueError("class_id no puede estar vacío.")
        if not self.common_name:
            raise ValueError("common_name no puede estar vacío.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("La confianza debe estar entre 0 y 1.")


@dataclass(frozen=True)
class PredictionResult:
    status: PredictionStatus
    primary: ClassPrediction
    alternatives: tuple[ClassPrediction, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in ("accepted", "no_cloud", "low_confidence"):
            raise ValueError("El estado de predicción no es válido.")
        if len(self.alternatives) > 2:
            raise ValueError("Una predicción admite como máximo dos alternativas.")
        if self.status == "no_cloud" and self.primary.class_id != "no_nube":
            raise ValueError("El estado no_cloud requiere la clase no_nube.")
        if self.status != "no_cloud" and self.primary.class_id == "no_nube":
            raise ValueError("La clase no_nube requiere el estado no_cloud.")
        if self.status == "no_cloud" and self.alternatives:
            raise ValueError("El estado no_cloud no admite alternativas de nubes.")


def format_prediction(result: PredictionResult) -> str:
    if result.status == "no_cloud":
        return (
            "## No se identificó una nube\n\n"
            f"Resultado: `{result.primary.class_id}` "
            f"({result.primary.confidence:.0%} de confianza).\n\n"
            "Probá encuadrar una parte del cielo donde haya nubes."
        )

    scientific_name = (
        f" ({result.primary.scientific_name})"
        if result.primary.scientific_name
        else ""
    )
    lines = [
        f"## {result.primary.common_name}{scientific_name} "
        f"— {result.primary.confidence:.0%}",
        "",
        (
            "**Estado:** Predicción aceptada."
            if result.status == "accepted"
            else "**Estado:** Confianza baja."
        ),
    ]
    if result.alternatives:
        lines.extend(
            [
                "",
                "**Otras posibilidades:**",
                *[
                    _format_candidate(candidate)
                    for candidate in result.alternatives
                ],
            ]
        )
    if result.status == "low_confidence":
        lines.extend(
            [
                "",
                "La confianza es baja; probá con otra foto, enfocada y con la nube más visible.",
            ]
        )
    return "\n".join(lines)


def _format_candidate(candidate: ClassPrediction) -> str:
    scientific_name = (
        f" ({candidate.scientific_name})" if candidate.scientific_name else ""
    )
    return (
        f"- {candidate.common_name}{scientific_name}: "
        f"{candidate.confidence:.0%}"
    )
