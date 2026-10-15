from dataclasses import dataclass
from typing import Literal

PredictionStatus = Literal["accepted", "no_cloud", "low_confidence"]


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
