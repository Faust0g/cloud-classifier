from typing import Final

from cloud_classifier.prediction import ClassPrediction, PredictionResult

SIMULATED_PREDICTIONS: Final[dict[str, PredictionResult]] = {
    "Nube reconocida": PredictionResult(
        status="accepted",
        primary=ClassPrediction(
            class_id="Cu",
            common_name="Cúmulo",
            scientific_name="Cumulus",
            confidence=0.88,
        ),
        alternatives=(
            ClassPrediction(
                class_id="Ac",
                common_name="Altocúmulo",
                scientific_name="Altocumulus",
                confidence=0.07,
            ),
            ClassPrediction(
                class_id="Sc",
                common_name="Estratocúmulo",
                scientific_name="Stratocumulus",
                confidence=0.03,
            ),
        ),
    ),
    "No se identificó una nube": PredictionResult(
        status="no_cloud",
        primary=ClassPrediction(
            class_id="no_nube",
            common_name="No se identificó una nube",
            scientific_name=None,
            confidence=0.94,
        ),
    ),
    "Confianza baja": PredictionResult(
        status="low_confidence",
        primary=ClassPrediction(
            class_id="Cu",
            common_name="Cúmulo",
            scientific_name="Cumulus",
            confidence=0.46,
        ),
        alternatives=(
            ClassPrediction(
                class_id="Ac",
                common_name="Altocúmulo",
                scientific_name="Altocumulus",
                confidence=0.32,
            ),
            ClassPrediction(
                class_id="Sc",
                common_name="Estratocúmulo",
                scientific_name="Stratocumulus",
                confidence=0.17,
            ),
        ),
    ),
}


def simulate_prediction(scenario: str) -> PredictionResult:
    prediction = SIMULATED_PREDICTIONS.get(scenario)
    if prediction is None:
        raise ValueError("Elegí uno de los escenarios de demostración disponibles.")
    return prediction
