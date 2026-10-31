from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import onnxruntime as ort
from PIL import Image

LogitDecision = Literal["firme", "par", "repetir", "no_nube"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "clasificador.onnx"
CALIBRATION_PATH = PROJECT_ROOT / "manifests" / "calibration-v1.json"
CONTRACT_PATH = Path(__file__).with_name("model_contract.json")


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
    for name, threshold in (("firme", firm_threshold), ("par", pair_threshold)):
        if threshold is not None and (
            not math.isfinite(threshold) or not 0 <= threshold <= 1
        ):
            raise ValueError(f"El umbral {name} debe estar entre 0 y 1.")

    scaled = values / temperature
    probabilities_array = np.exp(scaled - np.max(scaled))
    probabilities_array /= probabilities_array.sum()
    order = np.argsort(-probabilities_array, kind="stable")[:4]
    classes = tuple(class_ids[int(index)] for index in order)
    confidence = float(probabilities_array[order[0]])
    top2_probability = float(probabilities_array[order[:2]].sum())

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
class InferenceModel:
    session: ort.InferenceSession
    contract: dict[str, Any]
    calibration: dict[str, Any]
    class_ids: tuple[str, ...]
    input_name: str
    output_name: str

    def classify(self, image: Image.Image | str | Path) -> LogitClassification:
        from cloud_classifier.preprocessing import preprocess_image

        tensor = preprocess_image(image, self.contract)
        logits = self.session.run(
            [self.output_name],
            {self.input_name: tensor},
        )[0]
        if logits.shape != (1, len(self.class_ids)):
            raise RuntimeError("El modelo devolvió logits de dimensiones inesperadas.")
        return classify_logits(
            logits[0],
            self.calibration["T"],
            self.calibration["t_firme"],
            self.calibration["s_par"],
            self.class_ids,
        )


def load_inference_model(
    model_path: Path = MODEL_PATH,
    calibration_path: Path = CALIBRATION_PATH,
    contract_path: Path = CONTRACT_PATH,
) -> InferenceModel:
    from cloud_classifier.preprocessing import load_model_contract

    try:
        raw_contract = contract_path.read_bytes()
        contract = load_model_contract(contract_path)
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(
            "No se pudo cargar el contrato y la calibración del modelo."
        ) from error

    try:
        class_ids = _validate_runtime_contract(contract)
        _validate_calibration(calibration)
    except ValueError as error:
        raise RuntimeError(
            "El contrato o la calibración del modelo está incompleto o no es válido."
        ) from error
    if not model_path.is_file():
        raise RuntimeError(f"No se encontró el modelo ONNX: {model_path}.")

    try:
        session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
    except (OSError, RuntimeError) as error:
        raise RuntimeError(f"No se pudo cargar el modelo ONNX: {model_path}.") from error

    metadata = session.get_modelmeta().custom_metadata_map
    expected_metadata = {
        "source_checkpoint_sha256": calibration["best_pt_sha256"],
        "contract_sha256": hashlib.sha256(raw_contract).hexdigest(),
        "class_ids": json.dumps(class_ids),
    }
    for key, expected_value in expected_metadata.items():
        if metadata.get(key) != expected_value:
            raise RuntimeError(
                f"El modelo ONNX no coincide con el contrato o la calibración ({key})."
            )

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if (
        len(inputs) != 1
        or inputs[0].shape != [1, 3, 224, 224]
        or inputs[0].type != "tensor(float)"
        or len(outputs) != 1
        or outputs[0].shape != [1, len(class_ids)]
        or outputs[0].type != "tensor(float)"
    ):
        raise RuntimeError(
            "La entrada o salida del modelo ONNX no coincide con el contrato."
        )

    return InferenceModel(
        session=session,
        contract=contract,
        calibration=calibration,
        class_ids=class_ids,
        input_name=inputs[0].name,
        output_name=outputs[0].name,
    )


def _validate_runtime_contract(contract: dict[str, Any]) -> tuple[str, ...]:
    classes = contract.get("classes")
    calibration_config = contract.get("calibration")
    output = contract.get("output")
    if (
        not isinstance(classes, list)
        or len(classes) != 11
        or any(not isinstance(item, dict) for item in classes)
        or not isinstance(calibration_config, dict)
        or not isinstance(output, dict)
        or output.get("values") != "logits"
        or output.get("shape") != ["batch", 11]
        or calibration_config.get("manifest") != "manifests/calibration-v1.json"
        or calibration_config.get("temperature_field") != "T"
        or calibration_config.get("firm_threshold_field") != "t_firme"
        or calibration_config.get("pair_threshold_field") != "s_par"
        or calibration_config.get("pair_percentage_enabled_field")
        != "porcentaje_par_habilitado"
    ):
        raise ValueError("El contrato del modelo está incompleto o no es compatible.")

    class_ids = tuple(item.get("id") for item in classes)
    if (
        any(not isinstance(class_id, str) or not class_id for class_id in class_ids)
        or len(set(class_ids)) != len(class_ids)
        or class_ids[-1] != "no_nube"
    ):
        raise ValueError("El orden de clases del contrato no es válido.")
    return class_ids


def _validate_calibration(calibration: object) -> None:
    required_hashes = ("partition_sha256", "exclusions_sha256", "best_pt_sha256")
    if not isinstance(calibration, dict) or calibration.get("calibration_version") != 1:
        raise ValueError("La calibración no es válida.")
    required_fields = {
        "T",
        "t_firme",
        "s_par",
        "porcentaje_par_habilitado",
        "objetivo_precision",
        "tolerancia_suma_top2",
        "minimo_fotos_por_cajon",
        "seed",
        *required_hashes,
    }
    if not required_fields.issubset(calibration):
        raise ValueError("La calibración está incompleta.")
    for key in required_hashes:
        value = calibration.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"El hash {key} falta o no es válido.")

    temperature = calibration.get("T")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("La temperatura calibrada debe ser positiva y finita.")
    for key in ("t_firme", "s_par"):
        threshold = calibration.get(key)
        if threshold is not None and (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError(f"El umbral calibrado {key} no es válido.")
    if not isinstance(calibration.get("porcentaje_par_habilitado"), bool):
        raise ValueError("La configuración del porcentaje par falta o no es válida.")
    for key in ("objetivo_precision", "tolerancia_suma_top2"):
        value = calibration.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"El valor calibrado {key} no es válido.")
    for key in ("minimo_fotos_por_cajon", "seed"):
        value = calibration.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"El valor calibrado {key} no es válido.")


def format_prediction(
    classification: LogitClassification,
    cloud_config: dict[str, Any],
    image_directory: Path,
    show_pair_percentage: bool = False,
) -> tuple[str, tuple[str | None, str | None]]:
    clouds = {cloud["id"]: cloud for cloud in cloud_config["clases"]}
    caution = cloud_config["cautela_general"].strip()
    references: list[str | None] = [None, None]
    lines: list[str] = []

    if classification.decision == "no_nube":
        lines.extend(
            [
                f"## {cloud_config['no_nube']['mensaje']}",
                "",
                "La app no determina las condiciones actuales del cielo.",
            ]
        )
        return "\n".join(lines), (None, None)

    if classification.decision == "repetir":
        lines.extend(
            [
                "## Probá con otra foto",
                "",
                "No pude distinguir una nube con claridad. Tomá otra foto donde se vea mejor el cielo.",
            ]
        )
        _append_alternatives(lines, classification.classes[:2], clouds)
    elif classification.decision == "firme":
        primary_id = classification.classes[0]
        primary = clouds[primary_id]
        lines.extend(
            [
                f"## Probablemente es {primary['nombre']}",
                "",
                primary["descripcion"].strip(),
            ]
        )
        _append_cloud_explanation(lines, primary)
        references[0] = str(image_directory / primary["imagen"]["archivo"])
        _append_alternatives(lines, classification.classes[1:3], clouds)
    else:
        first_id, second_id = classification.classes[:2]
        first, second = clouds[first_id], clouds[second_id]
        pair = frozenset((first_id, second_id))
        event_note = _event_distinct_note(pair, cloud_config)
        lines.extend(
            [
                f"## Puede ser {first['nombre']} o {second['nombre']}",
                "",
                f"**{first['nombre']}:** {first['descripcion'].strip()}",
            ]
        )
        _append_cloud_explanation(lines, first)
        lines.extend(["", f"**{second['nombre']}:** {second['descripcion'].strip()}"])
        _append_cloud_explanation(lines, second)
        if event_note is not None:
            lines.extend(["", f"**Diferencia entre ambas:** {event_note}"])
        else:
            lines.extend(
                [
                    "",
                    "Se parecen visualmente y la foto no permite distinguirlas con seguridad.",
                ]
            )
        references[0] = str(image_directory / first["imagen"]["archivo"])
        references[1] = str(image_directory / second["imagen"]["archivo"])
        if show_pair_percentage:
            lines.extend(
                [
                    "",
                    f"Suma de probabilidades del par: {classification.top2_probability:.0%}.",
                ]
            )
        _append_alternatives(lines, classification.classes[2:4], clouds)

    lines.extend(["", f"**Cautela:** {caution}"])
    return "\n".join(lines), (references[0], references[1])


def _append_cloud_explanation(
    lines: list[str],
    cloud: dict[str, Any],
) -> None:
    lines.extend(
        [
            "",
            f"**Altura:** {cloud['altura']}",
            "",
            "**Cómo reconocerla:**",
            *[f"- {cue}" for cue in cloud["como_reconocerla"]],
            "",
            f"**Información general:** {cloud['asociaciones'].strip()}",
            "",
            "**Fuentes:**",
            *[
                f"- [{source['titulo']}]({source['url']})"
                for source in cloud["fuentes"]
            ],
        ]
    )


def _event_distinct_note(
    pair: frozenset[str],
    cloud_config: dict[str, Any],
) -> str | None:
    for event in cloud_config.get("eventos_distintos", []):
        if frozenset(event["clases"]) == pair:
            return event["nota"]
    return None


def _append_alternatives(
    lines: list[str],
    class_ids: Sequence[str],
    clouds: dict[str, dict[str, Any]],
) -> None:
    alternatives = [
        clouds[class_id]["nombre"]
        for class_id in class_ids
        if class_id != "no_nube"
    ][:2]
    if alternatives:
        lines.extend(
            ["", "**Otras posibilidades:**", *[f"- {name}" for name in alternatives]]
        )
