from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import timm

from cloud_classifier.prediction import _validate_calibration
from cloud_classifier.preprocessing import load_model_contract
from cloud_classifier.training import MODEL_NAME, class_order_from_contract

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "best.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "clasificador.onnx"
CALIBRATION = PROJECT_ROOT / "manifests" / "calibration-v1.json"
CONTRACT_PATH = PROJECT_ROOT / "src" / "cloud_classifier" / "model_contract.json"
LOGIT_TOLERANCE = 1e-4


def export_model(weights_path: Path, output_path: Path) -> None:
    raw_contract = CONTRACT_PATH.read_bytes()
    contract = load_model_contract(CONTRACT_PATH)
    class_ids = class_order_from_contract(contract)
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    _validate_calibration(calibration)

    checkpoint_hash = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    if checkpoint_hash != calibration["best_pt_sha256"]:
        raise ValueError("best.pt no coincide con el hash calibrado.")

    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(class_ids))
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_inputs = np.random.default_rng(42).standard_normal(
        (3, 1, 3, 224, 224),
        dtype=np.float32,
    )
    example = torch.from_numpy(synthetic_inputs[0])
    torch.onnx.export(
        model,
        example,
        str(output_path),
        input_names=["images"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    metadata = {
        "source_checkpoint_sha256": checkpoint_hash,
        "contract_sha256": hashlib.sha256(raw_contract).hexdigest(),
        "class_ids": json.dumps(class_ids),
    }
    del onnx_model.metadata_props[:]
    for key, value in metadata.items():
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(onnx_model, str(output_path))

    session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    for synthetic_input in synthetic_inputs:
        with torch.inference_mode():
            expected = model(torch.from_numpy(synthetic_input)).numpy()
        actual = session.run([output_name], {input_name: synthetic_input})[0]
        if expected.shape != (1, len(class_ids)) or actual.shape != expected.shape:
            raise RuntimeError("La exportación ONNX no produjo logits con la forma esperada.")
        if not np.allclose(expected, actual, rtol=0.0, atol=LOGIT_TOLERANCE):
            maximum_error = float(np.max(np.abs(expected - actual)))
            raise RuntimeError(
                f"PyTorch y ONNX Runtime difieren en logits (máximo {maximum_error:.2g})."
            )

    print(f"Modelo ONNX exportado y verificado: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta runs/best.pt a ONNX y verifica tres entradas sintéticas."
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    export_model(arguments.weights, arguments.output)
