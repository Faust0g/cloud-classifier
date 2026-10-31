import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cloud_classifier.demo import _load_cloud_config
from cloud_classifier.prediction import (
    classify_logits,
    format_prediction,
    load_inference_model,
)
from cloud_classifier.preprocessing import load_model_contract


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_IDS = tuple(item["id"] for item in load_model_contract()["classes"])
HASH = "a" * 64


def _calibration() -> dict:
    return {
        "calibration_version": 1,
        "T": 1.0,
        "t_firme": 0.8,
        "s_par": 0.85,
        "porcentaje_par_habilitado": False,
        "objetivo_precision": 0.85,
        "tolerancia_suma_top2": 0.08,
        "minimo_fotos_por_cajon": 30,
        "partition_sha256": HASH,
        "exclusions_sha256": HASH,
        "best_pt_sha256": HASH,
        "seed": 42,
    }


class PredictionDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = _load_cloud_config(CLASS_IDS)
        cls.image_directory = PROJECT_ROOT / "app" / "assets" / "clouds"

    def classify(self, logits: list[float]):
        return classify_logits(logits, 1.0, 0.8, 0.85, CLASS_IDS)

    def test_synthetic_logits_cover_all_three_results(self) -> None:
        firm = self.classify([8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        pair = self.classify([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 5.9, -1.0])
        retry = self.classify([0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.0])
        no_cloud = self.classify([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0])

        self.assertEqual(firm.decision, "firme")
        self.assertEqual(pair.decision, "par")
        self.assertEqual(retry.decision, "repetir")
        self.assertEqual(no_cloud.decision, "no_nube")

    def test_firm_result_omits_confidence_and_has_two_named_alternatives(self) -> None:
        prediction = self.classify([8.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])

        message, references = format_prediction(
            prediction,
            self.config,
            self.image_directory,
        )

        self.assertIn("Probablemente es Cirro", message)
        self.assertIn("Filamentos blancos", message)
        self.assertIn("**Altura:** alta", message)
        self.assertIn("**Cómo reconocerla:**", message)
        self.assertIn("Aspecto fibroso o sedoso", message)
        self.assertIn("**Suele asociarse con (información general):**", message)
        self.assertIn("no anuncian lluvia", message)
        self.assertIn("**Fuentes:**", message)
        self.assertIn(
            "Atlas Internacional de Nubes: Cirrus (Ci)",
            message,
        )
        self.assertIn("- Cirrocúmulo", message)
        self.assertIn("- Cirrostrato", message)
        self.assertNotRegex(message, r"\d+%")
        self.assertIsNotNone(references[0])

    def test_pair_has_both_references_and_no_percentage_by_default(self) -> None:
        prediction = self.classify(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 5.9, -1.0]
        )

        message, references = format_prediction(
            prediction,
            self.config,
            self.image_directory,
        )

        self.assertIn("Puede ser Cúmulo o Cumulonimbo", message)
        self.assertIn("Se parecen visualmente", message)
        self.assertIn("**Cúmulo:**", message)
        self.assertIn("**Cumulonimbo:**", message)
        self.assertIn("**Altura:** baja", message)
        self.assertIn("**Cómo reconocerla:**", message)
        self.assertIn("Base más oscura y plana", message)
        self.assertIn("Gran desarrollo vertical", message)
        self.assertEqual(message.count("**Suele asociarse con (información general):**"), 2)
        self.assertEqual(message.count("**Fuentes:**"), 2)
        self.assertNotIn("%", message)
        self.assertTrue(all(references))

    def test_documented_distinct_pair_shows_its_yaml_note(self) -> None:
        config = {
            **self.config,
            "eventos_distintos": [
                {
                    "clases": ["cumulus", "cumulonimbus"],
                    "nota": "Nota curada para este par.",
                }
            ],
        }
        prediction = self.classify(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 5.9, -1.0]
        )

        message, _ = format_prediction(prediction, config, self.image_directory)

        self.assertIn("**Diferencia entre ambas:** Nota curada para este par.", message)
        self.assertNotIn("Se parecen visualmente", message)

    def test_retry_and_no_cloud_have_distinct_spanish_messages(self) -> None:
        retry = self.classify([0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.0])
        no_cloud = self.classify([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0])

        retry_message, _ = format_prediction(retry, self.config, self.image_directory)
        no_cloud_message, _ = format_prediction(no_cloud, self.config, self.image_directory)

        self.assertIn("Probá con otra foto", retry_message)
        self.assertIn("No veo una nube que pueda clasificar", no_cloud_message)
        self.assertNotEqual(retry_message, no_cloud_message)


class InferenceStartupTests(unittest.TestCase):
    def _fake_session(self, metadata: dict):
        return SimpleNamespace(
            get_modelmeta=lambda: SimpleNamespace(custom_metadata_map=metadata),
            get_inputs=lambda: [
                SimpleNamespace(
                    name="images",
                    shape=[1, 3, 224, 224],
                    type="tensor(float)",
                )
            ],
            get_outputs=lambda: [
                SimpleNamespace(
                    name="logits",
                    shape=[1, 11],
                    type="tensor(float)",
                )
            ],
        )

    def _write_json(self, path: Path, value: object) -> None:
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_startup_rejects_model_hash_mismatch(self) -> None:
        contract_path = PROJECT_ROOT / "src" / "cloud_classifier" / "model_contract.json"
        contract_bytes = contract_path.read_bytes()
        contract_hash = hashlib.sha256(contract_bytes).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.onnx"
            model_path.write_bytes(b"fixture")
            calibration_path = root / "calibration.json"
            self._write_json(calibration_path, _calibration())
            session = self._fake_session(
                {
                    "source_checkpoint_sha256": "b" * 64,
                    "contract_sha256": contract_hash,
                    "class_ids": json.dumps(CLASS_IDS),
                }
            )
            with patch(
                "cloud_classifier.prediction.ort.InferenceSession",
                return_value=session,
            ):
                with self.assertRaisesRegex(RuntimeError, "no coincide"):
                    load_inference_model(
                        model_path,
                        calibration_path,
                        contract_path,
                    )

    def test_startup_rejects_incomplete_contract(self) -> None:
        contract = json.loads(
            (PROJECT_ROOT / "src" / "cloud_classifier" / "model_contract.json").read_text(
                encoding="utf-8"
            )
        )
        contract["output"]["shape"] = ["batch", 10]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract_path = root / "contract.json"
            self._write_json(contract_path, contract)
            calibration_path = root / "calibration.json"
            self._write_json(calibration_path, _calibration())
            model_path = root / "model.onnx"
            model_path.write_bytes(b"fixture")

            with self.assertRaisesRegex(RuntimeError, "incompleto"):
                load_inference_model(model_path, calibration_path, contract_path)

    def test_startup_rejects_missing_calibration_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract_path = root / "contract.json"
            contract = json.loads(
                (PROJECT_ROOT / "src" / "cloud_classifier" / "model_contract.json").read_text(
                    encoding="utf-8"
                )
            )
            self._write_json(contract_path, contract)
            calibration_path = root / "calibration.json"
            calibration = _calibration()
            del calibration["T"]
            self._write_json(calibration_path, calibration)
            model_path = root / "model.onnx"
            model_path.write_bytes(b"fixture")

            with self.assertRaisesRegex(RuntimeError, "calibración"):
                load_inference_model(model_path, calibration_path, contract_path)


if __name__ == "__main__":
    unittest.main()
