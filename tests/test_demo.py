from io import BytesIO
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from cloud_classifier.demo import (
    MAX_IMAGE_BYTES,
    InvalidImageError,
    process_image,
    validate_image,
)
from cloud_classifier.prediction import (
    ClassPrediction,
    PredictionResult,
    format_prediction,
)
from cloud_classifier.simulated_predictor import (
    simulate_prediction,
)


class ValidateImageTests(unittest.TestCase):
    def create_image(self, path: Path, image_format: str = "PNG") -> None:
        Image.new("RGB", (2, 2), color="blue").save(path, format=image_format)

    def test_accepts_jpeg_and_png(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for image_format, extension in (("JPEG", ".jpg"), ("PNG", ".png")):
                path = Path(directory) / f"image{extension}"
                self.create_image(path, image_format)

                with self.subTest(image_format=image_format):
                    validate_image(path)

    def test_rejects_unsupported_or_invalid_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.gif"
            self.create_image(path, "GIF")
            with self.assertRaisesRegex(InvalidImageError, "extensión"):
                validate_image(path)

            path = Path(directory) / "image.png"
            path.write_bytes(b"not an image")
            with self.assertRaisesRegex(InvalidImageError, "válida"):
                validate_image(path)

    def test_rejects_images_over_10_mb(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.png"
            path.write_bytes(b"x" * (MAX_IMAGE_BYTES + 1))

            with self.assertRaisesRegex(InvalidImageError, "10 MB"):
                validate_image(path)

    def test_accepts_valid_image_exactly_at_10_mb(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "maximum.png"
            image_bytes = BytesIO()
            Image.new("RGB", (2, 2)).save(image_bytes, format="PNG")
            content = image_bytes.getvalue()
            path.write_bytes(content + b"\0" * (MAX_IMAGE_BYTES - len(content)))

            validate_image(path)

    def test_rejects_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(InvalidImageError, "No se pudo leer"):
                validate_image(Path(directory) / "missing.png")


class SimulatedPredictionTests(unittest.TestCase):
    def test_scenarios_produce_distinct_user_messages(self) -> None:
        cloud = format_prediction(simulate_prediction("Nube reconocida"))
        no_cloud = format_prediction(simulate_prediction("No se identificó una nube"))
        low_confidence = format_prediction(simulate_prediction("Confianza baja"))

        self.assertIn("Cúmulo (Cumulus)", cloud)
        self.assertIn("Predicción aceptada", cloud)
        self.assertIn("Otras posibilidades:", cloud)
        self.assertIn("Altocúmulo (Altocumulus): 7%", cloud)
        self.assertIn("No se identificó una nube", no_cloud)
        self.assertIn("no_nube", no_cloud)
        self.assertIn("confianza es baja", low_confidence)
        self.assertIn("probá con otra foto", low_confidence)

    def test_unknown_scenario_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "escenarios"):
            simulate_prediction("modelo real")

    def test_prediction_format_is_independent_of_prediction_source(self) -> None:
        result = PredictionResult(
            status="accepted",
            primary=ClassPrediction("Cb", "Cumulonimbo", "Cumulonimbus", 0.81),
            alternatives=(
                ClassPrediction("Ns", "Nimbostrato", "Nimbostratus", 0.12),
            ),
        )

        formatted = format_prediction(result)

        self.assertIn("Cumulonimbo (Cumulonimbus) — 81%", formatted)
        self.assertIn("Nimbostrato (Nimbostratus): 12%", formatted)

    def test_prediction_rejects_more_than_two_alternatives(self) -> None:
        candidate = ClassPrediction("Cu", "Cúmulo", "Cumulus", 0.1)
        with self.assertRaisesRegex(ValueError, "como máximo dos"):
            PredictionResult(
                status="accepted",
                primary=candidate,
                alternatives=(candidate, candidate, candidate),
            )

    def test_prediction_status_matches_no_cloud_class(self) -> None:
        candidate = ClassPrediction("Cu", "Cúmulo", "Cumulus", 0.9)
        with self.assertRaisesRegex(ValueError, "requiere la clase no_nube"):
            PredictionResult(status="no_cloud", primary=candidate)

    def test_process_image_requires_an_image(self) -> None:
        result, upload, camera = process_image(None, None, "Nube reconocida")
        self.assertIn("Subí una imagen", result)
        self.assertIsNone(upload)
        self.assertIsNone(camera)

    def test_process_image_removes_cached_upload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            path = cache_dir / "upload.png"
            Image.new("RGB", (2, 2)).save(path, format="PNG")

            with patch(
                "cloud_classifier.demo.gr.utils.get_upload_folder",
                return_value=directory,
            ):
                result, upload, camera = process_image(
                    str(path),
                    None,
                    "Nube reconocida",
                )

            self.assertIn("Cúmulo (Cumulus)", result)
            self.assertIsNone(upload)
            self.assertIsNone(camera)
            self.assertFalse(path.exists())

    def test_invalid_cached_upload_is_removed_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            path = cache_dir / "upload.png"
            path.write_text("not an image", encoding="utf-8")

            with (
                patch(
                    "cloud_classifier.demo.gr.utils.get_upload_folder",
                    return_value=directory,
                ),
            ):
                result, upload, camera = process_image(
                    str(path),
                    None,
                    "Nube reconocida",
                )

            self.assertIn("no contiene una imagen", result)
            self.assertIsNone(upload)
            self.assertIsNone(camera)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
