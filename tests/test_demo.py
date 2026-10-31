from io import BytesIO
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from cloud_classifier.demo import (
    MAX_IMAGE_BYTES,
    InvalidImageError,
    _load_cloud_config,
    process_image,
    validate_image,
)
from cloud_classifier.prediction import classify_logits
from cloud_classifier.preprocessing import load_model_contract


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


class PredictionFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.class_ids = tuple(
            item["id"] for item in load_model_contract()["classes"]
        )
        cls.cloud_config = _load_cloud_config(cls.class_ids)

    def classify_firm(self, image_path: str | Path):
        return classify_logits(
            [8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            1.0,
            0.8,
            0.9,
            self.class_ids,
        )

    def test_process_image_requires_an_image(self) -> None:
        result, upload, camera = process_image(
            None,
            None,
            self.classify_firm,
            self.cloud_config,
        )
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
                    self.classify_firm,
                    self.cloud_config,
                )

            self.assertIn("Probablemente es Cirro", result)
            self.assertIsNotNone(upload)
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
                    self.classify_firm,
                    self.cloud_config,
                )

            self.assertIn("no contiene una imagen", result)
            self.assertIsNone(upload)
            self.assertIsNone(camera)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
