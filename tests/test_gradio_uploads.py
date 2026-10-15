import os
import socket
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from gradio_client import Client, handle_file
from PIL import Image

from cloud_classifier.demo import MAX_IMAGE_BYTES, create_demo


class GradioUploadIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_directory = tempfile.TemporaryDirectory()
        cls.upload_directory = Path(cls.temp_directory.name) / "uploads"
        cls.upload_directory.mkdir()
        cls.previous_temp_directory = os.environ.get("GRADIO_TEMP_DIR")
        os.environ["GRADIO_TEMP_DIR"] = str(cls.upload_directory)

        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            cls.port = listener.getsockname()[1]

        cls.demo = create_demo()
        cls.demo.launch(
            server_name="127.0.0.1",
            server_port=cls.port,
            quiet=True,
            prevent_thread_lock=True,
        )
        cls.client = Client(
            f"http://127.0.0.1:{cls.port}",
            verbose=False,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()
        cls.demo.close()
        if cls.previous_temp_directory is None:
            os.environ.pop("GRADIO_TEMP_DIR", None)
        else:
            os.environ["GRADIO_TEMP_DIR"] = cls.previous_temp_directory
        cls.temp_directory.cleanup()

    def create_uploads(self, directory: Path) -> dict[str, Path]:
        invalid_text = directory / "renamed.png"
        invalid_text.write_text("esto no es una imagen", encoding="utf-8")

        empty_file = directory / "empty.png"
        empty_file.touch()

        wrong_extension = directory / "image.txt"
        Image.new("RGB", (2, 2), color="blue").save(wrong_extension, format="PNG")

        oversized_image = directory / "oversized.png"
        image_buffer = BytesIO()
        Image.new("RGB", (2, 2), color="blue").save(image_buffer, format="PNG")
        image_bytes = image_buffer.getvalue()
        oversized_image.write_bytes(
            image_bytes + b"\0" * (MAX_IMAGE_BYTES + 1 - len(image_bytes))
        )
        self.assertGreater(oversized_image.stat().st_size, MAX_IMAGE_BYTES)

        return {
            "archivo de texto": invalid_text,
            "archivo vacío": empty_file,
            "extensión no permitida": wrong_extension,
            "imagen mayor a 10 MB": oversized_image,
        }

    def test_invalid_uploads_return_spanish_errors_and_remove_gradio_temporaries(
        self,
    ) -> None:
        expected_messages = {
            "archivo de texto": "El archivo no contiene una imagen JPEG o PNG válida.",
            "archivo vacío": "El archivo está vacío.",
            "extensión no permitida": "El archivo debe tener extensión .jpg, .jpeg o .png.",
            "imagen mayor a 10 MB": "La imagen supera el límite de 10 MB.",
        }

        with tempfile.TemporaryDirectory() as directory:
            for case, path in self.create_uploads(Path(directory)).items():
                with self.subTest(case=case):
                    result = self.client.predict(
                        handle_file(str(path)),
                        None,
                        "Nube reconocida",
                        api_name="/classify",
                    )

                    self.assertIn(expected_messages[case], result[0])
                    self.assertEqual(result[1:], (None, None))
                    remaining_files = {
                        item
                        for item in self.upload_directory.rglob("*")
                        if item.is_file()
                    }
                    self.assertEqual(remaining_files, set())

    def test_valid_upload_is_processed_and_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "cloud.png"
            Image.new("RGB", (2, 2), color="blue").save(image_path, format="PNG")

            result = self.client.predict(
                handle_file(str(image_path)),
                None,
                "Nube reconocida",
                api_name="/classify",
            )

        self.assertIn("Cúmulo (Cumulus)", result[0])
        self.assertEqual(result[1:], (None, None))
        remaining_files = {
            item for item in self.upload_directory.rglob("*") if item.is_file()
        }
        self.assertEqual(remaining_files, set())


if __name__ == "__main__":
    unittest.main()
