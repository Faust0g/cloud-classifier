import socket
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

from cloud_classifier.demo import CLOUD_IMAGES_PATH, create_demo
from cloud_classifier.prediction import classify_logits
from cloud_classifier.preprocessing import load_model_contract


class BrowserSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            cls.port = listener.getsockname()[1]

        cls.class_ids = tuple(
            item["id"] for item in load_model_contract()["classes"]
        )
        cls.decision = {"value": "firme"}

        def fake_classifier(image_path: str | Path):
            if cls.decision["value"] == "firme":
                logits = [8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
            else:
                logits = [
                    0.11,
                    0.10,
                    0.09,
                    0.08,
                    0.07,
                    0.06,
                    0.05,
                    0.04,
                    0.03,
                    0.02,
                    0.0,
                ]
            return classify_logits(logits, 1.0, 0.8, 0.85, cls.class_ids)

        cls.demo = create_demo(classifier=fake_classifier)
        cls.demo.launch(
            server_name="127.0.0.1",
            server_port=cls.port,
            quiet=True,
            prevent_thread_lock=True,
            allowed_paths=[str(CLOUD_IMAGES_PATH)],
        )
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.image_paths = []
        for index in range(2):
            image_path = Path(cls.temporary_directory.name) / f"smoke-{index}.png"
            Image.new("RGB", (32, 32), color=(100 + index, 150, 200)).save(image_path)
            cls.image_paths.append(image_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.demo.close()
        cls.temporary_directory.cleanup()

    def test_browser_firm_and_retry_results_render_without_frontend_errors(self) -> None:
        page_errors: list[str] = []
        console_errors: list[str] = []
        failed_component_assets: list[str] = []

        with sync_playwright() as playwright:
            with playwright.chromium.launch() as browser:
                page = browser.new_page()
                page.on("pageerror", lambda error: page_errors.append(str(error)))
                page.on(
                    "console",
                    lambda message: console_errors.append(message.text)
                    if message.type == "error"
                    else None,
                )
                page.on(
                    "response",
                    lambda response: failed_component_assets.append(response.url)
                    if "/gradio_api/custom_component/" in response.url
                    and response.status >= 400
                    else None,
                )
                page.goto(
                    f"http://127.0.0.1:{self.port}",
                    wait_until="domcontentloaded",
                )
                page.get_by_text("Loading...", exact=True).wait_for(
                    state="hidden",
                    timeout=15_000,
                )
                page.get_by_role(
                    "button",
                    name="Click to upload or drop files",
                ).wait_for(
                    state="visible",
                    timeout=15_000,
                )
                self.assertTrue(
                    page.get_by_text("Imagen del cielo", exact=True).is_visible()
                )
                self.assertTrue(
                    page.get_by_role(
                        "button",
                        name="Click to Access Webcam",
                    ).is_visible()
                )

                upload = page.locator('input[type="file"]').first
                classify_button = page.get_by_role("button", name="Clasificar")
                upload.set_input_files(str(self.image_paths[0]))
                classify_button.click()
                page.get_by_text("Probablemente es Cirro", exact=False).wait_for(
                    state="visible",
                    timeout=15_000,
                )
                self.assertTrue(page.get_by_text("Cautela:", exact=False).is_visible())
                self.assertTrue(
                    page.get_by_text("Otra referencia", exact=True).is_hidden()
                )

                self.decision["value"] = "repetir"
                page.reload(wait_until="domcontentloaded")
                page.get_by_text("Loading...", exact=True).wait_for(
                    state="hidden",
                    timeout=15_000,
                )
                upload.set_input_files(str(self.image_paths[1]))
                classify_button.click()
                page.get_by_text("Probá con otra foto", exact=False).wait_for(
                    state="visible",
                    timeout=15_000,
                )

                self.assertNotIn("Loading...", page.locator("body").inner_text())
                self.assertEqual(page_errors, [])
                self.assertEqual(console_errors, [])
                self.assertEqual(failed_component_assets, [])


if __name__ == "__main__":
    unittest.main()
