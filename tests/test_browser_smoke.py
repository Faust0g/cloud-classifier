import socket
import unittest

from playwright.sync_api import sync_playwright

from cloud_classifier.demo import create_demo


class BrowserSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
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

    @classmethod
    def tearDownClass(cls) -> None:
        cls.demo.close()

    def test_upload_camera_and_scenario_controls_render_without_frontend_errors(
        self,
    ) -> None:
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
                self.assertTrue(
                    page.get_by_role(
                        "radio",
                        name="Nube reconocida",
                    ).is_visible()
                )
                self.assertNotIn("Loading...", page.locator("body").inner_text())
                self.assertEqual(page_errors, [])
                self.assertEqual(console_errors, [])
                self.assertEqual(failed_component_assets, [])


if __name__ == "__main__":
    unittest.main()
