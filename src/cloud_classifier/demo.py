from __future__ import annotations

from pathlib import Path

import gradio as gr
from gradio.data_classes import FileData
from PIL import Image

from cloud_classifier.prediction import PredictionResult, format_prediction
from cloud_classifier.simulated_predictor import (
    SIMULATED_PREDICTIONS,
    simulate_prediction,
)

MAX_IMAGE_BYTES = 10_000_000
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG"}
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class UploadPathFile(gr.File):
    def get_block_name(self) -> str:
        # Use Gradio's built-in File frontend, not a custom component bundle.
        return "file"

    def _process_single_file(self, file: FileData) -> str:
        return file.path


class InvalidImageError(ValueError):
    pass


def validate_image(image_path: str | Path) -> None:
    path = Path(image_path)
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise InvalidImageError("El archivo debe tener extensión .jpg, .jpeg o .png.")

    try:
        size = path.stat().st_size
    except OSError as error:
        raise InvalidImageError("No se pudo leer la imagen.") from error

    if size > MAX_IMAGE_BYTES:
        raise InvalidImageError("La imagen supera el límite de 10 MB.")
    if size == 0:
        raise InvalidImageError("El archivo está vacío.")

    try:
        with Image.open(path) as image:
            if image.format not in SUPPORTED_IMAGE_FORMATS:
                raise InvalidImageError("El archivo debe ser una imagen JPEG o PNG.")
            image.verify()
    except InvalidImageError:
        raise
    except (Image.DecompressionBombError, OSError, SyntaxError, ValueError) as error:
        raise InvalidImageError("El archivo no contiene una imagen JPEG o PNG válida.") from error


def _remove_gradio_upload(image_path: str | Path) -> None:
    path = Path(image_path)
    try:
        cache_dir = Path(gr.utils.get_upload_folder()).resolve()
        cached_path = path.resolve(strict=True)
    except FileNotFoundError:
        return

    if cached_path.is_relative_to(cache_dir) and cached_path.is_file():
        cached_path.unlink()


def process_image(
    upload_path: str | None,
    camera_path: str | None,
    scenario: str,
) -> tuple[str, None, None]:
    image_path = upload_path or camera_path
    if image_path is None:
        return "⚠️ Subí una imagen o tomá una foto con la cámara.", None, None

    try:
        validate_image(image_path)
        prediction: PredictionResult = simulate_prediction(scenario)
    except InvalidImageError as error:
        return f"⚠️ {error}", None, None
    except ValueError as error:
        return f"⚠️ {error}", None, None
    finally:
        if upload_path is not None:
            _remove_gradio_upload(upload_path)
        if camera_path is not None:
            _remove_gradio_upload(camera_path)

    return format_prediction(prediction), None, None


def create_demo() -> gr.Blocks:
    with gr.Blocks(
        analytics_enabled=False,
        title="Cloud Classifier — demostración",
        delete_cache=(60, 60),
    ) as demo:
        gr.Markdown(
            "# Clasificador de nubes\n"
            "Elegí o capturá una imagen y probá los distintos estados del flujo. "
            "Los resultados son simulados: todavía no hay un modelo conectado. "
            "Las imágenes se procesan para esta solicitud y no se guardan."
        )
        with gr.Row():
            upload = UploadPathFile(
                label="Imagen del cielo",
                type="filepath",
                file_count="single",
                file_types=None,
            )
            camera = gr.Image(
                label="Tomar una foto",
                sources=["webcam"],
                type="filepath",
                image_mode=None,
                format="png",
            )
            scenario = gr.Radio(
                choices=list(SIMULATED_PREDICTIONS),
                value="Nube reconocida",
                label="Escenario para probar",
                info="Solo controla el resultado simulado; no depende del contenido de la foto.",
            )
        submit = gr.Button("Clasificar", variant="primary")
        result = gr.Markdown()
        submit.click(
            fn=process_image,
            inputs=[upload, camera, scenario],
            outputs=[result, upload, camera],
            api_name="classify",
        )
        gr.Markdown(
            "La cámara requiere permiso del navegador y un contexto seguro "
            "(localhost o HTTPS). Si no está disponible, podés cargar un archivo."
        )
    return demo


def main() -> None:
    create_demo().launch(server_name="127.0.0.1", share=False)
