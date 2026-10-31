from __future__ import annotations

from pathlib import Path
from typing import Callable

import gradio as gr
import yaml
from gradio.data_classes import FileData
from PIL import Image

from cloud_classifier.prediction import (
    InferenceModel,
    LogitClassification,
    format_prediction,
    load_inference_model,
)
from cloud_classifier.preprocessing import load_model_contract, preprocess_image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLOUDS_CONFIG_PATH = PROJECT_ROOT / "clouds.yaml"
CLOUD_IMAGES_PATH = PROJECT_ROOT / "app" / "assets" / "clouds"
MAX_IMAGE_BYTES = 10_000_000
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG"}
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
Classifier = Callable[[str | Path], LogitClassification]


class UploadPathFile(gr.File):
    def get_block_name(self) -> str:
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
        raise InvalidImageError(
            "El archivo no contiene una imagen JPEG o PNG válida."
        ) from error


def _remove_gradio_upload(image_path: str | Path) -> None:
    path = Path(image_path)
    try:
        cache_dir = Path(gr.utils.get_upload_folder()).resolve()
        cached_path = path.resolve(strict=True)
    except FileNotFoundError:
        return

    if cached_path.is_relative_to(cache_dir) and cached_path.is_file():
        cached_path.unlink()


def _load_cloud_config(contract_class_ids: tuple[str, ...]) -> dict:
    try:
        config = yaml.safe_load(CLOUDS_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeError(f"No se pudo cargar la configuración: {CLOUDS_CONFIG_PATH}.") from error
    if (
        not isinstance(config, dict)
        or not isinstance(config.get("clases"), list)
        or not isinstance(config.get("no_nube"), dict)
        or not isinstance(config["no_nube"].get("mensaje"), str)
        or not isinstance(config.get("cautela_general"), str)
        or not config["cautela_general"].strip()
    ):
        raise ValueError("clouds.yaml está incompleto.")

    class_ids = tuple(
        cloud.get("id") for cloud in config["clases"] if isinstance(cloud, dict)
    )
    if class_ids != contract_class_ids[:-1]:
        raise ValueError("El orden de clases de clouds.yaml no coincide con el contrato.")
    for cloud in config["clases"]:
        image = cloud.get("imagen")
        if (
            not isinstance(cloud.get("nombre"), str)
            or not isinstance(cloud.get("descripcion"), str)
            or not isinstance(cloud.get("altura"), str)
            or not isinstance(cloud.get("como_reconocerla"), list)
            or not cloud["como_reconocerla"]
            or any(not isinstance(cue, str) or not cue.strip() for cue in cloud["como_reconocerla"])
            or not isinstance(cloud.get("asociaciones"), str)
            or not cloud["asociaciones"].strip()
            or not isinstance(cloud.get("fuentes"), list)
            or not cloud["fuentes"]
            or any(
                not isinstance(source, dict)
                or not isinstance(source.get("titulo"), str)
                or not source["titulo"].strip()
                or not isinstance(source.get("url"), str)
                or not source["url"].strip()
                for source in cloud["fuentes"]
            )
            or not isinstance(image, dict)
            or not isinstance(image.get("archivo"), str)
            or Path(image["archivo"]).name != image["archivo"]
            or not (CLOUD_IMAGES_PATH / image["archivo"]).is_file()
        ):
            raise ValueError(f"Faltan datos de presentación para {cloud.get('id')}.")

    show_percentage = config.get("mostrar_porcentaje_par", False)
    if not isinstance(show_percentage, bool):
        raise ValueError("mostrar_porcentaje_par debe ser true o false.")
    distinct_pairs = config.get("eventos_distintos", [])
    if not isinstance(distinct_pairs, list) or any(
        not isinstance(pair, dict)
        or not isinstance(pair.get("clases"), list)
        or len(pair["clases"]) != 2
        or pair["clases"][0] == pair["clases"][1]
        or any(not isinstance(class_id, str) for class_id in pair["clases"])
        or any(class_id not in contract_class_ids[:-1] for class_id in pair["clases"])
        or not isinstance(pair.get("nota"), str)
        or not pair["nota"].strip()
        for pair in distinct_pairs
    ):
        raise ValueError("Cada evento_distinto requiere dos clases y una nota.")
    return config


def process_image(
    upload_path: str | None,
    camera_path: str | None,
    classifier: Classifier,
    cloud_config: dict,
    percentage_calibrated: bool = False,
) -> tuple[str, str | None, str | None]:
    image_path = upload_path or camera_path
    if image_path is None:
        return "⚠️ Subí una imagen o tomá una foto con la cámara.", None, None

    try:
        validate_image(image_path)
        classification = classifier(image_path)
        message, references = format_prediction(
            classification,
            cloud_config,
            CLOUD_IMAGES_PATH,
            show_pair_percentage=(
                cloud_config["mostrar_porcentaje_par"] and percentage_calibrated
            ),
        )
    except InvalidImageError as error:
        return f"⚠️ {error}", None, None
    except ValueError as error:
        return f"⚠️ {error}", None, None
    finally:
        if upload_path is not None:
            _remove_gradio_upload(upload_path)
        if camera_path is not None:
            _remove_gradio_upload(camera_path)

    return message, references[0], references[1]


def create_demo(classifier: Classifier | None = None) -> gr.Blocks:
    model: InferenceModel | None = None
    if classifier is None:
        model = load_inference_model()
        classifier = model.classify
        class_ids = model.class_ids
    else:
        class_ids = tuple(
            item["id"] for item in load_model_contract()["classes"]
        )
    cloud_config = _load_cloud_config(class_ids)
    percentage_calibrated = (
        model is not None
        and model.calibration["porcentaje_par_habilitado"] is True
    )

    def classify(
        upload_path: str | None,
        camera_path: str | None,
    ) -> tuple[str, dict, dict, None, None]:
        message, first_reference, second_reference = process_image(
            upload_path,
            camera_path,
            classifier,
            cloud_config,
            percentage_calibrated,
        )
        return (
            message,
            gr.update(value=first_reference, visible=first_reference is not None),
            gr.update(value=second_reference, visible=second_reference is not None),
            None,
            None,
        )

    with gr.Blocks(
        analytics_enabled=False,
        title="Clasificador de nubes",
        delete_cache=(60, 60),
    ) as demo:
        gr.Markdown(
            "# Clasificador de nubes\n"
            "Cargá una imagen del cielo o tomá una foto. "
            "La imagen se procesa para esta solicitud y no se guarda."
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
        submit = gr.Button("Clasificar", variant="primary")
        result = gr.Markdown()
        with gr.Row():
            first_reference = gr.Image(
                label="Referencia",
                type="filepath",
                interactive=False,
                visible=False,
                height=180,
            )
            second_reference = gr.Image(
                label="Otra referencia",
                type="filepath",
                interactive=False,
                visible=False,
                height=180,
            )
        submit.click(
            fn=classify,
            inputs=[upload, camera],
            outputs=[result, first_reference, second_reference, upload, camera],
            api_name="classify",
        )
        gr.Markdown(
            "La cámara requiere permiso del navegador y un contexto seguro "
            "(localhost o HTTPS). Si no está disponible, podés cargar un archivo."
        )
    return demo


def main() -> None:
    create_demo().launch(
        server_name="127.0.0.1",
        share=False,
        allowed_paths=[str(CLOUD_IMAGES_PATH)],
    )
