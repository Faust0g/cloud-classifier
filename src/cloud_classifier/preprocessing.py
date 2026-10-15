from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

MODEL_CONTRACT_PATH = Path(__file__).with_name("model_contract.json")


@lru_cache(maxsize=1)
def load_model_contract() -> dict[str, Any]:
    try:
        contract = json.loads(MODEL_CONTRACT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"No se pudo cargar el contrato del modelo en {MODEL_CONTRACT_PATH}."
        ) from error

    if not isinstance(contract, dict) or contract.get("contract_version") != 1:
        raise ValueError("La versión del contrato del modelo no es compatible.")
    _validate_input_contract(contract.get("input"))
    return contract


def _validate_input_contract(input_config: Any) -> None:
    if not isinstance(input_config, dict):
        raise ValueError("El contrato del modelo no define una entrada válida.")

    resize = input_config.get("resize")
    crop = input_config.get("crop")
    tensor = input_config.get("tensor")
    normalization = input_config.get("normalization")
    if (
        input_config.get("orientation") != "apply_exif"
        or input_config.get("color_mode") != "RGB"
        or not isinstance(resize, dict)
        or resize.get("short_side") != 256
        or resize.get("preserve_aspect_ratio") is not True
        or resize.get("rounding") != "nearest_half_up"
        or resize.get("method") != "Pillow_BILINEAR"
        or resize.get("antialias") is not True
        or not isinstance(crop, dict)
        or crop.get("method") != "center"
        or crop.get("size") != [224, 224]
        or crop.get("odd_remainder") != "discard_right_and_bottom"
        or not isinstance(tensor, dict)
        or tensor.get("dtype") != "float32"
        or tensor.get("layout") != "NCHW"
        or tensor.get("shape") != ["batch", 3, 224, 224]
        or tensor.get("channel_order") != "RGB"
        or tensor.get("pixel_range_before_normalization") != [0.0, 1.0]
        or not isinstance(normalization, dict)
    ):
        raise ValueError("El preprocesamiento del contrato no es compatible.")

    mean = normalization.get("mean")
    std = normalization.get("std")
    if (
        not isinstance(mean, list)
        or not isinstance(std, list)
        or len(mean) != 3
        or len(std) != 3
        or any(not isinstance(value, (int, float)) for value in mean + std)
        or any(value <= 0 for value in std)
    ):
        raise ValueError("Media y desvío deben contener tres valores válidos.")


def preprocess_image(
    image: Image.Image | str | Path,
    contract: dict[str, Any] | None = None,
) -> np.ndarray:
    """Convert an image to a normalized float32 NCHW batch per the model contract."""
    model_contract = load_model_contract() if contract is None else contract
    input_config = model_contract["input"]
    resize_config = input_config["resize"]
    crop_config = input_config["crop"]
    normalization = input_config["normalization"]

    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            pixels = _prepare_rgb_pixels(source, input_config, resize_config, crop_config)
    else:
        pixels = _prepare_rgb_pixels(image, input_config, resize_config, crop_config)

    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    normalized = (pixels - mean) / std
    tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
    return np.ascontiguousarray(tensor, dtype=np.float32)


def _prepare_rgb_pixels(
    source: Image.Image,
    input_config: dict[str, Any],
    resize_config: dict[str, Any],
    crop_config: dict[str, Any],
) -> np.ndarray:
    oriented = ImageOps.exif_transpose(source)
    with oriented:
        rgb = oriented.convert(input_config["color_mode"])
    with rgb:
        width, height = rgb.size
        short_side = resize_config["short_side"]
        scale = short_side / min(width, height)
        resized_size = (
            math.floor(width * scale + 0.5),
            math.floor(height * scale + 0.5),
        )
        resized = rgb.resize(resized_size, Image.Resampling.BILINEAR)
    with resized:
        crop_width, crop_height = crop_config["size"]
        left = (resized.width - crop_width) // 2
        top = (resized.height - crop_height) // 2
        cropped = resized.crop((left, top, left + crop_width, top + crop_height))
    with cropped:
        return np.asarray(cropped, dtype=np.float32) / np.float32(255.0)
