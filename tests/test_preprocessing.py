import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from cloud_classifier.demo import preprocess_image as app_preprocess_image
from cloud_classifier.preprocessing import (
    load_model_contract,
    preprocess_image,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CLASS_IDS = [
    "cirrus",
    "cirrocumulus",
    "cirrostratus",
    "altocumulus",
    "altostratus",
    "nimbostratus",
    "stratocumulus",
    "stratus",
    "cumulus",
    "cumulonimbus",
    "no_nube",
]
EXPECTED_CLASS_CODES = [
    "Ci",
    "Cc",
    "Cs",
    "Ac",
    "As",
    "Ns",
    "Sc",
    "St",
    "Cu",
    "Cb",
    None,
]


class ModelContractTests(unittest.TestCase):
    def test_contract_has_canonical_classes_and_explicit_logits(self) -> None:
        contract = load_model_contract()
        class_ids = [entry["id"] for entry in contract["classes"]]
        class_codes = [entry["code"] for entry in contract["classes"]]

        self.assertEqual(class_ids, EXPECTED_CLASS_IDS)
        self.assertEqual(class_codes, EXPECTED_CLASS_CODES)
        self.assertEqual(contract["output"]["values"], "logits")
        self.assertEqual(contract["output"]["shape"], ["batch", 11])
        self.assertEqual(contract["output"]["class_axis"], 1)
        self.assertEqual(
            contract["output"]["confidence"],
            "softmax_over_logits_along_class_axis",
        )

    def test_contract_ids_and_codes_match_clouds_yaml(self) -> None:
        contract = load_model_contract()
        cloud_config = yaml.safe_load(
            (PROJECT_ROOT / "clouds.yaml").read_text(encoding="utf-8")
        )
        yaml_classes = cloud_config["clases"]

        self.assertEqual(
            [entry["id"] for entry in contract["classes"][:-1]],
            [entry["id"] for entry in yaml_classes],
        )
        self.assertEqual(
            [entry["code"] for entry in contract["classes"][:-1]],
            [entry["codigo"] for entry in yaml_classes],
        )

    def test_contract_leaves_threshold_pending_with_calibration_provenance(self) -> None:
        threshold = load_model_contract()["confidence_threshold"]

        self.assertEqual(threshold["status"], "pending_calibration")
        self.assertIsNone(threshold["value"])
        self.assertEqual(threshold["provenance"]["partition"], "calibration")
        self.assertIsNone(threshold["provenance"]["target_precision"])

    def test_contract_preprocessing_matches_required_parameters(self) -> None:
        model_input = load_model_contract()["input"]

        self.assertEqual(model_input["orientation"], "apply_exif")
        self.assertEqual(model_input["color_mode"], "RGB")
        self.assertEqual(model_input["resize"]["short_side"], 256)
        self.assertTrue(model_input["resize"]["preserve_aspect_ratio"])
        self.assertEqual(model_input["resize"]["method"], "Pillow_BILINEAR")
        self.assertEqual(model_input["crop"]["method"], "center")
        self.assertEqual(model_input["crop"]["size"], [224, 224])
        self.assertEqual(model_input["tensor"]["dtype"], "float32")
        self.assertEqual(model_input["tensor"]["layout"], "NCHW")
        self.assertEqual(model_input["tensor"]["shape"], ["batch", 3, 224, 224])
        self.assertEqual(
            model_input["tensor"]["pixel_range_before_normalization"], [0.0, 1.0]
        )
        self.assertEqual(
            model_input["normalization"]["mean"], [0.485, 0.456, 0.406]
        )
        self.assertEqual(
            model_input["normalization"]["std"], [0.229, 0.224, 0.225]
        )
        self.assertEqual(
            model_input["normalization"]["source"],
            "ImageNet standard values; provisional until model selection",
        )
        self.assertEqual(load_model_contract()["augmentation"]["inference"], "none")
        self.assertFalse(load_model_contract()["augmentation"]["vertical_flip"])


class PreprocessImageTests(unittest.TestCase):
    def test_output_is_contiguous_float32_nchw_with_batch_dimension(self) -> None:
        image = Image.new("RGB", (640, 480), color=(128, 64, 32))
        tensor = preprocess_image(image)

        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        expected = (
            np.array([128, 64, 32], dtype=np.float32) / 255.0
            - np.array([0.485, 0.456, 0.406], dtype=np.float32)
        ) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        np.testing.assert_allclose(tensor[0, :, 112, 112], expected, atol=1e-6)

    def test_preserves_aspect_ratio_before_center_crop(self) -> None:
        image = Image.new("RGB", (400, 200), color="white")
        ImageDraw.Draw(image).ellipse((160, 60, 240, 140), fill="black")

        tensor = preprocess_image(image)
        dark_pixels = tensor[0, 0] < -1.0
        rows, columns = np.where(dark_pixels)
        width = columns.max() - columns.min() + 1
        height = rows.max() - rows.min() + 1

        self.assertLessEqual(abs(width - height), 2)
        self.assertLessEqual(abs(columns.mean() - 111.5), 1.0)
        self.assertLessEqual(abs(rows.mean() - 111.5), 1.0)

    def test_applies_exif_orientation_before_preprocessing(self) -> None:
        original = Image.new("RGB", (400, 200), color="black")
        draw = ImageDraw.Draw(original)
        draw.rectangle((0, 0, 199, 199), fill="red")
        draw.rectangle((200, 0, 399, 199), fill="blue")
        exif = Image.Exif()
        exif[274] = 6

        with tempfile.TemporaryDirectory() as directory:
            oriented_path = Path(directory) / "oriented.png"
            original.save(oriented_path, exif=exif)
            expected_orientation = original.transpose(Image.Transpose.ROTATE_270)

            actual = preprocess_image(oriented_path)
            expected = preprocess_image(expected_orientation)

        np.testing.assert_array_equal(actual, expected)

    def test_app_and_evaluation_share_the_same_preprocessor_for_fixed_image(
        self,
    ) -> None:
        self.assertIs(app_preprocess_image, preprocess_image)
        image = Image.new("RGB", (320, 512), color=(12, 100, 240))

        app_tensor = app_preprocess_image(image)
        evaluation_tensor = preprocess_image(image)

        np.testing.assert_array_equal(app_tensor, evaluation_tensor)


if __name__ == "__main__":
    unittest.main()
