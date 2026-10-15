import hashlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from scripts import download_ccsn


def create_test_archive() -> bytes:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as output:
        for source_class in download_ccsn.CLASS_MAPPING:
            image = io.BytesIO()
            Image.new("RGB", (2, 2), color="blue").save(image, format="JPEG")
            output.writestr(
                f"CCSN_v2/{source_class}/{source_class}-001.jpg",
                image.getvalue(),
            )
        output.writestr(
            "CCSN_v2/Ct/._Ct-001.jpg",
            b"AppleDouble metadata is ignored",
        )
    return archive.getvalue()


class DownloadCcsnTests(unittest.TestCase):
    def test_extracts_all_classes_and_maps_contrail_to_no_nube(self) -> None:
        archive_bytes = create_test_archive()
        response = io.BytesIO(archive_bytes)
        expected_count = len(download_ccsn.CLASS_MAPPING)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(download_ccsn, "EXPECTED_IMAGE_COUNT", expected_count),
            patch.object(
                download_ccsn,
                "CCSN_FILE_MD5",
                hashlib.md5(archive_bytes, usedforsecurity=False).hexdigest(),
            ),
            patch.object(download_ccsn, "urlopen", return_value=response),
        ):
            output_dir = Path(directory) / "images"
            counts = download_ccsn.prepare_dataset(output_dir)

            self.assertEqual(set(counts), set(download_ccsn.CLASS_MAPPING))
            self.assertTrue((output_dir / "no_nube" / "Ct-001.jpg").is_file())
            self.assertFalse((output_dir / "Ct").exists())
            self.assertFalse((output_dir / "no_nube" / "._Ct-001.jpg").exists())
            for source_class, target_class in download_ccsn.CLASS_MAPPING.items():
                self.assertTrue(
                    (output_dir / target_class / f"{source_class}-001.jpg").is_file()
                )

    def test_rejects_archive_with_unexpected_image_count(self) -> None:
        archive_bytes = create_test_archive()
        response = io.BytesIO(archive_bytes)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(download_ccsn, "EXPECTED_IMAGE_COUNT", 999),
            patch.object(
                download_ccsn,
                "CCSN_FILE_MD5",
                hashlib.md5(archive_bytes, usedforsecurity=False).hexdigest(),
            ),
            patch.object(download_ccsn, "urlopen", return_value=response),
        ):
            with self.assertRaisesRegex(ValueError, "cantidad esperada"):
                download_ccsn.prepare_dataset(Path(directory) / "images")

    def test_rejects_archive_with_invalid_checksum(self) -> None:
        response = io.BytesIO(create_test_archive())

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(download_ccsn, "CCSN_FILE_MD5", "invalid"),
            patch.object(download_ccsn, "urlopen", return_value=response),
        ):
            with self.assertRaisesRegex(ValueError, "checksum MD5"):
                download_ccsn.prepare_dataset(Path(directory) / "images")


if __name__ == "__main__":
    unittest.main()
