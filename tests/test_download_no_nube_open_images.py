import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.download_no_nube_open_images import (
    CC_BY_2_LICENSE,
    DEFAULT_SPLIT,
    EXCLUDED_LABEL_NAMES,
    _collect_candidates,
    _download_image,
    _load_image_metadata,
    _resolve_excluded_label_ids,
    main,
)

CLASS_NAMES = {
    "/m/01bqvp": "Sky",
    "/m/0csby": "Cloud",
    "/m/01ctsf": "Atmosphere",
    "/m/0866r": "Weather",
    "/m/0g2z8": "Fog",
    "/m/03xxq7": "Haze",
    "/m/01bqwh": "Mist",
    "/m/0z71l": "Storm",
    "/m/0jb2l": "Thunderstorm",
    "/m/06mb1": "Rain",
    "/m/01b2w5": "Sunset",
    "/m/0d1n2": "Horizon",
    "/m/05czz6l": "Airplane",
    "/m/015p6": "Bird",
    "/m/0k5j": "Aircraft",
    "/m/0fx72": "Airship",
    "/m/0d3qh": "Biplane",
    "/m/0dms2": "Blimp",
    "/m/077hhl": "Cargo aircraft",
    "/m/0g2bc": "Drone",
    "/m/02p18m": "Experimental aircraft",
    "/m/02yjc": "Fighter aircraft",
    "/m/0cmf2": "Fixed-wing aircraft",
    "/m/03m3mlb": "Floatplane",
    "/m/05h5q_s": "Glider",
    "/m/018rc1": "Ground attack aircraft",
    "/m/09ct_": "Helicopter",
    "/m/09j06": "Hot air balloon",
    "/m/015y8h": "Jet aircraft",
    "/m/07txbq": "Light aircraft",
    "/m/0f5lx": "Military aircraft",
    "/m/0275lwb": "Military helicopter",
    "/m/03hfwtf": "Military transport aircraft",
    "/m/0j_5b": "Monoplane",
    "/m/0272thc": "Motor glider",
    "/m/0dp7vs": "Narrow-body aircraft",
    "/m/0gd_ym": "Powered hang glider",
    "/m/0bp37nz": "Propeller-driven aircraft",
    "/m/053scx": "Radio-controlled aircraft",
    "/m/074l3b": "Radio-controlled helicopter",
    "/m/09g1yy": "Rigid airship",
    "/m/03w2bb": "Rocket-powered aircraft",
    "/m/01qnxs": "Seaplane",
    "/m/02bl11": "Spaceplane",
    "/m/025sb2w": "Supersonic aircraft",
    "/m/0h8mhmy": "Toy airplane",
    "/m/09n5xs": "Trijet",
    "/m/03mmj9": "Triplane",
    "/m/09n60k": "Twinjet",
    "/m/018rqw": "Wide-body aircraft",
    "/m/01g5v": "Animal",
}


class CollectCandidatesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.class_names = CLASS_NAMES

    def collect(self, rows: str, limit: int = 20) -> list[tuple[str, str]]:
        return _collect_candidates(
            io.StringIO(rows), self.class_names, limit=limit, seed=42
        )

    def test_selects_human_verified_positive_nonexcluded_labels(self) -> None:
        rows = (
            "ImageID,Source,LabelName,Confidence\n"
            "good,verification,/m/01g5v,1\n"
            "bad-cloud,verification,/m/01g5v,1\n"
            "bad-cloud,verification,/m/0csby,1\n"
            "bad-sky,verification,/m/01bqvp,1\n"
            "machine,machine,/m/01g5v,1\n"
            "negative,verification,/m/01g5v,0\n"
        )

        self.assertEqual(self.collect(rows), [("good", "/m/01g5v")])

    def test_any_excluded_positive_label_disqualifies_image(self) -> None:
        rows = (
            "ImageID,Source,LabelName,Confidence\n"
            "first,verification,/m/01g5v,1\n"
            "second,verification,/m/01g5v,1\n"
            "first,crowdsource-verification,/m/01ctsf,1\n"
        )

        self.assertEqual(self.collect(rows), [("second", "/m/01g5v")])

    def test_limit_and_seed_produce_repeatable_bounded_sample(self) -> None:
        rows = "ImageID,Source,LabelName,Confidence\n" + "".join(
            f"image-{i},verification,/m/01g5v,1\n" for i in range(100)
        )

        sample = self.collect(rows, limit=8)

        self.assertEqual(len(sample), 8)
        self.assertEqual(sample, self.collect(rows, limit=8))


class LoadImageMetadataTests(unittest.TestCase):
    def test_only_keeps_cc_by_rows_with_attribution_metadata(self) -> None:
        rows = (
            "ImageID,OriginalURL,OriginalLandingURL,License,Author,Title\n"
            f"valid,https://image,https://source,{CC_BY_2_LICENSE},Creator,Title\n"
            "other-license,https://image,https://source,"
            "https://creativecommons.org/licenses/by-nc/2.0/,Creator,Title\n"
            f"no-author,https://image,https://source,{CC_BY_2_LICENSE},,Title\n"
        )

        metadata = _load_image_metadata(
            io.StringIO(rows), {"valid", "other-license", "no-author"}, "metadata"
        )

        self.assertEqual(set(metadata), {"valid"})

    def test_image_download_uses_public_s3_bucket(self) -> None:
        response = io.BytesIO(b"\xff\xd8\xfftest")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "image.jpg"
            with patch(
                "scripts.download_no_nube_open_images.urlopen",
                return_value=response,
            ) as open_url:
                _download_image("image-id", "validation", destination)

            self.assertEqual(
                open_url.call_args.args[0].full_url,
                "https://open-images-dataset.s3.amazonaws.com/validation/image-id.jpg",
            )
            self.assertEqual(destination.read_bytes(), b"\xff\xd8\xfftest")


class ExcludedLabelsTests(unittest.TestCase):
    def test_every_excluded_name_resolves_to_one_class_id(self) -> None:
        label_ids = _resolve_excluded_label_ids(EXCLUDED_LABEL_NAMES, CLASS_NAMES)

        self.assertEqual(len(label_ids), len(EXCLUDED_LABEL_NAMES))
        self.assertEqual(
            {CLASS_NAMES[label_id] for label_id in label_ids},
            set(EXCLUDED_LABEL_NAMES),
        )

    def test_missing_excluded_name_aborts_with_explicit_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Sunset"):
            _resolve_excluded_label_ids(("Sky", "Sunset"), {"/m/sky": "Sky"})

    def test_script_stops_if_official_class_descriptions_omit_a_name(self) -> None:
        rows = "LabelName,DisplayName\n/m/01bqvp,Sky\n"
        with (
            patch(
                "scripts.download_no_nube_open_images._open_csv",
                return_value=io.StringIO(rows),
            ),
            patch(
                "scripts.download_no_nube_open_images.sys.stderr",
                new_callable=io.StringIO,
            ) as stderr,
        ):
            result = main([])

        self.assertEqual(result, 1)
        self.assertIn("Cloud", stderr.getvalue())
        self.assertIn("Sunset", stderr.getvalue())


class CommandLineDefaultsTests(unittest.TestCase):
    def test_validation_split_is_the_default(self) -> None:
        with patch("scripts.download_no_nube_open_images.download_subset") as download:
            download.return_value = (0, 0)

            self.assertEqual(main([]), 0)

        self.assertEqual(DEFAULT_SPLIT, "validation")
        download.assert_called_once_with(
            "validation",
            1000,
            42,
            Path("data") / "no_nube" / "open-images-v7" / "validation",
            Path("docs")
            / "datasets"
            / "no_nube"
            / "open-images-v7"
            / "validation",
        )


if __name__ == "__main__":
    unittest.main()
