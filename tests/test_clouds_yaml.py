import unittest
from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).resolve().parents[1] / "clouds.yaml"
EXPECTED_CLASS_IDS = {
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
}
REQUIRED_CLASS_FIELDS = {
    "id",
    "codigo",
    "nombre_cientifico",
    "altura",
    "descripcion",
    "como_reconocerla",
    "asociaciones",
    "confundible_con",
    "imagen",
    "fuentes",
}
MAX_TEXT_LENGTH = 500


def _text_values(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested_value in value.values():
            yield from _text_values(nested_value)
    elif isinstance(value, list):
        for nested_value in value:
            yield from _text_values(nested_value)


class CloudsYamlTests(unittest.TestCase):
    def load_config(self) -> dict:
        try:
            return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            self.fail(f"clouds.yaml no es YAML válido: {error}")

    def test_cloud_content_matches_contract_and_has_bounded_complete_fields(
        self,
    ) -> None:
        config = self.load_config()
        self.assertIsInstance(config, dict)
        classes = config.get("clases")
        self.assertIsInstance(classes, list)
        self.assertEqual(len(classes), len(EXPECTED_CLASS_IDS))

        class_ids = [cloud.get("id") for cloud in classes]
        self.assertEqual(set(class_ids), EXPECTED_CLASS_IDS)
        self.assertEqual(len(class_ids), len(set(class_ids)), "Hay IDs duplicados.")

        for cloud in classes:
            with self.subTest(class_id=cloud.get("id")):
                missing_fields = REQUIRED_CLASS_FIELDS - cloud.keys()
                self.assertFalse(
                    missing_fields,
                    f"Faltan campos: {', '.join(sorted(missing_fields))}",
                )
                for field in (
                    "codigo",
                    "nombre_cientifico",
                    "altura",
                    "descripcion",
                    "asociaciones",
                ):
                    self.assertIsInstance(cloud[field], str)
                    self.assertTrue(cloud[field].strip(), f"{field} está vacío.")

                recognition_cues = cloud["como_reconocerla"]
                self.assertIsInstance(recognition_cues, list)
                self.assertTrue(recognition_cues, "Faltan señales de reconocimiento.")
                self.assertTrue(
                    all(
                        isinstance(cue, str) and cue.strip()
                        for cue in recognition_cues
                    ),
                    "Las señales de reconocimiento deben ser textos no vacíos.",
                )

                confusing_classes = cloud["confundible_con"]
                self.assertIsInstance(confusing_classes, list)
                for confusing_class in confusing_classes:
                    self.assertIn(
                        confusing_class,
                        EXPECTED_CLASS_IDS,
                        f"ID confundible inexistente: {confusing_class!r}",
                    )

                image = cloud["imagen"]
                self.assertIsInstance(image, dict)
                for field in ("archivo", "credito"):
                    self.assertIsInstance(image.get(field), str)
                    self.assertTrue(image[field].strip(), f"imagen.{field} está vacío.")

                sources = cloud["fuentes"]
                self.assertIsInstance(sources, list)
                self.assertTrue(sources, "Cada clase debe tener al menos una fuente.")
                for source in sources:
                    self.assertIsInstance(source, dict)
                    for field in ("titulo", "url"):
                        self.assertIsInstance(source.get(field), str)
                        self.assertTrue(
                            source[field].strip(),
                            f"El campo de fuente {field} está vacío.",
                        )

        oversized_text = [
            (index, len(text))
            for index, text in enumerate(_text_values(config))
            if len(text) > MAX_TEXT_LENGTH
        ]
        self.assertFalse(
            oversized_text,
            f"Hay textos que superan {MAX_TEXT_LENGTH} caracteres: {oversized_text}",
        )


if __name__ == "__main__":
    unittest.main()
