# Cloud Classifier

## Objetivo

Proyecto para clasificar imágenes de nubes mediante técnicas de machine learning.

## Vocabulario

- **Dataset:** conjunto de imágenes utilizado para entrenar o evaluar el modelo.
- **Clase:** tipo de nube.
- **Certeza:** probabilidad de que esa nube sea del tipo propuesto.
- **Significado:** lo que suele indicar ese tipo de nube.
- **Entrenamiento:** proceso de ajuste del modelo a partir de datos.
- **Inferencia:** uso de un modelo entrenado para predecir la clase de una imagen.
- **ONNX:** formato interoperable para representar y ejecutar modelos de machine learning.
- **Gradio:** herramienta para crear una interfaz web para probar el clasificador.

## Estructura

- `data/`: datos locales, excluidos de Git.
- `docs/adr/`: decisiones de arquitectura.
- `src/clouds/`: código de datos, entrenamiento, evaluación y predicción.
- `app/`: interfaz de usuario.
- `tests/`: pruebas automatizadas.
- `clouds.yaml`: definición de las clases y sus significados.
