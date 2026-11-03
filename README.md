# Cloud Classifier

Aplicación local en español para identificar nubes en imágenes y explicar qué condiciones suelen asociarse a ellas. La foto se procesa localmente y no se conserva.

Clasifica una foto del cielo en uno de los 10 géneros de nubes de la Organización Meteorológica Mundial (OMM) o en `no_nube`. Muestra la predicción principal, hasta dos alternativas, una explicación breve con fuentes y una foto de referencia. Si el modelo duda entre dos clases lo dice, y si la evidencia es baja pide otra foto.

> **No es un pronóstico del tiempo.** Las asociaciones meteorológicas de cada nube son información general sobre ese tipo de nube, no un aviso sobre el cielo de la foto.

Es un proyecto de aprendizaje y portfolio.

## Resultados

El modelo es MobileNetV3 preentrenado (transfer learning). 
Estos números salen de una evaluación final que se ejecutó una sola vez, con la calibración ya fijada:

| Conjunto | Fotos | Top-1 | Top-2 | Precisión de los resultados "firmes" |
|---|---:|---:|---:|---:|
| Prueba (CCSN, Open Images, Sociedad) | 411 | 51,1 % (46,3–55,9) | 69,8 % (65,2–74,1) | 59,6 % (51,8–67,0) |
| Mis fotos | 122 | 36,9 % (28,8–45,7) | 51,6 % (42,9–60,3) | 60,5 % (45,6–73,6) |

Entre paréntesis, intervalos de confianza de 95 % (Wilson).

Lo que significan:

- Un resultado "firme" acierta aproximadamente 6 de cada 10 veces. El objetivo de calibración era 75 % y no se cumplió en la prueba: el umbral se eligió con el grupo de calibración y tiende a ser optimista. Por eso la app no muestra un porcentaje de confianza en los resultados firmes.
- La clase correcta está entre las dos primeras unas 7 de cada 10 veces (en fotos propias, 5 de cada 10). Por eso existe el resultado de par ("puede ser A o B").
- Las clases más débiles son `stratus` y `stratocumulus` (en fotos propias, 1 de 17 y 0 de 14). Se confunden entre sí y con `nimbostratus`.
- Con fotos personales el modelo rinde peor que con las de prueba.

## Ejecutar localmente

Requiere Python 3.14 y [uv](https://docs.astral.sh/uv/).

```powershell
uv sync
uv run cloud-classifier
```

Abrí la dirección local que imprime Gradio. La interfaz permite cargar JPEG/PNG de hasta 10 MB o tomar una foto con la cámara. La cámara requiere permiso del navegador y contexto seguro; localhost sirve para desarrollo. Si se deniega el permiso o no hay cámara, se puede cargar un archivo.

La analítica de uso de Gradio está desactivada; las imágenes no se usan para telemetría.

La app necesita un modelo exportado que **no viene en el repositorio** (ver [Contrato del modelo](#contrato-del-modelo)).

## Cómo decide la app

1. El modelo devuelve logits crudos para las 11 clases.
2. Se dividen por una temperatura `T` (escalado de temperatura, calibrado con el grupo de calibración) y se aplica softmax.
3. Se clasifica en uno de tres resultados:
   - **Firme:** la confianza calibrada de la clase principal supera un umbral. Se muestra "Probablemente es X".
   - **Par:** no es firme, pero la suma de las dos primeras probabilidades es alta. Se muestra "Puede ser A o B", con la descripción de ambas.
   - **Repetir foto:** la evidencia es baja. Si la predicción es `no_nube`, el mensaje es distinto.
4. `T` y los umbrales viven en un archivo versionado junto con los hashes del modelo y de las particiones. La app no arranca si algo no coincide.

El porcentaje del par está apagado por defecto: su verificación de calibración fue débil (un solo cajón con 64 fotos) y el resultado firme no alcanzó su objetivo en la prueba.

## Fotos de referencia de nubes

Las fotos que acompañan las explicaciones se guardan en `app/assets/clouds/`, una por género OMM. Usá el código de clase de `clouds.yaml` como nombre en minúsculas y JPEG o PNG, por ejemplo `ci.jpg` para Cirrus. Estas imágenes son referencias visuales de la clase, no fotos cargadas por quien usa la app.

## Contrato del modelo

El contrato versionado está en `src/cloud_classifier/model_contract.json`. Define el orden de las 10 clases OMM seguido de `no_nube`, las salidas como logits y el preprocesamiento compartido implementado en `src/cloud_classifier/preprocessing.py`. La aplicación requiere `manifests/calibration-v1.json` y un `models/clasificador.onnx` exportado desde el `runs/best.pt` correspondiente; valida hashes, orden de clases y dimensiones al arrancar.

Los pesos y el ONNX no se versionan por las condiciones de licencia de Open Images. Para exportar el checkpoint local, instalá las dependencias de entrenamiento/exportación y ejecutá:

```powershell
uv sync --extra training
uv run python scripts/export_onnx.py
```

El exportador produce logits crudos con entrada `1x3x224x224` y verifica la equivalencia con PyTorch en tres entradas sintéticas antes de aceptar el artefacto. `models/` está excluido de Git.

## Método

Casi todo el esfuerzo fue evitar medir mal.

- **Particiones por grupos de parecido.** Imágenes del mismo autor o casi idénticas (hash perceptual de 64 bits, distancia de Hamming ≤ 8) forman un grupo que no se parte entre entrenamiento, calibración y prueba. Reparto 70/15/15 estratificado por clase, con semilla fija.
- **Manifiestos congelados y atados por hash.** Las asignaciones no se recalculan. Los manifiestos de exclusión, validación y calibración registran el SHA-256 de lo que dependen.
- **Cada grupo tiene un único rol.** Entrenamiento ajusta pesos, validación elige la época, calibración fija `T` y umbrales, prueba solo mide. Los criterios (precisión objetivo, tolerancia de calibración de 8 puntos, mínimo de 30 fotos por cajón) se escribieron **antes** de ver los resultados.
- **Evaluación final de una sola corrida.** El script exige confirmación y deja un archivo de bloqueo.
- **Auditoría de etiquetas.** Al revisar a ojo apareció que CCSN contiene fotos con dos tipos de nube guardadas en dos carpetas, con etiquetas contradictorias. Se revisaron 130 grupos y se confirmaron 128; esas 271 imágenes se excluyeron de entrenamiento y evaluación y se guardaron aparte como un set de **doble etiqueta**, que solo se usa para mirar cómo responde el modelo (a veces dice par, a veces firme). Quitar las ambiguas deja ejemplos más fáciles, así que las métricas pueden ser algo optimistas frente a fotos reales.
- **Qué se probó y se descartó:** una tasa de aprendizaje menor y pesos por clase en la pérdida no superaron el criterio fijado de antemano (macro-F1 de validación > 0,60), así que se mantuvo el primer modelo.

## Datos y atribuciones

Las imágenes **no están en el repositorio** y el modelo entrenado **tampoco se publica**. Las fuentes:

| Fuente | Uso | Licencia / permiso |
|---|---|---|
| CCSN (Zhang et al., 2018) | 10 géneros de nube y estelas (`no_nube`) | CC0 |
| Open Images V7 (validación) | `no_nube` | CC BY 2.0, con atribución por imagen |
| Fotos propias | Evaluación final y cielos despejados | Propias |

Las fotos de referencia que muestra la app son de CCSN. Las atribuciones de Open Images están en `docs/datasets/`.

Si usás CCSN, citá: Zhang, J., Liu, P., Zhang, F., Song, Q. (2018). *CloudNet: Ground-Based Cloud Classification With Deep Convolutional Neural Network.* Geophysical Research Letters. https://doi.org/10.1029/2018GL077787

Datasets evaluados y descartados: HBMCD (el enlace de descarga ya no funciona) y el dataset multietiqueta de Rosenberger et al. (30 clases, imágenes de 64×100, no encaja con los 10 géneros).

## Preparar conjuntos públicos

Las imágenes se descargan a `data/`, que está excluido de Git. Para preparar CCSN (2.543 JPEG, con `Ct` asignado a `no_nube`) ejecutá:

```powershell
uv run python scripts/download_ccsn.py
```

Para `no_nube` de Open Images V7 se usa el descargador filtrado, que conserva autoría y licencia CC BY 2.0 por imagen en los metadatos versionados. Este proyecto usó `--limit 300`:

```powershell
uv run python scripts/download_no_nube_open_images.py --limit 300
```

Las citas y licencias de CCSN se conservan en [docs/datasets/ccsn/CITATION.txt](./docs/datasets/ccsn/CITATION.txt); Open Images genera `ATTRIBUTION.csv` y `CITATION.txt` dentro de `docs/datasets/`.

## Reproducir el entrenamiento

Requiere los datos en `data/` con la estructura de los manifiestos y una GPU (probado en Google Colab). En Colab (Python 3.13) no se instala el paquete: se instala `timm` y se usa `PYTHONPATH=src`.

```bash
PYTHONPATH=src python scripts/train_model.py --max-epochs 20 --patience 4 --device cuda
PYTHONPATH=src python scripts/calibrate.py --model <ruta>/best.pt --target-precision 0.75 \
    --output manifests/calibration-v1.json --report calibration-report.md
```

La evaluación final (`scripts/evaluate_final.py`) se ejecuta una sola vez.

Los hashes de los manifiestos dependen de los bytes exactos de cada archivo. Si clonás el repositorio en Linux o macOS, los saltos de línea pueden diferir de los de Windows y las verificaciones fallar con "el manifiesto no corresponde". [Los manifiestos se guardan sin conversión de saltos de línea (.gitattributes)]

## Limitaciones conocidas

- Modelo de precisión moderada; mejor con nubes de capas bajas distintivas (`cumulonimbus`, `cumulus`) y peor con las capas grises parecidas.
- CCSN no trae autores, así que se agrupa solo por hash perceptual (control más débil). El hash no detecta copias recortadas o a otra escala.
- Mis 122 fotos no tienen sesión registrada: el intervalo real es más ancho que el reportado.
- Hay clases con menos de 30 fotos en calibración o prueba (`altocumulus`, `altostratus`, `cirrus`, `cumulus` y `stratus`), así que las métricas por clase son orientativas.
- La app es local: no guarda fotos, no tiene cuentas ni historial.

## Verificación manual de la interfaz

Seguí el [checklist manual](./docs/manual-ui-checklist.md) en escritorio y móvil, incluyendo permisos de cámara y carga alternativa.

## Pruebas automatizadas

La primera vez, instalá Chromium para la prueba de interfaz:

```powershell
uv run playwright install chromium
```

Después ejecutá:

```powershell
uv run python -m unittest discover -s tests
```

## Estructura

```text
src/cloud_classifier/   código del paquete (predictor, calibración, entrenamiento)
app/                    interfaz Gradio y fotos de referencia
scripts/                descarga, entrenamiento, calibración, evaluación final, exportación
manifests/              particiones, exclusiones, validación y calibración
clouds.yaml             contenido curado de cada clase
docs/adr/               decisiones de diseño
```

## Licencia

[Código bajo licencia MIT (ver [LICENSE](./LICENSE)). Los datos, las imágenes y el modelo entrenado tienen sus propias condiciones, descritas en "Datos y atribuciones.]