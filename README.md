# Cloud Classifier

Aplicación local en español para identificar nubes en imágenes y explicar qué condiciones suelen asociarse a ellas. La primera rebanada usa resultados simulados para probar la interfaz antes de integrar un modelo.

## Ejecutar localmente

Requiere Python 3.14 y [uv](https://docs.astral.sh/uv/).

```powershell
uv sync
uv run cloud-classifier
```

Abrí la dirección local que imprime Gradio. La interfaz permite cargar JPEG/PNG de hasta 10 MB o tomar una foto con la cámara. La cámara requiere permiso del navegador y contexto seguro; localhost sirve para desarrollo. Si se deniega el permiso o no hay cámara, se puede cargar un archivo.

Los resultados están simulados y se pueden seleccionar para probar los estados de nube reconocida, `no_nube` y baja confianza. No representan una predicción del contenido de la imagen.
La analítica de uso de Gradio está desactivada; las imágenes no se usan para telemetría.

## Fotos de referencia de nubes

Las fotos que acompañan las explicaciones se guardan en `app/assets/clouds/`, una por género OMM. Usá el código de clase de `clouds.yaml` como nombre en minúsculas y JPEG o PNG, por ejemplo `ci.jpg` para Cirrus. Estas imágenes son referencias visuales de la clase, no fotos cargadas por quien usa la app.

## Preparar conjuntos públicos

Las imágenes se descargan a `data/`, que está excluido de Git. Para preparar CCSN (2.543 JPEG, con `Ct` asignado a `no_nube`) ejecutá:

```powershell
uv run python scripts/download_ccsn.py
```

Para `no_nube` de Open Images V7 se usa el descargador filtrado, que conserva autoría y licencia CC BY 2.0 por imagen en los metadatos versionados:

```powershell
uv run python scripts/download_no_nube_open_images.py --limit 50
```

Las citas y licencias de CCSN se conservan en [docs/datasets/ccsn/CITATION.txt](./docs/datasets/ccsn/CITATION.txt); Open Images genera `ATTRIBUTION.csv` y `CITATION.txt` dentro de `docs/datasets/`.

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
