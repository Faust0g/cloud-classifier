# Checklist manual de la primera rebanada web

Ejecutar `uv run cloud-classifier` y abrir la dirección local que muestra Gradio. Los resultados son simulados; esta verificación no mide la calidad de un modelo.

## Escritorio

- [ ] Cargar un JPEG válido de hasta 10 MB y obtener el escenario seleccionado.
- [ ] Cargar un PNG válido de hasta 10 MB.
- [ ] Intentar cargar una imagen mayor a 10 MB; debe mostrarse un error claro y no una predicción.
- [ ] Intentar cargar un formato distinto de JPEG/PNG o un archivo dañado; debe mostrarse un error claro.
- [ ] Probar cada escenario: nube reconocida con alternativas y confianzas, `no_nube` con su mensaje, y confianza baja solicitando otra foto.
- [ ] Conceder permiso a la cámara, capturar una imagen y clasificarla.
- [ ] Denegar el permiso de cámara; la carga de archivo debe seguir disponible.
- [ ] Probar sin cámara conectada; la carga de archivo debe seguir disponible.
- [ ] Confirmar que la imagen no aparece en historial ni se conserva en el directorio temporal de Gradio tras procesarse.

## Móvil

- [ ] Abrir la app en un contexto seguro (HTTPS o localhost) y comprobar que los controles se pueden usar sin desbordamiento horizontal.
- [ ] Conceder permiso y capturar con la cámara del dispositivo.
- [ ] Denegar el permiso o simular una cámara ausente y comprobar que la carga de archivo sigue disponible.
- [ ] Cargar una imagen y recorrer los tres escenarios; los textos y resultados deben ser legibles y distinguibles.
