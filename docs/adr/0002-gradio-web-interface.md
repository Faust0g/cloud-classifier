# Gradio para la interfaz web del MVP

La interfaz web local del MVP se implementará con Gradio, en coherencia con el stack Python existente y el vocabulario del proyecto, que ya identifica Gradio como herramienta para probar el clasificador. Se elige frente a construir una interfaz web y API propias para entregar antes el flujo completo de carga/cámara a resultado, con menor superficie de implementación; la decisión puede revisarse si la experiencia requiere capacidades que Gradio no cubra. El acceso a cámara requiere permiso explícito del navegador y un contexto seguro (HTTPS o localhost); la carga de archivo seguirá disponible como alternativa.

## Fragilidad conocida

`UploadPathFile` hereda de `gr.File` para entregar la ruta subida directamente al handler, que valida y elimina el archivo. Su método `get_block_name()` debe devolver `"file"` para reutilizar el frontend incorporado de Gradio: si Gradio cambia el contrato de componentes o esta adaptación deja de resolverse como `file`, puede solicitar recursos `/gradio_api/custom_component/` inexistentes y dejar la interfaz en `Loading...`. La prueba de humo del navegador detecta errores de consola y estas respuestas fallidas; al actualizar Gradio, ejecutar la suite completa y revisar `UploadPathFile`.
