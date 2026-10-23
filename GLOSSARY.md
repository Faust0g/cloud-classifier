# Cloud Classifier

## Objetivo

Proyecto para clasificar imágenes de nubes mediante técnicas de machine learning.

## Vocabulario

- **Dataset:** conjunto de imágenes utilizado para entrenar o evaluar el modelo.
- **Clase:** categoría que identifica una imagen, como un tipo de nube o No nube.
- **No nube:** categoría para imágenes que no muestran ninguna de las nubes identificables por este proyecto.
- **Certeza:** probabilidad de que esa nube sea del tipo propuesto.
- **Resultado firme:** resultado que muestra una clase principal cuando su confianza calibrada y el margen sobre la segunda clase satisfacen los criterios de calibración.
- **Resultado de par:** resultado que comunica dos clases de nube plausibles cuando la suma calibrada de sus probabilidades es alta y la diferencia entre ellas es pequeña.
- **Par de clases confundibles:** dos clases de nube que el modelo puede confundir y que se presentan juntas en un resultado de par.
- **Rama parecidas:** presentación de un par cuyas clases tienen significados meteorológicos similares; comunica ambas clases y sus descripciones breves, sin explicación adicional específica del par.
- **Rama eventos distintos:** presentación de un par cuyas clases se asocian a fenómenos distintos; además de describirlas, explica la diferencia y cómo distinguirlas.
- **Repetir foto:** resultado que solicita otra imagen cuando la evidencia no alcanza para un resultado firme o de par; no equivale a **No nube**.
- **Grupo de parecido:** conjunto de imágenes relacionadas por autor conocido o similitud visual, que se mantiene unido al evaluar el clasificador.
- **Grupo multiclase:** grupo de parecido cuyas imágenes tienen más de una etiqueta de clase.
- **Partición de datos:** división del conjunto de imágenes en entrenamiento, calibración y prueba.
- **Asignación congelada:** pertenencia de una imagen y su grupo a una partición que no cambia al incorporar datos posteriores.
- **Validación:** subconjunto de entrenamiento separado por grupos de parecido y usado para elegir la época del modelo; no sustituye los conjuntos de calibración o prueba.
- **Asignación de validación versionada:** archivo reproducible que fija qué grupos de entrenamiento se usan para validación; queda ligado por hash a los manifiestos de partición y, cuando aplica, exclusión.
- **Exclusión por etiquetas contradictorias:** omisión de todas las copias de una misma foto cuando aparecen etiquetadas con clases distintas; se aplica antes de abrir las imágenes para entrenar o evaluar.
- **Set de doble etiqueta:** referencias a fotos duplicadas con sus distintas clases, reservado para un chequeo descriptivo y separado del entrenamiento, la validación, la calibración y la prueba.
- **Parada temprana:** criterio que detiene el entrenamiento cuando la métrica elegida sobre validación deja de mejorar durante un número configurado de épocas.
- **Carpeta de corrida:** directorio local ignorado por Git que reúne los pesos, la configuración, las curvas y las métricas de una ejecución de entrenamiento.
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
