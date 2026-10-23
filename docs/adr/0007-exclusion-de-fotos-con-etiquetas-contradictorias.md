# Exclusión de fotos duplicadas con etiquetas contradictorias

## Estado

La exclusión, el set de doble etiqueta y validation-v2 están aprobados y
congelados.

## Contexto

La auditoría reunió 130 grupos multiclase de `partitions-v1.json`. La revisión
visual confirmó que 128 grupos (271 rutas) representan fotos repetidas en
carpetas con etiquetas distintas: 181 filas de train, 48 de calibration y 42 de
test. Hay 259 imágenes de CCSN y 12 de sociedad; Open Images no aporta grupos
multiclase. Los 14 grupos con tres o más imágenes contienen copias de la misma
foto y se excluyen completos.

La inspección visual determinó que `group_3279ca96570ee968` y
`group_a5a19a1698ce22b8` son fotos distintas y deben permanecer en las
particiones. En el informe de auditoría sus estados previos son, respectivamente,
“dudoso” y “bajo contraste”. SSIM, MAE, Hamming y contraste se conservan como
evidencia, pero no determinan la exclusión.

## Decisión

- Excluir las 271 rutas de los 128 grupos confirmados de entrenamiento,
  validación, calibración y prueba. El lector valida el SHA-256 de
  `partitions-v1.json` y filtra las rutas antes de abrir imágenes.
- Guardar la exclusión en `manifests/exclusions-v1.json` y el set descriptivo en
  `manifests/double-label-v1.json`, ambos ligados al SHA-256
  `0ebcf4d3c72ce3344cb51a4024ff5e7333dc9fcfe7e6c5e6bd0c7d5c53c0eae6`.
- Mantener sin cambios `partitions-v1.json` y `validation-v1.json`. La asignación
  `validation-v1.json` queda como historial y no se usa para entrenar después de
  las exclusiones.
- Generar `validation-v2.json` sobre las 1.944 filas train restantes, con el
  mismo algoritmo agrupado/estratificado, fracción objetivo del 10% y semilla
  fija 42. Su asignación se liga a los hashes de `partitions-v1.json` y
  `exclusions-v1.json`; cualquier cambio invalida el archivo. No se usan modelos,
  métricas ni predicciones para asignar grupos. El entrenamiento requiere v2 y
  rechaza v1 explícitamente.
- `manifests/validation-v2.json` se congeló después de revisar y aprobar el
  informe y candidato. Su SHA-256 es
  `2077ee5459ed31129208eb6809702ff2a37ccb09b93f5ca4f625946b2905c3a0`.
- El reparto determinista aprobado selecciona 195 imágenes en 191 grupos.
  Cirrus queda con 11 imágenes en 10 grupos, frente a 11/11 en v1 y 6/6 en v1
  tras exclusiones; todas las clases superan el mínimo de 8 grupos. Las
  proporciones multiimagen son 2,09% en validación y 1,92% en el resto de train.
  El informe candidato conserva la comparación completa por clase.
- Versionar por separado la lista de exclusiones y un set descriptivo de 128
  fotos únicas con las clases y rutas de sus copias. Las imágenes permanecen en
  `data/`, fuera del repositorio. El set de doble etiqueta no se usa en
  entrenamiento, validación, selección de época, calibración ni umbrales. #11
  puede consultarlo solo después de calibrar para un chequeo descriptivo de
  resultados de par.
- validation-v1 pierde 29 imágenes y 14 grupos al aplicar exclusiones y queda
  con 182 imágenes en 182 grupos; cirrus pasa de 11 a 6 grupos. Por esa pérdida
  de soporte se prepara v2 antes de cualquier entrenamiento o inspección de
  métricas, manteniendo v1 inmutable como historial.
- La proporción de grupos multiimagen pasa de 7,14% a 0% en validación y de
  6,02% a 2,15% en el resto de train (37 de 1.724 grupos). Los grupos
  multiimagen restantes de train son de una sola clase. No hay fuga ni
  concentración en validación; no se rebalancea ni se cambia la asignación
  congelada.
- Tras la exclusión quedan 1.944 imágenes train, 409 calibration y 411 test.
  Quedan menos de 30 imágenes en calibration para altocumulus (29), altostratus
  (27), cirrus (22), cumulus (27) y stratus (28); en test, para altostratus
  (27), cirrus (22), cumulus (26) y stratus (28). Se reporta el soporte
  reducido, no se corrige alterando las particiones.

## Consecuencias y limitaciones

Excluir fotos ambiguas favorece ejemplos más claros de etiquetar. Por eso las
métricas de calibración y prueba pueden ser algo optimistas frente a fotos reales,
que pueden mostrar dos capas de nubes. El set de doble etiqueta compensa este
sesgo solo parcialmente. Tanto CCSN como sociedad contienen fotos repetidas con
etiquetas distintas.

La agrupación usa autor o pHash con distancia de Hamming ≤ 8. Este método puede
no detectar copias recortadas o guardadas a otra escala. Las comparaciones SSIM,
MAE y contraste son evidencia del relevamiento y no un clasificador de
duplicados.

## Referencias

- [Script de auditoría](../../scripts/audit_label_conflicts.py)
- [Generador del informe candidato](../../scripts/report_label_conflict_decision.py)
- [ADR 0005: particiones de datos congeladas](./0005-particiones-de-datos-congeladas.md)
- [ADR 0006: entrenamiento inicial reproducible](./0006-entrenamiento-inicial-reproducible.md)
