# Sistema de puntuación

> 🌐 [English](../SCORING.md) · [Français](../fr/SCORING.md) · [Deutsch](../de/SCORING.md) · [Italiano](../it/SCORING.md) · **Español** · [Português](../pt/SCORING.md) · [简体中文](../zh/SCORING.md)

Las fotos se clasifican en una categoría y luego se puntúan con los pesos de esa categoría.

## Cómo funciona la puntuación

1. **Detección de categoría** - La foto se analiza por su contenido (rostros, etiquetas, datos EXIF)
2. **Evaluación de filtros** - Las categorías se evalúan por orden de prioridad hasta que una coincide (un *contexto de puntuación* puede promover/excluir categorías por álbum o foto sin cambiar el orden base — consulta [Contextos de puntuación](#contextos-de-puntuación))
3. **Aplicación de pesos** - Se aplican los pesos específicos de la categoría a las métricas
4. **Aplicación de modificadores** - Se aplican bonificaciones, penalizaciones e indicadores de comportamiento
5. **Puntuación final** - Suma ponderada acotada al rango 0-10

## Categorías

`scoring_config.json` define 34 categorías (33 con nombre más `default`), evaluadas en orden ascendente de prioridad hasta que una coincide. Gana la prioridad más baja. La lista completa se encuentra en el array `categories`; las principales son:

| Prioridad | Categoría | Método de detección |
|----------|----------|------------------|
| 8 | `art` | Etiquetas: painting, statue, drawing, cartoon, anime |
| 10 | `astro` | Etiquetas: aurora, astrophotography, stars, milky way, Y luminancia media ≤ 0.30 |
| 15 | `concert` | Etiquetas: concert |
| 35 | `group_portrait` | Proporción facial ≥ 5% Y is_group_portrait |
| 42 | `silhouette` | Tiene rostro Y `is_silhouette` (el rostro más grande es oscuro frente a un entorno más claro) |
| 45 | `portrait` | Proporción facial ≥ 5%, no silueta/grupo/mono |
| 46 | `portrait_bw` | Retrato monocromo (rostro ≥ 5%) |
| 55 | `macro` | Etiquetas: macro, insect, butterfly, dewdrop, ... |
| 65 | `wildlife` | Etiquetas: animal, bird, marine, reptile, primate |
| 80 | `long_exposure` | Obturación 1-10 segundos |
| 85 | `night` | Luminancia < 0.15 |
| 88 | `monochrome` | is_monochrome (saturación < 5%) |
| 95 | `street` | Etiquetas: street, urban_culture |
| 96 | `human_others` | Tiene rostro Y proporción facial < 5% |
| 100 | `landscape` | Etiquetas: landscape, mountain, beach, forest, ... |
| 999 | `default` | Reserva (sin filtro) |

Otras categorías basadas en etiquetas son `aerial`, `food`, `sports`, `vehicle`, `travel`, `fashion`, `candid`, `product`, `architecture`, `urban`, `golden_hour`, `blue_hour`, `cinematic`, `vintage`, `abstract`, `minimalist`, `dramatic` y `weather`.

En los perfiles con etiquetado CLIP (`legacy`/`8gb`, y `16gb`/`24gb` cuando su etiquetador VLM no se ejecuta), un escaneo multipasada etiqueta cada foto durante la pasada de embeddings, antes de calcular su categoría y agregado. La categoría que guarda un escaneo usa por tanto las etiquetas, igual que `--recompute-average`. El paso de etiquetado del final del escaneo solo recoge las fotos que siguen sin etiquetar. Una ejecución con un solo `--pass NAME` no etiqueta.

**Silueta.** `is_silhouette` sale de una regla de píxeles basada en el rostro (ver [Filtros booleanos](#filtros-booleanos)), no de una etiqueta ni de una prueba de histograma. El indicador almacenado afecta a la puntuación además de a la categoría. Cuando `exposure.silhouette_detection` es `true` (el valor por defecto), el agregado de una silueta omite la penalización por recorte. El `exposure_score` almacenado depende del comando: un escaneo lo guarda con la penalización por recorte, mientras que `--recompute-average` lo recalcula sin ella para toda foto marcada `is_silhouette`, sea cual sea el valor de `exposure.silhouette_detection`. Para volver a derivar los indicadores de una biblioteca existente, ejecuta `python facet.py --recompute-silhouette` y después `--recompute-average` dos veces (ver [COMMANDS.md](COMMANDS.md)).

**Astro.** Una foto necesita una etiqueta astro *y* una luminancia media de como máximo 0.30 (`luminance_max`): una escena luminosa etiquetada como `aurora` o `stars` no es astro, y una foto sin luminancia almacenada nunca coincide. Un `scoring_config.json` con su propio array `categories` conserva su propia entrada `astro` y no recibe este filtro hasta que añadas `"luminance_max": 0.30` a `astro.filters`; `python facet.py --doctor` te avisa cuando a la tuya le falta.

## Contextos de puntuación

El orden de prioridad anterior es global — todas las fotos se evalúan contra la misma lista. Un **contexto de puntuación** es un *delta* con nombre sobre ese orden base: promueve una lista corta de categorías al frente y excluye otras por completo, sin renumerar nada. `default` (`promote`/`excluded` vacíos) es el contexto neutro (no-op), así que nada cambia para una foto a menos que se le asigne explícitamente un contexto.

**Orden efectivo** = `promote` (en el orden dado) → el orden de prioridad global con los nombres promovidos y excluidos eliminados → `default` al final. Un nombre que aparece a la vez en `promote` y en `excluded` se elimina por completo — gana `excluded`. `ScoringConfig.resolve_context_order()` (`config/scoring_config.py`) calcula y memoiza esto una vez por nombre de contexto.

Preajustes incluidos — editables desde la pestaña **Contexto de puntuación** del visor (`PUT /api/config/scoring_contexts/{name}`, restringido al modo edición) o directamente en el JSON; consulta [Contextos de puntuación](CONFIGURATION.md#contextos-de-puntuación) para la referencia completa de campos:

| Contexto | Promueve | Excluye |
|---------|----------|----------|
| `default` | — | — |
| `action_stage` | `sports`, `concert`, `candid` | `silhouette` |
| `party_event` | `group_portrait`, `candid`, `food` | — |
| `portrait_session` | `portrait`, `portrait_bw`, `fashion` | — |
| `wildlife` | `wildlife` | — |
| `landscape` | `landscape`, `golden_hour`, `blue_hour` | — |
| `motorsport` | `sports`, `vehicle` | `silhouette` |

Solo el *delta* es editable — arrastra la cabeza promovida al orden que quieras (o usa los botones Subir/Bajar), alterna la exclusión de una categoría — nunca un orden completo independiente por contexto: las categorías no promovidas siempre conservan el orden de prioridad global, así que una categoría añadida más tarde nunca puede faltar en silencio en seis listas separadas. Consulta [Editar un contexto](CONFIGURATION.md#editar-un-contexto) para las reglas de validación.

Un contexto se asigna por álbum (`PUT /api/albums/{id}/scoring_context`, que lo materializa en cada foto que es miembro en ese momento — para un álbum inteligente, una instantánea, no una suscripción, consulta [Contextos de puntuación](CONFIGURATION.md#contextos-de-puntuación)) o, para una foto concreta y difícil, se aplica como una anulación de categoría persistente (`POST /api/comparison/override_category`). Ambas palancas persisten en una tabla auxiliar `photo_scoring_overrides` en lugar de como columnas en `photos` — `save_photo`/`save_photos_batch` escriben las filas de foto con `INSERT OR REPLACE`, lo que borraría en silencio cualquier columna nueva de esa fila en el siguiente reescaneo. Establecer una palanca deja la otra intacta, y cada una se puede borrar de forma independiente. **Ninguna de las dos surte efecto en fotos ya puntuadas hasta un recálculo** — `python facet.py --recompute-average`, o `POST /api/scan/recompute` desde el visor (protegido entre procesos frente a que se ejecuten dos a la vez — consulta [Cambiar las prioridades requiere un recálculo](CONFIGURATION.md#reordenar-la-prioridad-global)). Si `normalization.per_category` está activado, ejecuta el recálculo dos veces — consulta [Normalización](CONFIGURATION.md#normalización) para saber por qué el primer pase normaliza frente a la categoría anterior de cada foto.

### La trampa de los EXIF faltantes

Reordenar — ya sea editando la prioridad global o promoviendo mediante un contexto — solo cambia qué categoría se *prueba primero*. No puede hacer que los filtros de una categoría coincidan con una foto con la que de otro modo no coincidirían. `config/category_filter.py:122-128` hace fallar un filtro de rango numérico por completo siempre que el valor subyacente de la foto falte o no se pueda interpretar, en lugar de omitir solo ese límite — un valor ausente y un valor fuera de rango se tratan de forma idéntica, y la categoría se descarta en cualquiera de los dos casos.

En concreto: `sports` (prioridad 71) lleva `shutter_speed_max: 0.02`. Un fotograma de baile disparado más lento que 1/50s, o sin una velocidad de obturación EXIF legible, no supera ese filtro sin importar dónde se sitúe `sports` en el orden de evaluación — incluso promovida al frente por un contexto como `action_stage`. La foto cae entonces en la siguiente categoría que coincida, típicamente `fashion` (prioridad 43, etiquetada `fashion`, tiene rostro) o `silhouette` (prioridad 42, a contraluz con rostro). **Esto es lo más útil que se puede comprobar cuando una foto acaba en una categoría inesperada:** antes de reordenar o promover nada, confirma que los filtros numéricos de la categoría de destino realmente pueden coincidir con el EXIF almacenado de la foto, no solo con sus etiquetas.

### La trampa de la etiqueta faltante

La trampa de los EXIF de arriba supone que la capa de etiquetas ya encontró coincidencia — `required_tags` tiene su propia versión del mismo problema: un filtro no puede coincidir con una etiqueta que no existe en absoluto en el vocabulario CLIP. Antes del commit `917dd94`, nada en `scoring_config.json` cubría el baile: un fotograma que el etiquetador describía como baile, actuación o escenario no coincidía con ninguno de los `required_tags` de `sports` (`sports`, `motion`, `athlete`, `competition`, `action_sport`) y caía directamente en la reserva `default` — promover `sports` mediante `action_stage` no cambiaba nada, porque la promoción solo reordena *cuándo* se prueba una categoría, nunca si sus filtros coinciden.

`sports.tags.dance` ahora lleva 11 prompts de CLIP (`dance performance`, `dancer on stage`, `ballet dancer`, `contemporary dance`, `ballroom dancing`, `latin dance`, `hip hop dance`, `dance competition`, `dance troupe performing`, `dancer mid leap`, `dancer in motion`), y `dance` se añadió a `sports.filters.required_tags`, así que una foto de baile etiquetada ya puede superar ese filtro. Esto hace que `sports` sea *alcanzable* para contenido de baile — no exime de la trampa de la velocidad de obturación de arriba: un fotograma de baile a obturación lenta o sin EXIF ahora puede pasar la comprobación de etiquetas y aun así fallar en `shutter_speed_max: 0.02`, cayendo en `fashion` o `silhouette` exactamente como se describió allí.

**Las fotos existentes conservan las etiquetas que ya tienen** — el nuevo vocabulario solo cambia lo que el etiquetador asigna de aquí en adelante. Reetiqueta la biblioteca con `python facet.py --recompute-tags` para aplicarlo retroactivamente.

### Reordenar la prioridad global

`GET/POST /api/config/category_priorities` (requiere edición) lee y reescribe el orden base contra el que aplica su delta cada contexto. `POST` recibe `{"order": [name, ...]}` — una permutación de igual conjunto de todos los nombres de categoría distintos de `default` — y **permuta los valores de prioridad existentes sobre el nuevo orden** en lugar de renumerar (10/20/30/…): el multiconjunto de prioridades no cambia, así que los números de la tabla anterior siguen siendo válidos y la unicidad se mantiene por construcción. `default` (prioridad 999) queda fijada al final y excluida del reordenamiento. Cada escritura toma antes una copia con marca de tiempo `.backup.<timestamp>` de `scoring_config.json`; este escritor y el editor de pesos (`update_category_weights`) ahora comparten un único bloqueo, ya que el anterior ciclo de lectura-modificación-escritura sin protección permitía que un guardado concurrente de cada uno descartara en silencio los cambios del otro.

Reordenar no toca por sí solo la `category` almacenada de ninguna foto — ejecuta un recálculo después (`--recompute-average` o `POST /api/scan/recompute`) para aplicarlo.

**Limitación conocida:** `api/types.py` construye la lista desplegable de tipo/filtro de la galería a partir de `ScoringConfig.get_categories()` una sola vez al importar el módulo. Un reordenamiento de prioridad se recoge de inmediato para la coincidencia real de categorías (cada llamada de puntuación y recálculo vuelve a leer la configuración desde disco), pero el desplegable de tipo de la galería conserva su orden antiguo hasta que se reinicia el proceso del visor. El filtrado en sí no se ve afectado — reordenar no añade ni quita ningún nombre de categoría.

## Definición de categoría

Cada categoría en `scoring_config.json` tiene estos componentes:

```json
{
  "name": "portrait",
  "priority": 45,
  "filters": {
    "face_ratio_min": 0.05,
    "has_face": true,
    "is_silhouette": false,
    "is_group_portrait": false,
    "is_monochrome": false
  },
  "weights": {
    "aesthetic_percent": 32,
    "eye_sharpness_percent": 16,
    "face_quality_percent": 14,
    "composition_percent": 12,
    "liqe_percent": 8,
    "exposure_percent": 4,
    "tech_sharpness_percent": 4,
    "color_percent": 4,
    "contrast_percent": 4,
    "aesthetic_iaa_percent": 2
  },
  "modifiers": {
    "bonus": 0.419,
    "_apply_blink_penalty": true,
    "noise_tolerance_multiplier": 0.006,
    "_clipping_multiplier": 0.5
  },
  "tags": {}
}
```

## Referencia de filtros

### Filtros de rango numérico

| Filtro | Campo | Descripción |
|--------|-------|-------------|
| `face_ratio_min` / `face_ratio_max` | `face_ratio` | Área facial como fracción (0.0-1.0) |
| `face_count_min` / `face_count_max` | `face_count` | Número de rostros |
| `iso_min` / `iso_max` | `ISO` | ISO de la cámara |
| `shutter_speed_min` / `shutter_speed_max` | `shutter_speed` | Tiempo de exposición (segundos) |
| `luminance_min` / `luminance_max` | `mean_luminance` | Brillo (0.0-1.0) |
| `focal_length_min` / `focal_length_max` | `focal_length` | Distancia focal (mm) |
| `f_stop_min` / `f_stop_max` | `f_stop` | Número f de apertura |

### Filtros booleanos

| Filtro | Descripción |
|--------|-------------|
| `has_face` | Al menos un rostro detectado |
| `is_monochrome` | Saturación < 5% |
| `is_silhouette` | El rostro más grande es oscuro y su entorno es claro: el brillo medio (luma, 0-1) del 60 % central del cuadro del rostro es inferior a 0.12, y la banda alrededor del cuadro (de 0.2 a 0.75 del tamaño del cuadro hacia fuera por cada lado, recortada al encuadre) es más clara en más de 0.20. Ni la etiqueta CLIP `silhouette` ni el histograma del fotograma completo lo establecen ya. Una foto sin rostro nunca es una silueta |
| `is_group_portrait` | face_count >= `min_faces_for_group` (configurable, predeterminado: 4) |

### Filtros de etiquetas

| Filtro | Descripción |
|--------|-------------|
| `required_tags` | Lista de etiquetas que la foto debe tener |
| `excluded_tags` | Lista de etiquetas que la foto NO debe tener |
| `tag_match_mode` | `"any"` (predeterminado) o `"all"` |

## Claves de peso

Todos los pesos usan el sufijo `_percent`. Se normalizan mediante `get_weights()`, por lo que los totales no tienen por qué sumar exactamente 100, pero mantenerlos en 100 conserva las puntuaciones en la escala 0-10.

| Clave | Métrica | Origen | Mejor para |
|-----|--------|--------|----------|
| `aesthetic_percent` | Atractivo visual | TOPIQ o CLIP+MLP | Todas |
| `quality_percent` | Calidad heredada | Sumada a `aesthetic_percent` antes de puntuar (sin señal separada) | — |
| `face_quality_percent` | Claridad facial | InsightFace | Retratos |
| `eye_sharpness_percent` | Nitidez ocular | Puntos de referencia de InsightFace | Retratos |
| `tech_sharpness_percent` | Nitidez general | Varianza de Laplaciano | Paisajes |
| `composition_percent` | Composición | SAMP-Net o basado en reglas | Todas |
| `exposure_percent` | Equilibrio de exposición | Análisis de histograma | Todas |
| `color_percent` | Armonía cromática | Análisis HSV | Fotos en color |
| `contrast_percent` | Contraste tonal | Dispersión de histograma | B/N |
| `dynamic_range_percent` | Rango tonal | Análisis de histograma | HDR, paisajes |
| `isolation_percent` | Separación del sujeto | Rostro frente al fondo | Retratos, fauna |
| `leading_lines_percent` | Líneas guía | Detección de bordes | Arquitectura |
| `power_point_percent` | Regla de los tercios | Ubicación del sujeto | Todas |
| `saturation_percent` | Saturación cromática | Análisis HSV | Fotos vibrantes |
| `noise_percent` | Nivel de ruido | Estimación de ruido | Baja iluminación |
| `face_sharpness_percent` | Nitidez de la zona facial | Análisis facial | Retratos |
| `aesthetic_iaa_percent` | Mérito estético artístico | TOPIQ IAA (entrenado con AVA) | Arte, creativo |
| `face_quality_iqa_percent` | Calidad facial (IQA) | TOPIQ NR-Face | Retratos |
| `liqe_percent` | Puntuación de calidad LIQE | LIQE | Diagnósticos |
| `subject_sharpness_percent` | Nitidez de la zona del sujeto | BiRefNet + Laplaciano | Retratos, fauna |
| `subject_prominence_percent` | Proporción de área del sujeto | BiRefNet | Macro, fauna |
| `subject_placement_percent` | Regla de los tercios del sujeto | BiRefNet | Todas |
| `bg_separation_percent` | Separación del fondo | BiRefNet | Retratos, macro |

**`quality_percent` cuenta como `aesthetic`.** La métrica de calidad no tiene puntuación propia, así que el `quality_percent` de una categoría se suma a su `aesthetic_percent` antes de puntuar. Los pesos siguen sumando 100. Las versiones anteriores puntuaban la parte `quality` con una métrica que siempre valía 0, así que esa parte se perdía en el agregado. En su próximo `--recompute-average`, una biblioteca agregada por una versión anterior recupera esa parte. En una copia de una biblioteca de 129.737 fotos, 35.139 agregados subieron, 544 bajaron (media +0,087) y ninguna categoría cambió.

## Modificadores

Ajustan el comportamiento de la puntuación por categoría:

| Modificador | Tipo | Descripción |
|----------|------|-------------|
| `bonus` | float | Se añade a la puntuación final (p. ej., 0.5) |
| `noise_tolerance_multiplier` | float | Escala la penalización por ruido (0.5 = mitad) |
| `_skip_clipping_penalty` | bool | Omitir la penalización por recorte de exposición |
| `_skip_oversaturation_penalty` | bool | Omitir la penalización por sobresaturación |
| `_clipping_multiplier` | float | Escala la penalización por recorte |
| `_apply_blink_penalty` | bool | Aplicar la penalización por detección de parpadeo |

## Dimensiones de saliencia del sujeto

Cuatro dimensiones derivadas de la segmentación del sujeto por BiRefNet:

| Clave de peso | Métrica | Descripción |
|-----------|--------|-------------|
| `subject_sharpness_percent` | Nitidez del sujeto | Calidad de enfoque de la zona del sujeto frente al fondo. Alta = sujeto nítido, fondo suave. |
| `subject_prominence_percent` | Prominencia del sujeto | Área del sujeto como fracción del encuadre. Alta para macro y sujetos muy encuadrados, baja para escenas amplias. |
| `subject_placement_percent` | Ubicación del sujeto | Puntuación de la regla de los tercios para el centro de masa del sujeto. |
| `bg_separation_percent` | Separación del fondo | Diferencia de gradiente de bordes en el límite del sujeto (calidad del bokeh). |

Usa `subject_sharpness_percent` y `bg_separation_percent` para retrato/fauna; `subject_prominence_percent` para macro.

**Sin sujeto, sin número.** Cuando no se encuentra sujeto, `subject_sharpness`, `subject_placement` y `bg_separation` se almacenan como NULL (igual que `power_point_score` cuando no hay cuadro de sujeto), `subject_prominence` es 0.0 y `subject_bbox` es NULL; las versiones anteriores almacenaban un 5.0 neutro. Los filtros de rango mín/máx de la galería omiten un NULL en esas métricas (`min_subject_sharpness=0` ya no devuelve una foto sin sujeto), las estadísticas lo dejan fuera de los promedios y la interfaz lo muestra como ausente. El agregado sigue puntuando un NULL como el 5.0 neutro, así que ningún agregado cambia. Las filas existentes conservan su 5.0 hasta que `--recompute-saliency` y `--recompute-composition-cpu` las reescriban.

**Cuando falla la saliencia.** En CUDA, el tamaño de lote de saliencia se deriva de la VRAM libre (unos 2,4 GiB por imagen a 1024 px, entre 1 y 8 imágenes) y se reduce a la mitad tras un error de memoria agotada, hasta llegar a una sola imagen. Una foto cuya saliencia sigue fallando registra un ERROR y no guarda nada: una foto que nunca se puntuó conserva NULL en las cinco columnas de saliencia (`subject_sharpness`, `subject_prominence`, `subject_placement`, `bg_separation`, `subject_bbox`), y `--recompute-saliency` y `--pass saliency` conservan los valores guardados de una foto cuando su saliencia falla. Un reescaneo completo de la foto (por ejemplo con `--force`) guarda NULL para ella, igual que para una foto nunca puntuada. Facet nunca escribe un número neutro en lugar de un resultado fallido. En el agregado, un NULL puntúa como neutro (5,0), igual que una foto escaneada antes de que existiera la saliencia, así que ni ayuda ni perjudica a la puntuación. `--recompute-saliency` vuelve a seleccionar esa foto (consulta [Saliencia del sujeto](COMMANDS.md#saliencia-del-sujeto)).

## Dimensiones IQA complementarias

Tres modelos de calidad adicionales:

| Clave de peso | Modelo | Descripción |
|-----------|-------|-------------|
| `aesthetic_iaa_percent` | TOPIQ IAA | Mérito estético entrenado con AVA, distinto de la puntuación estética de calidad técnica. Mejor para categorías de arte/creativas. |
| `face_quality_iqa_percent` | TOPIQ NR-Face | Evaluación de la calidad de la zona facial. Mejor para categorías de retrato. |
| `liqe_percent` | LIQE | Puntuación de calidad más un diagnóstico de distorsión (desenfoque de movimiento, sobreexposición, ruido). |

Estos modelos se ejecutan como parte de la canalización de puntuación predeterminada y comparten la VRAM con TOPIQ. Añade sus claves de peso a cualquier categoría donde la evaluación sea útil.

**Cuando un modelo suplementario no puede puntuar una foto.** TOPIQ NR-Face necesita una cara que encuentre su propio detector, y LIQE una foto cuyo lado corto mida al menos 224 px; Aesthetic V2.5, si está activado, puede fallar con cualquier imagen. Una foto que un modelo no puede puntuar guarda NULL en la columna de ese modelo (`face_quality_iqa`, `liqe_score`, `aesthetic_iaa`, `aesthetic_v25`) en lugar de un 5,0 neutro. Un error de memoria agotada de la GPU se reintenta con la mitad del tamaño de lote, hasta llegar a una sola imagen, antes de que la foto se dé por fallida; un error de memoria de la CPU no se reintenta así, sigue el camino ordinario por imagen, y una foto que sigue fallando guarda NULL. Un fallo inesperado se registra como ERROR con el número de fotos afectadas, mientras que los dos casos esperados (no se encontró ninguna cara, lado corto inferior a 224 px) solo se suman en el registro cuando se libera el modelo. `--recompute-iqa` no reintenta en cada ejecución ninguno de los dos casos esperados: el del lado corto se decide por el tamaño de la miniatura almacenada, y un rechazo de NR-Face se recuerda hasta que la foto se reescanea. Un fallo de memoria agotada o inesperado se reintenta en la siguiente ejecución. En el agregado, un NULL puntúa como neutro (5,0), así que las puntuaciones agregadas, las categorías y las clasificaciones no cambian. **Un filtro de rango de la galería sobre Estética (IAA), Calidad facial (IQA), LIQE o Aesthetic V2.5 ya no incluye una foto sin puntuación**: una foto sin caras antes tenía un 5,0 guardado y coincidía con `min_face_quality_iqa=0`, y ahora no; una ordenación ascendente por Estética (IAA), Calidad facial (IQA) o LIQE coloca esas fotos al final, tanto en la galería como en los álbumes (inteligentes y compartidos incluidos). Los valores guardados por versiones anteriores se mantienen hasta que `--repair-neutral-iqa` borre el 5,0 neutro de `aesthetic_iaa`, `face_quality_iqa` y `liqe_score` (después ejecuta `--recompute-iqa` y `--recompute-average`). Un reescaneo que reescribe la foto (por ejemplo con `--force`) guarda NULL para una puntuación que el modelo no pudo producir; `--recompute-iqa` solo rellena las columnas que son NULL, y `--pass quality-iaa`, `--pass quality-face` y `--pass quality-liqe` conservan el valor guardado cuando un modelo falla, de modo que ninguno sustituye un 5,0 antiguo. Si el modelo de calidad principal (TOPIQ en los perfiles `16gb` y `24gb`) falla para una foto, la foto se registra como fallida en `scan_failures`, en lugar de guardarse con una puntuación estética neutra de 5,0. El siguiente escaneo recoge una foto que aún no está en la base de datos; una ya guardada (por ejemplo tras un reescaneo con `--force`) conserva su fila y la reintenta `--retry-failed all` (un `--retry-failed` sin argumento solo cubre la última ejecución que registró fallos) u otro reescaneo con `--force`. Con `--pass quality` se conserva el valor guardado y no se registra ninguna fila de fallo. Consulta [Modelos de calidad suplementarios](COMMANDS.md#modelos-de-calidad-suplementarios).

### Señales complementarias (no en el agregado predeterminado)

| Columna | Origen | Descripción |
|--------|--------|-------------|
| `aesthetic_clip` | `analyzers/aesthetic_clip.py` + embedding CLIP/SigLIP en caché | Una puntuación estética complementaria gratuita (0-10) derivada de los embeddings de imagen en caché proyectándolos sobre un "eje estético" construido a partir de prompts de texto positivos/negativos. Cero inferencia de imagen adicional durante el escaneo. **No** forma parte del `aggregate` predeterminado. Pobla con `python scripts/compute_aesthetic_clip.py --db <path>`. Compara con `python scripts/benchmark_aesthetic.py --db <path> --ava AVA.txt --photo-dir <dir>`. AVA SRCC ≈ 0.52 en el conjunto `ava_test/` de 500 fotos (frente a 0.94 para `aesthetic_iaa`): útil como prefiltro económico o cuando TOPIQ-IAA no está disponible. |

## Etiquetas de categoría (vocabulario CLIP)

Las etiquetas activan las categorías basadas en etiquetas y se emparejan mediante similitud CLIP:

```json
{
  "tags": {
    "landscape": ["landscape", "scenic view", "nature scene"],
    "mountain": ["mountain", "alpine", "peaks"],
    "beach": ["beach", "ocean", "seaside", "coastal"]
  }
}
```

Cada clave es el nombre canónico de la etiqueta, y el array contiene sinónimos para el emparejamiento CLIP.

## Puntuación de Selección destacada

El filtro "Selección destacada" del visor usa una puntuación ponderada personalizada:

```json
"top_picks_weights": {
  "aggregate_percent": 30,
  "aesthetic_percent": 28,
  "composition_percent": 18,
  "face_quality_percent": 24
}
```

**Cálculo de la puntuación:**
- Con rostro (proporción facial ≥ 20%): las cuatro métricas contribuyen
- Sin rostro: `face_quality_percent` se redistribuye a `aesthetic` y `composition`

## Consideraciones sobre el perfil de VRAM

Los pesos predeterminados están optimizados para **TOPIQ** (0.93 SRCC), el modelo estético para todos los perfiles.

| Perfil | Modelo estético | Embeddings | Etiquetador | Recomendaciones |
|---------|-----------------|-----------|--------|-----------------|
| `24gb` | TOPIQ (0.93 SRCC) | SigLIP 2 NaFlex SO400M | Qwen3.5-4B | Mejor precisión, pesos predeterminados |
| `16gb` | TOPIQ (0.93 SRCC) | SigLIP 2 NaFlex SO400M | Qwen3.5-2B | Pesos predeterminados |
| `8gb` | CLIP+MLP (0.76 SRCC) | CLIP ViT-L-14 | similitud CLIP | Los pesos predeterminados funcionan bien |
| `legacy` | CLIP+MLP en CPU | CLIP ViT-L-14 | similitud CLIP | Pesos predeterminados, más lento |

Todos los perfiles ejecutan además modelos complementarios de PyIQA (TOPIQ IAA, TOPIQ NR-Face, LIQE) y, opcionalmente, BiRefNet_dynamic para la saliencia del sujeto.

Ejecuta `--compute-recommendations` después de cambiar de perfil para analizar las distribuciones de puntuación.

## Flujo de ajuste de pesos

### Opción A: a través del visor (recomendada)

1. Abre `/stats` → pestaña **Categorías** → subpestaña **Pesos**
2. Desbloquea el modo edición
3. Selecciona una categoría en el desplegable del editor
4. Ajusta los controles deslizantes — la **vista previa de distribución de puntuaciones** en vivo muestra el impacto estimado
5. Haz clic en **Guardar** y luego en **Recalcular puntuaciones** para aplicar

El visor ejecuta `--recompute-category` internamente, actualizando solo las fotos de esa categoría.

### Opción B: a través de la CLI

#### 1. Analizar las puntuaciones actuales

```bash
python facet.py --compute-recommendations
```

Muestra:
- Distribuciones de puntuación por categoría
- Análisis de correlación de pesos
- Ajustes sugeridos

#### 2. Ajustar los pesos

Edita los pesos de categoría en `scoring_config.json`. Asegúrate de que sumen 100.

#### 3. Recalcular las puntuaciones

```bash
python facet.py --recompute-average               # Todas las categorías
python facet.py --recompute-category portrait      # Una sola categoría (más rápido)
```

Usa los embeddings almacenados: no se necesita GPU.

#### 4. Validar los cambios

```bash
python facet.py --compute-recommendations
```

Compara las distribuciones antes/después.

## Modo de comparación por pares

Entrena los pesos comparando pares de fotos:

### Configuración

1. Establece un `edition_password` no vacío en la configuración: `"viewer": { "edition_password": "your-password" }`
2. Inicia el visor: `python viewer.py`
3. Haz clic en el botón "Comparar"

### Interfaz de comparación

- Fotos en paralelo
- Teclado: A (gana la izquierda), B (gana la derecha), T (empate), S (omitir)
- La barra de progreso muestra las comparaciones hacia el mínimo de 50

### Orígenes de comparación

Las comparaciones llevan un marcador `source` para que el optimizador pueda ponderarlas por fiabilidad:

- `vote` — votos A/B explícitos desde la interfaz de comparación
- `culling` — derivados automáticamente de las decisiones de selección de ráfagas/similares: cada
  foto rechazada se empareja con hasta dos fotos conservadas del mismo grupo
  (con un tope de 12 pares por grupo). Las fotos conservadas ganan. Los votos explícitos sobre el mismo
  par nunca se sobrescriben.
- `rating` — pares sintéticos generados a partir de valoraciones por estrellas y favoritos

Por lo tanto, revisar los grupos de ráfaga en el visor amplía el conjunto de entrenamiento para la
optimización de pesos sin ningún esfuerzo adicional.

### Optimización de pesos

```bash
# Comprobar las estadísticas de comparación
python facet.py --comparison-stats

# Optimizar los pesos a partir de las comparaciones (se aplica solo si generaliza)
python facet.py --optimize-weights --optimize-category portrait

# Restringir los datos de entrenamiento a orígenes específicos
python facet.py --optimize-weights --optimize-category portrait --optimize-sources vote,culling

# Aplicar aunque no se cumpla el umbral de validación reservada
python facet.py --optimize-weights --optimize-category portrait --optimize-force

# Aplicar a todas las fotos
python facet.py --recompute-average
```

### Canalización de etiqueta a peso

Más allá de los votos A/B explícitos, otros dos flujos de etiquetas alimentan el optimizador:

1. Las **decisiones de selección** se capturan automáticamente en cada
   confirmación de ráfaga/similar (`source='culling'`).
2. Las **valoraciones por estrellas, los favoritos y los rechazos** se materializan en pares
   sintéticos con `python facet.py --sync-label-comparisons` (`source='rating'`).
   Volver a ejecutarlo re-sincroniza a partir de las etiquetas actuales, de modo que las valoraciones retiradas desaparecen.

El optimizador ajusta los pesos que mejor predicen qué foto ganó cada comparación (un modelo de Bradley-Terry) y cuenta cada fuente según su fiabilidad: vote 1.0, rating 0.7, culling 0.5. Un **empate** cuenta como media victoria más media derrota: es evidencia de que las dos fotos deberían puntuar parecido. El ajuste está **anclado a tus pesos actuales**: con pocas comparaciones la sugerencia se queda cerca de ellos, y se aleja más a medida que se acumulan votos. Entrena con cada foto tal como la ve el puntuador: las mismas métricas 0-10 que lee el agregado (incluidos `liqe`, `aesthetic_iaa`, `face_quality_iqa` y las métricas de saliencia del sujeto), con la nitidez ocular, la nitidez técnica y el color normalizados por percentil, la puntuación de exposición reconstruida a partir del histograma almacenado y el indicador de retrato de grupo. Los percentiles son los que guardó el último `--recompute-average`; antes del primer `--recompute-average` se usan tal cual las columnas almacenadas. Parte de los pesos que realmente usa el puntuador y escribe esos mismos: `quality_percent` se suma a `aesthetic` como hace el puntuador, y al aplicar se escribe `quality_percent: 0` en las categorías que lo tienen.

Los pesos **solo se aplican si generalizan**. Los pesos finales se ajustan con todas las comparaciones, pero la decisión de escribirlos se toma con comparaciones que el ajuste no ha visto: validación cruzada de 5 particiones con **particiones agrupadas por foto** (ninguna foto está a la vez en los pares de entrenamiento y en los reservados de una partición). Las predicciones sobre los pares reservados, agrupadas entre particiones, se comparan con los pesos actuales sobre esos mismos pares mediante una **prueba exacta de McNemar unilateral**, que pregunta si los pesos nuevos ganan significativamente más pares de aquellos en los que ambos discrepan. Los pesos solo se aplican cuando `p < 0.05` **y** la ganancia reservada es de al menos 2 puntos porcentuales; en caso contrario la ejecución registra las cifras y no escribe nada. Pasa `--optimize-force` para aplicarlos igualmente. Si la puerta no puede ejecutarse (muy pocos grupos independientes de fotos), no se aplica nada salvo que se fuerce. La optimización es por categoría y necesita al menos 30 comparaciones **decisivas** (ganó una foto) **en esa categoría**; los empates y las comparaciones de otras categorías no cuentan. Una categoría sin votos no puede ajustarse a partir de datos. El registro muestra, por ejemplo:

```
Held-out accuracy:      69.2% +/- 4.0 (5-fold, photo-grouped) - the honest estimate
Held-out gain:          +8.5 pp over the current weights on the same 389 pairs
McNemar (one-sided):    p = 0.0006 (only the fit right: 66, only the current weights right: 33)
NOT applying: the held-out gate needs p < 0.05 and a gain of at least 2.0 pp. ...   (logged when the gate fails)
```

Aplicar un resultado con `--optimize-weights` cambia cómo se ordenan entre sí las fotos de la categoría, no dónde queda la categoría frente a las demás. En la misma escritura de configuración que los pesos, `modifiers.bonus` se desplaza en sentido contrario al desplazamiento medio proyectado de la puntuación de las fotos de la categoría, de modo que el agregado medio de la categoría se queda donde estaba. El bonus se mantiene dentro de [-5, 5], y el registro indica cuándo ese límite deja corta la corrección. Una categoría sin entrada en `categories` (`others`) no tiene bonus y no se recentra. La instantánea de pesos registrada antes de escribir incluye el bonus, y restaurar una instantánea restaura pesos **y** bonus (las instantáneas registradas antes de este cambio, por `calibrate.py` o para `others` no tienen bonus y lo dejan intacto). Límite conocido: el botón **Aplicar** del visor y una edición manual en la pestaña Pesos escriben los pesos mediante `POST /api/config/update_weights`, que no recentra el bonus; solo lo hace `--optimize-weights`.

**Tras actualizar, vuelve a entrenar el ranker personal y el cabezal keeper (`--train-ranker`, `--train-keeper`), y después ejecuta `--recompute-average`.** El optimizador, el ranker y el cabezal keeper leen las mismas métricas por foto, que ahora incluyen la nitidez ocular, la nitidez técnica y el color normalizados por percentil; un modelo guardado por una versión anterior no puede detectar el cambio.

Cadencia recomendada:

```bash
python facet.py --mine-insights          # qué señal existe, deriva, salud
python facet.py --sync-label-comparisons # actualizar los pares derivados de valoraciones
python facet.py --optimize-weights       # aprender los pesos de todos los orígenes
python facet.py --recompute-average      # aplicar + persistir la instantánea de percentiles
```

### Ajuste de pesos en la interfaz

1. Abre el panel de vista previa de pesos durante la comparación
2. Ajusta los controles deslizantes para ver los cambios de puntuación en tiempo real
3. Haz clic en "Sugerir pesos" para obtener valores optimizados
4. Actualiza manualmente la configuración

**Sugerir pesos** también responde a una pregunta más precisa que la puerta reservada anterior: ¿en qué medida los pesos *actualmente en uso* de esta categoría ya coinciden con tus propias comparaciones? Al pulsarlo devuelve `accuracy_before` — el porcentaje de los pares etiquetados de esa categoría (votos A/B, descartes y pares derivados de valoraciones) cuyo ganador predicen correctamente los pesos en uso — junto a `accuracy_after`, la misma cifra para los pesos sugeridos. Ambas se miden sobre las mismas comparaciones con las que se ajustaron los pesos, así que `accuracy_after` es optimista. Se muestran lado a lado en la pestaña Sugerencias de pesos y en la barra lateral de la pestaña Comparación A/B cada vez que lo ejecutas (`GET /api/comparison/learned_weights`). Que se ofrezca **Aplicar** sigue la misma puerta reservada de McNemar que la CLI, no esa ganancia optimista: `suggest_changes` es verdadero solo cuando se supera la puerta. La petición ejecuta la validación cruzada solo cuando la ganancia optimista ya alcanza 2 puntos porcentuales (`?use_cv=true` la ejecuta en todos los casos). El visor necesita `min_comparisons_for_optimization` (valor incluido 50) comparaciones decisivas de esa categoría, mientras que la CLI necesita 30; por debajo de eso el botón informa del déficit («Need at least N decisive comparisons (have K)») en lugar de un número.

## Añadir categorías personalizadas

```json
{
  "name": "underwater",
  "priority": 62,
  "filters": {
    "required_tags": ["underwater"],
    "tag_match_mode": "any"
  },
  "weights": {
    "aesthetic_percent": 40,
    "color_percent": 25,
    "composition_percent": 20,
    "exposure_percent": 15
  },
  "modifiers": {
    "noise_tolerance_multiplier": 0.3,
    "bonus": 0.5
  },
  "tags": {
    "underwater": ["underwater", "scuba", "diving", "ocean"],
    "fish": ["fish", "coral", "reef"]
  }
}
```

Añádela al array `categories` en `scoring_config.json`, luego ejecuta `--recompute-average` (o `--recompute-category underwater` solo para la nueva categoría).

## Ejemplos de flujo de trabajo

### Ajustar la categoría de conciertos

```bash
# Edita scoring_config.json:
# Busca la categoría "concert", ajusta:
#   "noise_tolerance_multiplier": 0.05
#   "exposure_percent": 5

python facet.py --recompute-category concert
```

O usa el editor de pesos del visor en `/stats` → Categorías → Pesos para vista previa en vivo y recálculo con un clic.

### Cambiar al perfil 8gb

```bash
# Edita: "vram_profile": "8gb"
python facet.py --compute-recommendations  # Analizar
# Reduce aesthetic_percent en las categorías si es necesario
python facet.py --recompute-average
```

### Añadir la categoría submarina

1. Añade la definición de la categoría (ver arriba)
2. Ejecuta `python facet.py --validate-categories`
3. Ejecuta `python facet.py --recompute-average`
