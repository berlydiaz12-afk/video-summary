# preproc_study — estudio del resumen de video sobre un solo video

Aplica el procedimiento del Cap. 4 a **un video** y muestra cada decisión:

- qué cuadros se eliminan y por qué;
- dónde están los cortes;
- qué candidatas se generan (ventanas y tomas cortas) y cuáles se descartan;
- qué miniclips resultan y qué recibe Swin.

Sirve para entender el procedimiento, calibrar θ_L, θ_σ, θ_E, θ_H y r, y documentar el capítulo con ejemplos reales.

- Pensado para **cuadernos Jupyter / Colab**: no tiene interfaz de línea de comandos.
- Todas las decisiones se toman con las funciones de `preproc`, sin reimplementarlas.
- **ffmpeg** lee el video y escribe los `.mp4`; **OpenCV** procesa y dibuja las imágenes; **matplotlib** dibuja los gráficos.
- Lee el mismo `config.yaml` que `preproc`.

## Procedimiento que estudia

Es el mismo de `preproc` (ver `REGISTRO_CAMBIOS.md`, cambio 3d). **No hay selección por movimiento**: toda candidata conservada es un miniclip.

1. Decodificación a 25 fps (gris de 64 × 36 y RGB de 48 × 27 para TransNet V2).
2. Filtrado de cuadros negros, uniformes y pantallas fijas.
3. Tomas con TransNet V2 y candidatas:
   - ventanas de L = T·r = 64 cuadros (r = 2) en las tomas de ≥ 64 cuadros;
   - con `short_shot_windows: true`, una candidata de 32 cuadros consecutivos (r = 1), centrada, en cada toma de 32 a 63 cuadros.
4. Eliminación de redundancia por pHash dentro de cada toma (θ_H). Solo se comparan las ventanas; las candidatas de tomas cortas son las únicas de su toma y se conservan siempre.
5. Miniclips: todas las candidatas conservadas, en orden temporal. Si no hay ninguna (N = 0), un único miniclip de la toma más larga.
6. Extracción a 224 × 224 y almacenamiento de **un `.mp4` por miniclip** (H.264 `yuv420p`, `output.crf`, `output.preset`), a 25 / r fps.

## Instalación en Colab

Requisitos: `preproc` ya configurado (TransNet V2 clonado y pesos convertidos), ffmpeg **con libx264** (lo exige el almacenamiento de los miniclips, igual que en `preproc`), `opencv-python` y `matplotlib`. Los tres últimos vienen en Colab.

```python
# Celda 1: rutas y dependencias
import sys
sys.path.insert(0, "/content/preprocesamiento")   # carpeta que contiene preproc/ y preproc_study/
!pip install -q -e /content/preprocesamiento      # dependencias de preproc (numpy, scipy, PyYAML, torch)

# Celda 2: estudio completo de un video
from preproc_study import study_video
summary = study_video(
    "/content/videos/ejemplo.mp4",
    "/content/preprocesamiento/config.yaml",
    "/content/estudio/ejemplo",
    gpu=0,                      # -1 = CPU
    annotated_video=True,
    max_images=200,             # máximo de tiras por tipo (los CSV siempre están completos)
    strides=(1, 2, 4, 8),
    hamming_thresholds=range(0, 33, 2),
    sheets=True,                # una hoja PNG con los 32 cuadros de cada miniclip
    measure_storage=True,       # relee cada .mp4: PSNR del almacenamiento y estadísticas de lo que recibe Swin
    verify=False,               # True: compara con preproc.pipeline.process_video
)
summary["miniclips"]
```

En `config.yaml`, `transnet.repo_dir` y `transnet.weights` deben apuntar a `/content/TransNetV2` y `/content/weights/...`. Las rutas relativas se resuelven respecto de la carpeta del YAML.

## Uso procedimiento por procedimiento

```python
import pandas as pd
from IPython.display import display
from preproc_study import VideoStudy, plots

study = VideoStudy("/content/videos/ejemplo.mp4", "/content/preprocesamiento/config.yaml", gpu=0)

decoded = study.decode()                         # gray (n, 36, 64), rgb48 (n, 27, 48, 3), frame mapping
filtering = study.filter_frames()                # umbrales del YAML
filtering_20 = study.filter_frames(luma_threshold=20, static_threshold=0.8)   # otra prueba, sin tocar el YAML
display(plots.plot_frame_filtering(filtering_20, decoded.fps))

shots = study.detect_shots(filtering_20)         # TransNet V2 se reutiliza mientras los segmentos no cambien
shots_r4 = study.detect_shots(filtering_20, stride=4)
display(plots.plot_shot_lengths(shots, decoded.fps))

redundancy = study.remove_redundancy(shots, hamming_threshold=12)
miniclips = study.plan_miniclips(shots, redundancy)
display(pd.DataFrame(miniclips.rows()))          # tipo, toma, cuadro inicial, r, fps de reproducción, m_j

rows = study.sweep_hamming_threshold(range(0, 33, 2))
display(pd.DataFrame(rows))                      # ventanas descartadas y miniclips resultantes por θ_H
display(plots.plot_hamming_threshold_sweep(rows, current=12))

clip = study.read_miniclip(0)                    # (32, 224, 224, 3) uint8, sin escribir nada
frames = study.read_frames([100, 200, 300])      # cuadros sueltos a 320 px, por índice de la línea a 25 fps
```

Cada método guarda su resultado en el objeto (`study.decoded`, `study.filtering`, `study.shots`, `study.redundancy`, `study.miniclips`, `study.swin_input`) y lo devuelve. Los argumentos con nombre reemplazan campos de la sección correspondiente del YAML solo para esa llamada:

| Método | Parámetros que se pueden cambiar |
|---|---|
| `filter_frames` | `luma_threshold`, `std_threshold`, `static_threshold`, `static_min_frames` |
| `detect_shots` | `threshold`, `frames_per_clip`, `stride`, `short_shot_windows` |
| `remove_redundancy` | `hamming_threshold`, `phash_step` |
| `plan_miniclips` | ninguno: la regla es fija (todas las candidatas conservadas) |
| `extract_miniclips` | `folder` (sin carpeta no escribe nada), `sheets`, `measure_storage` |

`study.run(out, ...)` recalcula todo con `study.config` y escribe las salidas. Para correrlo con otros umbrales, modifica la configuración antes, por ejemplo `study.config.summary.hamming_threshold = 12`.

## Salidas (`out/`)

| Carpeta / archivo | Contenido |
|---|---|
| `01_decoding/metadata.json` | Datos originales (ffprobe), cuadros a 25 fps, correspondencia estimada (repetidos y omitidos), versiones |
| `01_decoding/frame_mapping.csv` | `frame, time_s, source_frame, source_time_s` |
| `02_frame_filtering/frames.csv` | Por cuadro: luminancia media, desviación, d_k, indicadores `black`, `uniform`, `static`, `removed` y motivo |
| `02_frame_filtering/series.png` | Las tres series con θ_L, θ_σ, θ_E y los tramos eliminados sombreados por motivo |
| `02_frame_filtering/removed_runs.csv`, `removed_runs/run_<i>.png` | Cada tramo eliminado; la tira muestra el cuadro anterior, el primero, el del medio, el último y el posterior |
| `02_frame_filtering/segments.csv` | Segmentos que sobreviven |
| `03_shot_detection/probabilities.csv`, `.png` | p de TransNet V2 por cuadro, umbral, límites de segmento |
| `03_shot_detection/transitions.csv`, `transitions/transition_<i>.png` | Cada transición; la tira muestra 3 cuadros antes, la transición y 3 después, con p |
| `03_shot_detection/shots.csv` | Tomas, con cuántas ventanas generan y si aportan una candidata de toma corta |
| `03_shot_detection/candidates.csv` | Todas las candidatas con el índice de `preproc` (primero las ventanas, después las de tomas cortas), tipo y r |
| `03_shot_detection/shot_lengths.png` | Histograma de duración de tomas con L para r = 8, 4 y 2, y T (mínimo para una candidata de toma corta) |
| `03_shot_detection/stride_sweep.csv` | Para cada r: tomas con ventana, ventanas, candidatas de tomas cortas, tomas sin candidata, cobertura del tiempo útil |
| `04_redundancy_removal/decisions.csv` | Por ventana: referencia, distancia, decisión, ventana anterior de la toma y su distancia |
| `04_redundancy_removal/signatures.csv` | pHash hexadecimal en cada posición (0, 25, 50…) |
| `04_redundancy_removal/matrices/shot_<n>.png` | Matriz de distancias entre las ventanas de una toma (× = descartada) |
| `04_redundancy_removal/discarded/window_<i>.png` | Ventana descartada bajo su referencia, con la distancia por posición |
| `04_redundancy_removal/hamming_threshold_sweep.csv`, `.png` | Para cada θ_H: ventanas conservadas y descartadas, **miniclips resultantes**, segundos cubiertos y caso de respaldo |
| `05_miniclips/miniclips.csv` | Por miniclip: tipo (`conservada`, `toma_corta`, `toma_mas_larga`), candidata, toma, cuadro inicial, r, duración, fps de reproducción, m_j (solo descriptor) |
| `05_miniclips/miniclips.json` | Conteos (por tipo y por r, cuadros únicos, tiempo cubierto, caso de respaldo) y los 32 índices de cada miniclip |
| `06_swin_input/miniclip_<k>.mp4` | **El mismo archivo que guarda `preproc`** en `miniclips/<id>/<id>_<k>.mp4`: 32 cuadros de 224 × 224, H.264 `yuv420p` (`output.crf`, `output.preset`), a 25 / r fps |
| `06_swin_input/miniclip_<k>.png` | Los 32 cuadros en cuadrícula, con índice, tiempo, r y fps de almacenamiento (`sheets=False` para omitirlas) |
| `06_swin_input/tensors.json` | Forma del tensor `(3, 32, 224, 224)`, media y desviación por canal antes y después de normalizar, PSNR del almacenamiento, bytes y fps de cada archivo |
| `timeline.png` | Video completo: p, tramos eliminados, tomas, candidatas, conservadas y descartadas, miniclips |
| `annotated_video.mp4` | Video a 25 fps y 320 px con el estado de cada cuadro |
| `report.md`, `summary.json` | Números clave de cada procedimiento, avisos y tiempos |
| `verification.md` | Solo con `verify=True`: comparación con `process_video` |

Los colores son los mismos en todas las salidas:

- motivo de eliminación: violeta = negro, naranja = uniforme, aqua = pantalla fija;
- amarillo = transición;
- gris = ventana candidata; verde = ventana conservada; rojo = ventana descartada; marrón = candidata de toma corta;
- rosa = miniclip.

En el video anotado, la barra inferior repite ese color y un punto rosa marca los 32 cuadros que recibe Swin.

## Cómo calibrar

- **θ_L y θ_σ (cuadros negros o uniformes):**
  - En `series.png`, las curvas de luminancia y desviación deben quedar bajo la línea solo en fundidos a negro, cortinillas y placas lisas.
  - Revisa las tiras `removed_runs/` con motivo `black` o `uniform`: si muestran escenas oscuras con contenido, baja el umbral.
  - Si en el video anotado ves cuadros negros que se conservan, súbelo.
- **θ_E (pantalla fija):**
  - En la serie d_k (escala logarítmica), las pantallas fijas aparecen como mesetas cerca de 0.
  - Si las tiras `static` muestran escenas con movimiento leve (personas quietas, planos largos), baja θ_E o sube `static_min_frames`.
- **Umbral de transición:**
  - Las tiras `transitions/` con p apenas sobre el umbral son las dudosas.
  - Si son cortes falsos (destellos, movimiento rápido), sube el umbral con `detect_shots(threshold=...)`, sin volver a ejecutar el modelo.
- **r (stride) y tomas cortas:**
  - `stride_sweep.csv` y `shot_lengths.png` muestran cuántas tomas alcanzan L, cuántas aportan una candidata de toma corta y cuántas no aportan nada.
  - Con un r grande, más tomas quedan por debajo de L; con `short_shot_windows`, las de T a L − 1 cuadros siguen aportando una candidata de r = 1.
- **θ_H (el único control de cantidad de miniclips):**
  - `hamming_threshold_sweep.png` muestra cuántos miniclips da el video para cada θ_H; la línea punteada marca las candidatas de tomas cortas, que no dependen de θ_H.
  - Busca el codo de la curva y contrasta con las tiras `discarded/`: con un θ_H correcto, la ventana descartada debe verse casi igual a su referencia.
  - Las matrices muestran si una toma es homogénea (todo claro) o cambia (bloques oscuros).

## Diferencias con el paquete en masa

La decodificación, el filtrado, la detección de tomas, las candidatas, la redundancia, los miniclips, la extracción y el almacenamiento son **las mismas funciones** de `preproc`. Con la misma configuración, los resultados coinciden con `process_video`, y `verify=True` lo comprueba (los píxeles, con un SHA-1 por miniclip). Lo que este paquete agrega o hace distinto:

1. **Correspondencia con el video original (estimada).**
   - ffmpeg no informa qué cuadro original usó el filtro `fps` para cada cuadro de salida.
   - Se reconstruye con los instantes de los paquetes (ffprobe, sin decodificar), redondeando cada uno al cuadro de salida más cercano: el cuadro n muestra el último original cuyo instante redondeado es ≤ n.
   - Las listas de edición, los paquetes descartados o un tiempo de inicio distinto de cero pueden correrla un cuadro. Solo es informativa: ningún resultado depende de ella.
2. **Tercera pasada de decodificación.**
   - Una pasada de ffmpeg a 320 px (o al tamaño de miniatura si no se pide el video anotado) produce el video anotado y las miniaturas de las tiras.
   - Se guardan solo los cuadros que las tiras necesitan, así que la memoria no crece con la duración.
   - Usa el mismo filtro `fps`; si cuenta un número distinto de cuadros, se registra un aviso.
3. **Escalado de las miniaturas:**
   - ffmpeg escala por área al tamaño de visualización (con SAR y rotación);
   - OpenCV (`cv2.resize`, `INTER_AREA`) reduce a 160 px.

   Estas imágenes no intervienen en ninguna decisión.
4. **Miniclips uno por uno.**
   - Como en `preproc`, cada miniclip se extrae, se verifica, se codifica y se libera antes del siguiente: la memoria no crece con la cantidad de miniclips.
   - No se guarda ningún `.npy`: el formato de salida es el `.mp4` de cada miniclip.
5. **Pérdida por almacenamiento** (`measure_storage=True`).
   - Cada `.mp4` se vuelve a leer con `preproc.load_miniclip`; se reporta el PSNR frente a los cuadros extraídos y las estadísticas del tensor se calculan sobre lo decodificado, que es lo que recibe Swin.
   - `preproc` no hace esta medición en cada corrida (cuesta una lectura extra por miniclip).
6. **Video anotado con pérdida.** `annotated_video.mp4` se codifica en *streaming* con libx264, o mpeg4 si falta libx264. Solo sirve para mirar.
7. **Límite de imágenes.**
   - Con más tramos, transiciones o descartes que `max_images`, se dibuja un subconjunto espaciado uniformemente a lo largo del video.
   - Las matrices se dibujan para las tomas con más ventanas.
