# preproc — resumen y preprocesamiento de video (Cap. 4)

Convierte cada video en **miniclips de 32 cuadros a 224 × 224** para Video Swin (Swin-B), con los registros que usa el Cap. 5. Cada video produce tantos miniclips como candidatas conserve el filtro de redundancia: no hay selección por movimiento ni un número fijo por video.

## Instalación

1. **ffmpeg ≥ 4.0** con `ffprobe` en el `PATH` (con `libx264` para el modo de diagnóstico).
2. **PyTorch** según tu CUDA (https://pytorch.org/get-started/locally/). Instálalo antes que el paquete para que pip no lo reemplace.
3. El paquete: `pip install -e preprocesamiento` (con `[parquet]` para el resumen en Parquet).
4. **TransNet V2** (dependencia externa, no se instala con pip):
   ```bash
   git clone https://github.com/soCzech/TransNetV2   # con Git LFS para los pesos
   cd TransNetV2/inference-pytorch
   pip install tensorflow                              # solo para convertir los pesos, una vez
   python convert_weights.py --test                    # crea transnetv2-pytorch-weights.pth
   ```
   Indica `transnet.repo_dir` y `transnet.weights` en `config.yaml`.
5. Comprueba la instalación: `python -m preproc check --config config.yaml --gpu 0`

## Uso

```bash
python -m preproc run --input D:/datos/pornography2k --dataset pornography2k \
                      --config config.yaml --output salida --workers 8 --gpu 0
```

- `--gpu -1`: TransNet V2 en CPU dentro de cada trabajador. `--workers 0`: sin paralelismo.
- La corrida se puede reanudar: un video con `meta/<id>.json` y el mismo hash de configuración se omite. `--force` reprocesa.
- `python -m preproc report --output salida --dataset pornography2k` regenera el resumen y las estadísticas.

Desde Python:

```python
from preproc import load_config, check_ffmpeg, process_video, TransNetDetector, normalize, to_cthw
cfg = load_config("config.yaml")
clips, meta = process_video("video.mp4", "video.mp4", "video", cfg,
                            TransNetDetector(cfg.transnet, device="cuda:0"),
                            check_ffmpeg())
x = to_cthw(normalize(clips[0], cfg.output.mean, cfg.output.std))   # (3, 32, 224, 224) float32
```

Leer una salida para el entrenamiento:

```python
from preproc import check_ffmpeg, load_miniclip, load_miniclips
ff = check_ffmpeg()                                                              # una vez
clip = load_miniclip("salida/pornography2k/miniclips/vPorn000001/vPorn000001_000.mp4", ff)  # (32, 224, 224, 3) uint8
clips = load_miniclips("salida/pornography2k/miniclips/vPorn000001", ff)        # (K, 32, 224, 224, 3) uint8
```

Los `.mp4` también se leen con PyAV, decord o torchvision. Con muchos archivos, conviene trabajar en el disco local de Colab y copiar a Drive un solo `.tar` por conjunto.

## Salidas (`salida/<conjunto>/`)

| Archivo | Contenido |
|---|---|
| `miniclips/<id>/<id>_000.mp4`, `_001.mp4`, … | **Un `.mp4` por miniclip**, en orden temporal: 32 cuadros de 224 × 224, H.264 `yuv420p` (crf 18 por defecto), a `fps / r` (12,5 fps los de ventana, 25 fps los de tomas cortas). Se ven en cualquier reproductor y se leen con `load_miniclip` / `load_miniclips` |
| `meta/<id>.json` | datos originales, cuadros tras cada paso, tomas y ventanas, miniclips (ventana, cuadro inicial, toma, m_j, respaldo), ventanas descartadas con su distancia, tiempos por paso, versión y hash |
| `resumen_<conjunto>.csv` | una fila por video con las columnas numéricas de los JSON |
| `estadisticas_<conjunto>.json` | totales y medias, % de reducción, videos por caso de respaldo, fallidos, tiempos |
| `errores_<conjunto>.log` | `fecha  id  ruta  motivo` de cada video fallido |
| `logs/preproc_<fecha>.log` | registro completo de la corrida |

## Calibración con el modo de diagnóstico

`--debug-samples N` (los primeros N videos) o `--debug-ids a,b` guarda en `debug/<id>/` **un .mp4 por miniclip** con un nombre descriptivo. Su contenido es equivalente a `miniclips/<id>/`, que ya se puede ver directamente; el modo de diagnóstico solo agrega los nombres. El nombre indica el orden, la ventana, el inicio (mm:ss), la toma, m_j y el caso de respaldo. Los videos de diagnóstico se reprocesan siempre.

Para θ_L, θ_σ, θ_E y θ_H, revisa en `meta/<id>.json` los campos `cuadros`, `ventanas_descartadas` (con su distancia) y `respaldo`, y en `estadisticas_<conjunto>.json` la reducción y los casos de respaldo.

## Desviaciones y supuestos

1. **Umbral de transición:** `p > 0.5`, como `predictions_to_scenes` del código oficial (el Cap. 4 dice ≥).
2. **Pantalla fija:** el tramo i..j cumple d_k < θ_E para k = i+1..j y mide j − i + 1 ≥ 25; el primer cuadro del tramo también se elimina. Los cuadros negros y las pantallas fijas se evalúan sobre la línea completa y se unen.
3. **Luminancia:** BT.601 calculada desde RGB. No se usa `-pix_fmt gray`, que puede dejar el negro en 16. θ_L, θ_σ y θ_E dependen de la resolución de análisis (64 × 36).
4. **pHash:** reducción por área desde el gris de 64 × 36; la mediana incluye el coeficiente DC. Para buscar duplicados entre conjuntos hay que usar la misma función en ambos.
5. **Entrada de TransNet V2:** 48 × 27 con escalado bicúbico, después de `fps=25`. El código oficial escala sin cambiar la tasa. Cada segmento del paso 2 se procesa con su propio relleno.
6. **Sin selección por movimiento (paso 5 eliminado):** cambio respecto del Cap. 4. **Todas** las candidatas que conserva el paso 4 son miniclips, en orden temporal. No hay K fijo ni K variable, ni ventanas desplazadas ni copias: la cantidad de miniclips de cada video solo depende de sus tomas y del filtro de redundancia (θ_H). El movimiento m_j de cada miniclip se registra en el JSON solo como descriptor.
7. **N = 0** (ninguna candidata): un único miniclip de la toma más larga tras los pasos 2 y 3 (si hay empate, la primera), con índices `inicio + ⌊(2i+1)·longitud / 64⌋`. Si la toma tiene menos de 32 cuadros, se repiten cuadros en orden.
8. **Candidatas de tomas cortas** (`summary.short_shot_windows: true`): cambio respecto del Cap. 4. Toda toma de al menos L = T·r cuadros genera ventanas, y el paso 4 conserva siempre la primera de cada toma. Además, cada toma de T a L − 1 cuadros (1,28–2,5 s con r = 2) aporta una candidata de T cuadros consecutivos (r = 1) centrada en la toma, que también se conserva siempre. Los miniclips tienen entonces dos duraciones (2,56 s con r = 2 y 1,28 s con r = 1): cada uno registra su `r` en el JSON y se guarda a 25 / r fps. Las tomas de menos de T cuadros no aportan candidata (`conteos.tomas_sin_candidata`).
9. **Ampliar tomas cortas a 64 cuadros:** descartado. Ninguna forma da 2,56 s reales sin cruzar un corte o inventar cuadros; Swin recibe 32 cuadros reales en todos los casos.
10. **Extracción:** se vuelve a decodificar desde el inicio con el mismo `fps=25` y sin `-ss`, para garantizar los mismos índices. Se detiene al obtener el último cuadro pedido. La verificación de alineación es aproximada y solo genera un aviso.
11. **`short_side_crop`:** usa la proporción de pantalla calculada con ffprobe (SAR y rotación).
12. **Salida y determinismo:** cada miniclip se guarda como un `.mp4` H.264 `yuv420p` con pérdida casi imperceptible (`output.crf`, 18 por defecto), en lugar de `.npy`. Los cuadros que recibe Swin al decodificar **no son idénticos al bit** a los extraídos. No se mide la pérdida en cada corrida; la corrida de prueba con `medium` dio un PSNR medio de 39,3 dB (mínimo por video 32,5 dB). Los archivos son reproducibles con las mismas versiones de ffmpeg, numpy y torch y el mismo hardware. TransNet V2 en GPU y en CPU puede diferir en probabilidades cercanas a 0,5. El JSON incluye tiempos, así que nunca es idéntico entre corridas.
13. **Hash de configuración:** no incluye rutas, hilos, `batch_size` ni la sección `debug`. Sí incluye `output.crf` y `output.preset`, porque cambian los cuadros guardados. Cambiar el archivo de pesos no invalida las salidas existentes.
14. **Etiquetas:** el código no las lee. Los ids se derivan del nombre del archivo, que en Pornography-2k puede contener la clase (`vPorn…`, `vNonPorn…`).
