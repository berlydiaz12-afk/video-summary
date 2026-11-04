# Registro de cambios: resumen y preprocesamiento de video (Cap. 4)

Este registro documenta los cambios hechos al procedimiento del Cap. 4 durante su implementación, con el motivo y la evidencia de cada uno. Los cambios que afectan al texto de la tesis lo indican de forma explícita.

**Conjunto de prueba** usado en todas las mediciones: 20 videos de Pornography-2k (10 `vNonPorn`, 10 `vPorn`), 2,1 h en total, 189 117 cuadros a 25 fps, **556 MB** en su formato original.

---

## 1. Salida en `.mp4` sin pérdida: el resumen pesa más que los videos originales

**Fecha:** 27 oct 2026 · **Estado:** reemplazado por el cambio 1b (H.264 `yuv420p` crf 18, un archivo por miniclip).

### Qué se cambió

Los miniclips dejaron de guardarse en `.npy` (píxeles crudos) y pasaron a guardarse en `.mp4` con **H.264 sin pérdida en RGB** (`libx264rgb -qp 0 -pix_fmt rgb24`). Se escribe un archivo por video con sus K_v × 32 cuadros seguidos, donde el cuadro `k·32 + t` es el cuadro t del miniclip k. Al guardar, cada archivo se decodifica y se compara al bit con el arreglo original.

**Motivo:** el `.npy` ocupa 4,8 MB por miniclip (224 × 224 × 3 × 32 bytes) y no se comprime. Se buscaba un formato comprimido que, a la vez, entregara a Swin exactamente los mismos cuadros.

### Resultado (r = 2, D = 10 s)

| | Cuadros | Tamaño total | Por cuadro |
|---|---|---|---|
| Videos originales (H.264 con pérdida) | 189 117 | **556 MB** | ~2,9 KB |
| Miniclips en `.mp4` sin pérdida | 24 032 (751 miniclips × 32) | **1,21 GB** | ~50,4 KB |
| Referencia: píxeles crudos (`.npy`) | 24 032 | 3,62 GB | 147 KB |

- El resumen conserva solo el **12,7 % de los cuadros**, pero ocupa **2,2 veces más** que los videos originales.
- Por cuadro, la salida ocupa **~17 veces más** que el original.
- Frente al `.npy`, el formato sin pérdida reduce el tamaño apenas **3 veces**.
- Cada miniclip ocupa en promedio 1,6 MB. El video más pesado es vPorn000006: 201 miniclips, 345 MB.
- **Estimación para Pornography-2k completo** (44 GB originales): **~80–95 GB** de salida.

Todos los archivos pasaron la verificación al bit (20 procesados, 0 fallidos, `salida_verificado = True`).

### Por qué crece el tamaño

1. **Compresión sin pérdida frente a con pérdida.** El H.264 de los originales descarta la información poco visible (detalle fino, ruido) y aprovecha la semejanza entre cuadros, con lo que logra entre 50 y 100 veces menos que los píxeles crudos. Sin pérdida hay que conservar cada píxel exacto, y solo se consiguen unas 3 veces menos.
2. **Color completo.** Los originales guardan el color a un cuarto de resolución (YUV 4:2:0, 1,5 bytes por píxel). La salida guarda RGB completo (3 bytes por píxel): el doble de datos antes de comprimir.
3. **Se conservan los artefactos del original.** Los cuadros extraídos ya traen los bloques y el ruido de la compresión original. Guardarlos sin pérdida conserva también esos artefactos, que son lo más difícil de comprimir.

Reducir la resolución de 320 × 240 a 224 × 224 compensa muy poco frente a estos tres factores.

**Observación:** guardar sin pérdida **no aporta fidelidad respecto de la escena real**, porque los originales ya perdieron información al comprimirse. Solo garantiza que los cuadros sean idénticos a los que decodificó ffmpeg, con sus defectos incluidos.

### Problemas detectados

- **Los archivos no se pueden ver en reproductores comunes.** Usan el perfil H.264 *High 4:4:4 Predictive* (código 244, confirmado en la cabecera). El reproductor de Windows y los navegadores no lo soportan y muestran error o colores incorrectos, aunque el archivo está íntegro. VLC y ffplay sí lo abren.
- **La división en miniclips no se ve en el archivo.** Al reproducir el `.mp4` de un video, los miniclips pasan uno tras otro a 12,5 fps sin ninguna separación. La división solo la reconstruye `load_miniclips()`.

### Opciones en evaluación

| Opción | Idéntico al bit | Se ve en Windows | Prueba (20 videos) | Pornography-2k |
|---|---|---|---|---|
| Sin pérdida (actual) | Sí | No (solo VLC o ffplay) | 1,21 GB | ~80–95 GB |
| Sin pérdida + vista previa `yuv420p` de cada miniclip | Sí | Sí (la vista previa) | ~1,3–1,5 GB | ~85–100 GB |
| H.264 `yuv420p` crf 18, con pérdida casi imperceptible | No | Sí | ~120–250 MB (estimado) | ~8–17 GB (estimado) |

Además, se propone guardar **un archivo por miniclip** (`miniclips/<id>/<id>_000.mp4`, …) en lugar de uno por video.

**Impacto en la tesis:** el Cap. 4 debe indicar el formato en que se almacenan los miniclips y, si se elige la opción con pérdida, que los cuadros se recomprimen y que la normalización se aplica después de decodificarlos.

---

## 1b. Salida definitiva: un `.mp4` por miniclip, H.264 `yuv420p` crf 18

**Fecha:** 27 oct 2026 · **Estado:** implementado, sin probar todavía. **Afecta al Cap. 4.**

### Qué se cambió

- **Formato:** H.264 `yuv420p` con `libx264`, crf 18 (configurable en `output.crf`) y preset `veryfast` (antes `medium`; cambiado el 31 oct 2026 porque el paso 6 ocupaba el 69 % del tiempo con 1986 miniclips), con el espacio de color BT.601 declarado en el archivo. Reemplaza al modo sin pérdida en RGB.
- **Organización:** **un archivo por miniclip**, `miniclips/<id>/<id>_000.mp4`, `<id>_001.mp4`, … en orden temporal. Cada archivo tiene 32 cuadros de 224 × 224 y se reproduce a `fps / r` (12,5 fps con r = 2), la duración real de la ventana.
- **Control de calidad:** al principio se medía el **PSNR** de cada miniclip decodificado. **Se eliminó el 31 oct 2026**: no cambiaba ningún miniclip ni lo usaba el entrenamiento, y obligaba a lanzar un segundo ffmpeg por miniclip (~2000 procesos extra en la prueba). La corrida de prueba con `medium` dejó el dato para la tesis: PSNR medio 39,3 dB, mínimo por video 32,5 dB.
- **Lectura:** `load_miniclip(ruta)` devuelve `(32, 224, 224, 3)` y `load_miniclips(carpeta)` devuelve `(K_v, 32, 224, 224, 3)`.

### Motivo

- **Tamaño:** se estima que la salida quedará por debajo de los videos originales (~8–17 GB para Pornography-2k, frente a ~80–95 GB sin pérdida). Se confirmará con la próxima corrida.
- **Fidelidad:** la exactitud al bit no aportaba fidelidad real, porque los originales ya están comprimidos con pérdida. Además, el color de los originales ya era 4:2:0, así que submuestrear el color no pierde casi nada nuevo, y Swin se preentrenó con videos decodificados con pérdida.
- **Revisión:** los miniclips se pueden ver en cualquier reproductor, y la división en miniclips se ve directamente en las carpetas.

### Consecuencias

- Los cuadros que recibe Swin **no son idénticos al bit** a los extraídos (PSNR medio 39,3 dB en la prueba con `medium`).
- `output.crf` y `output.preset` ahora entran en el hash de la configuración, porque cambian los cuadros guardados. Las salidas anteriores se reprocesan.
- Con D = 10 s, Pornography-2k generaría unos 50 000 archivos. Conviene copiar a Google Drive un `.tar` por conjunto, no los archivos sueltos.

**Impacto en la tesis:** el Cap. 4 debe indicar que cada miniclip se almacena como H.264 crf 18 y que la normalización se aplica después de decodificarlo. Si hace falta reportar la pérdida por almacenamiento, el dato es el PSNR de la corrida de prueba (39,3 dB).

---

## 2. Tasa de muestreo temporal: r = 8 → r = 2

**Fecha:** oct 2026 · **Estado:** implementado. **Afecta al Cap. 4.**

### Qué se cambió

`summary.stride` pasa de 8 a 2. La longitud de ventana baja de L = 256 cuadros (10,24 s) a **L = 64 cuadros (2,56 s)**.

### Motivo (primera corrida, r = 8 y K = 8)

- Solo el 28 % de las tomas (111 de 403) alcanzaba los 256 cuadros. En los videos no explícitos, el 5 % (14 de 255).
- La mitad de los videos cayó en casos de respaldo con copias: 6 en N = 0 y 4 en N < K con repetición. Solo el 65 % de los cuadros guardados era distinto.
- **Sesgo por clase:** 6 de 10 videos no explícitos cayeron en N = 0, frente a 0 de 10 explícitos. En los no explícitos, solo el 39 % de los cuadros era distinto (90 % en los explícitos). El modelo podría aprender a reconocer el formato de la entrada en lugar del contenido.
- La duración media de una toma era de 4,6 s en los no explícitos y de 42 s en los explícitos.

Con la inspección de 7 videos no explícitos, las tomas que generan ventana pasan del **4 % (r = 8) al 36 % (r = 2)**. Además, r = 2 coincide con el muestreo de preentrenamiento de Video Swin en Kinetics (32 × 2).

### Consecuencias

- **θ_H debe recalibrarse.** Con r = 8, la distancia entre ventanas consecutivas de una misma toma estaba casi siempre entre 18 y 32, así que θ_H = 10 casi nunca descartaba nada (1 de 587 ventanas). Con ventanas de 2,56 s las distancias bajan.
- Las tomas de menos de 64 cuadros (< 2,56 s) siguen sin generar ventana.

---

## 3. Número de miniclips variable por video (parámetro D)

**Fecha:** 27 oct 2026 · **Estado:** eliminado en el cambio 3d (dependía de la selección por movimiento). **Afecta al Cap. 4.**

### Qué se cambió

Se agregó el parámetro `summary.seconds_per_clip` (D). Cada video produce **K_v = ⌈duración útil / D⌉** miniclips, con un mínimo de 1. La duración útil es la de los cuadros que quedan en tomas después de los pasos 2 y 3. Si `num_clips` (K) está definido, actúa como tope. Con `seconds_per_clip: null` se recupera el comportamiento original, con K fijo.

| Caso | Antes (K fijo) | Ahora (K variable) |
|---|---|---|
| N ≥ K_v | K ventanas elegidas por movimiento | K_v ventanas, con la misma regla |
| 0 < N < K_v | Ventanas desplazadas y copias hasta completar K | Las N ventanas, sin desplazadas ni copias (`N<K_todas`) |
| N = 0 | Un miniclip copiado K veces | Un único miniclip de la toma más larga |

**Motivo:** que la cantidad de miniclips respete la duración de cada video y eliminar las copias.

### Resultado (r = 2, D = 10 s, sin tope)

- 751 miniclips, **todos distintos**. K_v va de 1 (vNonPorn000001) a 201 (vPorn000006).
- Casos: 18 `ninguno`, 1 `N<K_todas` (vNonPorn000006), 1 `N=0` (vNonPorn000001).

### Consecuencia para el Cap. 5

Los videos tienen un número distinto de miniclips. El cargador de datos debe muestrear miniclips por video o agregar las predicciones por video, para que los videos largos no dominen el entrenamiento.

---

## 3b. Tomas cortas como candidatas y selección por movimiento desactivada

**Fecha:** 27 oct 2026 · **Estado:** las tomas cortas siguen vigentes; la opción `select_by_motion` se reemplazó por la eliminación del paso (cambio 3d). **Afecta al Cap. 4.**

### Problema

vNonPorn000006 (30 s, montaje rápido) produjo **un solo miniclip**: de sus 19 tomas, solo una llegaba a los 64 cuadros de una ventana (2,6 s). Las demás duraban entre 0,8 y 2,5 s, así que su contenido no aportaba ninguna candidata.

### Qué se cambió

Hay dos opciones nuevas en `summary`, activadas en `config.yaml`:

| Opción | Valor | Efecto |
|---|---|---|
| `short_shot_windows` | `true` | Una toma de T a L − 1 cuadros (1,28–2,5 s) aporta una candidata de **32 cuadros consecutivos (r = 1)**, centrada en la toma. Al ser la única de su toma, siempre se conserva en el paso 4 |
| `select_by_motion` | `false` | **Todas** las candidatas conservadas pasan a ser miniclips. K y D dejan de usarse |

Cada miniclip registra su `r` en el JSON y se guarda a 25 / r fps, para que dure lo mismo que en el video original. Si se reactiva la selección por movimiento, compara el movimiento por cuadro (m_j / r).

### Resultado esperado para vNonPorn000006

1 ventana normal + 8 candidatas de tomas cortas = **9 miniclips**. Las 10 tomas de menos de 1,28 s siguen sin aportar (`conteos.tomas_sin_candidata`).

### Consecuencias

- **El número de miniclips solo depende de las tomas y de θ_H.** Ya no hay un control de cantidad: un video de 33 min puede generar cientos de miniclips. **Calibrar θ_H pasa a ser prioritario**, porque el filtro de redundancia es el único que reduce la cantidad.
- **Dos duraciones de miniclip:** 2,56 s (r = 2) y 1,28 s (r = 1). Las candidatas de tomas cortas se concentran en los videos no explícitos, así que la proporción de miniclips con r = 1 puede diferir entre clases. Conviene revisarla en el Cap. 5.
- Al desactivar la selección por movimiento, el procedimiento se aparta del paso 5 del Cap. 4.

---

## 3c. Corrección: falta de memoria en el paso 6

**Fecha:** 27 oct 2026 · **Estado:** corregido, sin probar todavía. No afecta a la tesis.

### Problema

Con la selección por movimiento desactivada (3b), la corrida en Colab falló dos veces: "un proceso de trabajo terminó abruptamente", a los 18 y a los 13 de 20 videos. El sistema mató un proceso por falta de RAM. El paso 6 tenía dos defectos de memoria:

1. Armaba **todos** los miniclips del video en RAM, más una segunda copia al ordenarlos.
2. La verificación de alineación convertía **todos** los cuadros extraídos a `float32` de una vez (4 veces su tamaño).

En total se usaban ~1,1 MB por cuadro extraído: **~8,6 GB** para vPorn000004 (~244 miniclips) y **~26 GB** para vPorn000006 (~743). Mientras la selección limitaba la cantidad, el problema no aparecía.

### Corrección

El paso 6 ahora trabaja **miniclip por miniclip**: ffmpeg entrega los cuadros en orden, y cada miniclip se arma apenas se completan sus 32 cuadros, se verifica su alineación, se codifica (`storage.ClipWriter`) y se libera. La memoria del paso 6 queda en **unos pocos MB**, independiente de la cantidad de miniclips. La salida es la misma que antes.

---

## 3d. Eliminación del paso de selección de miniclips por movimiento

**Fecha:** 30 oct 2026 · **Estado:** implementado, sin probar todavía. **Afecta al Cap. 4.**

### Decisión

Se elimina del procedimiento el **paso 5 (selección de K miniclips guiada por el movimiento)**. Motivo: **quita contenido**. Descarta ventanas que ya pasaron el filtro de redundancia solo para ajustarse a una cantidad fija (K) o proporcional a la duración (K_v = ⌈útil / D⌉), y con ello pierde partes del video sin que sean redundantes.

### Procedimiento resultante

1. Normalización a 25 fps.
2. Filtrado de cuadros negros, uniformes y pantallas fijas.
3. Tomas con TransNet V2 y candidatas: ventanas de 64 cuadros (r = 2) en las tomas de ≥ 64 cuadros, y una candidata de 32 cuadros consecutivos (r = 1) en cada toma de 32 a 63 cuadros.
4. Eliminación de redundancia por pHash dentro de cada toma (θ_H).
5. **Todas las candidatas conservadas son miniclips**, en orden temporal. Si no hay ninguna (N = 0): un único miniclip de la toma más larga.
6. Extracción a 224 × 224 y almacenamiento en `.mp4` (un archivo por miniclip).

### Qué se eliminó del código

| Elemento | Antes | Ahora |
|---|---|---|
| Regla F_j ≥ (k − 0,5)/K y elección de la ventana libre más cercana | `selection.select_by_motion` | Eliminada |
| K fijo (`num_clips`), K variable (`seconds_per_clip` = D), ε (`epsilon`), opción `select_by_motion` | En `summary` | Eliminados de la configuración. Un `config.yaml` que todavía los tenga **no carga** (clave desconocida) |
| Respaldos de K fijo: ventanas desplazadas L/2 y repetición cíclica | `shots.shifted_windows`, casos `N<K_desplazadas` y `N<K_repetidas` | Eliminados |
| Caso `N<K_todas` (menos ventanas que K_v) | Existía | Eliminado: ya no hay un objetivo con el que comparar |
| Campos del JSON: `seleccion.miniclips_objetivo`, `segundos_por_miniclip`, `por_movimiento`, `movimiento_conservadas`, `respaldo.ventanas_desplazadas`, `respaldo.miniclips_repetidos`, `miniclips[].repetido` | Existían | Eliminados |

Se conservan el movimiento m_j de cada miniclip (en el JSON, solo como descriptor) y los casos de respaldo `ninguno` y `N=0`.

### Consecuencias

- **El número de miniclips por video solo depende de sus tomas y de θ_H.** No hay ningún tope: un video de 33 min puede generar cientos de miniclips. **La calibración de θ_H es ahora prioritaria**, porque es el único filtro de cantidad. El paso 6 ya procesa miniclip por miniclip (cambio 3c), así que la cantidad no afecta a la memoria, pero sí al disco.
- **Entrenamiento (Cap. 5):** los videos largos aportan muchos más miniclips que los cortos. El cargador de datos debe equilibrarlos (por ejemplo, muestrear un número fijo de miniclips por video y por época, o agregar las predicciones por video).
- **Texto de la tesis:** el Cap. 4 debe eliminar la descripción del paso 5 (movimiento, distribución acumulada, K y casos de respaldo N < K) y reemplazarla por la regla "todas las candidatas conservadas".
- **`preproc_study` (paquete de estudio):** su etapa de selección por movimiento (`motion_selection.py`, el barrido de D y el gráfico de F_j) usaba lo que se eliminó y **dejará de funcionar** hasta adaptarla. Adaptado en el cambio 3e.

---

## 3e. Adaptación de `preproc_study` al procedimiento actual

**Fecha:** 4 nov 2026 · **Estado:** implementado, sin probar todavía. No afecta a la tesis.

### Problemas que tenía

- Fallaba en el paso 5: llamaba a `plan_miniclips` con `fps` y usaba K, D, ε, F_j y `Miniclip.repeated`, eliminados en el cambio 3d. La verificación leía `seleccion.miniclips_objetivo`, que ya no existe en el JSON.
- No generaba las candidatas de tomas cortas (cambio 3b): sus miniclips no coincidían con los de `process_video`.
- Guardaba todos los miniclips en RAM y verificaba la alineación de todos los cuadros a la vez, el mismo defecto que se corrigió en el cambio 3c (~3,6 GB + ~14 GB para vPorn000006).
- Codificaba los `.mp4` con los ajustes del modo de diagnóstico y a 25 / r fijo, no con `encode_clip` a 25 / r de cada miniclip.

### Qué se cambió (versión 0.2.0)

| Antes | Ahora |
|---|---|
| `motion_selection.py`, `select_miniclips`, carpeta `05_motion_selection/` (CDF, barrido de D) | `miniclips.py`, `plan_miniclips`, carpeta `05_miniclips/`: todas las candidatas conservadas, con tipo, r y fps de cada miniclip; m_j solo como descriptor |
| Solo ventanas de L cuadros | Ventanas + candidatas de tomas cortas, con el mismo orden de índices que `preproc`; `windows.csv` pasa a `candidates.csv` |
| Barrido de θ_H: ventanas descartadas | Además, miniclips resultantes y segundos cubiertos, con gráfico (`hamming_threshold_sweep.png`) |
| Miniclips en RAM, `save_npy` | Miniclip por miniclip (`iter_miniclip_frames`), un `.mp4` con `encode_clip` (igual que `preproc`); se eliminó `save_npy` |
| — | `measure_storage`: relee cada `.mp4`, reporta el PSNR del almacenamiento y calcula las estadísticas del tensor sobre lo que recibe Swin |
| Verificación con claves antiguas y píxeles en RAM | Claves actuales del JSON (candidatas, tomas cortas, tipo, r e índices de cada miniclip) y un SHA-1 por miniclip vía `clip_sink` |

---

## 4. Propuestas descartadas

| Propuesta | Por qué se descartó |
|---|---|
| Selección repartida por toma (los K miniclips distribuidos entre tomas) | La rechazó el autor de la tesis. Después se eliminó toda la selección (cambio 3d) |
| Miniclip directo (r = 1, 1,28 s) para tomas de 32 a 63 cuadros, sin pasar por la redundancia ni por la selección | Revertida en su primera versión. Se retomó como candidata en el cambio 3b |
| Ampliar las tomas cortas hasta 64 cuadros (2,56 s): extender sobre las tomas vecinas, duplicar o interpolar cuadros, o repartir los 32 cuadros en toda la toma (r' no entero) | Ninguna obtiene 2,56 s reales: o cruza un corte, o inventa cuadros, o solo cambia el espaciado. Duplicar cada cuadro y muestrear con r = 2 da los mismos 32 cuadros consecutivos que r = 1. Se decidió que **32 cuadros reales bastan**: Swin recibe 32 cuadros en todos los casos |

---

## Pendientes

- Correr el conjunto de prueba con la versión actual (formato 1b, tomas cortas, sin selección, corrección de memoria) y registrar el tamaño real y los miniclips por video.
- **Calibrar θ_H con r = 2: es el único filtro de cantidad de miniclips.** Calibrar también θ_L, θ_σ y θ_E (hoy provisionales: 16, 5 y 0.5).
- Probar `preproc_study` 0.2.0 (cambio 3e) con `verify=True` sobre un video con tomas cortas (por ejemplo, vNonPorn000006).
- Actualizar el Cap. 4 (paso 5) y definir en el Cap. 5 cómo se equilibran videos con distinta cantidad de miniclips.
- Revisar en los videos originales si los tramos fijos eliminados en vNonPorn000001, 000002 y 000005 contienen información (por ejemplo, presentaciones de fotos).
- Revisar si vPorn000003, vPorn000004 y vNonPorn000008 son realmente de una sola toma, o si TransNet V2 no detectó cortes.
- Contar en el conjunto completo los videos de menos de ~10 fps: en vNonPorn000010 (2,9 fps) TransNet V2 detectó cortes falsos.
