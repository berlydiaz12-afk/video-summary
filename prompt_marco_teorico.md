# Encargo: redactar el marco teórico de los algoritmos del resumen de video

## 1. Contexto de la tesis

- La tesis clasifica videos (conjunto **Pornography-2k**: videos explícitos y no explícitos) con **Video Swin Transformer** (Swin-B).
- El **Capítulo 4** describe un procedimiento de **resumen y preprocesamiento de video**. Cada video se reduce a un conjunto de **miniclips de 32 cuadros a 224 × 224**, que son la entrada del modelo.
- El objetivo del resumen es quitar los cuadros sin información y las partes repetidas sin perder contenido. No se limita la cantidad de miniclips por video.
- El marco teórico debe explicar los **seis algoritmos** en que se apoya el procedimiento, **en el orden en que se aplican**:

| # | Algoritmo | Función en el procedimiento |
|---|---|---|
| 1 | Diferencia entre cuadros (*frame differencing*) | Detectar y eliminar pantallas fijas |
| 2 | TransNet V2 | Detectar los cortes entre tomas |
| 3 | Muestreo temporal por ventanas con paso fijo | Formar las ventanas candidatas de 32 cuadros |
| 4 | pHash con transformada discreta del coseno (DCT) | Obtener una huella perceptual de cada ventana |
| 5 | Distancia de Hamming | Comparar las huellas de dos ventanas |
| 6 | Filtrado greedy por umbral | Descartar las ventanas repetidas dentro de cada toma |

## 2. Qué se pide

Redacta una sección de marco teórico, en **español**, con un subapartado por algoritmo y en el orden de la tabla.

Cada subapartado debe incluir:

1. **Definición** breve.
2. **Origen**: autor, año y problema que resolvía.
3. **Formulación**: las ecuaciones principales en LaTeX, definiendo cada símbolo.
4. **Propiedades y limitaciones** relevantes para el resumen de video.
5. **Relación con este trabajo**: un párrafo corto que conecte el concepto con su uso (sección 4). No describas la implementación paso a paso; eso corresponde al Cap. 4.

Agrega una **introducción breve** (un párrafo) sobre el resumen de video: qué es y la diferencia entre resumen por cuadros clave y por fragmentos. Explica también por qué la detección de tomas y la eliminación de redundancia son etapas habituales.

### Estilo

- Tono académico, impersonal, sin viñetas en el cuerpo del texto. Las viñetas solo se permiten en listas de propiedades si aportan claridad.
- Extensión orientativa: **400–700 palabras por algoritmo**. El más largo es TransNet V2 y los más cortos son Hamming y el filtrado greedy.
- Citas en formato **APA 7**, con lista de referencias al final.
- **No inventes referencias.** Usa las de la sección 5 y, si agregas otras, deben ser verificables (DOI, arXiv o editorial). Si no estás seguro de un dato bibliográfico, márcalo con `[VERIFICAR]`.
- No menciones nombres de funciones, archivos ni detalles de código.
- No incluyas algoritmos fuera de la lista. En particular, no incluyas la selección de miniclips por movimiento, que se eliminó del procedimiento.

## 3. Símbolos que debes usar (coinciden con el Cap. 4)

| Símbolo | Significado | Valor en este trabajo |
|---|---|---|
| T | Cuadros por miniclip | 32 |
| r | Paso de muestreo temporal (*stride*) | 2 (1 en tomas cortas) |
| L = T · r | Longitud de la ventana en cuadros | 64 (2,56 s a 25 fps) |
| θ_E | Umbral de diferencia entre cuadros (pantalla fija) | Por calibrar (provisional 0,5) |
| θ_H | Umbral de distancia de Hamming (redundancia) | Por calibrar |
| d_k | Diferencia media entre el cuadro k y el k − 1 | — |
| p_k | Probabilidad de transición del cuadro k (TransNet V2) | Umbral 0,5 |

## 4. Cómo se usa cada algoritmo en este trabajo

Esta sección sirve para escribir el párrafo «Relación con este trabajo». Todo el video se lleva antes a **25 fps constantes**.

### 4.1 Diferencia entre cuadros

- Se calcula sobre cuadros de análisis en escala de grises (luminancia BT.601) de 64 × 36.
- d_k = media de |I_k(x, y) − I_{k−1}(x, y)| sobre todos los píxeles.
- Un tramo es una **pantalla fija** si d_k < θ_E de forma continua durante al menos **25 cuadros (1 s)**. Se elimina el tramo completo.
- En el mismo paso también se eliminan los cuadros negros o uniformes (brillo medio o desviación por debajo de un umbral). Esto es contexto y no tiene subapartado propio.

### 4.2 TransNet V2

- Red convolucional 3D preentrenada para detectar transiciones entre tomas (cortes y transiciones graduales).
- Entrada: cuadros RGB de **48 × 27**. Inferencia por **ventanas de 100 cuadros** que avanzan **50 cuadros**, usando solo los 50 centrales de cada ventana (25 de contexto a cada lado).
- Salida: una probabilidad por cuadro (sigmoide). Hay **transición si p_k > 0,5**. Una toma es un tramo continuo sin transiciones.
- Conceptos que conviene explicar: convolución 3D, convoluciones dilatadas (celdas DDCNN), convoluciones (2+1)D o factorizadas, y las dos salidas de la red (*single-frame* y *all-frames*). En este trabajo se usa la salida *single-frame*.

### 4.3 Muestreo temporal por ventanas con paso fijo

- Cada toma se divide **desde su inicio** en ventanas consecutivas **sin solapamiento** de L = 64 cuadros. El sobrante final se descarta.
- De cada ventana se toman T = 32 cuadros con paso r = 2: índices s, s + 2, …, s + 62.
- Las tomas de 32 a 63 cuadros aportan una sola candidata de 32 cuadros consecutivos (r = 1), centrada en la toma.
- Relación con la literatura: el muestreo 32 × 2 es el mismo con que se preentrenó Video Swin en Kinetics, y el muestreo disperso es habitual en el reconocimiento de acciones (TSN, SlowFast).

### 4.4 pHash con DCT

- Cada imagen en gris se reduce a **32 × 32** promediando por área.
- Se aplica la **DCT-II 2D** (ortonormal) y se conserva el bloque **8 × 8 de baja frecuencia**, incluido el coeficiente DC.
- Cada uno de los 64 coeficientes se compara con la **mediana** del bloque: 1 si es mayor y 0 si no. El resultado es un hash de **64 bits**.
- La **firma de una ventana** es el pHash de un cuadro por segundo (cada 25 cuadros) dentro de ella. En una ventana de 64 cuadros son 3 hashes.
- Explica por qué la DCT concentra la energía en las bajas frecuencias y por qué el hash resiste cambios de compresión, escala y brillo, pero no cambios de contenido.

### 4.5 Distancia de Hamming

- Número de bits distintos entre dos hashes: d_H(a, b) = popcount(a XOR b), con valores de 0 a 64.
- La distancia entre dos ventanas es el **promedio** de las distancias de Hamming entre sus hashes en las mismas posiciones.
- Conviene mencionar que es una métrica en el espacio de cadenas binarias y que se calcula muy rápido.

### 4.6 Filtrado greedy por umbral

- Las ventanas candidatas se recorren en **orden temporal**, **dentro de cada toma**.
- La primera ventana de cada toma siempre se conserva.
- Cada ventana nueva se compara **solo con la última conservada de su toma**. Si la distancia es ≤ θ_H se descarta y, si no, se conserva y pasa a ser la nueva referencia.
- Es un algoritmo greedy de una sola pasada, con costo lineal en el número de ventanas. Conserva el orden temporal, pero no garantiza el subconjunto óptimo.
- Relación con la literatura: es análogo a la extracción secuencial de cuadros clave por umbral de disimilitud.
- Todas las ventanas conservadas se convierten en miniclips. Si no queda ninguna, se genera un único miniclip de la toma más larga.

## 5. Referencias sugeridas

Verifica los datos antes de citar.

**Resumen de video y detección de tomas**
- Truong, B. T., & Venkatesh, S. (2007). Video abstraction: A systematic review and classification. *ACM Transactions on Multimedia Computing, Communications, and Applications, 3*(1).
- Apostolidis, E., Adamantidou, E., Metsai, A. I., Mezaris, V., & Patras, I. (2021). Video summarization using deep neural networks: A survey. *Proceedings of the IEEE, 109*(11), 1838–1863.
- Hanjalic, A. (2002). Shot-boundary detection: Unraveled and resolved? *IEEE Transactions on Circuits and Systems for Video Technology, 12*(2), 90–105.

**Diferencia entre cuadros**
- Zhang, H., Kankanhalli, A., & Smoliar, S. W. (1993). Automatic partitioning of full-motion video. *Multimedia Systems, 1*(1), 10–28.

**TransNet V2**
- Souček, T., & Lokoč, J. (2020). TransNet V2: An effective deep network architecture for fast shot transition detection. *arXiv:2008.04838*. (También publicado en ACM Multimedia 2024 [VERIFICAR].)
- Souček, T., Moravec, J., & Lokoč, J. (2019). TransNet: A deep network for fast detection of common shot transitions. *arXiv:1906.03363*.
- Tran, D., Bourdev, L., Fergus, R., Torresani, L., & Paluri, M. (2015). Learning spatiotemporal features with 3D convolutional networks. *ICCV*.
- Yu, F., & Koltun, V. (2016). Multi-scale context aggregation by dilated convolutions. *ICLR*.

**Muestreo temporal**
- Wang, L., et al. (2016). Temporal segment networks: Towards good practices for deep action recognition. *ECCV*.
- Feichtenhofer, C., Fan, H., Malik, J., & He, K. (2019). SlowFast networks for video recognition. *ICCV*.
- Liu, Z., et al. (2022). Video Swin Transformer. *CVPR*.
- Kay, W., et al. (2017). The Kinetics human action video dataset. *arXiv:1705.06950*.

**pHash y DCT**
- Ahmed, N., Natarajan, T., & Rao, K. R. (1974). Discrete cosine transform. *IEEE Transactions on Computers, C-23*(1), 90–93.
- Zauner, C. (2010). *Implementation and benchmarking of perceptual image hash functions* (Tesis de maestría). Upper Austria University of Applied Sciences, Hagenberg.
- Krawetz, N. (2011). *Looks like it*. The Hacker Factor Blog. (Fuente divulgativa del algoritmo pHash basado en DCT; úsala solo como apoyo.)

**Distancia de Hamming**
- Hamming, R. W. (1950). Error detecting and error correcting codes. *Bell System Technical Journal, 29*(2), 147–160.

**Algoritmos greedy**
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to algorithms* (4.ª ed.). MIT Press.

## 6. Formato de entrega

- Markdown con encabezados `##` por algoritmo, numerados en el orden de la tabla de la sección 1.
- Ecuaciones en LaTeX (`$...$` en línea y `$$...$$` en bloque).
- Al final, la sección **Referencias** en APA 7, solo con las obras citadas en el texto.
