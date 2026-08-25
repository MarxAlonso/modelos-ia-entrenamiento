# Aprendizaje Fase 2 — Preprocesamiento de imágenes para CNN

Material de estudio para estudiantes de ingeniería de software. Cada término que aparece en `src/preprocesamiento.py` se explica aquí desde cero.

---

## 1. ¿Qué es "preprocesar" y por qué no se puede saltar?

Una red neuronal es una función matemática que multiplica números. No "ve" radiografías: recibe **tensores** (arrays multidimensionales). Preprocesar es convertir el archivo del disco (`IM-0115-0001.jpeg`, 130 KB de bytes comprimidos) en exactamente el formato numérico que la red espera:

```text
archivo JPEG          tensor de entrada
┌──────────┐         ┌─────────────────────────┐
│ bytes... │   →→→   │ 224 × 224 × 3 floats    │   224 alto × 224 ancho
│ comprimi │         │ valores 0..1            │   × 3 canales RGB
└──────────┘         └─────────────────────────┘   = 150 528 números por imagen
```

Si le das a la red imágenes con formatos mezclados (unas grises, otras RGB; unas gigantes, otras chicas; unos píxeles 0-255 y otros 0-1), los gradientes se vuelven basura y el modelo no converge. **El preprocesamiento es un contrato**: la misma transformación SIEMPRE, en entrenamiento y en producción.

---

## 2. Concepto por concepto

### Tensor y shape

Un tensor es un array N-dimensional. En Deep Learning las imágenes viven como tensores rank-4 cuando van en lotes:

```text
(batch, alto, ancho, canales)
(32,    224,  224,   3     )     ← nuestro batch de entrenamiento
```

`32` imágenes apiladas: la GPU/CPU procesa el lote entero en paralelo (vectorización).

### dtype: uint8 vs float32

En disco, un píxel JPEG es `uint8` (entero 0-255, 1 byte). Las redes necesitan decimales para calcular gradientes → convertimos a `float32`. Es también el momento de reescalar: dividir entre 255 lleva todo a `[0, 1]`.

```python
tf.image.convert_image_dtype(img, tf.float32)   # hace ambas cosas a la vez
```

### Resize 224×224 y interpolación bilinear

Las convoluciones exigen tamaño fijo. Elegimos 224×224 porque **ResNet50 y MobileNetV2 nacieron entrenadas con ese tamaño en ImageNet** (1,2 M de imágenes): sus capas internas ya "saben" leer el mundo a esa escala.

Redimensionar de ~1328px a 224px requiere inventar píxeles: la **interpolación bilinear** calcula cada píxel nuevo promediando los 4 vecinos más cercanos con pesos por distancia. Suaviza la imagen; suficiente porque lo que define la neumonía son patrones globales de opacidad, no textura fina.

### Normalización: cada modelo habla su idioma

Este es el error clásico #2 de principiante: usar `preprocess_input` equivocado. La regla es brutal:

> Un modelo preentrenado solo funciona si su entrada tiene EXACTAMENTE la distribución con la que fue entrenado originalmente.

| Modelo | Espera | Por qué ese rango |
|---|---|---|
| CNN propia | `[0, 1]` | Elección libre; estable para gradientes simples |
| ResNet50 | `x*255` convertido a **BGR**, restando medias de ImageNet (~[-124, 152]) | Herencia del framework Caffe donde fue creada |
| MobileNetV2 | `[-1, 1]` | Diseñada así para funcionar bien tras ReLU6 |

Por eso el módulo expone `NORMALIZADORES = {"rescale", "resnet50", "mobilenetv2"}`: el script de entrenamiento de cada versión elige el suyo y nada más.

### Data augmentation: fabricar variabilidad

Idea: tu dataset tiene 4 671 fotos, pero el mundo tiene infinitas variantes (paciente ligeramente rotado, equipo con más/menos contraste, pulmón desplazado unos píxeles). Augmentation aplica transformaciones ALEATORIAS a cada imagen EN CADA ÉPOCA, así la red casi nunca ve dos veces la foto idéntica. Es regularización: combate el overfitting sin recolectar datos nuevos.

Nuestras 4 transformaciones permitidas:

| Capa Keras | Qué hace | Simula clínicamente |
|---|---|---|
| `RandomRotation(5/360)` | gira ±5° | paciente levemente inclinado al tomar la placa |
| `RandomZoom(0.1)` | acerca/aleja ≤10% | variación de distancia/enfoque |
| `RandomTranslation(0.1, 0.1)` | mueve ≤10% vertical/horizontal | centrado imperfecto |
| `RandomContrast(0.08)` | estira/comprime contraste ±8% | calibración distinta del equipo |

Y la prohibición clave — **sin flip horizontal**: voltear una radiografía pone el corazón (que está a la IZQUIERDA, condición llamada *situs solitus*) a la derecha. Le estarías enseñando anatomía imposible. En gatos/fotos genéricas el flip es gratis; en medicina, cada transformación necesita justificación clínica.

Detalles finos:

- Augmentación SOLO en train. Validation/test deben ser deterministas (son el examen).
- `training=True`: las capas Random* se comportan distinto según el modo; en evaluación pasan inactivas automáticamente.
- RandomContrast puede producir valores fuera de `[0,1]` (vimos máx 1.022 en la verificación) → recortamos con `clip_by_value` antes de normalizar. Rigor numérico barato.

### El pipeline tf.data: una cinta transportadora

Entrenar rápido significa que la CPU prepara datos mientras la "GPU/CPU-matemática" procesa el batch anterior. `tf.data` orquesta eso como línea de ensamble:

```python
ds = from_tensor_slices((rutas, etiquetas))   # lista de trabajos pendientes
     .shuffle(4671, seed=42)                  # barajar órdenes (solo train)
     .map(decodificar, num_parallel_calls=AUTOTUNE)  # estación 1: decodificar
     .map(augmentar,   num_parallel_calls=AUTOTUNE)  # estación 2: augmentar
     .map(normalizar,  num_parallel_calls=AUTOTUNE)  # estación 3: normalizar
     .batch(32)                               # empaquetar cajas de 32
     .prefetch(AUTOTUNE)                      # preparar el siguiente lote MIENTRAS el modelo consume el actual
```

Términos:

- **AUTOTUNE**: TensorFlow decide cuántos hilos paralelos usar según tu CPU (tenemos 8 hilos).
- **shuffle(buffer)**: desorden aleatorio dentro de una ventana de 4 671 elementos; evita que la red aprenda el orden de las fotos.
- **batch(32)**: agrupa en lotes; el gradiente se promedia por lote.
- **prefetch**: solapa producción y consumo (el secreto del throughput).
- Decisión documentada: NO usamos `.cache()` porque guardar 5 191 imágenes decodificadas en RAM consumiría ~3 GB extra de tus 16 GB; con 181 img/s el pipeline no es el cuello de botella.

### Batch size, época, steps

- **batch_size=32**: cuántas imágenes por paso de gradiente. 32 es equilibrio clásico estabilidad/velocidad en CPU.
- **época**: una pasada completa por train = 146 batches (4 671/32 ≈ 146).
- **steps_per_epoch** = nº de batches por época. Con `fit()` Keras lo calcula solo.

### Pesos de clase (repaso Fase 1)

`pesos_de_clase(df)` devuelve `{0: 1.937, 1: 0.674}`. En la loss, cada imagen NORMAL cuenta 1.937 veces y cada PNEUMONIA 0.674 → ambas clases aportan igual aunque haya 2.89× más neumonías.

---

## 3. Cómo verificarás que esto funciona (lo que corrimos)

```text
[rescale]      batch (32, 224, 224, 3) float32 · rango [0.000, 1.000]
[resnet50]     batch (32, 224, 224, 3) float32 · rango [-123.680, 151.061]
[mobilenetv2]  batch (32, 224, 224, 3) float32 · rango [-1.000, 1.000]
train: 146 batches · validation: 17 · test: 20
throughput: 181 img/s (CPU i5-10300H)
```

Leer resultados como ingeniero:

- Shapes correctos ⇒ geometría OK.
- Rangos exactos ⇒ contrato de normalización OK.
- 181 img/s ⇒ una época de train tardará ~26 s solo en datos; el modelo añadirá lo suyo. Viable en CPU.

---

## 4. Glosario rápido

| Término | Significado |
|---|---|
| **Tensor / rank / shape** | Array N-D / nº de dimensiones / tamaño en cada dimensión |
| **Interpolación bilinear** | Inventar píxeles nuevos promediando 4 vecinos ponderados |
| **Normalización** | Llevar los valores al rango/distribución que el modelo espera |
| **Augmentation** | Transformaciones aleatorias que multiplican la variedad efectiva de datos |
| **Regularización** | Técnicas contra overfitting (augment, dropout, weight decay…) |
| **tf.data** | API de pipelines de datos con paralelismo y prefetch |
| **prefetch / AUTOTUNE** | Solapar preparación y consumo / TF elige el paralelismo óptimo |
| **batch / época / step** | Lote / pasada completa al dataset / un forward+backward |

---

## 5. Ejercicios sugeridos

1. Cambia `RandomZoom(0.1)` por `RandomZoom(0.9)` mentalmente: ¿qué aprendería mal la red? ¿Por qué "zoom extremo" destruye el contexto clínico?
2. Si mañana entrenas con dataset v2 en escala DICOM real (16 bits), ¿qué líneas del módulo cambiarían? (Pista: `decode_jpeg` ya no sirve; ventana/pulmonar HU.)
3. Explica con tus palabras por qué aplicar augmentación también en test falsearía las métricas.

---
*Referencias: TensorFlow tf.data guide · keras.applications (resnet50.preprocess_input, mobilenet_v2.preprocess_input) · Shorten & Khoshgoftaar, "A survey on Image Data Augmentation for Deep Learning", Journal of Big Data 2019.*
