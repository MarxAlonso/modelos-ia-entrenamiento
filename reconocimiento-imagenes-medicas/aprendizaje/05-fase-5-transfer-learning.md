# Aprendizaje Fase 5 — Transfer Learning

Material de estudio basado en la versión REAL v003 de este proyecto (MobileNetV2 congelada). Aquí está el concepto que separa los proyectos amateur de los profesionales en visión.

---

## 1. El problema que resuelve transfer learning

Nuestra CNN propia aprendió desde cero: empezó sin saber NADA del mundo visual. Por eso necesito 35 épocas para distinguir texturas básicas.

Pero alguien ya entrenó una red con **1.4 millones de imágenes de ImageNet** (perros, autos, comida...). Lo sorprendente: las primeras capas de esa red aprendieron bordes, curvas, texturas y gradientes de luz — rasgos UNIVERSALES de cualquier imagen natural... y también de radiografías.

```text
Entrenar desde cero            Transfer learning
─────────────────────          ─────────────────────────────
época 1-5: aprende bordes      época 1: ¡ya distingue patrones!
época 6-20: aprende formas     época 1-6: solo ajusta la decisión
época 20+: aprende neumonía    final: f1_macro 0.84 en 21 min

v002 (desde cero): 84 min → ROC-AUC 0.891
v003 (transfer):    21 min → ROC-AUC 0.967
```

## 2. Feature extraction: la arquitectura de v003

```python
base = MobileNetV2(include_top=False, weights="imagenet", input_shape=(224,224,3))
base.trainable = False                    # 🔒 congelada: sus pesos NO se actualizan

x = base(entradas, training=False)        # BatchNorm SIEMPRE en modo inferencia
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
salida = Dense(1, activation="sigmoid")(x)
```

Piezas clave:

- `include_top=False`: descartamos la cabeza original de ImageNet (clasificaba 1000 objetos); nos quedamos SOLO con el extractor de características (2.26M params).
- `trainable=False`: congelamos. Solo entrenamos nuestra cabeza: **164 097 params = 6.8% del total**. Entrenar el 6.8% es lo que hace la época rápida (~100s vs 130s de v002 pese a una base 22× más grande).
- `training=False` al llamar la base: crítico e invisible para principiantes. MobileNetV2 contiene capas **BatchNorm**, que tienen dos modos: entrenamiento (normaliza con estadísticas DEL LOTE ACTUAL) e inferencia (usa medias guardadas de ImageNet). Si dejáramos training=True, esas estadísticas cambiarían en cada paso mientras los pesos están congelados → inconsistencia y degradación silenciosa. Regla: **base congelada ⇒ siempre training=False**.

### ¿Por qué funciona tan bien con radiografías si ImageNet no tiene rayos X?

Las características genéricas (bordes, frecuencias, contrastes) son reutilizables; la cabeza nueva solo aprende a COMBINARLAS para decidir NORMAL/PNEUMONIA. Es como contratar un ojo entrenado durante años y enseñarle solo el criterio diagnóstico.

---

## 3. El hallazgo clínico de v003: el balance FN/FP

```text
matriz v003: TN=155 · FP=79 · FN=5 · TP=385

recall_pneumonia = 385/390 = 0.987   ← casi no escapa ningún enfermo ✔
recall_normal    = 155/234 = 0.662   ← pero 34% de sanos van a "estudio extra"
```

El modelo es MUY sensible y menos específico. Dos causas combinadas:

1. **class_weights** (NORMAL pesa 1.937): empujan la decisión hacia "no perder enfermos".
2. **Umbral 0.5 fijo**: moverlo a ~0.6-0.7 intercambiaría algo de recall_pneumonia por muchos FP menos.

¿Cuál es mejor? Depende del costo clínico: en triaje pediátrico, un FN (alta a un enfermo) es peor que varios FP (radiología revisa de más). Por eso el proyecto prioriza recall_pneumonia como desempate del registro — pero documentar el trade-off es obligación ética del ingeniero. La Fase 5 (fine tuning) debería mejorar AMBOS lados.

---

## 4. Cuantización: la lección de velocidad de v003

| Formato | Tamaño | F1 macro | Latencia/img CPU |
|---|---|---|---|
| fp32 keras | 11.6 MB | 0.8442 | ~ |
| TFLite **dyn** | 2.70 MB | 0.8696 | **56.9 ms** |
| TFLite **int8** | 2.90 MB | 0.8504 | **7.1 ms** |

Dos descubrimientos reales:

1. **Dynamic range quedó lento (57 ms)**: conserva activaciones float; MobileNetV2 está llena de *depthwise convolutions* que en float no aprovechan bien los kernels enteros de XNNPACK. Full int8 ejecuta TODO en enteros → **8× más rápido**.
2. **Ambos formatos superaron al fp32 en f1** (0.87/0.85 > 0.84): la cuantización actuó como regularizador involuntario — redondear pesos amortigua sobreajustes finos. Fenómeno conocido, pero verlo en tu propio modelo es otra cosa.

Regla práctica resultante para este proyecto: **desplegar siempre int8** (rápido, pequeño, métricas aceptables); usar el .keras fp32 solo como maestro de reentrenamiento.

---

## 5. Glosario rápido

| Término | Significado |
|---|---|
| **Transfer learning** | Reutilizar pesos preentrenados de otra tarea/dataset |
| **Feature extraction** | Congelar la base y entrenar solo una cabeza nueva |
| **Fine tuning** | (Fase siguiente) descongelar parte de la base y reentrenar con lr bajo |
| **BatchNorm modo train/inference** | Normaliza con stats del lote / con stats congeladas |
| **Sensibilidad (recall)** | Capacidad de detectar enfermos: TP/(TP+FN) |
| **Especificidad** | Capacidad de confirmar sanos: TN/(TN+FP) |
| **Depthwise convolution** | Convolución ligera por canal (clave de MobileNet) |
| **XNNPACK** | Librería de kernels optimizados que ejecuta TFLite en CPU |

---

## 6. Ejercicios sugeridos

1. Con la matriz v003 y umbral 0.5: si subimos el umbral a 0.7, ¿FN sube o baja? ¿FP? ¿Qué métrica del registro podría bajar?
2. ¿Por qué congelar hace la época MÁS rápida aunque el forward pase igual por 2.26M params? (Pista: backward es ~2× el costo del forward, y desaparece.)
3. La época 6 fue la mejor (val_auc 0.985) y luego val_loss empeoró mientras train loss seguía bajando: ¿qué fenómeno es y qué lo contenía?

---
*Referencias: Keras transfer learning guide · Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018 · Yosinski et al., "How transferable are features in deep neural networks?", NeurIPS 2014.*
