# Aprendizaje Fase 3 — CNNs y cuantización

Material de estudio para estudiantes de ingeniería de software. Explica cada pieza de `src/entrenar.py` y `src/cuantizar.py`, incluido el **error real que cometimos y cómo lo detectamos** (aprender de errores propios es el objetivo del versionamiento).

---

## 1. Cómo "ve" una CNN: convolución

Una imagen es una parrilla de números. Un **filtro** (kernel) es una parrillita pequeña (3×3) que se desliza sobre TODA la imagen haciendo multiplicación y suma en cada posición:

```text
imagen 5×5            filtro 3×3              mapa de rasgos
■ □ ■ □ ■             ┌─────────┐
□ ■ □ ■ □      ✕      │ a b c d │      =   un número por cada
■ □ ■ □ ■             │ e f g h │           posición donde pasó
□ ■ □ ■ □             │ i j k l │           el filtro
■ □ ■ □ ■             └─────────┘
```

Cada filtro aprende a detectar UN patrón: bordes, texturas, manchas difusas... Las primeras capas detectan cosas simples (líneas); las profundas combinan esas detecciones en conceptos complejos ("zona pulmonar opaca"). Eso es lo que significa "aprendizaje jerárquico de características": **la red inventa sus propias features; nadie se las programa**.

- `Conv2D(32, 3)` = 32 filtros distintos de 3×3 → 32 mapas de rasgos.
- Parámetros: un filtro 3×3 sobre RGB pesa 3×3×3+1 = 28 números. ¡28! Por eso una CNN de 109 mil parámetros puede competir con redes gigantes: **comparte pesos en toda la imagen**.

### ReLU

`activation="relu"` aplica max(0, x): elimina los negativos. Sin funciones no lineales entre capas, apilar convoluciones sería equivalente a UNA sola operación lineal: la red no aprendería nada interesante.

### MaxPooling2D

Reduce cada mapa de rasgos a la mitad (224→112→56...) quedándose con el valor máximo de cada ventana 2×2. Dos beneficios: menos cómputo y **tolerancia a pequeñas traslaciones** (si la mancha se movió 3 px, sigue estando en la misma zona del mapa reducido).

### GlobalAveragePooling2D vs Flatten

- `Flatten`: aplana todo (56×56×128 = 401 408 valores) → capa densa gigante → overfitting seguro.
- `GAP`: promedia CADA mapa completo en UN número (128 valores). La red resume "¿cuánto se activó este detector en total?".

Elegimos GAP como dice el mapa: 401 408 → 128 números antes de decidir.

### Dense + Dropout + sigmoid final

- `Dense(128)`: combina los 128 resúmenes en una representación de decisión.
- `Dropout(0.5)`: durante el ENTRENAMIENTO apaga aleatoriamente el 50% de neuronas en cada paso. Obliga a la red a no depender de caminos únicos → anti-overfitting. En predicción se apaga solo (Keras lo maneja).
- `Dense(1, sigmoid)`: comprime la salida a un número entre 0 y 1 = P(PNEUMONIA).

### binary_crossentropy + class_weight

La loss mide qué tan mentirosa fue la probabilidad dicha:

```text
loss = -[ w_y·y·log(p) + w_1-y·(1-y)·log(1-p) ]
```

Si dice 0.9 para una neumonía real → pérdida baja; si dice 0.9 para una normal → pérdida altísima. El `class_weight={NORMAL:1.94}` multiplica los errores de NORMAL para compensar que hay 2.89× más neumonías: sin esto, la red obtendría mejor loss ignorando a las normales.

---

## 2. EarlyStopping — y EL ERROR QUE COMETIMOS

```python
EarlyStopping(monitor="val_recall", mode="max", patience=5, restore_best_weights=True)
```

Concepto correcto: vigilar una métrica en validation y frenar cuando deja de mejorar (patience = épocas de gracia), restaurando los mejores pesos. Evita overfitting y horas desperdiciadas.

**Lo que pasó en v001** (lección registrada):

```text
Época 1: val_recall = 0.7714   ← "mejor" por azar (519 imágenes de val)
Épocas 2-6: val_recall ≈ 0.55-0.60 (pero val_auc subía: 0.72 → 0.80 ✔)
→ patience agotado → restauró los pesos de la ÉPOCA 1 (casi sin entrenar)
```

El recall con class_weights en 519 imágenes es una métrica **ruidosa**: fluctúa mucho época a época. Vigilarlo directo hizo que early stopping eligiera un modelo casi virgen. El AUC sí mejoraba establemente — la red estaba APRENDIENDO bien, pero la regla de parada era mala.

Lecciones de ingeniería (para v002):

1. Monitoriza métricas estables (`val_loss`, `val_auc`) o suavizadas.
2. Una métrica ruidosa + patience corto = decisiones prematuras.
3. Siempre mira las curvas (`curvas_entrenamiento.png`) antes de interpretar métricas finales.

---

## 3. Leer las métricas desde la matriz de confusión

```text
                    pred NORMAL    pred PNEUMONIA
real NORMAL    │      TN=76     │     FP=158    │
real PNEUMONIA │      FN=110    │     TP=280    │
```

- **Accuracy** = (TN+TP)/total = 336/624 = 0.57 — engañosa sola.
- **Recall pneumonia** = TP/(TP+FN) = 280/390 = 0.718 — de los enfermos, ¿cuántos pillamos?
- **Precision pneumonia** = TP/(TP+FP) = 280/438 = 0.639 — cuando gritamos "neumonía", ¿acertamos?
- **F1** = media armónica precision·recall — equilibra ambas.
- **ROC-AUC** = capacidad de ordenar (enfermo arriba de sano) INDEPENDIENTE del umbral 0.5. Es la métrica honesta del modelo crudo: 0.58 en test vs 0.80 en validation tardía confirma el diagnóstico del punto anterior.
- **f1_macro**: promedia F1 de ambas clases → ninguna clase "no existe" para el modelo.

Clínico: FP=158 (decir neumonía a sanos) genera estudios extra; FN=110 (dar alta a enfermos) es el error peligroso. Por eso el registro prioriza recall_pneumonia como desempate.

---

## 4. Cuantización: float32 → int8

### El problema

Un float32 ocupa 4 bytes. Nuestro modelo: 109 889 parámetros × 4 B ≈ 430 KB (+ overhead Keras = 1.36 MB). En producción móvil/edge, modelos de cientos de MB no caben ni arrancan rápido. Y cada operación float32 gasta CPU/batería.

### La idea

Guardar los pesos en int8 (1 byte, 256 niveles) y operar con enteros:

```text
real = escala × (entero - zero_point)

peso float 0.0372  →  entero 19   (escala≈0.00196, zp=0)
```

Se pierde precisión decimal, pero las redes son robustas: pequeños cambios en pesos ⇒ pequeños cambios en salida. La clave es elegir bien `escala` y `zero_point`.

### Los dos modos que generamos

| Modo | Qué cuantifica | Requiere datos | Resultado |
|---|---|---|---|
| **dynamic range** | solo pesos (activaciones float, convertidas al vuelo) | No | ~4× menor; fácil |
| **full int8** | pesos + activaciones + cálculos internos | **dataset representativo** | aún menor y más rápido en CPUs edge |

El dataset representativo sirve de **calibración**: pasamos 200 radiografías reales y el conversor registra los rangos min/max de cada activación para fijar sus escalas. Si calibraras con imágenes que no representan el uso real, las activaciones se recortarían mal y el modelo degradaría.

Nuestros números reales (verificados en test):

```text
modelo.keras        1.36 MB   f1_macro 0.5191   latencia ~? 
quant_dyn.tflite    0.12 MB   f1_macro 0.5191   7.98 ms/img   ← 11.1× más chico, idéntico
quant_int8.tflite   0.12 MB   f1_macro 0.5120   7.01 ms/img   ← -0.7pp f1, más rápido
```

Regla del plan maestro cumplida: pérdida < 1% ⇒ cuantización aprobada. Nota técnica: al ser un mini-modelo (109k params), el tamaño ya era pequeño y TFLite comprime fuerte; con ResNet50 (~25M params) veremos reducciones absolutas mucho mayores.

### XNNPACK

El log mostró `Created TensorFlow Lite XNNPACK delegate`: es la librería de kernels optimizados (SIMD AVX2 de tu i5) que ejecuta el .tflite. Por eso medimos 7 ms con 4 hilos.

---

## 5. Flujo completo de la fase (mentalidad MLOps)

```text
entrenar.py                          cuantizar.py
────────────                        ────────────────────────────────
manifiesto -> pipeline tf.data       modelo.keras
CNN propia + class_weights             ├→ dyn.tflite ─┐
EarlyStopping (val_recall ⚠)           └→ int8.tflite ┤→ evaluar en MISMO test
fit() → curvas                         metricas.json += cuantizacion
evaluar test (UNA vez)                 registro_versiones.json ← dashboard Astro
guardar config/metricas/modelo
```

Todo queda versionado en `models/versiones/v001_<fecha>_cnn/`: si mañana v002 duplica el f1_macro, podremos EXPLICAR por qué comparando configs línea a línea.

---

## 6. Glosario rápido

| Término | Significado |
|---|---|
| **Filtro/kernel** | Parrilla pequeña de pesos que se desliza detectando un patrón |
| **Mapa de rasgos** | Salida de aplicar un filtro a toda la imagen |
| **MaxPooling / GAP** | Reducción espacial por máximo local / promedio global por canal |
| **Dropout** | Apagar neuronas al azar en training para regularizar |
| **Sigmoid + BCE** | Probabilidad binaria + su función de pérdida |
| **EarlyStopping/patience** | Parar cuando validation deja de mejorar / épocas de tolerancia |
| **Umbral (0.5)** | Corte de decisión sobre la probabilidad; moverlo intercambia recall↔precision |
| **Cuantización** | Representar floats con enteros (int8) reduciendo tamaño/cómputo |
| **Dataset representativo** | Muestras reales usadas para CALIBRAR rangos de activación int8 |
| **Dynamic range / full int8** | Cuantizar solo pesos / pesos+activaciones+cómputo |

---

## 7. Ejercicios sugeridos

1. Con la matriz TN=76 FP=158 FN=110 TP=280, recalcula accuracy, recall y precision a mano. ¿Cambiarías el umbral a 0.35 para subir recall_pneumonia? ¿Qué pagarías?
2. ¿Por qué full int8 dio recall_pneumonia 0.7231 (¡mayor!) que fp32 con f1 levemente menor? (Pista: cuantización como regularizador involuntario.)
3. Si v002 monitorea `val_auc` con patience 8, predice: ¿mejorará el ROC-AUC de test respecto a 0.579? Justifica con las curvas de v001.

---
*Referencias: Goodfellow et al., Deep Learning caps. 6-9 · Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", arXiv:1712.05877 · TensorFlow Lite docs.*
