# Fase 2 — Preprocesamiento y pipeline tf.data

Fecha: 2026-08-23 · Estado: ✅ Completada · Material de estudio: [`aprendizaje/02-fase-2-preprocesamiento.md`](../aprendizaje/02-fase-2-preprocesamiento.md)

---

## 1. Qué se construyó

Módulo único `src/preprocesamiento.py` que centraliza TODO el preprocesamiento del proyecto. Ningún script de entrenamiento decodifica o transforma imágenes por su cuenta; todos pasan por aquí. Esto garantiza que las comparaciones entre versiones midan diferencias del **modelo**, no del preprocesamiento.

### API pública

```python
from preprocesamiento import (
    cargar_manifiesto,      # -> DataFrame del manifiesto congelado
    crear_pipeline,         # -> tf.data.Dataset listo para model.fit()
    pesos_de_clase,         # -> {0: 1.937, 1: 0.674}
    capa_augmentacion,      # -> Sequential de capas Random* (inspeccionable)
    IMG_SIZE, SEMILLA, MAPA_CLASES,
)

crear_pipeline(df, split, batch_size=32, augment=None, normalizacion="rescale")
#   split: "train" | "validation" | "test"
#   augment: None = automático (True solo en train)
#   normalizacion: "rescale" | "resnet50" | "mobilenetv2"
```

## 2. Pipeline implementado

```text
ruta desde manifiesto (NO desde carpetas)
   ↓ read_file + decode_jpeg(channels=0)          respeta canales originales
   ↓ convert_image_dtype(float32)                 0..255 -> 0..1
   ↓ cond: 1 canal -> tile a RGB / >3 canales -> recorte a RGB
   ↓ resize bilinear 224×224
   ↓ [train] RandomRotation(±5°) Zoom(10%) Translation(10%) Contrast(8%)
   ↓ clip_by_value(0,1)                           limpia exceso del contraste
   ↓ normalizador elegido (rescale | resnet50 | mobilenetv2)
   ↓ batch(32) + prefetch(AUTOTUNE)
```

Decisiones de diseño:

| Decisión | Alternativa descartada | Razón |
|---|---|---|
| `decode_jpeg(channels=0)` + cond de canales | `channels=3` directo | channels=3 fuerza conversión interna inconsistente; así controlamos L→RGB explícitamente |
| Normalizador por parámetro (`NORMALIZADORES` dict) | hardcodear `/255` en todo lado | ResNet50/MobileNetV2 exigen SU preprocess_input original (BGR+medias caffe, [-1,1]) |
| Sin `.cache()` en RAM | cachear tensores decodificados | ~3 GB extra sobre 16 GB totales; throughput medido no lo necesita |
| `shuffle` solo train + `seed=42` + reshuffle por época | shuffle global determinista | reproducibilidad con variedad entre épocas |
| Labels via `StaticHashTable` ("NORMAL"→0) | one-hot | loss `SparseCategoricalCrossentropy`/binaria acepta entero; menos memoria |
| Augmentación como módulo aparte inspeccionable | augment inline anónimo | se puede graficar, auditar y reutilizar en Grad-CAM |

## 3. Verificación ejecutada (salida real)

| Normalización | Shape batch | dtype | Rango resultante | Estado |
|---|---|---|---|---|
| rescale | (32, 224, 224, 3) | float32 | [0.000, 1.000] | ✔ |
| resnet50 | (32, 224, 224, 3) | float32 | [-123.680, 151.061] | ✔ rango caffe esperado |
| mobilenetv2 | (32, 224, 224, 3) | float32 | [-1.000, 1.000] | ✔ |

```text
batches:    train=146 · validation=17 · test=20     (batch_size=32)
throughput: 181 img/s en CPU (i5-10300H)            → ~26 s/época solo datos
pesos:      {NORMAL(0): 1.937, PNEUMONIA(1): 0.674}
evidencia:  data/reportes/v1_preprocesamiento/augmentacion_muestras.png
```

Bug corregido durante la verificación: `RandomContrast` producía máximos de 1.022 (fuera de [0,1]); añadido `tf.clip_by_value` dentro de cada normalizador.

## 4. Contrato para las siguientes fases

1. **Fase 3 (v001 CNN propia)** usará `normalizacion="rescale"` + `class_weights`.
2. **Fases 4-5 (transfer learning)** usarán `"mobilenetv2"` / `"resnet50"` según la base.
3. Test SIEMPRE con `augment=False` implícito (split ≠ train).
4. Cualquier cambio aquí exige: actualizar docs + re-entrenar comparativamente al menos una versión anterior.

---

**Siguiente paso → Fase 3:** entrenar v001 (CNN propia), cuantizar TFLite, registrar versión.
