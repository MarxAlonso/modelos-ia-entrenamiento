# Plan Maestro — Reconocimiento de Imágenes Médicas con CNN

Proyecto académico y experimental. Clasificador de imágenes médicas que **mejora versión tras versión** mediante más datos, más entrenamiento y mejores técnicas, con modelos **cuantizados** para no sobrecargar la laptop.

> Documento de referencia original: [`mapa_reconocimiento_imagenes_medicas.md`](../mapa_reconocimiento_imagenes_medicas.md)

---

## 1. Alcance

### Qué ES este proyecto

- Clasificar radiografías de tórax (NORMAL vs PNEUMONIA) con CNN.
- Entrenamiento **iterativo y versionado**: cada nueva versión parte del aprendizaje anterior y añade mejoras (más épocas, mejor augmentación, más datos, mejores arquitecturas).
- Modelos exportados **cuantizados** (TFLite) → inferencia ligera en laptop.
- Visualización web con **Astro** leyendo los registros de versiones.

### Qué NO es este proyecto

- ❌ No se creará ningún chatbot / mini-GPT / NLP. Solo visión por computadora.
- ❌ No es un producto de diagnóstico médico real.
- ❌ No replica el sistema recomendador; es un proyecto nuevo e independiente.

---

## 2. Decisiones técnicas

| Tema | Decisión | Justificación |
|---|---|---|
| Framework | TensorFlow + Keras | Ya dominado, soporta TFLite |
| Entrada | 224×224×3 | Compatible con ResNet50/MobileNetV2 |
| Dataset v1 | Kaggle Chest X-Ray Pneumonia (~5 800 img, 2 clases) | Público, pequeño, ideal para empezar |
| Arquitecturas | CNN propia → MobileNetV2 → ResNet50 | De simple a complejo, comparables |
| Cuantización | Post-training: dynamic range + int8 (+ float16 opcional) | Reduce modelo ~4-8×, inferencia rápida en CPU |
| Métricas | Accuracy, Precision, Recall, F1, ROC-AUC, matriz de confusión | Recall es la métrica crítica en medicina |
| Interpretabilidad | Grad-CAM sobre el modelo Keras | Explica dónde "mira" el modelo |
| Visualización | Astro estático | Sin backend pesado; lee JSONs de registro |

---

## 3. Estrategia de cuantización (núcleo del proyecto)

Flujo por cada versión de entrenamiento:

```text
Entrenamiento (Keras, float32)
        │
        ├── models/versiones/vXXX/modelo.keras        ← maestro, sirve para
        │                                               seguir entrenando después
        │
        └── Conversión TFLite (post-training)
                │
                ├── modelo_quant_dyn.tflite   ← dynamic range (~1/4 del tamaño)
                └── modelo_quant_int8.tflite  ← int8 con dataset representativo
                                                (~1/4 tamaño + más rápido en CPU)

Evaluación SIEMPRE comparativa:
    métricas float32  VS  métricas cuantizado
    → si el cuantizado pierde > 1% de recall/F1, revisar calibración
```

Reglas:

1. El `.keras` es el **maestro**: las siguientes versiones se fine-tunean desde él.
2. Los `.tflite` son solo para **inferencia/demo**: nunca se reentrenan.
3. Cada versión registra en `metricas.json` el bloque `cuantizacion`:

```json
{
  "cuantizacion": {
    "dynamic_range": { "tamano_mb": 2.3, "accuracy": 0.961 },
    "int8":          { "tamano_mb": 1.9, "accuracy": 0.958, "dataset_representativo": 200 }
  }
}
```

---

## 4. Estructura de carpetas

```text
reconocimiento-imagenes-medicas/
│
├── mapa_reconocimiento_imagenes_medicas.md   # mapa original (referencia)
├── docs/                                     # planificación y avances
│   ├── 00-plan-maestro.md                    # ESTE documento
│   └── 01-politica-versionamiento.md
│
├── data/
│   ├── registro_datasets.json                # índice versionado (se sube a git)
│   └── versiones/
│       ├── v1_kaggle_chest_xray/             # imágenes (NO se sube a git)
│       └── v2_...                            # futuras ampliaciones
│
├── src/                                      # scripts reutilizables
│   ├── preparar_datos.py
│   ├── preprocesamiento.py
│   ├── entrenar.py                           # entrena una versión completa
│   ├── cuantizar.py                          # keras -> tflite + evaluación
│   ├── gradcam.py
│   └── predecir_tflite.py
│
├── models/
│   ├── registro_versiones.json               # índice central (se sube a git)
│   └── versiones/
│       └── v001_2026-MM-DD_hhmm_cnn/
│           ├── config.json                   # hiperparámetros exactos
│           ├── metricas.json                 # fp32 vs cuantizado
│           ├── matriz_confusion.png
│           ├── curvas_entrenamiento.png
│           ├── modelo.keras                  # maestro (no se sube si pesa)
│           ├── modelo_quant_dyn.tflite
│           └── modelo_quant_int8.tflite
│
├── notebooks/                                # exploración y análisis
│
└── viz/                                      # app Astro (dashboard)
```

---

## 5. Ciclo de mejora continua

El corazón del proyecto: cada iteración produce una versión nueva y comparable.

```text
        ┌────────────────────────────────────────────┐
        │                                            │
        ▼                                            │
  Analizar versión actual                            │
  (¿dónde falla? matriz de confusión + Grad-CAM)     │
        │                                            │
        ▼                                            │
  Decidir mejora:                                    │
  · más épocas / lr schedule                         │
  · mejor augmentación                               │
  · dataset v2 (más imágenes u otra fuente)          │
  · arquitectura superior / fine tuning              │
        │                                            │
        ▼                                            │
  Entrenar nueva versión (desde .keras anterior      │
  o desde cero según la mejora)                      │
        │                                            │
        ▼                                            │
  Cuantizar + evaluar + registrar                    │
        │                                            │
        ▼                                            │
  Actualizar dashboard Astro ────────────────────────┘
```

Una versión se considera **mejor** solo si supera a la anterior en Recall y F1 en el mismo conjunto de test, sin crecer demasiado en tamaño.

---

## 6. Roadmap paso a paso

### Fase 0 — Entorno

- [x] Crear `venv` propio del proyecto e instalar dependencias (`requirements.txt`)
- [x] Verificar TensorFlow, GPU disponible, CPU/RAM → `docs/03-fase-0-entorno.md`
- [x] Definir semillas aleatorias para reproducibilidad (SEED=42)

### Fase 1 — Dataset v1

- [x] Descargar Kaggle Chest X-Ray Pneumonia → `data/versiones/v1_kaggle_chest_xray/`
- [x] Validar imágenes (dañadas, duplicados, resoluciones)
- [x] Analizar balance de clases (este dataset está desbalanceado: usar class weights)
- [x] Notebook de exploración con gráficos
- [x] Registrar en `registro_datasets.json`

### Fase 2 — Preprocesamiento

- [x] `src/preprocesamiento.py`: resize 224×224, normalización, augmentación médicamente razonable
- [x] Splits fijos train/validation/test guardados como manifiesto CSV (para que TODAS las versiones evalúen sobre lo mismo)

### Fase 3 — Versión v001: CNN propia

- [x] Entrenar CNN básica (arquitectura del mapa, sección 8)
- [x] Guardar `modelo.keras` + curvas
- [x] `src/cuantizar.py`: convertir a tflite dyn + int8, comparar métricas
- [x] Matriz de confusión + registrar versión

### Fase 4 — Versión v002+: Transfer Learning

- [x] MobileNetV2 feature extraction → entrenar → cuantizar → registrar
- [ ] ResNet50 feature extraction → entrenar → cuantizar → registrar

### Fase 5 — Versión v00X: Fine Tuning

- [x] Descongelar capas superiores, lr pequeño (1e-5), reentrenar
- [x] Comparar contra feature extraction en el registro

### Fase 6 — Grad-CAM

- [x] `src/gradcam.py` sobre los modelos Keras
- [x] Galería de mapas de calor por versión (aciertos y errores)

### Fase 7 — Inferencia cuantizada

- [x] `src/predecir_tflite.py`: carga `.tflite`, predice clase + probabilidad
- [x] Medir latencia promedio en CPU local

### Fase 8 — Dashboard Astro

- [x] App en `viz/` que lee `models/registro_versiones.json`
- [x] Páginas: evolución de métricas por versión, tabla comparativa, galería Grad-CAM
- [x] Build estático, sin backend

### Fase 9+ — Mejora continua (loop de la sección 5)

- [x] Dataset v2: más imágenes o fuente adicional (NIH ChestX-ray14 ✔, RSNA pendiente)
- [ ] Nuevas versiones partiendo del mejor `.keras`

---

## 7. Métricas mínimas registradas por versión

```text
Accuracy · Precision · Recall · F1 · ROC-AUC
Matriz de confusión (TN FP FN TP)
Tiempo de entrenamiento · Tamaño fp32 vs tflite
Latencia media de inferencia cuantizada (CPU)
Dataset usado (versión) · Configuración completa
```

---

## 8. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Desbalance NORMAL/PNEUMONIA | Class weights + priorizar Recall/F1 |
| Overfitting con dataset pequeño | Augmentación razonable, dropout, early stopping |
| Test contaminado entre versiones | Splits fijos en manifiesto, nunca regenerarlos sin documentarlo |
| Pérdida de calidad al cuantizar | Comparar siempre fp32 vs tflite; int8 usa dataset representativo |
| Laptop sin recursos | Tamaño 224², batch pequeño, MobileNetV2 primero, early stopping |
| Fugas de datos (mismo paciente en train/test) | Verificar en exploración; documentar origen de cada split |

---

## 9. Estado

| Fase | Estado |
|---|---|
| Planificación | ✅ Completada |
| 0 — Entorno | ✅ Completada (`docs/03-fase-0-entorno.md`) |
| 1 — Dataset v1 | ✅ Completada (`docs/04-fase-1-dataset.md`) |
| 2 — Preprocesamiento | ✅ Completada (`docs/05-fase-2-preprocesamiento.md`) |
| 3 — v001 CNN propia + cuantización | ✅ Completada (`docs/06-fase-3-v001.md`) · f1_macro 0.519 baseline, mejoras definidas para v002 |
| 3b — Ciclo: v002 | ✅ Completada (`docs/07-v002-ciclo-de-mejora.md`) · f1_macro 0.801, ROC-AUC 0.891 · v002 vigente |
| 4 — Transfer Learning v003 | ✅ Completada (`docs/08-v003-mobilenetv2.md`) · f1_macro 0.844, ROC-AUC 0.967, recall PNEU 0.987 · int8 7 ms |
| 5 — Fine Tuning v004 | ✅ Completada (`docs/09-v004-fine-tuning.md`) · recall PNEU 0.992 y ROC-AUC 0.972, pero f1_macro 0.816 → v003 sigue vigente; v005 = ajuste de umbral |
| 6 — Umbral v005 | ✅ Completada (`docs/10-v005-umbral.md`) · hipótesis refutada (sobreajuste de decisión) · v003 sigue vigente |
| 7 — Grad-CAM v003 | ✅ Completada (`docs/11-v003-gradcam.md`) · 64% energía en pulmones, 27% en bordes → auditoría "en observación"; propuesto experimento de control |
| 8 — Experimento de control v006 | ✅ Completada (`docs/12-v006-experimento-control.md`) · atajo refutado: sin bordes f1 0.835 ≈ 0.844 · v003 vigente y auditada |
| 9 — Dataset v2 NIH | ✅ Completada (`docs/13-fase-9-dataset-v2.md`) · 112 120 imágenes adulto, manifiesto sin fuga, dominio separado de v1 |
| 10 — v007 NIH | ✅ Completada (`docs/14-v007-nih.md`) · f1 0.334 / AUC 0.684: etiquetas débiles + dominio duro, análisis y opciones v008 · registro ahora por dominio (v003 campeón v1, v007 campeón v2) |
| Dashboard Astro | ✅ Implementado (`viz/`, build verificado: 11 páginas) |
| Inferencia TFLite | ✅ Completada (`docs/15-fase-inferencia-tflite.md`) |
| 11 — v008 NIH | ✅ Completada (`docs/16-v008-negativos-duros.md`) · f1 0.334→0.475, FP −50% · v008 campeón de v2 |
| Resto | ⬜ Pendiente (WSL2+GPU DenseNet / RSNA v3) |
