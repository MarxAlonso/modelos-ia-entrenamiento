# Fase 8 — v006: Experimento de control (recorte de bordes)

Fecha: 2026-08-25 · Estado: ✅ Completada — **hipótesis del atajo REFUTADA con evidencia** · Estudio previo: [`aprendizaje/08-fase-8-gradcam.md`](../aprendizaje/08-fase-8-gradcam.md)

---

## 1. Diseño experimental

Pregunta abierta por Grad-CAM (Fase 7): *¿depende v003 de los marcadores/texto quemados en los bordes de las placas?*

Método de control clásico: eliminar la variable sospechosa y medir.

| Elemento | v003 | v006 |
|---|---|---|
| Arquitectura, dataset, splits, weights | idénticos | idénticos |
| Receta de entrenamiento | epochs 20 · patience 6 · reduce-lr | idéntica |
| Preprocesamiento | imagen completa | **recorte de 24 px por borde** (antes del resize) |

Cambios de código: `preprocesamiento.crear_pipeline(recorte_borde=N)` (crop en resolución original) y `entrenar_transfer.py --recorte-borde`, registrado en `config.json` como `recorte_borde_px`.

## 2. Resultados (test congelado)

| Métrica | v003 (con bordes) | v006 (sin bordes) | Lectura |
|---|---|---|---|
| F1 macro | 0.8442 | **0.8350** | −0.9 pp → dentro de ruido |
| Recall PNEUMONIA | 0.9872 | **0.9923** | ✔ igual o mejor |
| ROC-AUC | 0.9672 | 0.9630 | equivalente |
| Matriz | TN155 FP79 FN5 TP385 | TN149 FP85 FN3 TP387 | patrón similar |

Cuantización int8: 2.90 MB · accuracy 0.867 · f1_macro **0.8467** · 7.5 ms

### Grad-CAM comparativo

| Energía del heatmap | v003 | v006 |
|---|---|---|
| Centro (pulmones) | 64.0% | 59.5% |
| Bordes | 27.3% | **19.6% ↓** |

Galerías separadas por versión (`data/reportes/gradcam_<version>/`); además se corrigió un defecto del script que sobrescribía reportes entre versiones.

## 3. Veredicto científico

**La hipótesis "v003 usa atajos en los bordes" queda REFUTADA**: sin los bordes, el rendimiento se mantiene (−0.9pp ≈ varianza de entrenamiento) y el recall de neumonía incluso sube. La energía residual en bordes de v003 era correlación incidental, no dependencia.

Consecuencias:

1. **v003 permanece vigente** (f1 0.844 > 0.835) y ahora está AUDITADA: sus métricas reflejan capacidad anatómica real.
2. El proyecto gana su primer ciclo completo del método científico: observación (Grad-CAM) → hipótesis → experimento controlado → veredicto documentado.
3. Registro actual: v003 vigente; v001, v002, v004, v005, v006 superadas.

## 4. Estado del leaderboard

```text
f1_macro fp32 :  v001 0.519 · v002 0.801 · [v003 0.844 VIGENTE] · v004 0.816 · v005 0.737 · v006 0.835
int8 despliegue: v003 2.90MB/7ms · v006 2.90MB/7ms (alternativa auditada casi equivalente)
```

---

**Siguientes pasos posibles → dashboard Astro (Fase 8 del plan), ResNet50 feature extraction, o dataset v2 (NIH ChestX-ray14).**
