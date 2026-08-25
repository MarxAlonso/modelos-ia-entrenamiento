# Fase 7 — Grad-CAM sobre v003 (el campeón bajo auditoría visual)

Fecha: 2026-08-25 · Estado: ✅ Completada · Material de estudio: [`aprendizaje/08-fase-8-gradcam.md`](../aprendizaje/08-fase-8-gradcam.md)

---

## 1. Qué se construyó

`src/gradcam.py`: Grad-CAM clásico implementado desde cero (~30 líneas de núcleo), con:

- División robusta del modelo recargado en `cam_base` (última conv `out_relu` 7×7×1280) + `head` re-aplicada — tras un reload el grafo interno no es trazable desde los inputs externos, partir por la frontera de la base siempre funciona.
- Selección automática de los casos MÁS representativos por categoría del test: TP/TN/FP/FN ordenados por confianza.
- **Chequeo objetivo de plausibilidad**: fracción de energía del heatmap dentro de una elipse central (campos pulmonares) vs bordes (marcadores/texto de placa).

## 2. Artefactos generados

```text
data/reportes/v1_gradcam/
├── gradcam_TP_0/1.png · TN · FP · FN    # original | heatmap | overlay (p mostrada)
├── gradcam_galeria.png                  # grid resumen 4×3
└── resumen_gradcam.json
```

23 mapas generados sobre v003 (`models/versiones/v003_20260823-2049_mobilenetv2_fe`).

## 3. Resultados y lectura honesta

```json
{
  "energia_media_en_centro": 0.6396,
  "energia_media_en_borde": 0.2734,
  "interpretacion": "Revisar: parte relevante del heatmap cae fuera de la zona pulmonar"
}
```

| Hallazgo | Evidencia | Conclusión |
|---|---|---|
| La mayoría de la atención es anatómica | 64% de energía central (pulmones) | ✔ base legítima |
| Posible uso de artefactos de placa | 27% de energía en bordes | ⚠ auditar con experimento de control |
| Errores (FP/FN) muestran atención dispersa | visible en galería FP/FN | coherente con su naturaleza |

**Estado de v003 tras auditoría: "en observación"** — métricas válidas pero con sospecha razonable de atajos (shortcuts) vía marcadores laterales, problema documentado en la literatura para este dataset.

## 4. Experimento de control propuesto (v006 candidato)

Recortar/ocultar los 24 px de borde (donde viven marcadores L/R y texto quemado) en preprocesamiento → reentrenar feature extraction idéntico → comparar:

- Si f1_macro cae mucho ⇒ el modelo DEPENDÍA de artefactos.
- Si se mantiene ≈0.84 ⇒ los bordes eran ruido secundario; ganamos modelo más honesto.
- Comparación Grad-CAM antes/después como evidencia visual final.

Costo estimado: ~25 min CPU + cambio de una línea en preprocesamiento.

---

**Siguiente paso sugerido → v006 (recorte de bordes + reentrenar) o dashboard Astro con las 5 versiones ya registradas.**
