# Fase 11 — v008: negativos duros en NIH (mejora del dominio adulto)

Fecha: 2026-08-25 · Estado: ✅ Completada — **v008 nuevo campeón de v2** (f1 0.475 vs 0.334)

---

## 1. Qué se cambió (sobre v007, mismo test congelado)

| Cambio | Implementación |
|---|---|
| **Negativos duros** | 3 000 hallazgos tipo-infiltración/consolidación (visualmente parecidos a neumonía) añadidos como no-neumonía en train/val. Nuevo manifiesto `v2_splits_duros.csv` generado por `generar_manifiesto_v2.py --negativos-duros 3000 --salida v2_splits_duros.csv`; columna `rol=negativo_duro` para auditoría; TEST intacto |
| **Oversampling de positivos ×2** | `entrenar_transfer.py --sobremuestrear-positivos 2` duplica filas PNEUMONIA de train (754→1 508 efectivas) |
| Registro de datasets | Nueva variante `v2duros` en `preprocesamiento.DATASETS` con el mismo `id` de dominio → comparaciones válidas por política §4 |

## 2. Resultados (mismo test NIH, 10 416 imágenes)

| Métrica | v007 | **v008** | Δ |
|---|---|---|---|
| F1 macro | 0.334 | **0.475** | **+0.141** ✔ |
| Accuracy | 0.395 | 0.676 | +0.281 |
| Recall NORMAL | 0.368 | 0.684 | +0.316 |
| FP | 6 233 | **3 115** | −50% |
| Recall PNEUMONIA | 0.870 | 0.535 | trade-off asumido |
| ROC-AUC | 0.684 | 0.661 | ≈ igual |

Cuantización int8: 2.90 MB · accuracy 0.696 · f1_macro 0.482 · **6.5 ms**

## 3. Lectura honesta

1. La hipótesis central se confirmó: los hallazgos miméticos eran la causa principal de los FP masivos; enseñándoselos como "no neumonía", el modelo dejó de disparar ante infiltraciones.
2. El recall de neumonía bajó: el punto de operación se movió hacia especificidad. Con AUC ≈ 0.66, existe un umbral intermedio mejor que 0.5 para ambas clases (lección v005).
3. val_auc decayó desde la época 1 → early stopping restauró pesos tempranos otra vez. Para v009: más épocas con lr inicial menor (5e-4) y patience mayor, o cosine schedule.
4. El techo sigue siendo el **label noise** de NIH (etiquetas por NLP). Mejora estructural real = dataset RSNA (labels de radiólogos).

## 4. Estado

```text
mejor_por_dataset: { v1_kaggle_chest_xray: v003 (f1 0.844) · v2_nih_chest_xray14: v008 (f1 0.475) }
```

**Siguientes pasos → WSL2+CUDA para DenseNet121/EfficientNet fine tuning completo, o dataset v3 RSNA.**
