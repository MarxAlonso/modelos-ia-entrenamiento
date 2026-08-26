# Fase 12 — Dataset v3: RSNA Pneumonia Detection + v009

Fecha: 2026-08-25 · Estado: ✅ Completada — **tercer dominio con etiquetas de radiólogos** · Material de estudio: [`aprendizaje/09-fase-10-dominio-y-etiquetas-debiles.md`](../aprendizaje/09-fase-10-dominio-y-etiquetas-debiles.md)

---

## 1. Dataset v3 incorporado

| Campo | Valor |
|---|---|
| Fuente | Espejo `kellly/RSNA-Pneumonia-Detection` (zip oficial del challenge, 3.75 GB) |
| Imágenes | **26 684** (1 por paciente → fuga imposible por construcción) |
| Etiquetas | Por **radiólogos**: Target=1 ⟺ Lung Opacity confirmada; cajas por consenso |
| Resolución | 1024×1024 (L), 0 dañadas en muestra |
| Desbalance | 3.44:1 (NORMAL 20 672 / PNEUMONIA 6 012) — manejable |

Lección de infraestructura: descargar 26 686 archivos individuales disparó el rate-limiting de HF (429 + esperas de minutos). Solución: espejo con **un solo ZIP** = una petición. Regla práctica registrada.

## 2. Manifiesto propio

RSNA no trae split oficial → `generar_manifiesto_v3.py` crea split patient-wise estratificado 80/10/10 (semilla 42):

```text
              NORMAL   PNEUMONIA   total
train          16538        4810   21348
validation      2067         601    2668
test            2067         601    2668
```

Cajas conservadas en columnas (`cajas`) para localización futura.

## 3. Versión v009 (primera sobre RSNA)

```text
MobileNetV2 congelada · NORMAL train limitado a 5000 (--max-normal-train)
8 épocas (early stopping val_auc, best época 4) · 26 min CPU
```

| Métrica | test RSNA (2 668 img) |
|---|---|
| F1 macro | **0.636** |
| ROC-AUC | **0.838** |
| Recall / Precision PNEUMONIA | 0.890 / 0.389 |
| Matriz | TN=1226 · FP=841 · FN=66 · TP=535 |
| int8 | 2.90 MB · f1 **0.659** · 7.3 ms |

## 4. Los tres dominios, cada uno con su campeón

```json
"mejor_por_dataset": {
  "v1_kaggle_chest_xray":  { "id": "v003", "f1_macro": 0.844 },  // pediátrico
  "v2_nih_chest_xray14":   { "id": "v008", "f1_macro": 0.475 },  // adulto, labels NLP
  "v3_rsna":               { "id": "v009", "f1_macro": 0.636 }   // adulto, labels radiólogo
}
```

Lectura científica: NIH y RSNA son ambos adultos PERO con calidad de etiqueta distinta — la diferencia de dificultad entre v008 (0.475) y v009 (0.636) es, en buena parte, **el costo del label noise** medido empíricamente.

## 5. Caminos siguientes para RSNA

1. Más NORMAL en train (--max-normal-train mayor o sin límite) ahora que hay GPU pendiente.
2. Fine tuning completo + DenseNet121 (requiere WSL2+CUDA).
3. Localización con las cajas: segundo modelo o activaciones guiadas.
4. Cross-dataset: entrenar en RSNA y evaluar en test NIH/v1 para cuantificar shift.
