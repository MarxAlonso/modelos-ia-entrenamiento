# Fase 4 — Versión v003: MobileNetV2 feature extraction

Fecha: 2026-08-23 · Estado: ✅ Completada · Material de estudio: [`aprendizaje/05-fase-5-transfer-learning.md`](../aprendizaje/05-fase-5-transfer-learning.md)

---

## 1. Qué se construyó

- `src/entrenar_transfer.py`: script genérico de transfer learning (soporta `mobilenetv2` y `resnet50`, base congelada con `training=False`, cabeza GAP→Dense128→Dropout0.3→sigmoid).
- `src/cuantizar.py` mejorado: ahora lee la **normalización de cada versión desde su config.json** (v003 usa la de mobilenetv2, no rescale) tanto para calibrar int8 como para evaluar.

Carpeta: `models/versiones/v003_20260823-2049_mobilenetv2_fe/`

```text
params totales:   2,422,081
entrenables:        164,097  (6.8% — solo la cabeza)
base:               MobileNetV2 ImageNet, congelada
normalización:      mobilenetv2 ([-1,1])
epochs reales:      12 (early stopping en val_auc, best época 6)
tiempo CPU:         21.3 min
```

## 2. Resultados (test congelado)

| Métrica | v002 (cnn propia) | **v003 (transfer)** | Δ |
|---|---|---|---|
| Accuracy | 0.8077 | **0.8654** | +0.058 |
| Recall PNEUMONIA | 0.7897 | **0.9872** | +0.198 |
| F1 macro | 0.8013 | **0.8442** | +0.043 |
| ROC-AUC | 0.8906 | **0.9672** | +0.077 |
| Tiempo entrenamiento | 83.7 min | **21.3 min** | −62 min |

Matriz fp32: TN=155 · FP=79 · FN=**5** · TP=385

### Cuantización y latencia

| Formato | Tamaño | F1 macro | Latencia/img |
|---|---|---|---|
| TFLite dyn | 2.70 MB (4.3×↓) | **0.8696** (> fp32) | 56.9 ms |
| TFLite int8 | 2.90 MB (4.0×↓) | 0.8504 | **7.1 ms** ← despliegue recomendado |

Hallazgos técnicos:
1. int8 es 8× más rápido que dyn en esta arquitectura (depthwise convolutions aprovechan kernels enteros de XNNPACK).
2. Ambos formatos cuantizados superaron el f1 del fp32 → cuantización como regularizador.
3. Trade-off clínico visible: recall_normal 0.662 vs recall_pneumonia 0.987 (class weights + umbral). Documentado; el fine tuning debe mejorar el lado NORMAL.

## 3. Registro

v003 = `vigente`; v002 y v001 = `superadas`. El dashboard leerá directamente esta progresión:

```text
f1_macro:  v001 0.519 → v002 0.801 → v003 0.844
roc_auc:   v001 0.579 → v002 0.891 → v003 0.967
```

---

**Siguiente paso → Fase 5:** fine tuning (descongelar capas superiores, lr 1e-5) sobre v003 o ResNet50 feature extraction según orden del mapa.
