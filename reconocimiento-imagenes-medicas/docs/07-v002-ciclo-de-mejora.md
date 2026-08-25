# Fase 4 (ciclo) — Versión v002: early stopping corregido + LR schedule

Fecha: 2026-08-23 · Estado: ✅ Completada · Material de estudio: [`aprendizaje/04-fase-4-ciclo-de-mejora.md`](../aprendizaje/04-fase-4-ciclo-de-mejora.md)

---

## 1. Qué se hizo

Primera iteración formal del ciclo de mejora continua. Cambio controlado respecto a v001 (arquitectura, dataset, splits y weights idénticos):

| Parámetro | v001 | v002 |
|---|---|---|
| EarlyStopping monitor | val_recall (ruidoso) ⚠ | **val_auc** (estable) |
| patience | 5 | 8 |
| epochs máx | 25 | 35 |
| ReduceLROnPlateau | no | sí (÷3, patience 2, sobre val_auc) |

`src/entrenar.py` quedó parametrizado por CLI para que futuras versiones no toquen código:

```powershell
python src/entrenar.py --version-id v002 --monitor val_auc --patience 8 --epochs 35 --reduce-lr
```

Carpeta: `models/versiones/v002_20260823-1919_cnn/` (mismos artefactos que v001).

## 2. Resultados (test congelado, mismo test de siempre)

| Métrica | v001 fp32 | v002 fp32 | Δ |
|---|---|---|---|
| Accuracy | 0.5705 | **0.8077** | +0.237 |
| Recall PNEUMONIA | 0.7179 | **0.7897** | +0.072 |
| F1 macro | 0.5191 | **0.8013** | **+0.282** |
| ROC-AUC | 0.579 | **0.891** | +0.312 |
| Épocas efectivas | 1 (restaurada por error) | 35 | — |
| Tiempo CPU | 13.7 min | 83.7 min | +70 min |
| Tamaño / cuantización | idénticos (109k params → 0.12 MB int8) | | |

Matriz v002 fp32: TN=196 · FP=38 · FN=82 · TP=308 (antes TN=76 FP=158 FN=110 TP=280)

### Cuantización v002

| Formato | Tamaño | Accuracy | F1 macro | Latencia |
|---|---|---|---|---|
| dyn | 0.12 MB | 0.8189 | **0.8121** (mejor que fp32) | 11.5 ms |
| int8 | 0.12 MB | 0.8109 | 0.8039 | 11.4 ms |

Hipótesis declarada en `docs/06-fase-3-v001.md`: "ROC-AUC > 0.80 y f1_macro > 0.70" → **confirmada en ambas**.

## 3. Observaciones del entrenamiento

1. `ReduceLROnPlateau` actuó 4 veces (épocas 5, 15, 22, 32); cada reducción desbloqueó mejoras inmediatas (val_auc: 0.76→0.86→0.95→0.959).
2. La época 35 fue la mejor: el modelo aún no convergía → margen para v003.
3. Señales leves de overfitting desde época ~9 (train↑ constante, val_loss oscilante), contenidas por dropout+augmentación.

## 4. Registro

`registro_versiones.json`: v002 `vigente`, v001 `superada`. Mejor versión por f1_macro con desempate recall_pneumonia.

## 5. Propuestas para la siguiente iteración

- **v003** (opcional, ciclo): epochs 50 + cosine decay; esperado f1_macro ≥ 0.85.
- **Fase 4 del mapa (recomendada ahora)**: transfer learning MobileNetV2 feature extraction — salto cualitativo esperado (f1_macro ≥ 0.90) gracias a ImageNet.

---

**Siguiente paso → Fase 4 del plan:** MobileNetV2 transfer learning (v003).
