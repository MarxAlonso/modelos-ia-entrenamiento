# Fase 6 — Versión v005: ajuste de umbral (hipótesis refutada, lección registrada)

Fecha: 2026-08-25 · Estado: ✅ Completada (v005 NO supera a v003; registro la marca `superada`) · Material de estudio: [`aprendizaje/07-fase-7-umbrales-y-sobreajuste.md`](../aprendizaje/07-fase-7-umbrales-y-sobreajuste.md)

---

## 1. Qué se construyó

`src/ajustar_umbral.py`: barrido de umbrales sobre validation con restricción recall_pneumonia ≥ 0.97, gráfico del barrido, creación de versión sin reentrenar (hereda modelo + tflites de la base), evaluación única en test y registro.

```powershell
python src/ajustar_umbral.py --version-id v005 --desde models/versiones/v004_20260823-2126_ft
```

Carpeta: `models/versiones/v005_20260825-1335_umbral/`

```text
├── config.json               # umbral_decision=0.09, elegido en validation
├── metricas.json             # test fp32 e int8 con t*
├── barrido_umbral.png        # evidencia visual de la elección
├── modelo_quant_int8.tflite  # heredado de v004 (2.90 MB)
└── modelo_quant_dyn.tflite   # heredado de v004
```

## 2. Resultados

| Etapa | f1_macro | Recall PNEU | Recall NORMAL |
|---|---|---|---|
| Barrido en **validation** | **0.9378** @ t=0.09 | ≥0.97 (restricción) | — |
| Test fp32 con t=0.09 | 0.7372 | 0.9974 | 0.4487 |
| Test int8 con t=0.09 | 0.7116 | 0.9974 | — |

Matriz fp32 test: TN=105 · FP≈129 · FN=1 · TP=389

## 3. Diagnóstico del fallo

El umbral 0.09 que maximizaba f1 en las 519 imágenes de validation era un artefacto de:

1. **Sobreajuste de decisión**: 91 umbrales probados contra un split chico.
2. **Shift validation↔test**: v004 polarizaba perfectamente EN validation (val_precision histórica = 1.0000) pero no en test.
3. **Probabilidades no calibradas**: class_weights deforman los scores; ordenan bien (AUC alto) pero no significan probabilidad literal.

## 4. Veredicto y estado del registro

v003 sigue vigente (f1_macro 0.844). Historial de hipótesis:

| Versión | Hipótesis | Resultado |
|---|---|---|
| v002 | early stopping corregido ↑↑ | ✔ confirmada (+0.28 f1) |
| v003 | transfer learning ↑↑ | ✔ confirmada (+0.04 f1, AUC 0.967) |
| v004 | fine tuning equilibra clases | ✘ refutada (sesgo amplificado) |
| v005 | umbral optimizado ↑ | ✘ refutada (sobreajuste de decisión) |

Costo total de los dos experimentos fallidos: ~35 min de cómputo. Información ganada: dos caminos cerrados con evidencia y documentados para siempre.

## 5. Opciones reales para v006

1. Umbral robusto por validación cruzada k-fold dentro de train+val.
2. Recalibración de probabilidades (Platt/isotonic).
3. Reentrenamiento desde v003 con class_weights suavizados.
4. Dataset v2 (NIH ChestX-ray14) para atacar el shift de raíz.

---

**Siguiente paso sugerido → Grad-CAM sobre v003 (Fase 7 del mapa): entender DÓNDE mira el mejor modelo antes de seguir afinando números.**
