# Fase — Inferencia local cuantizada (demo)

Fecha: 2026-08-25 · Estado: ✅ Completada

---

## Qué se construyó

`src/predecir_tflite.py`: predictor de línea de comandos que carga el `.tflite`
de cualquier versión y aplica EXACTAMENTE su preprocesamiento declarado en
`config.json` (normalización por arquitectura + recorte de borde si existe).

```powershell
python src/predecir_tflite.py `
  --modelo models/versiones/v003_20260823-2049_mobilenetv2_fe/modelo_quant_int8.tflite `
  --imagen data/versiones/v1_kaggle_chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg `
  --benchmark 30
```

Salida real con v003 int8:

```text
Clase predicha: PNEUMONIA      ← etiqueta real: PNEUMONIA ✔
Probabilidad:   72.66%
Latencia promedio CPU (30 repeticiones): 9.36 ms ± 0.64 ms

Clase predicha: NORMAL         ← etiqueta real: NORMAL ✔
Probabilidad:   1.95%
```

## Detalles técnicos

- Lee `config.json` hermano del `.tflite` → garantiza mismo preproceso que en entrenamiento (contrato de Fase 2).
- Warmup de 3 invocaciones antes de medir (los primeros `invoke` cargan kernels XNNPACK).
- Respeta `umbral_decision` de la versión (v005-style overrides funcionan).

## Estado del mapa original

Con esto quedan cubiertas las fases del mapa: exploración ✔, preprocesamiento ✔,
CNN propia ✔, transfer learning ✔, fine tuning ✔, comparación ✔, Grad-CAM ✔,
**inferencia cuantizada ✔**. Pendientes opcionales: demo Streamlit (sustituida
por dashboard Astro), ResNet50 feature extraction y dataset RSNA.
