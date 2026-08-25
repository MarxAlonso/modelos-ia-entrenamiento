# Política de Versionamiento

Reglas oficiales de versionamiento de datasets y entrenamientos. Toda versión debe seguir esta convención sin excepciones, porque el dashboard Astro y las comparaciones dependen de ella.

---

## 1. Dos líneas de versiones independientes

| Línea | Carpeta | Formato | Ejemplo |
|---|---|---|---|
| **Datasets** | `data/versiones/` | `vN_<descripcion>` | `v1_kaggle_chest_xray` |
| **Entrenamientos** | `models/versiones/` | `vNNN_<fecha>_<arquitectura>` | `v003_2026-08-24_1430_resnet50_ft` |

- Los datasets crecen en contenido (más imágenes, otra fuente).
- Los entrenamientos crecen en calidad (mejor Recall/F1 sobre el mismo test).
- Un entrenamiento SIEMPRE declara qué versión de dataset usó.

---

## 2. Contenido obligatorio de un entrenamiento

```text
models/versiones/v003_2026-08-24_1430_resnet50_ft/
├── config.json               # TODO lo necesario para reproducir
├── metricas.json             # resultados fp32 + cuantizado
├── matriz_confusion.png
├── curvas_entrenamiento.png
├── modelo.keras              # maestro reentrenable
├── modelo_quant_dyn.tflite
└── modelo_quant_int8.tflite
```

### config.json mínimo

```json
{
  "version": "v003",
  "dataset": "v1_kaggle_chest_xray",
  "base_anterior": "v002",
  "arquitectura": "resnet50_finetuning",
  "capas_descongeladas": 30,
  "img_size": [224, 224],
  "batch_size": 16,
  "epochs": 20,
  "epochs_reales": 14,
  "learning_rate": 0.00001,
  "augmentacion": ["rotation_5", "zoom_0.1", "width_shift_0.1"],
  "class_weights": { "NORMAL": 1.9, "PNEUMONIA": 0.67 },
  "semilla": 42,
  "notas": "Fine tuning desde v002; mejora recall de NORMAL"
}
```

La regla de oro: **leyendo `config.json` + `metricas.json` se debe entender la versión completa sin abrir código.**

---

## 3. Registro central

Un único archivo índice por línea, actualizado al terminar cada versión:

### `models/registro_versiones.json`

```json
{
  "actualizada": "2026-08-24T14:30:00",
  "mejor_version": "v003",
  "metrica_referencia": "recall_pneumonia_test",
  "versiones": [
    {
      "id": "v001",
      "carpeta": "v001_2026-08-23_1000_cnn",
      "dataset": "v1_kaggle_chest_xray",
      "arquitectura": "cnn_propia",
      "accuracy": 0.91,
      "precision": 0.94,
      "recall": 0.90,
      "f1": 0.92,
      "roc_auc": 0.96,
      "cuantizado": { "formato": "int8", "tamano_mb": 1.8, "accuracy": 0.905 },
      "tamano_mb_fp32": 28.4,
      "tiempo_entrenamiento_min": 12,
      "estado": "superada"
    }
  ]
}
```

### `data/registro_datasets.json`

```json
{
  "datasets": [
    {
      "id": "v1_kaggle_chest_xray",
      "origen": "Kaggle paulmothi/chest-xray-pneumonia-dataset",
      "clases": { "NORMAL": 1583, "PNEUMONIA": 4273 },
      "total_imagenes": 5856,
      "splits": { "train": 5216, "val": 16, "test": 624 },
      "observaciones": "Val muy pequeño: regenerar split propio con manifiesto"
    }
  ]
}
```

---

## 4. Reglas de comparación entre versiones

1. **Mismo test siempre**: todas las versiones evalúan sobre el manifiesto de test fijado en Fase 2. Cambiar el test invalida la comparación.
2. **Criterio de mejora**: una versión es mejor si `recall` y `f1` suben respecto a la anterior (accuracy es secundaria por el desbalance).
3. **Criterio de tamaño**: aceptable hasta 2× el tamaño del cuantizado anterior; más que eso, justificarlo en `config.json → notas`.
4. **Nunca borrar versiones antiguas**: quedan como histórico; el registro marca su `estado` (`vigente`, `superada`, `descartada`).
5. **Mejora de datos vs técnica**: si cambia el dataset, la nueva versión debe indicarlo en `dataset`; las comparaciones directas solo son válidas entre versiones con el mismo dataset y mismo test.

---

## 5. Qué se sube a git y qué no

| Artefacto | Git |
|---|---|
| `docs/`, `src/`, `notebooks/` | ✅ Sí |
| `registro_versiones.json`, `registro_datasets.json` | ✅ Sí |
| `config.json`, `metricas.json`, PNGs de cada versión | ✅ Sí |
| `modelo_quant_int8.tflite` / `dyn.tflite` | ✅ Sí si < ~25 MB |
| `modelo.keras` maestro | ⚠️ Solo si < ~50 MB; si pesa más, queda local y se anota en el registro |
| Imágenes de datasets (`data/versiones/*/`) | ❌ Nunca (regenerables/descargables) |

---

## 6. Convención de nombres — resumen rápido

```text
Dataset:     vN_descripcion            ej: v1_kaggle_chest_xray
Entrenam.:   vNNN_fecha_arq[_detalle]  ej: v002_2026-08-23_1800_mobilenet_fe
                                       fe = feature extraction, ft = fine tuning
```
