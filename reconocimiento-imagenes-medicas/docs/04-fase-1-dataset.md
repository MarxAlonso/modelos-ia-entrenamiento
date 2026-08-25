# Fase 1 — Dataset v1: exploración y manifiesto

Fecha: 2026-08-23 · Estado: ✅ Completada · Material de estudio: [`aprendizaje/01-fase-1-datasets-medicos.md`](../aprendizaje/01-fase-1-datasets-medicos.md)

---

## 1. Qué se hizo

| # | Acción | Script / salida |
|---|---|---|
| 1 | Descarga del dataset (shards Parquet desde HuggingFace, espejo oficial de Kermany et al. 2018) y extracción a carpetas train/validation/test | `src/preparar_datos_v1.py` |
| 2 | Validación de las 5 856 imágenes (integridad, dimensiones, modo de color, duplicados MD5, fuga entre splits) | `src/explorar_dataset.py` |
| 3 | Gráficos de distribución, resoluciones y muestras visuales | `data/reportes/v1_exploracion/*.png` |
| 4 | Manifiesto fijo de splits (test intacto + regeneración estratificada de train/validation con deduplicación) | `src/generar_manifiesto.py` → `data/manifiestos/v1_splits.csv` |
| 5 | Registro del dataset en el índice central | `data/registro_datasets.json` |

Nota sobre el origen: Kaggle exige autenticación API; se usó el espejo `hf-vision/chest-xray-pneumonia` de HuggingFace, que contiene exactamente el mismo dataset (5 856 JPEG, mismos conteos por split) publicado a partir de Mendeley Data rscbjbr9sj v2.

## 2. Resultados de la exploración

```text
Total validado:        5 856 imágenes   (0 dañadas)
Modo de color:         L=5 573 · RGB=283   -> uniformar en preprocesamiento
Resoluciones:          min 384×127 · media 1328×971 · max 2916×2713
Desbalance train:      2.89 : 1  (PNEUMONIA domina)
Duplicados MD5:        26 internos en train (eliminados) · 6 internos en test (conservados)
Fuga train↔test:       0 imágenes compartidas  ✔
```

Gráficos generados:

- `distribucion_clases.png` — evidencia del desbalance por split
- `resoluciones.png` — dispersión de tamaños originales (justifica resize 224²)
- `muestras.png` — ejemplos visuales NORMAL vs PNEUMONIA

## 3. Decisiones tomadas

| Decisión | Justificación |
|---|---|
| Test = test oficial intacto (234/390) | Verificado sin fuga ni solape con train; mantenerlo permite comparar con la literatura que usa este benchmark |
| Train/validation regenerados con semilla 42, estratificados 90/10 | La validation oficial tiene solo 16 imágenes → métricas de ruido |
| Deduplicación MD5 en el pool de train (−26) | Evitar inflar el aprendizaje con copias exactas |
| class_weights NORMAL=1.937 · PNEUMONIA=0.674 | Compensan el desbalance 2.89:1; los usará TODA versión entrenada con v1 |

## 4. Splits finales (congelados)

```text
                NORMAL   PNEUMONIA   total
train             1206        3465    4671
validation         134         385     519
test               234         390     624
```

Cualquier versión futura debe cargar estos splits desde `data/manifiestos/v1_splits.csv`, nunca re-splittear.

## 5. Implicaciones para Fase 2 (preprocesamiento)

1. Convertir todo a un único modo (RGB de 3 canales para compatibilidad con ResNet/MobileNet preentrenadas).
2. Resize 224×224 + normalización según modelo base.
3. Augmentación razonable (rotación ±5°, zoom ≤10%, shift ≤10%) — sin flips horizontales masivos: la anatomía cardiaca es lateral.
4. Cargar rutas desde el manifiesto, no desde la estructura de carpetas.

---

**Siguiente paso → Fase 2:** módulo `src/preprocesamiento.py` + pipeline tf.data.
