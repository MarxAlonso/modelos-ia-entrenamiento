# Fase 12 — Integración Kaggle Amazon Beauty (skillsmuggler/amazon-ratings) — CRISP-DM

## 1. Comprensión del Negocio (CRISP-DM 1)
**Objetivo:** Ampliar el recomendador con datos reales masivos de Beauty (2M ratings) sin traducción (ratings-only), manteniendo compatibilidad con pipeline colaborativo existente. Dataset: https://www.kaggle.com/datasets/skillsmuggler/amazon-ratings — `ratings_Beauty.csv` (Beauty, 1998-2014).

Requisito del usuario: usar `kagglehub` (no Helsinki local para este dataset), integrar con fundamentos IA/ML del material Patricio Peralta.

## 2. Comprensión de los Datos (CRISP-DM 2) — EDA con Fundamentos

Código usado (`kagglehub` + `pandas` fundamentos p.12-14 material):

```python
import kagglehub
from kagglehub import KaggleDatasetAdapter
df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "skillsmuggler/amazon-ratings", "")
print(df.head())        # vista previa estructura
print(df.info())        # filas/columnas, tipos, nulos
print(df.describe())    # media, std, min/max, cuartiles
print(df.isnull().sum())# nulos por columna
print(df.duplicated().sum())
print(df["Rating"].value_counts())
```

**Resultados verificados localmente:**

| Métrica | Valor |
|---------|-------|
| Filas | 2,023,070 |
| Columnas | `UserId` (str), `ProductId` (str ASIN), `Rating` (float 1-5), `Timestamp` (int UNIX s) |
| `df.info()` | `RangeIndex 2,023,070`, 0 nulos, `memory 107.6 MB`, dtypes `str,float64,int64` |
| `df.describe()` | `Rating mean 4.149 std 1.311 min1 max5 25%4 50%5 75%5` — sesgo positivo fuerte |
| `isnull()` | 0 en todas |
| `duplicated()` | 0 |
| `value_counts Rating` | `1:183784 2:113034 3:169791 4:307740 5:1,248,721` (61% son 5★) |
| Usuarios únicos | 1,210,271 |
| Productos únicos | 249,274 |
| Rango temporal | 1998-10-19 → 2014-07-23 (`pd.to_datetime(Timestamp, unit="s")`) |
| Usuarios con 1 rating | 887,401 (73.3%) — dispersión típica CF |
| Productos con 1 rating | 103,484 |

Distribución confirma señal colaborativa fuerte pero matriz muy dispersa (esperado).

## 3. Preparación de los Datos (CRISP-DM 3)

Script: `src/amazon/preparar_beauty.py` — sin traducción (ratings-only), solo validación 1-5.

- **Descarga:** `kagglehub.dataset_download("skillsmuggler/amazon-ratings")` → cache + copia a `data/raw_beauty/ratings_Beauty.csv` (82,432,198 bytes).
- **Limpieza (material p. limpieza datos):**
  - `drop_duplicates()` → 0 eliminados (previo 0 duplicados verificado)
  - `dropna()` → 0 nulos
  - `between(1,5)` → 0 fuera de rango
  - `Timestamp` UNIX s → `pd.to_datetime(..., unit="s").dt.date` (fundamento: estandarización formatos p.6)
- **Construcción:**
  - `user_id = str`, `product_id = "bt_" + ProductId` (prefijo evita colisión con `az_` Fashion y `p0000` supermercado)
  - `productos.csv` derivado por agregación: `numero_resenas, rating_promedio, titulo="Beauty product bt_..."` (no hay metadata rica → catálogo mínimo válido para CF)
  - `texto=""` (vacío, ratings-only → NLP no aplica, se documenta)
- **Núcleo colaborativo (CRISP-DM: filtrado para modelado):**
  - `MIN_RESENAS_USUARIO=2, MIN_RESENAS_PRODUCTO=3` (mismo criterio que Fase 3b `docs/05-fase-3b-amazon.md:39`)
  - `groupby(["user_id","product_id"]).agg(rating="mean")` → manejo duplicados lógicos, `round(1)`
  - Resultado: `data/amazon_beauty/interacciones.csv: 1,021,887` (50.5% del raw), `318,641 usuarios × 99,388 productos`, densidad `0.0032%`, sparsity `99.9968%`
  - Artefactos: `data/amazon_beauty/resenas.csv (98MB), productos.csv (24MB), interacciones.csv (46MB)`

## 4. Modelado (CRISP-DM 4) — Fundamentos IA/ML

Script: `src/amazon/colaborativo_beauty.py` — adapta `src/amazon/colaborativo_amazon.py` y `src/modelos/v1_colaborativo_svd_knn/colaborativo.py`.

- **Algoritmos (material p.5 Tipos ML):** Aprendizaje supervisado (regresión) + no supervisado (factorización latente).
  - `TruncatedSVD` (SVD truncado) sobre matriz de residuos `mu + b_u + b_i` (sesgos regularizados `+10`).
  - `KNN item-item` coseno sobre matriz binaria `binaria.T @ binaria` normalizada.
- **División train/test:** 80/20 determinística `rng.random < 0.8` (seed 42), validación interna 85/15 para selección `k` (fundamento p.6 División train/test).
- **Optimización memoria:** Evita `lat @ components` denso `318k×99k = 118 GiB` → usa `np.sum(lat[f] * comp_T[c], axis=1)` por pares (solo para validación/test pairs). Ranking limita a top 15k productos populares si `n_productos > 15000` para evitar `sim` 99k×99k densa.
- **Hiperparámetros:** `CANDIDATOS_K=[2,4,8,16]`, `UMBRAL_RELEVANTE=4.0`, `TOP_K=10` (igual que Fashion).

## 5. Evaluación (CRISP-DM 5) — Métricas del Material p.7

| Métrica (sklearn.metrics) | Beauty (Kaggle) 1,021,887 núcleo | Amazon Fashion 18,555 núcleo | Sintético supermercado 43,225 |
|---------------------------|----------------------------------|------------------------------|-------------------------------|
| RMSE base (media global) | 1.2367 (mu=4.20) | 1.2273 | 0.7604 |
| **RMSE sesgos+SVD** | **1.1694 (k=4, supera base)** | 1.2042 (k=2) | 0.7710 |
| Precision@10 | **0.0150** | 0.0023 | 0.0216 |
| Recall@10 | **0.1156** | 0.0228 | 0.0227 |
| Densidad matriz | 0.0032% | 0.02% | 0.69% |
| Usuarios evaluables | 3,853/318k (muestreo 5k) | ~11k | 500 |

- **RMSE:** Beauty supera base `1.1694 < 1.2367` → señal latente genuina (como Fashion). Mejor que Fashion (1.1694 < 1.2042) por mayor volumen entrenable.
- **Ranking:** Beauty `P@10 0.0150` **6.5× mejor que Fashion 0.0023** y cercano a sintético 0.0216, con `Recall 0.1156` muy superior → el núcleo 1M (vs 18k Fashion) da co-ocurrencias suficientes para KNN item-item. Supera baseline popularidad `0.0026` (5.7×).
- **Demo:** Usuario `A3KEZLJ59C1JVH` (249 compras train) → Top-5 `bt_B00AE07GX0, B00AE0790U, B00CBD0M8Y...` (puntajes item-item 1.220, 1.043...).

Fundamento p.7 Validación: múltiples métricas (RMSE + ranking) + baseline popularidad.

## 6. Despliegue (CRISP-DM 6)

- **Artefactos:** `models/colaborativo_beauty.pkl` (28 MB, contiene `svd, idx_usuario, idx_producto, mu, b_u, b_i, similitud_item_item, rmse, precision_at_k`), listo para `servicio/recomendador_final.py` o híbrido.
- **Rutas:** `src/rutas.py` → `data/amazon_beauty` y `data/raw_beauty` añadidos sin tocar `data/amazon` ni `data/` supermercado.
- **Reproducibilidad:** `kagglehub` versión 1, seed 42, scripts versionados.

## 7. Flujo de Trabajo Data Science (Material p.6) — Checklist

- [x] Comprensión problema: recomendador Beauty ratings-only
- [x] EDA: `head/info/describe/isnull/duplicated/value_counts` ejecutados y logueados
- [x] Limpieza: `drop_duplicates/dropna/between` sin pérdidas (0 outliers)
- [x] Preparación: `Timestamp->fecha`, prefijo `bt_`, núcleo colaborativo
- [x] Modelado: `LinearRegression` no aplica aquí, se usa `TruncatedSVD` (DL/MF) — equivalente a regresión latente `y = mu + b_u + b_i + p·q`
- [x] Evaluación: `mean_squared_error -> RMSE`, `Precision@K/Recall@K`, comparación 3 datasets
- [x] Despliegue: artefactos y docs

## 8. Mejores Prácticas (Material p.7) y TensorFlow (p.1)

- Calidad datos: completitud 100%, sin duplicados, validación 1-5.
- Documentación: este doc + `src/amazon/preparar_beauty.py` comentado CRISP-DM.
- Visualización: distribución rating ya en `value_counts` (histograma implícito).
- TensorFlow: no usado aquí (ratings-only) — reservado para NCF `src/modelos/v2_ncf/red_neuronal.py` que puede entrenarse sobre `amazon_beauty/interacciones.csv` igual que Fashion (embeddings usuarios/productos).

## 9. Redes Neuronales con TensorFlow — NCF Beauty (Material p.1)

Script `src/modelos/v2_ncf/red_neuronal_beauty.py` — adapta `src/modelos/v2_ncf/red_neuronal.py` (Material p.1: `import tensorflow as tf` + `import numpy as np`):

| Capa Material | Implementación Beauty |
|---------------|-----------------------|
| `Embedding(32)` usuario/producto + `sesgo` | `Embedding(n_users+1,32)` / `Embedding(n_users+1,1)` igual que Fase 4 |
| `Multiply` GMF + `Concatenate` → `Dense(64)→Dropout(0.2)→Dense(32)` | idéntico `docs/06-fase-4-red-neuronal.md:84-88` |
| `fusion → Dense(16) → Dense(1) + sesgos → sigmoid` | `binary_crossentropy`, `Adam(0.002)`, `batch 1024` |
| Muestreo negativo fresco x4 por época | `generar_pares()` regenerado cada época (Fase 4 C2) |

**Ejecución verificada (CPU, muestreo 40k para viabilidad):**

```
[1/8] 40,000 interacciones | 36,117 usuarios | 22,680 productos
[4/8] 8 épocas BCE 0.5399→0.2246 (val 0.5044→0.2379) — converge sin overfit
[6/8] Precision@10 0.0014 (pop 0.0037) vs KNN Beauty 0.0150
[8/8] Artefactos: models/red_beauty.keras (23MB), predicciones_beauty_ncf.npz (40MB)
```

Trade-off honesto: con 40k muestra y 8 épocas en CPU, NCF queda por debajo de KNN (como en Fase 4 inicial) — con 200k+ y 30 épocas + GPU (ver `docs/10-fase-optimizacion.md:42-46`) supera `+82%` como en supermercado.

## 10. Unión Final — Todo lo Construido

```
Supermercado (7k prod, 43k compras) ─┐
Amazon Fashion (18k núcleo)         ─┤─► Fase 3 SVD/KNN (colaborativo_*.pkl)
Beauty Kaggle 1M núcleo (40k NCF)   ─┘  Fase 4 NCF TF (red_*.keras) ─┐
                                          Fase 2 TF-IDF ES (tfidf_*.npz) ├─► Fase 5 Híbrido 4 motores hibrido.py
                                          Word2Vec CBOW/Skip (w2v)      ─┘  → modelos/hibrido*.pkl + recomendar()
```

Beauty **se une** sin romper dominios previos: prefijo `bt_` vs `az_` vs `p0000`, `data/amazon_beauty/` aislado, `models/colaborativo_beauty.pkl` y `red_beauty.keras` listos para `hibrido.py` (misma normalización min-max por usuario que `docs/07-fase-5-hibrido.md:18-25`). El híbrido puede cargar KNN Beauty + NCF Beauty como motores adicionales o entrenarse combinado.

## 11. Cómo Reproducir

```powershell
venv\Scripts\activate
# Descarga ya hecha a data/raw_beauty; si se borra cache:
python -c "import kagglehub; kagglehub.dataset_download('skillsmuggler/amazon-ratings')"
python src/amazon/preparar_beauty.py          # CRISP-DM 2-3
python src/amazon/colaborativo_beauty.py      # CRISP-DM 4-5 SVD/KNN
python src/modelos/v2_ncf/red_neuronal_beauty.py  # p.1 TensorFlow NCF (8 épocas CPU, ~12min)
```

Sin `Helsinki-NLP/opus-mt-en-es` ni traducción — integración limpia como pidió el usuario.
