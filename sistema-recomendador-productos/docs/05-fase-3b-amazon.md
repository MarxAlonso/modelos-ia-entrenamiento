# Fase 3b — Integración del dataset real de Amazon (reseñas en español)

## Objetivo

Fortalecer el proyecto con **datos reales**: reseñas auténticas de usuarios de Amazon Fashion (`minhth2nh/amazon_product_review_283K`), traducidas al español localmente, aplicando el mismo pipeline NLP y reentrenando el filtrado colaborativo para comparar contra los datos sintéticos.

## 1. Dataset original

| Aspecto | Valor |
|---------|-------|
| Fuente | HuggingFace `minhth2nh/amazon_product_review_283K` |
| Formato | 3 archivos parquet |
| Reseñas | **204 382** |
| Usuarios únicos | 186 622 |
| Productos únicos | 71 214 |
| Dominio | AMAZON FASHION (2004–2023) |
| Idioma | Inglés |

### Columnas aprovechadas

| Columna original | Renombrada | Uso |
|------------------|-----------|-----|
| `user_id` | `user_id` | Clave de usuario |
| `parent_asin` | `product_id` (prefijo `az_`) | Clave de producto |
| `rating` | `rating` | Feedback 1–5 |
| `title_x + text` | `texto` | **Reseña real** (material NLP) |
| `timestamp` (ms) | `fecha` | Análisis temporal |
| `title_y`, `features`, `description`, `price`, `store` | catálogo | Contenido del producto |

### El problema de dispersión (hallazgo clave)

| Métrica | Valor |
|---------|-------|
| Usuarios con exactamente 1 reseña | **93 %** |
| Usuarios con ≥2 reseñas | 13 979 |
| Productos con ≥5 reseñas | 5 882 |
| Densidad matriz completa | 0.0015 % (inutilizable para CF) |

**Solución — núcleo colaborativo:** filtrar a usuarios con ≥2 reseñas Y productos con ≥3 reseñas:

| Tabla resultante | Filas | Descripción |
|------------------|-------|-------------|
| `data/amazon/productos.csv` | 71 214 | Catálogo completo con texto |
| `data/amazon/resenas.csv` | 204 382 | Todas las reseñas limpias |
| `data/amazon/interacciones.csv` | **18 555** | Núcleo: 11 856 usuarios × 6809 productos (densidad 0.02 %) |

## 2. Traducción al español (local, incremental)

**Tecnología:** modelo neuronal seq2seq `Helsinki-NLP/opus-mt-en-es` (~310 MB, PyTorch CPU).

**Diseño incremental** (requisito del proyecto: no traducir todo de golpe):

```
Cada ejecución de src/traducir_resenas.py:
1. Construye la muestra determinística (2000 reseñas, semilla 42, 20–450 caracteres)
2. Lee data/amazon/resenas_traducidas.csv si existe -> set de índices ya traducidos
3. Traduce solo las siguientes 250 pendientes (lotes de 32, num_beams=2)
4. APPEND al CSV -> progreso nunca se pierde
5. Reporta "Progreso total: X/2000"
```

- Velocidad medida: ~16 textos/segundo en CPU → 250 reseñas ≈ 30 s por ejecución.
- Las 2000 reseñas quedaron en `resenas_traducidas.csv` con columnas `texto` (EN) y `texto_es`.
- ¿Por qué solo muestra? Traducir las 204 382 tomaría ≈ 3.5 h de CPU continuo; la muestra es representativa y suficiente para el módulo NLP.

**Ejemplos de calidad:**

| EN | ES |
|----|----|
| Wonderful. I bought this to complete my son's outfit... | Maravilloso. Compré esto para completar el traje de mi hijo... |
| Great product and very stylish | Gran producto y muy elegante |
| Safety. Great product and very stylish | Seguridad. Gran producto y muy elegante |

## 3. NLP sobre reseñas en español (`src/nlp_resenas.py`)

Las mismas 5 etapas de la Fase 2 aplicadas al corpus traducido:

```
Matriz TF-IDF resultante: 2000 reseñas × 2983 términos
Artefactos: models/tfidf_resenas.pkl, models/tfidf_resenas_matrix.npz
```

Demostración sobre `"¡Muy bonito!. Bien, esposa lo adora"`:
limpieza → `['bonito','esposa','adora']` → lemas `['bonito','esposar','adorar']`.

### Análisis de sentimiento léxico (validación del texto)

Se construyó un léxico español simple (positivos/negativos) y se calculó `sentimiento = (pos − neg) / (pos + neg)`:

| Resultado | Valor | Interpretación |
|-----------|-------|----------------|
| Correlación sentimiento ↔ rating | **r = 0.617** (p ≈ 1e−138) | El texto traducido conserva la señal de opinión |
| Reseñas con señal léxica detectable | 1312/2000 | Cobertura razonable del léxico simple |

### Vocabulario discriminativo (TF-IDF por grupo de rating)

| Negativas (1–2 ★) | Positivas (4–5 ★) |
|-------------------|-------------------|
| pequeño (120), tamaño (57), barato (47) | encantar (373), cómodo (287), calidad (235) |
| estrella (44), calidad (41), material (41) | perfecto (185), color (165) |

→ Los clientes negativos se quejan de talla/precio; los positivos destacan comodidad y calidad. Insight de negocio directo desde texto no estructurado.

## 4. Filtrado colaborativo con interacciones REALES (`src/colaborativo_amazon.py`)

Mismo diseño que la Fase 3 (sesgos μ+b_u+b_i, k por validación, KNN ítem-ítem coseno), ejecutado sobre el núcleo real:

- k seleccionado: **2** (RMSE validación plano entre 2/4/8)
- División 80/20: train 14 887 / test 3 668

## 5. Comparación sintético vs real (resultado principal)

| Métrica | Sintético | Amazon real |
|---------|-----------|-------------|
| Densidad matriz | 0.62 % | **0.02 %** |
| RMSE línea base (media) | 0.7604 | 1.2273 |
| RMSE sesgos+SVD | 0.7699 | **1.2042** ✓ supera a la base |
| Precision@10 | 0.0118 | 0.0023 |
| Recall@10 | 0.0210 | 0.0228 |

### Lectura honesta de la comparación

1. **RMSE — gana el mundo real como señal:** con datos sintéticos el SVD NO superaba a predecir la media (ruido uniforme); con datos reales SÍ la supera (1.2042 < 1.2273). Los ratings humanos tienen estructura latente genuina (géneros, marcas, expectativas) que los factores latentes capturan.
2. **Ranking — gana lo sintético por construcción:** con densidad 0.02 % y usuarios con 1 sola compra en train, el ítem-ítem casi no tiene co-ocurrencias; incluso pierde contra popularidad pura (0.0023 vs 0.0078). En el dataset sintético la densidad era 30× mayor y el patrón de categorías artificialmente fuerte.
3. **Conclusión para el proyecto:** los datos reales son más difíciles pero más veraces; el sistema híbrido (Fase 5) existe precisamente para compensar la debilidad del colaborativo puro en datos escasos combinándolo con contenido (TF-IDF de descripciones y reseñas) y la red neuronal.

## Artefactos generados

| Archivo | Contenido |
|---------|-----------|
| `models/tfidf_resenas.pkl` + `.npz` | Vectorizador y matriz TF-IDF de reseñas (Fase NLP-resenas) |
| `models/colaborativo_amazon.pkl` | Modelo CF entrenado con datos reales + métricas |
| `data/amazon/resenas_traducidas.csv` | Muestra bilingüe EN/ES con lemas y sentimiento |

## Resultados de la fase

- [x] Dataset real descargado, limpiado y esquematizado en español
- [x] Núcleo colaborativo extraído con criterio de actividad mínima
- [x] Traducción neuronal local en→es con sistema incremental reanudable
- [x] Pipeline NLP de 5 etapas sobre reseñas españolas
- [x] Sentimiento validado contra rating (r=0.617)
- [x] CF reentrenado con datos reales y comparado honestamente contra sintéticos
