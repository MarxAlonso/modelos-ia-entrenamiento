# Fase 4 — Red neuronal con embeddings (Deep Learning, TensorFlow/Keras)

## Objetivo

Construir el componente **Deep Learning** del recomendador: una red neuronal que aprende embeddings de usuarios y productos para generar rankings top-N personalizados sobre el dominio 100 % en español (catálogo de supermercado + descripciones TF-IDF de la Fase 2).

## 1. Evolución del modelo: tres iteraciones documentadas

Esta fase fue un proceso de depuración real — cada intento fallido se documenta porque enseña el porqué del diseño final.

### Iteración A — MLP sobre embeddings concatenados (MSE)

```
Embedding(u) ⊕ Embedding(p) → Dense(64) → Dense(32) → rating
Pérdida: MSE sobre ratings | 232k parámetros
```

| Problema detectado | Evidencia |
|--------------------|-----------|
| La red debía aprender las medias de rating "a fuerza" dentro del MLP | RMSE 0.9039 (peor que predecir la media global: 0.7601) |
| Sobreajuste severo | train loss bajó a 0.28 mientras val_loss se estancaba en ~0.86 |
| Ranking inútil | Precision@10 = 0.0004; demo dispersa (vinos para amante de pasabocas) |

### Iteración B — NeuMF con ramas de sesgo explícitas

Se aplicó el patrón que funcionó en la Fase 3 (sesgos μ + b_u + b_i), ahora como capas de la red:

```
sesgo_u = Embedding(nU+1, 1) inicializada en la media global
sesgo_p = Embedding(nP+1, 1) inicializada en ceros
GMF     = producto elemento a elemento de los embeddings (interacción latente)
MLP     = concat(u,p) → Dense(64) → Dense(32)
salida  = sesgo_u + sesgo_p + Dense(1)([GMF ‖ MLP])
Pérdida: MSE
```

| Resultado | Valor |
|-----------|-------|
| RMSE test | **0.7617** vs línea base 0.7601 → la red arranca exactamente en la base y ya no la empeora |
| Precision@10 | 0.0016 — sigue sin superar popularidad (0.0043) |

**Diagnóstico:** en regresión MSE, las parejas nunca vistas quedan ≈ μ + b_u + b_i + ruido. El ranking no tiene señal personalizada.

### Iteración C (final) — NCF con pérdida binaria y muestreo negativo

Cambio de formulación según He et al. (Neural Collaborative Filtering): para top-N, el objetivo correcto es **distinguir compra de no-compra**, no predecir el valor del rating.

| Elemento | Diseño final |
|----------|--------------|
| Positivos | Cada compra real de train (20 709) |
| Negativos | 4 productos NO comprados muestreados al azar por cada compra (**negativos frescos regenerados en cada época**) |
| Pérdida | `binary_crossentropy` sobre salida sigmoide (`probabilidad_compra`) |
| Optimizador | Adam(0.002), batch 512 |
| Épocas | 30 (la curva val_loss siguió bajando hasta el final: 0.48 → 0.087) |

### Arquitectura final (`recomendador_neural_hibrido`)

| Capa | Salida | Parámetros | Papel |
|------|--------|------------|-------|
| entrada usuario / producto / contenido_espanol | — | 0 | IDs + vector LSA del texto español |
| embedding_usuario (32 dim) | 32 | 16 032 | Vector latente del usuario |
| embedding_producto (32 dim) | 32 | 209 856 | Vector latente del producto |
| interaccion_gmf (Multiply) | 32 | 0 | Compatibilidad usuario-producto |
| oculta_1 → dropout → oculta_2 | 32 | 6 240 | MLP no lineal |
| **texto_espanol (Dense 32)** | 32 | 2 080 | Rama de contenido: TF-IDF reducido con LSA a 64 dim (varianza explicada 28.7 %) |
| fusion (96) → fusion_oculta (16) | 16 | 1 552 | Combina GMF + MLP + contenido |
| sesgo_usuario / sesgo_producto | 1 | 7 059 | Prior de popularidad/actividad |
| logit → sigmoide | 1 | 17 | Probabilidad de compra |
| **Total** | | **242 836 (~950 KB)** | |

## 2. Por qué "negativos frescos por época" fue decisivo

| Versión | Negativos | Mejor época | Precision@10 |
|---------|-----------|-------------|--------------|
| C1 | Fijos una sola vez + EarlyStopping | 1 | 0.0037 |
| **C2 (final)** | Regenerados cada época | 30 | **0.0086** |

Con negativos fijos la red los memoriza y deja de aprender (val_loss sube desde la época 2). Al regenerarlos, cada época plantea un problema nuevo y los embeddings siguen convergiendo durante 30 épocas sin sobreajustar.

## 3. Resultados finales en test

| Modelo | Precision@10 | Recall@10 |
|--------|--------------|-----------|
| Popularidad (línea base) | 0.0043 | — |
| KNN ítem-ítem (Fase 3) | **0.0118** | 0.0210 |
| Red neuronal NCF (Fase 4) | 0.0086 | 0.0161 |

→ La red **duplica a la popularidad** y queda segunda tras el KNN. Recordar el techo analizado en la Fase 3: con elección uniforme dentro de categoría, ~0.01–0.02 es el máximo alcanzable; ambos modelos colaborativos operan cerca de ese techo.

## 4. Validación cualitativa — evolución de la demo (u0007, favoritas: pasabocas/frutas-y-verduras/despensa)

| Iteración | Top-5 generado | Lectura |
|-----------|----------------|---------|
| A (MPL-MSE) | vinos, avena, queso… disperso | Sin aprendizaje |
| B (NeuMF-MSE) | pescado, carne, ponqué, agua… disperso | Solo sesgos |
| C1 (NCF 1 época) | 5/5 carnes idénticas | Clustering sí, personalización no |
| **C2 (final)** | arepas quinua, caldo gallina, natilla, arepas paisa (lacteos/despensa/pasabocas-afines) | **Concentración + variedad personalizada** |

La demo final muestra dos capacidades emergentes: los scores casi idénticos (0.94–0.95) revelan que el espacio de embeddings agrupa alimentos afines, y el filtro por usuario acerca el ranking a sus categorías favoritas.

## 5. Artefactos generados

| Archivo | Contenido |
|---------|-----------|
| `models/red_neuronal.keras` | Modelo Keras entrenado (formato nativo, recargable) |
| `models/predicciones_red_neuronal.npz` | Matriz completa de probabilidades 500×6557 (entrada del híbrido) |
| `models/red_neuronal_metricas.pkl` | Índices, métricas, configuración (dim_embedding=32, dim_contenido=64, n_negativos=4) |
| `src/red_neuronal.py` | Pipeline reproducible completo |

## 6. Lecciones aprendidas (para sustentación)

1. **Formulación > arquitectura:** cambiar MSE→BCE con muestreo negativo mejoró el ranking más que cualquier cambio de capas.
2. **Los sesgos explícitos estabilizan:** inicializar la salida exactamente en la línea base evita que la red la empeore mientras aprende.
3. **Negativos estáticos se memorizan:** regenerarlos por época es lo que permitió entrenar 30 épocas.
4. **El DL aporta representaciones:** el clustering de productos en el espacio latente (demo C1/C2) es algo que ni SVD ni KNN ofrecen explícitamente.
5. El KNN gana aquí porque explota co-ocurrencia directa; el híbrido de la Fase 5 combinará ambas fortalezas + contenido TF-IDF.

## Resultados de la fase

- [x] Tres iteraciones documentadas con evidencia numérica
- [x] Arquitectura NCF híbrida final (embeddings + GMF + MLP + texto español)
- [x] Entrenamiento estable de 30 épocas con negativos frescos
- [x] Precision@10 = 0.0086 (2× popularidad), Recall@10 = 0.0161
- [x] Demo cualitativa validada en español
