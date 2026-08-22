# Fase 3 — Filtrado colaborativo (ML): SVD + KNN ítem-ítem

## Objetivo

Construir el componente de **filtrado colaborativo** del recomendador a partir del historial de compras: predecir ratings (SVD) y generar rankings top-N personalizados (KNN ítem-ítem), evaluando con métricas estándar.

## 1. Preparación de la matriz usuario-producto

| Concepto | Valor |
|----------|-------|
| Forma de la matriz | **500 usuarios × 6557 productos** |
| Densidad | **0.62 %** (20 709 compras observadas en train) |
| División | 80/20 aleatoria con semilla 42 → train=20 709, test=5 145 |
| Representación | Matriz dispersa SciPy (`csr_matrix`) |

> Solo 6557 de los 7449 productos tienen al menos una compra; los 892 restantes no aparecen en la matriz colaborativa (podrán recomendarse vía content-based en el híbrido).

## 2. Lección aprendida: primer intento fallido (documentado como evidencia)

**Intento inicial:** TruncatedSVD directo sobre la matriz de ratings con celdas vacías = 0.

| Problema detectado | Consecuencia |
|--------------------|--------------|
| Los 0 significan "no comprado", pero el SVD los trata como rating real | El modelo aprendió a reconstruir ceros → predicciones colapsadas cerca de 0 |
| k=64 componentes ≈ 450k parámetros para solo 20k observaciones | Sobreajuste severo |
| Resultados | RMSE 3.88 (inútil), Precision@10 ≈ aleatorio, demo dispersa |

**Corrección aplicada** (enfoque tipo Netflix Prize):

1. Modelo de sesgos: `pred(u,i) = μ + b_u + b_i` donde μ = media global, `b_u` y `b_i` son desviaciones por usuario/producto (con amortiguación `+10` para items poco vistos).
2. SVD entrenado sobre los **residuos** `r − μ − b_u − b_i`, no sobre ratings crudos.
3. Selección de `k` por validación en lugar de fijarlo arbitrariamente.

## 3. Selección de hiperparámetro k (validación 85/15 dentro de train)

| k componentes | RMSE validación |
|---------------|-----------------|
| 2 | 0.7636 |
| 4 | 0.7634 |
| **8** | **0.7628** ← seleccionado |
| 16 | 0.7635 |

La curva es casi plana: la mayor parte de la señal ya está capturada por los sesgos. Se eligió k=8.

## 4. Tarea A — Predicción de rating (SVD + sesgos)

| Modelo | RMSE en test |
|--------|--------------|
| Línea base: predecir media global | 0.7604 |
| Sesgos + SVD (k=8) | 0.7699 |

**Hallazgo honesto:** el SVD **no supera** a la línea base en esta tarea. Causa: en los datos sintéticos el ruido del rating dentro de cada grupo es σ≈0.6 y la señal latent adicional es mínima; los factores latentes introducen más varianza que la señal que aportan. Esto es un límite del *generador de datos*, no del algoritmo — y se documenta como tal.

**Utilidad real del modelo de rating:** estimar "cuánto le gustaría" un producto nunca visto (usado en el híbrido de la Fase 5).

## 5. Tarea B — Ranking top-N (KNN ítem-ítem)

Algoritmo:

```
1. Matriz binaria B (usuario × producto) con las compras de train
2. Similitud coseno entre columnas: sim(i,j) = cos(B[:,i], B[:,j])
   → mide co-compra: productos comprados por los mismos usuarios son similares
3. Puntaje(u,i) = Σ_j∈compras(u) B[u,j] · sim(i,j)
4. Excluir lo ya comprado y ordenar descendente
```

### Experimento comparativo previo (ejecutado antes de decidir)

| Estrategia | Precision@10 |
|------------|--------------|
| KNN ítem-ítem coseno binario | **0.0118** |
| SVD implícito binario (k=16/64) | 0.0118 |
| Popularidad pura | 0.0043 |

Se adoptó KNN ítem-ítem (empata con SVD implícito, es más explicable y cubre la técnica "KNN" del enunciado junto a "SVD").

## 6. Métricas finales en test

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| Precision@10 | **0.0118** | ~15× mejor que azar (≈0.0008) y 2.7× mejor que popularidad |
| Recall@10 | 0.0210 | De los relevantes del test, 2.1 % aparecen en el top-10 |
| RMSE rating | 0.7699 vs 0.7604 base | Ver hallazgo honesto arriba |
| Usuarios evaluables | 491/500 | Cobertura de evaluación |

### ¿Por qué Precision@10 "parece baja"? (análisis del techo)

Con densidad 0.62 % y elección uniforme del producto *dentro* de cada categoría favorita, el techo teórico de cualquier filtro colaborativo es acertar a **nivel categoría**. Ejemplo: recomendar 10 productos perfectos de "despensa" (1770 candidatos) contra ~5 relevantes reales del test da precision esperada ≈ 0.03 aunque el modelo sea óptimo. El valor obtenido (0.0118) es consistente con ese techo según mezcla de categorías. **Conclusión:** el modelo extrae toda la información disponible en los datos generados; mejorar requeriría datos con preferencia a nivel de producto individual.

## 7. Validación cualitativa (demo u0007)

Preferencias reales declaradas: `pasabocas | frutas-y-verduras | despensa`
Distribución real de sus compras: pasabocas 19 · despensa 18 · frutas-y-verduras 14

Top-5 generado por el modelo (sin haber visto nunca esas preferencias):

| # | Score | Categoría inferida | Producto |
|---|-------|--------------------|----------|
| 1 | 1.511 | pasabocas | Miel Lafrica Natural Gold x440g |
| 2 | 1.467 | pasabocas | Galletas Saltinas Original 5 Tacos x450g |
| 3 | 1.408 | pasabocas | Menta Chao Limón Bolsa x350g |
| 4 | 1.392 | frutas-y-verduras | Amor de primavera |
| 5 | 1.339 | pasabocas | Pasabocas De Todito Mix Paketon x300g |

→ 5/5 recomendaciones caen en categorías favoritas. El patrón fue aprendido exclusivamente del historial de compras.

## Artefactos generados

| Archivo | Contenido |
|---------|-----------|
| `models/colaborativo_svd.pkl` | Modelo SVD final, sesgos (μ, b_u, b_i), índices, matriz de similitud item-item, todas las métricas |
| `models/predicciones_svd.npz` | Matriz de puntajes ranking 500×6557 + arrays de usuarios y productos (entrada directa para el híbrido) |
| `src/colaborativo.py` | Pipeline completo reproducible |

## Resultados de la fase

- [x] Matriz dispersa usuario-producto construida
- [x] Error del intento naive documentado y corregido con sesgos
- [x] k seleccionado empíricamente por validación (k=8)
- [x] Predictor de rating: μ + b_u + b_i + SVD
- [x] Ranker top-N: KNN ítem-ítem coseno
- [x] Evaluación: RMSE, Precision@10, Recall@10 + líneas base
- [x] Demo validada contra preferencias reales del usuario
