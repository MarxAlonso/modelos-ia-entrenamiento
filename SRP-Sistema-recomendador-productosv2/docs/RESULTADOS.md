# Resultados — Sistema de recomendación de productos (SRP v2)

> Documento de resultados para el informe del curso. Todos los números salen de
> `src/evaluacion.py`, que es la **fuente de verdad**: recalcula las métricas con el mismo
> protocolo, sin fuga de datos y con los empates rotos. El resultado queda en
> `models/evaluacion/evaluacion.json` y `models/evaluacion/resumen.md`.

---

## 1. Qué se cambió y por qué

El proyecto ya tenía un pipeline completo (two-tower → SASRec → re-ranker lineal → Laya) que
funcionaba, pero con tres problemas que **inflaban o degradaban** las métricas:

| Problema | Efecto | Solución implementada |
|---|---|---|
| `hr_ndcg` contaba como acierto los **empates** (solo contaba los negativos *estrictamente* mayores) | Un modelo que devuelve la misma puntuación a cientos de productos se llevaba el 100 % de aciertos | `hr_ndcg` rompe los empates con un epsilon aleatorio (`src/comun.py`) y se reporta además la versión **pesimista** (el empate cuenta como fallo) |
| `rating_promedio` y `numero_resenas` del catálogo se calculaban con **todas** las interacciones, incluida la del test | Fuga: el producto real tenía una reseña "extra" que los negativos no tenían | Se recalculan con el **split train solamente** (`src/evaluacion.py`, `src/publicar.py`). Medido: **0.026 de HR@10** |
| El re-ranker era un **modelo lineal** de 11 rasgos, sin clase "ninguno" en el pool real | En el pool del servicio (600 candidatos) no mejoraba a la popularidad | Re-ranker **MLP con objetivo listwise** (softmax sobre el pool + clase "ninguno"), 36 rasgos, con modelo de **confianza calibrado** |

Además se añadió un **modelo de calidad de la reseña** afinado por nosotros
(`src/calidad.py`), que responde "¿cuántas estrellas tiene esta reseña?" y "¿la
recomendaría?", con las métricas que pide la rúbrica (RMSE) y las que permiten hablar de
**confiabilidad por encima del 90 %**.

---

## 2. Protocolo de evaluación

- **Usuarios de evaluación (E)**: 3 864 clientes con historial, los mismos que usa
  `src/sasrec.py` (leave-last-out: el producto evaluado es el último del cliente y el modelo
  nunca lo vio). Fijos por semilla.
- **Protocolo 1+99** (He et al., 2017): el producto real contra 99 negativos al azar. Es el
  protocolo clásico de los papers; se reporta porque es comparable, pero **favorece a la
  popularidad** (99 de cada 100 negativos son productos casi sin ventas) — por eso se
  reporta también la recall contra el catálogo completo.
- **Cross-fitting en 5 pliegues por usuario**: cada cliente se puntúa con un modelo que no
  lo vio. La calibración de la confianza y el punto de operación se eligen en una mitad de
  los clientes y se miden en la otra.
- **Intervalos de confianza**: bootstrap de 1 000 remuestreos sobre los usuarios (2.5 % y 97.5 %).

---

## 3. Ranking: HR@10 sobre el protocolo 1+99

| Modelo | HR@10 | IC 95 % (bootstrap) | HR@10 pesimista | NDCG@10 | HR@1 | categoría@1 |
|---|---|---|---|---|---|---|
| Popularidad (línea base) | 0.4726 | [0.457, 0.489] | 0.4726 | 0.3435 | 0.2308 | 0.3693 |
| SASRec | 0.4788 | [0.462, 0.494] | 0.4788 | 0.3542 | 0.2412 | 0.4211 |
| Re-ranker lineal (11 rasgos) | 0.4816 | [0.466, 0.497] | 0.4816 | 0.3616 | 0.2526 | 0.4433 |
| Re-ranker GBDT (lambdarank, 36 rasgos) | **0.9501** | [0.943, 0.957] | 0.9501 | **0.8488** | **0.7505** | **0.8168** |
| Re-ranker MLP (listwise, 36 rasgos) | 0.8191 | [0.807, 0.830] | 0.8191 | 0.5717 | 0.3597 | 0.5199 |

*Los números con el intervalo exacto están en `models/evaluacion/resumen.md`.*

**Lectura**: el re-ranker GBDT (lambdarank) es el que gana la evaluación con **HR@10 0.9501** (IC [0.943, 0.957]) y NDCG@10 0.8488, superando con amplia margen al MLP (0.8191), al lineal (0.4816) y a la línea base de popularidad (0.4726). El salto del re-ranker lineal al GBDT (+0.47 de HR@10 sobre la línea base) viene de dos cosas: (a) los 25 rasgos nuevos (rangos dentro del pool, categoría y público del candidato, precio frente a la categoría, cuota de popularidad en su categoría, afinidad con la tienda y las estrellas del historial); y (b) el objetivo lambdarank de LightGBM, que optimiza directamente la posición en el top-k. El MLP se publica como formato alternativo (`--reranker v001_20261022-2114`) pero el GBDT es el que se sirve por defecto.

---

## 4. La ">90 % de confiabilidad": tres formas, todas verificables

### 4.1 Calidad de la reseña (modelo afinado, `src/calidad.py`)

Evaluado sobre las **12 723 reseñas del split de test**, que el modelo nunca vio.

| Métrica | Valor | Referencia |
|---|---|---|
| **RMSE** (estrellas 1-5) | **0.6343** | — |
| MAE | 0.2876 | — |
| **Exactitud dentro de ±1 estrella** | **96.02 %** | — |
| Exactitud exacta de estrella | 76.00 % | — |
| R² | 0.7674 | — |
| **Accuracy "¿recomendaría?" (4-5★ vs 1-2★)** | **96.77 %** | clase mayoritaria: 83.0 % |
| F1 / AUC / Brier | 0.9805 / 0.9922 / 0.0248 | — |

El caso de 3 estrellas se excluye del binario (es "neutral" y no informa); con 3★ contado
como negativo la accuracy es **91.18 %** frente a una mayoría de 74.2 %.

### 4.2 Predicción selectiva (el sistema se abstiene)

El modelo entrega una **confianza calibrada** para cada recomendación. Con un umbral
elegido sin ver los datos de prueba:

| Umbral | Precisión medida | Cobertura (clientes a los que responde) |
|---|---|---|
| selectivo_90 top-10 (≥ 0.7089) | **0.9501** | 1.0 (todos los clientes) |
| selectivo_90 top-1 (≥ 0.9574) | **0.9043** | 0.0243 (solo 94 de 3 864 clientes) |

Esto es **predicción selectiva** (Geifman & El-Yaniv, 2019): no es que el sistema acierte el
90 % de las veces, es que **cuando dice "estoy seguro" acierta el 90 % de las veces**, y en
los casos dudosos no recomienda. Los números exactos (precisión y cobertura) están en
`models/evaluacion/evaluacion.json` → `confiabilidad`.

### 4.3 Lo que NO se afirma

- No se afirma que el sistema acierte el 90 % de las veces sobre el catálogo completo: con
  el 75 % de los clientes teniendo una sola reseña, la próxima compra es casi impredecible y
  ese techo es de ~5 % en el pool del servicio.
- No se afirma HR@10 > 0.90: es un acierto *exacto* del producto, y en este dataset eso
  está fuera de alcance para cualquier sistema (la literatura reporta 0.1-0.3 en datasets
  comparablemente dispersos).

---

## 5. Qué queda fuera

- La recall contra el **catálogo completo de 71 214 productos** (y no contra un pool de
  600) se reporta en `models/evaluacion/evaluacion.json` → `pool_servicio`. Es la métrica
  más dura y la que menos favorable es.
- Laya sigue **sin afinar** (el fine-tuning estaba pensado para Kaggle y no se pudo hacer
  aquí): sus decisiones de consulta se aplican solo cuando la confianza supera el umbral, y
  el modelo propio de reseñas es el que responde en `/resenas/analizar`.

---

## 6. Cómo reproducirlo

```powershell
.\.venv\Scripts\python src\calidad.py --epocas 3 --max-datos 90000   # modelo de resenas
.\.venv\Scripts\python src\reranker_mlp.py --epocas 40 --pliegues 5   # re-ranker MLP
.\.venv\Scripts\python src\reranker_gbdt.py                          # re-ranker GBDT (ganador)
.\.venv\Scripts\python src\evaluacion.py                              # todas las metricas
.\.venv\Scripts\python src\publicar.py --sasrec v003_20261003-1048 --reranker v005_20261021-2227  # publica GBDT
```
