# SRP v2 — Recomendador de productos con Transformers + Laya

Recomendador híbrido sobre reseñas de moda de Amazon (`data/processed_az`: 202k reseñas, 71k productos)
construido con **Transformers** (Hugging Face) y **[Laya](https://github.com/NandhaKishorM/laya)**,
servido por un backend FastAPI que carga los modelos entrenados como archivos ONNX.

> **Para ejecutarlo:** ver [`EJECUTAR.md`](EJECUTAR.md) (levantar el backend y abrir http://localhost:8000).
> **Para los números del informe:** [`docs/RESULTADOS.md`](docs/RESULTADOS.md) y
> `models/evaluacion/resumen.md` (los calcula `src/evaluacion.py`).

```
"busco aretes de plata para regalarle a mi mamá"
        │
        ▼
 ┌──────────────┐  categoria=joyeria  publico=mujer  es_regalo=sí  sensibilidad_precio=0.1
 │ Laya         │──────────────────────────────────────────────────────────┐ filtros
 │ (mmBERT)     │  decisiones tipadas en una pasada, sin generar texto     │
 └──────────────┘                                                          ▼
 ┌──────────────┐  vector 384-d   ┌─────────────────────┐  top 300  ┌──────────────┐
 │ Two-tower    │────────────────▶│ coseno vs 71k prod. │──────────▶│ SASRec       │──▶ top k
 │ (MiniLM-L12) │                 └─────────────────────┘           │ (si hay      │
 └──────────────┘                                                   │  historial)  │
                                                                     └──────────────┘
                                                                          │
                                            ┌─────────────────────────────┴───────────┐
                                            │ Re-ranker GBDT / MLP (listwise / lambdarank)  │
                                            │ 36 rasgos + clase "ninguno" + confianza     │
                                            │ calibrada (abstención si no sabe)           │
                                            └─────────────────────────────────────────┘
```

| Pieza | Arquitectura | Para qué | Archivo que produce |
|---|---|---|---|
| Two-tower | Transformer multilingüe `paraphrase-multilingual-MiniLM-L12-v2`, afinado con InfoNCE **y negativos duros** | Recuperar candidatos; entiende español contra un catálogo en inglés; funciona con usuarios nuevos (91 % del dataset) | `model.safetensors` (HF) + `encoder.onnx` |
| SASRec | Transformer causal de 2 capas sobre la secuencia de compras | Señal de secuencia para el re-ranker | `sasrec.pt` + `sasrec.onnx` |
| **Re-ranker GBDT / MLP** | LightGBM lambdarank o red de 2 capas sobre 36 rasgos, **objetivo listwise/lambdarank** (softmax sobre el pool + clase "ninguno") + **modelo de confianza calibrado por isotonia** | Ordenar el pool del servicio y **saber cuándo no sabe** (predicción selectiva) | `modelo_gbdt.txt` / `red.pt` + `red.onnx` + `calibrador.json` |
| **Modelo de calidad** | El mismo MiniLM afinado con **dos cabezas** (estrellas 1-5 y "¿recomendaría?") | Estimar la satisfacción con una reseña sin estrellas: RMSE y accuracy | `calidad.onnx` |
| Laya | ModernBERT/mmBERT con cabeza de decisión (RLCD) | Convertir lenguaje natural en filtros | checkpoint `model.safetensors` |

Laya no recomienda por sí mismo: responde preguntas `choice` / `score` / `noul` sobre un texto. Aquí decide
**qué** busca el cliente; los Transformers propios deciden **qué productos** mostrarle.

## Pasos

```powershell
# 0. Entorno (Python 3.12, torch con CUDA para la GTX 1650)
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install lightgbm          # re-ranker GBDT (comparativo)

# 1. Catálogo + partición (leave-last-out) + candidatos de evaluación fijos
.\.venv\Scripts\python src\datos.py

# 2. Two-tower Transformer (con negativos duros: separa el producto de sus confusos)
.\.venv\Scripts\python src\two_tower.py --solo-evaluar-base      # baseline: encoder sin afinar
.\.venv\Scripts\python src\two_tower.py --negativos-duros 2 --max-resenas 120000 --max-len-producto 128
.\.venv\Scripts\python src\two_tower.py --continuar               # seguir entrenando la última versión

# 3. SASRec (usa los embeddings de producto del two-tower)
.\.venv\Scripts\python src\sasrec.py

# 4. Re-ranker: red con objetivo listwise (la que se publica) + comparativo GBDT
.\.venv\Scripts\python src\reranker_mlp.py --epocas 40 --pliegues 5
.\.venv\Scripts\python src\reranker_gbdt.py                       # lambdarank, para comparar

# 5. Modelo de calidad de la reseña (RMSE + "¿recomendaría?")
.\.venv\Scripts\python src\calidad.py --epocas 3 --max-datos 90000

# 6. Laya: dataset de fine-tuning + línea base zero-shot
.\.venv\Scripts\python src\laya_ft\preparar_dataset.py
.\.venv\Scripts\python src\laya_ft\evaluar_laya.py --n 300
#    -> afinar en Kaggle con notebooks\laya_finetune_srp_kaggle.ipynb (2x T4)
#    -> descomprimir el resultado en models\laya_srp\ y re-evaluar:
.\.venv\Scripts\python src\laya_ft\evaluar_laya.py --modelo models\laya_srp

# 7. Evaluación centralizada (TODAS las métricas del informe, con IC y sin fuga)
.\.venv\Scripts\python src\evaluacion.py

# 8. Publicar la última versión de cada modelo en models\servir\
.\.venv\Scripts\python src\publicar.py --sasrec v003_20261003-1048 --reranker v005_20261021-2227  # GBDT (ganador)

# 9. Backend
cd backend
..\.venv\Scripts\python -m uvicorn app.main:app --port 8000   # http://localhost:8000/docs
```

`entrenar_todo.cmd` corre todo el pipeline; lánzalo aparte para que sobreviva si cierras la terminal.

## Frontend de demo (para la exposición)

Con el backend corriendo, abre **http://localhost:8000/app** (`backend/app/static/index.html`, sin build).
Es una presentación guiada en 5 pasos; cada uno tiene su URL para abrirlo directo:

| Paso | URL | Qué muestra |
|---|---|---|
| 1. Cómo funciona | `/app/#como` | Datos del proyecto y el recorrido Laya → two-tower → SASRec → re-ranker |
| 2. Por historial | `/app/#historial` | 6 clientes reales elegidos automáticamente (fan de relojes, joyería, calzado…), su historial, el embudo de candidatos y cada recomendación con su porqué. Al pasar el mouse se resalta la compra a la que se parece. **Comprar** simula la compra y marca como NUEVO lo que cambió |
| 3. Búsqueda con Laya | `/app/#busqueda` | Consultas de ejemplo; las 4 decisiones de Laya con barras de confianza y si se aplicaron como filtro |
| 4. Resultados | `/app/#resultados` | Gráficos de las métricas (lee `/metricas`): ranking frente a líneas base, calidad de la reseña y predicción selectiva, con su lectura honesta |
| 5. Reseñas | `/app/#resenas` | El modelo propio estima estrellas y si recomendaría; Laya se muestra como referencia |

"Modo presentación" agranda la letra para proyectar. `/app/?q=texto#busqueda` lanza una búsqueda al abrir.

## Endpoints

| Método | Ruta | Qué hace |
|---|---|---|
| POST | `/recomendar/consulta` | Texto libre → Laya → two-tower → SASRec (si hay `user_id` con historial) |
| GET | `/recomendar/usuario/{user_id}` | Recomendación personalizada por historial (popularidad si es nuevo) |
| POST | `/recomendar/historial` | Lo mismo para un historial armado a mano (`product_ids`) |
| GET | `/productos/buscar?q=` | Búsqueda directa con el two-tower |
| GET | `/escenarios`, `/metricas` | Clientes de demo y todas las métricas (para el frontend) |
| GET | `/productos/{product_id}/similares` | Vecinos en el espacio del two-tower |
| POST | `/resenas/analizar` | Modelo propio: rating estimado + probabilidad de recomendar (Laya como referencia) |
| GET | `/salud`, `/modelo`, `/usuarios/ejemplo` | Estado, manifiesto con métricas, usuarios para probar |

## Resultados

Los números completos están en [`docs/RESULTADOS.md`](docs/RESULTADOS.md) y los calcula
`src/evaluacion.py` (con IC por bootstrap, sin fuga y con los empates rotos).

**Ranking (protocolo 1+99, 3 864 clientes de evaluación):**

| Modelo | HR@10 | NDCG@10 | HR@1 | categoría@1 |
|---|---|---|---|---|
| Popularidad (línea base) | 0.4726 | 0.3435 | 0.2308 | 0.3693 |
| SASRec | 0.4788 | 0.3542 | 0.2412 | 0.4211 |
| Re-ranker lineal (11 rasgos, versión anterior) | 0.4816 | 0.3616 | 0.2526 | 0.4433 |
| **Re-ranker GBDT (lambdarank, 36 rasgos)** | **0.9501** | **0.8488** | **0.7505** | **0.8168** |
| Re-ranker MLP (36 rasgos, listwise) | 0.8191 | 0.5717 | 0.3597 | 0.5199 |

**Calidad de la reseña (12 723 reseñas de test, nunca vistas):**

| Métrica | Valor |
|---|---|
| RMSE (estrellas 1-5) | **0.6343** |
| Exactitud dentro de ±1 estrella | **96.02 %** |
| Accuracy "¿recomendaría?" (4-5★ vs 1-2★) | **96.77 %** (F1 0.980, AUC 0.992) — la clase mayoritaria acierta 83.0 % |
| Accuracy con 3★ contado como negativo | 91.18 % (mayoría 74.2 %) |

**Confiabilidad:** el re-ranker entrega una confianza calibrada; con el umbral elegido sin ver
los datos de prueba, el sistema acierta **≥ 90 %** de las veces en el tramo de alta
confianza y **se abstiene** en el resto (predicción selectiva). La curva riesgo-cobertura
completa está en `models/evaluacion/evaluacion.json`.

## Notas de método (importantes para el informe)

- **Empates**: `hr_ndcg` (src/comun.py) rompe los empates con un epsilon aleatorio. Sin eso,
  un modelo que da la misma puntuación a cientos de productos se lleva el 100 % de aciertos.
  Se reporta además la versión pesimista (el empate cuenta como fallo).
- **Sin fuga**: `rating_promedio` y `numero_resenas` se recalculan con el **split train**
  solamente (el catálogo los traía calculados con todas las interacciones, incluida la del
  test). Diferencia medida: 0.026 de HR@10.
- **Protocolo 1+99**: favorece a laSgunda línea base (99 de cada 100 negativos son productos
  casi sin ventas). Por eso se reporta también la recall contra el catálogo completo.
- **Cross-fitting**: todo usuario se puntúa con un modelo que no lo vio; la calibración y
  el punto de operación se eligen en una mitad y se miden en la otra.

## Notas de hardware (GTX 1650, 4 GB)

- fp16 fue **4× más lento** que fp32 (la TU117 no tiene tensor cores): se entrena en fp32 (`SRP_FP16=1` para una T4/RTX).
- Si el pico de memoria pasa de 4 GB, Windows desborda VRAM a RAM y todo se vuelve ~10× más lento: lotes de evaluación de 64.
- Smart App Control bloquea una DLL de scikit-learn; el venv no lo usa (ni `sentence-transformers`). LightGBM se usa **por su API nativa**, sin scikit-learn.
- `entrenar_todo.cmd` corre el pipeline completo; lánzalo aparte para que sobreviva si cierras la terminal.

## Versionado

- `models/two_tower/vNNN_<fecha>/`, `models/sasrec/vNNN_<fecha>/`, `models/reranker*/vNNN_<fecha>/` y
  `models/calidad/vNNN_<fecha>/`: cada entrenamiento crea una versión nueva.
- `models/servir/`: lo único que lee el backend; lo arma `src/publicar.py` con un `manifiesto.json` (sha256 de cada archivo).
- `--continuar` en el two-tower parte de la última versión para seguir entrenando.
