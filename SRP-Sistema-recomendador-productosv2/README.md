# SRP v2 — Recomendador de productos con Transformers + Laya

Recomendador híbrido sobre reseñas de moda de Amazon (`data/processed_az`: 202k reseñas, 71k productos)
construido con **Transformers** (Hugging Face) y **[Laya](https://github.com/NandhaKishorM/laya)**,
servido por un backend FastAPI que carga los modelos entrenados como archivos ONNX.

```
"busco aretes de plata para regalarle a mi mamá"
        │
        ▼
 ┌──────────────┐  categoria=joyeria  publico=mujer  es_regalo=sí  sensibilidad_precio=0.1
 │ Laya         │──────────────────────────────────────────────────────────┐
 │ (mmBERT)     │  decisiones tipadas en una pasada, sin generar texto     │ filtros
 └──────────────┘                                                          ▼
 ┌──────────────┐  vector 384-d   ┌─────────────────────┐  top 300  ┌──────────────┐
 │ Two-tower    │────────────────▶│ coseno vs 71k prod. │──────────▶│ SASRec       │──▶ top k
 │ (MiniLM-L12) │                 └─────────────────────┘           │ (si hay      │
 └──────────────┘                                                   │  historial)  │
                                                                    └──────────────┘
```

| Pieza | Arquitectura | Para qué | Archivo que produce |
|---|---|---|---|
| Two-tower | Transformer multilingüe `paraphrase-multilingual-MiniLM-L12-v2`, afinado con InfoNCE | Recuperar candidatos; entiende español contra un catálogo en inglés; funciona con usuarios nuevos (91 % del dataset) | `model.safetensors` (HF) + `encoder.onnx` |
| SASRec | Transformer causal de 2 capas sobre la secuencia de compras | Re-rankear para usuarios con historial | `sasrec.pt` + `sasrec.onnx` |
| Laya | ModernBERT/mmBERT con cabeza de decisión (RLCD) | Convertir lenguaje natural en filtros; leer reseñas sin estrellas | checkpoint `model.safetensors` + `rl_agent_config.json` |

Laya no recomienda por sí mismo: responde preguntas `choice` / `score` / `noul` sobre un texto. Aquí decide
**qué** busca el cliente; los Transformers propios deciden **qué productos** mostrarle.

## Pasos

```powershell
# 0. Entorno (Python 3.12, torch con CUDA para la GTX 1650)
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -r requirements.txt

# 1. Catálogo + partición (leave-last-out) + candidatos de evaluación fijos
.\.venv\Scripts\python src\datos.py

# 2. Two-tower Transformer
.\.venv\Scripts\python src\two_tower.py --solo-evaluar-base   # baseline: encoder sin afinar
.\.venv\Scripts\python src\two_tower.py                       # afina y exporta
.\.venv\Scripts\python src\two_tower.py --continuar           # seguir entrenando la última versión

# 3. SASRec (usa los embeddings de producto del two-tower)
.\.venv\Scripts\python src\sasrec.py

# 4. Laya: dataset de fine-tuning + línea base zero-shot
.\.venv\Scripts\python src\laya_ft\preparar_dataset.py
.\.venv\Scripts\python src\laya_ft\evaluar_laya.py --n 300
#    -> afinar en Kaggle con notebooks\laya_finetune_srp_kaggle.ipynb (2x T4)
#    -> descomprimir el resultado en models\laya_srp\ y re-evaluar:
.\.venv\Scripts\python src\laya_ft\evaluar_laya.py --modelo models\laya_srp

# 5. Publicar la última versión de cada modelo en models\servir\
.\.venv\Scripts\python src\publicar.py

# 6. Backend
cd backend
..\.venv\Scripts\python -m uvicorn app.main:app --port 8000   # http://localhost:8000/docs
```

## Frontend de demo (para la exposición)

Con el backend corriendo, abre **http://localhost:8000/app** (`backend/app/static/index.html`, sin build).
Es una presentación guiada en 5 pasos; cada uno tiene su URL para abrirlo directo:

| Paso | URL | Qué muestra |
|---|---|---|
| 1. Cómo funciona | `/app/#como` | Datos del proyecto y el recorrido Laya → two-tower → SASRec |
| 2. Por historial | `/app/#historial` | 6 clientes reales elegidos automáticamente (fan de relojes, joyería, calzado…), su historial, el embudo de candidatos y cada recomendación con su porqué. Al pasar el mouse se resalta la compra a la que se parece. **Comprar** simula la compra y marca como NUEVO lo que cambió |
| 3. Búsqueda con Laya | `/app/#busqueda` | Consultas de ejemplo; las 4 decisiones de Laya con barras de confianza y si se aplicaron como filtro |
| 4. Resultados | `/app/#resultados` | Gráficos de las métricas (lee `/metricas`) con su lectura honesta |
| 5. Reseñas | `/app/#resenas` | Laya estima estrellas y si recomendaría, a partir de un texto |

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
| POST | `/resenas/analizar` | Laya afinado: reseña → rating estimado + probabilidad de recomendar |
| GET | `/salud`, `/modelo`, `/usuarios/ejemplo` | Estado, manifiesto con métricas, usuarios para probar |

## Resultados (versión publicada `srp-20261001-2330`)

**Two-tower** (`models/two_tower/v001_*`): MiniLM afinado 1 época (100k pares, ~50 min en la GTX 1650).

| Tarea | Métrica | Encoder base | Afinado |
|---|---|---|---|
| Búsqueda por texto, usuario nuevo (reseña → producto) | recall@10 sobre 71k | 0.038 | **0.080** (×2.1) |
| Historial → siguiente compra (historial codificado como texto) | HR@10 (1+99) | 0.240 | 0.186 |
| Historial → siguiente compra (promedio de embeddings, item-to-item) | HR@10 (1+99) | — | 0.255 |

El fine-tuning mejora la búsqueda, que es su función en el backend. Para usuarios con historial se usa
item-to-item porque codificar el historial como texto empeoró.

**Personalización** (3 864 usuarios de evaluación, `models/sasrec/v002_*`):

| Modelo | HR@10 (1+99) | recall@10 catálogo completo |
|---|---|---|
| Popularidad | 0.474 | 0.0411 |
| SASRec + sesgo de popularidad | 0.472 | 0.0401 |
| Two-tower item-to-item | 0.249 | 0.0173 |
| **RRF item-to-item + SASRec** (lo que usa `/recomendar/usuario`) | 0.360 | **0.0445** |

Lectura honesta: en este dataset el 91 % de los usuarios tiene una sola reseña y la siguiente compra suele ser
un producto popular de otra categoría, así que la popularidad es una línea base muy difícil. El protocolo
1+99 con negativos al azar la favorece (Krichene & Rendle, 2020); en recall sobre los 71k productos la
fusión es lo mejor y además recomienda cosas coherentes con el historial.

**Laya zero-shot** (`laya-multilingual`, `data/laya/reporte_*.json`): reseñas: satisfacción 0.37
(azar 0.20), recomendaría 0.64; consultas: categoría 0.58, público 0.59, es_regalo 0.37, precio 0.33.
Por eso, mientras no exista `models/laya_srp/` (fine-tuning en Kaggle), el backend solo aplica
categoría y público.

Todos los modelos se evalúan con los mismos candidatos (`data/splits/candidatos_test.npy`). Cada versión
guarda sus métricas en `models/<modelo>/vNNN_*/metricas.json`, y `src/publicar.py` las resume en
`models/registro_versiones.json`.

## Notas de hardware (GTX 1650, 4 GB)

- fp16 fue **4× más lento** que fp32 (la TU117 no tiene tensor cores): se entrena en fp32 (`SRP_FP16=1` para una T4/RTX).
- Si el pico de memoria pasa de 4 GB, Windows desborda VRAM a RAM y todo se vuelve ~10× más lento: lotes de evaluación de 64.
- Smart App Control bloquea una DLL de scikit-learn; el venv no lo usa (ni `sentence-transformers`).
- `entrenar_todo.cmd` corre el pipeline completo; lánzalo aparte para que sobreviva si cierras la terminal.

## Versionado

- `models/two_tower/vNNN_<fecha>/` y `models/sasrec/vNNN_<fecha>/`: cada entrenamiento crea una versión nueva.
- `models/servir/`: lo único que lee el backend; lo arma `src/publicar.py` con un `manifiesto.json` (sha256 de cada archivo).
- `--continuar` en el two-tower parte de la última versión para seguir entrenando.
