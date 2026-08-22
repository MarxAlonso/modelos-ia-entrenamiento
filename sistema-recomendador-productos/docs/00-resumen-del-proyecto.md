# Sistema de Recomendación de Productos con Machine Learning

Proyecto académico — Recomendador híbrido ML + DL para productos de supermercado.

## Objetivo

Diseñar un sistema que recomiende productos a usuarios según:

- **Historial de compras** → Filtrado colaborativo (Scikit-learn / TruncatedSVD)
- **Preferencias y comportamiento** → Content-based con NLP sobre descripciones (TF-IDF)
- **Deep Learning** → Red neuronal con embeddings (TensorFlow/Keras)
- **Híbrido** → Combinación ponderada de los puntajes de ambos modelos

## Pipeline general

```
Dataset real (catálogo supermercado)  +  Historial de compras simulado
                    │
        FASE 1: Limpieza y preparación de datos
                    │
        FASE 2: NLP (limpieza → tokenización → stopwords
                → lematización → TF-IDF)
                    │
        FASE 3: Modelo ML — Filtrado colaborativo (TruncatedSVD)
                    │
        FASE 4: Modelo DL — Red neuronal con embeddings (Keras)
                    │
        FASE 5: Sistema híbrido + métricas (Precision@K, RMSE)
```

## Estado del proyecto

| Fase | Descripción | Estado |
|------|-------------|--------|
| 0 | Entorno de desarrollo | ✅ Completada |
| 1 | Datos: catálogo real + interacciones | ✅ Completada |
| 2 | NLP: preprocesamiento y vectorización | ✅ Completada |
| 3 | Filtrado colaborativo (ML) | ✅ Completada |
| 3b | Dataset real Amazon + traducción ES + NLP reseñas | ✅ Completada |
| 4 | Red neuronal con embeddings (DL) | ✅ Completada |
| 5 | Híbrido + evaluación | ✅ Completada |

**Resultado final (mitad de reporte, sin fuga):** HÍBRIDO P@10 = 0.0070 > KNN 0.0061 > NCF 0.0044 > Contenido 0.0042 > Popularidad 0.0024 · Demo: 15/15 recomendaciones en categorías favoritas

## Estructura del proyecto

```
sistema-recomendador-productos/
├── data/
│   ├── raw/                      # Dataset supermercado original (HuggingFace)
│   ├── raw_amazon/               # Parquets de reseñas Amazon
│   ├── productos.csv             # Catálogo supermercado (7449)
│   ├── usuarios.csv              # Usuarios sintéticos con preferencias (500)
│   ├── interacciones.csv         # Historial de compras simulado (25 854)
│   └── amazon/                   # Datos reales: catálogo, reseñas, núcleo, traducciones
├── docs/                         # Esta documentación, fase por fase
├── models/                       # Artefactos entrenados
│   ├── tfidf_vectorizer.pkl      # Vectorizador TF-IDF (Fase 2)
│   ├── tfidf_matrix.npz          # Matriz 7449 × 5000 (Fase 2)
│   ├── colaborativo_svd.pkl      # SVD + sesgos + similitud ítem-ítem (Fase 3)
│   └── predicciones_svd.npz      # Puntajes ranking 500×6557 (Fase 3)
├── notebooks/                    # Notebooks Jupyter
├── src/                          # Scripts ejecutables por fase
│   ├── preparar_datos.py         # Fase 1
│   ├── preprocesar_nlp.py        # Fase 2
│   ├── colaborativo.py           # Fase 3 (sintético)
│   ├── preparar_amazon.py        # Fase 3b: limpieza dataset real
│   ├── traducir_resenas.py       # Fase 3b: traducción incremental en→es
│   ├── nlp_resenas.py            # Fase 3b: NLP sobre reseñas españolas
│   ├── colaborativo_amazon.py    # Fase 3b: CF con datos reales
│   ├── red_neuronal.py           # Fase 4: NCF con embeddings + contenido
│   └── hibrido.py                # Fase 5: fusión ponderada + recomendar()
├── venv/                         # Entorno virtual Python 3.12.10
└── requirements.txt
```

## Documentación por fase

| Documento | Contenido |
|-----------|-----------|
| [01-fase-0-entorno.md](01-fase-0-entorno.md) | Instalación, versiones, decisiones técnicas |
| [02-fase-1-datos.md](02-fase-1-datos.md) | Dataset, esquema de tablas, generación de interacciones, uniones |
| [03-fase-2-nlp.md](03-fase-2-nlp.md) | Las 5 etapas NLP aplicadas al catálogo |
| [04-fase-3-colaborativo.md](04-fase-3-colaborativo.md) | SVD + KNN ítem-ítem, métricas y validación |
| [05-fase-3b-amazon.md](05-fase-3b-amazon.md) | Dataset real Amazon, traducción ES, NLP de reseñas, comparación |
| [06-fase-4-red-neuronal.md](06-fase-4-red-neuronal.md) | NCF con embeddings, 3 iteraciones documentadas, resultados |
| [07-fase-5-hibrido.md](07-fase-5-hibrido.md) | Fusión ponderada anti-fuga, evaluación final, demo `recomendar()` |

## Cómo ejecutar el entorno

```powershell
cd sistema-recomendador-productos
venv\Scripts\activate
jupyter notebook        # o ejecutar los scripts de src\
```
