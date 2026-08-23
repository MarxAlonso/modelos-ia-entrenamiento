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
│   ├── amazon/                   # Catálogo + reseñas traducidas (real)
│   ├── titulos-ecommerce-es/     # Corpus externo de títulos ES
│   ├── clip-ecommerce/           # Corpus externo adicional
│   ├── productos.csv             # Catálogo supermercado (7449)
│   ├── usuarios.csv              # Usuarios sintéticos con preferencias (500)
│   ├── interacciones.csv         # Historial ampliado (43 225)
│   └── comportamiento.csv        # Embudo view→cart→purchase (236 353 eventos)
├── docs/                         # Documentación fase por fase
├── models/                       # Artefactos entrenados (+ versiones/)
├── ver_grafos/                   # Generador de JSONs + app React
├── venv/                         # Entorno virtual Python 3.12.10
├── requirements.txt
└── src/
    ├── rutas.py                  # Rutas compartidas del proyecto
    ├── datos/                    # Fase 1 y ampliaciones
    │   ├── preparar_datos.py     # Catálogo + interacciones base
    │   ├── generar_interacciones.py
    │   └── ampliar_datos.py      # Fase 10: +65% compras y comportamiento
    ├── nlp/
    │   ├── preprocesar_nlp.py    # Fase 2: 5 etapas → TF-IDF
    │   └── nlp_resenas.py        # Fase 3b: NLP reseñas Amazon
    ├── modelos/                  # Una carpeta por mejora de modelo
    │   ├── v1_colaborativo_svd_knn/colaborativo.py
    │   ├── v2_ncf/red_neuronal.py            # init Word2Vec (Fase 4b)
    │   ├── v3_word2vec/                      # entrenar_word2vec + semánticos SBERT
    │   ├── v4_hibrido_multimotor/            # hibrido 4 motores + mejorado + scoring
    │   ├── v5_torres_gnn_clip/               # Two-Tower, LightGCN, pipeline V4, CLIP
    │   └── v6_mini_gpt/lenguaje_mini_gpt.py  # Transformer propio
    ├── amazon/                   # Fase 3b: preparar, traducir, colaborativo real
    ├── servicio/
    │   ├── recomendaciones_app.py # recomendaciones.json para la app
    │   ├── recomendador_final.py
    │   └── recomendar_cli.py
    └── mlops/
        ├── autoentrenamiento.py   # ciclo vigilante de reentrenamiento
        ├── entrenar_cadena_wsl.sh # cadena completa en GPU
        ├── gpu_env.sh
        └── entrenar_gpu.ps1
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
| [08-fase-word2vec-spacy.md](08-fase-word2vec-spacy.md) | Word2Vec CBOW vs Skip-gram con tokenización spaCy, comparación y promoción |
| [09-fase-mini-gpt.md](09-fase-mini-gpt.md) | Mini-GPT Transformer desde cero sobre el corpus propio (generación de texto ES) |
| [10-fase-optimizacion.md](10-fase-optimizacion.md) | +65% datos, dataset de comportamiento, reentrenamiento GPU y app con react-icons |

## Cómo ejecutar el entorno

```powershell
cd sistema-recomendador-productos
venv\Scripts\activate
jupyter notebook        # o ejecutar los scripts de src\<carpeta>\
```

Los scripts se ejecutan desde la raíz del proyecto: `python src\modelos\v4_hibrido_multimotor\hibrido.py`. Todos resuelven sus rutas vía `src/rutas.py`, sin importar el directorio de ejecución. Para entrenar en GPU (WSL2): `powershell src\mlops\entrenar_gpu.ps1 -Pasos 5000` o la cadena completa con `src\mlops\entrenar_cadena_wsl.sh`.
