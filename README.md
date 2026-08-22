# 🛒 Sistema de Recomendación de Productos con Machine Learning

Proyecto académico: **recomendador híbrido ML + DL** de productos de supermercado, entrenado y ejecutado 100 % localmente. Combina filtrado colaborativo (Scikit-learn), redes neuronales con embeddings (TensorFlow/Keras) y NLP en español (spaCy + TF-IDF), integrando además un dataset real de reseñas de Amazon traducido al español con un modelo neuronal local.

## 🎯 Resultado final

| Modelo | Precision@10 |
|--------|--------------|
| Popularidad (línea base) | 0.0024 |
| Contenido TF-IDF | 0.0042 |
| NCF red neuronal | 0.0044 |
| KNN ítem-ítem | 0.0061 |
| **🏆 Híbrido ponderado** | **0.0070** |

Demo validada: **15/15 recomendaciones** caen en las categorías favoritas declaradas de los usuarios de prueba, con desglose explicable por componente (`[knn | ncf | texto]`).

## 🔄 Pipeline

```
Catálogo real supermercado (7449 productos)     Reseñas reales Amazon (204 382)
        │                                              │
   Fase 1: limpieza e interacciones               Fase 3b: núcleo colaborativo
        │                                              │
   Fase 2: NLP (5 etapas) → TF-IDF 7449×5000      Traducción EN→ES local (opus-mt)
        │                                              │
   Fase 3: SVD + sesgos / KNN ítem-ítem ──────────┤ NLP reseñas españolas (sentimiento r=0.617)
        │                                        │
   Fase 4: Red NCF embeddings + contenido ────────┘
        │
   Fase 5: HÍBRIDO ponderado → recomendar(usuario) → Top-N
```

## 📚 Documentación por fase

Cada fase está documentada punto por punto en [`docs/`](sistema-recomendador-productos/docs/):

| Doc | Contenido |
|-----|-----------|
| [00-resumen](sistema-recomendador-productos/docs/00-resumen-del-proyecto.md) | Visión general del proyecto |
| [01-fase-0](sistema-recomendador-productos/docs/01-fase-0-entorno.md) | Entorno Python 3.12 + TensorFlow + spaCy |
| [02-fase-1](sistema-recomendador-productos/docs/02-fase-1-datos.md) | Dataset, esquema relacional y generación de interacciones |
| [03-fase-2](sistema-recomendador-productos/docs/03-fase-2-nlp.md) | Las 5 etapas NLP aplicadas al catálogo |
| [04-fase-3](sistema-recomendador-productos/docs/04-fase-3-colaborativo.md) | Filtrado colaborativo SVD + KNN |
| [05-fase-3b](sistema-recomendador-productos/docs/05-fase-3b-amazon.md) | Dataset real Amazon + traducción ES + NLP de reseñas |
| [06-fase-4](sistema-recomendador-productos/docs/06-fase-4-red-neuronal.md) | Red neuronal NCF (3 iteraciones documentadas) |
| [07-fase-5](sistema-recomendador-productos/docs/07-fase-5-hibrido.md) | Sistema híbrido y evaluación final |

## 🚀 Cómo ejecutar

```powershell
cd sistema-recomendador-productos
python -m venv venv                 # Python 3.12 requerido (TensorFlow)
venv\Scripts\activate
pip install -r requirements.txt

# Pipeline completo en orden:
python src\preparar_datos.py        # Fase 1: datos
python src\preprocesar_nlp.py       # Fase 2: NLP + TF-IDF
python src\colaborativo.py          # Fase 3: filtrado colaborativo
python src\red_neuronal.py          # Fase 4: red neuronal NCF
python src\hibrido.py               # Fase 5: híbrido + demo recomendaciones
```

### Módulo opcional Amazon (Fase 3b)

Los archivos pesados no están en el repositorio; se regeneran así:

```powershell
# 1. Descargar parquets de https://huggingface.co/datasets/minhth2nh/amazon_product_review_283K
#    a data/raw_amazon/data/
python src\preparar_amazon.py      # limpia y genera el núcleo colaborativo
python src\traducir_resenas.py     # traducción incremental EN→ES (250/ejecución)
python src\nlp_resenas.py          # NLP sobre reseñas en español
python src\colaborativo_amazon.py  # CF comparativo con datos reales
```

## 📁 Estructura

```
modelos-ia-entrenamiento/
└── sistema-recomendador-productos/
    ├── data/          # catálogo, usuarios, interacciones, amazon/
    ├── docs/          # documentación completa por fase
    ├── models/        # modelos entrenados (.keras, .pkl, TF-IDF)
    ├── src/           # scripts del pipeline (9 fases)
    ├── notebooks/
    └── requirements.txt
```

## 🔑 Decisiones técnicas destacadas

- **SVD sobre residuos con sesgos explícitos** (estilo Netflix Prize): corrigió el fallo clásico del SVD naive sobre matrices dispersas.
- **NCF con muestreo negativo fresco por época**: cambió la formulación MSE→BCE y duplicó la calidad del ranking.
- **Selección de pesos del híbrido sin fuga de datos**: ajuste en una mitad del test, reporte en la otra.
- **Traducción neuronal local EN→ES** con reanudación incremental (Helsinki-NLP opus-mt).

## 👤 Autor

**MarxAlonso** — Proyecto de Machine Learning · Deep Learning · NLP
