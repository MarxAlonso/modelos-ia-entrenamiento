# Fase 2 — NLP: preprocesamiento y vectorización del texto

## Objetivo

Aplicar las 5 etapas clásicas de preprocesamiento NLP sobre las descripciones de los 7449 productos, para convertir texto libre en español a vectores numéricos que permitan medir **similitud semántica entre productos** (componente *content-based* del recomendador).

## Por qué el NLP es necesario aquí

El catálogo real trae `ai_generated_description`: texto rico en español ("bolsa con capacidad para 10 kg… adecuada para consumo familiar"). Los modelos no entienden texto; la vectorización convierte cada producto en un punto en un espacio de 5000 dimensiones donde **productos parecidos quedan cerca**. Eso habilita frases del tipo: *"a quien le compró arroz Diana blanco también le puede interesar…"* basado en contenido.

## Pipeline implementado (`src/preprocesar_nlp.py`)

```
texto crudo → [1] Limpieza → [2] Tokenización → [3] Stopwords
            → [4] Lematización → [5] TF-IDF → matriz 7449 × 5000
```

## Las 5 etapas aplicadas

### Demostración sobre un ejemplo real

Texto de prueba:
`"Los Paquetes de Arroz Diana blanco x10kg están corriendo en promoción!!! Compra ya."`

| Etapa | Técnica | Resultado |
|-------|---------|-----------|
| 1. Limpieza | minúsculas; regex elimina URLs, números y signos (`[^a-záéíóúüñ\s]`) | `los paquetes de arroz diana blanco x kg están corriendo en promoción compra ya` |
| 2. Tokenización | `nlp()` de spaCy divide en unidades mínimas | `['los','paquetes','de','arroz','diana','blanco','x','kg','están','corriendo','en','promoción','compra','ya']` |
| 3. Stopwords | filtro `token.is_stop` + lista propia de palabras ruidosas del dominio | `['paquetes','arroz','diana','blanco','x','kg','corriendo','promoción','compra']` |
| 4. Lematización | lematizador estadístico de `es_core_news_sm` | `['paquete','arroz','diana','blanco','x','kg','correr','promoción','compro']` |
| 5. Vectorización | `TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)` | vector disperso por producto |

### Detalle de cada decisión

**Etapa 1 — Limpieza**
- Minúsculas para unificar "Arroz" = "arroz".
- Se conservan tildes y ñ (español) pero se eliminan dígitos y puntuación.
- Colapsa espacios múltiples.

**Etapa 3 — Stopwords**
- Stopwords oficiales de spaCy para español (de, la, el, están, …).
- Lista extra del dominio supermercado detectada al leer descripciones: `producto(s), contiene, presentado/presenta, además, sugiere, tipo, forma, tamaño` — palabras que aparecen en casi todas las descripciones generadas y no aportan discriminación.

**Etapa 4 — Lematización (la etapa clave)**

Reduce cada palabra a su lema (forma base de diccionario), igualando variantes morfológicas:

| Palabra original | Lema |
|------------------|------|
| corriendo | correr |
| paquetes | paquete |
| compra | compro |
| presentes / presenta | presentar / presentar |

Efecto: "paquetes" y "paquete" cuentan como el mismo término → el TF-IDF no fragmenta la señal.

**Etapa 5 — Vectorización TF-IDF**

| Hiperparámetro | Valor | Efecto |
|----------------|-------|--------|
| `max_features` | 5000 | Vocabulario acotado a los términos más informativos |
| `ngram_range` | (1, 2) | Captura unigramas y bigramas ("leche entera", "arroz blanco") |
| `min_df` | 2 | Descarta términos que aparecen en un solo producto (ruido) |

Resultado: **matriz dispersa 7449 × 5000** donde cada celda es peso TF-IDF = frecuencia del término × rareza del término.

## Rendimiento del procesamiento

| Métrica | Valor |
|---------|-------|
| Documentos procesados | 7449 |
| Modo | `nlp.pipe(batch_size=128)` (desactivados `parser` y `ner`, innecesarios) |
| Dimensionalidad final | 5000 features |

## Artefactos generados

| Archivo | Contenido |
|---------|-----------|
| `models/tfidf_vectorizer.pkl` | Vectorizador entrenado (serializado con joblib) — reutilizable para vectorizar nuevos productos o consultas |
| `models/tfidf_matrix.npz` | Matriz TF-IDF dispersa (SciPy) |
| `data/productos.csv` | Actualizado con columna `texto_lematizado` |

## Validación: similitud coseno

Se calculó `cosine_similarity` entre el producto base y todo el catálogo:

**Producto base:** `Arroz Diana blanco x10kg`

| Similitud | Producto recuperado |
|-----------|--------------------|
| 1.000 | Arroz Diana blanco x10kg *(duplicado textual de otra presentación)* |
| 0.365 | Arroz Diana blanco x25und x500g |
| 0.353 | Arroz Diana blanco fideos x460g |
| 0.310 | Azúcar Providencia orgánica doypack x454g |
| 0.306 | Azúcar Providencia blanca x1kg |

**Interpretación:** sin haber visto jamás la columna `main_category`, el sistema agrupa correctamente arroces blancos granulados y azúcares — productos visual y funcionalmente parecidos. El componente content-based está operativo.

## Resultados de la fase

- [x] 5 etapas NLP implementadas y demostradas una a una
- [x] Lematización en español funcionando (corriendo → correr)
- [x] Corpus completo procesado: 7449 documentos
- [x] Matriz TF-IDF 7449 × 5000 entrenada y persistida
- [x] Búsqueda por similitud coseno validada
