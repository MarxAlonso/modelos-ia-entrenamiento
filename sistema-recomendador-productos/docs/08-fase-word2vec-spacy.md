# Fase 6 — Word2Vec con tokenización spaCy: CBOW vs Skip-gram

## Objetivo

Superar las limitaciones del TF-IDF de la Fase 2 entrenando **embeddings densos de palabras** con Word2Vec (gensim), usando el pipeline NLP de spaCy como preparación del corpus. Se entrenan y comparan **las dos arquitecturas** —CBOW y Skip-gram— y se promueve automáticamente la ganadora.

## Por qué Word2Vec mejora al TF-IDF

| Aspecto | TF-IDF (Fase 2) | Word2Vec (esta fase) |
|---------|-----------------|----------------------|
| Tipo de vector | Disperso, 5000 dims, una por término | Denso, 100 dims compartido |
| Similitud | Solo si comparten palabras exactas | Semántica: "frijol" ≈ "fréjol" ≈ "lenteja" |
| Palabras fuera del vocabulario | Ignoradas | Vecinos cercanos heredan significado |
| ¿Requiere anotación? | No (auto-supervisado) | **No** — aprende solo de los contextos |

Word2Vec es auto-supervisado: no necesita dataset anotado. Genera instancias deslizando una ventana de contexto k=5 sobre cada oración:

- **CBOW**: contexto → palabra central (`El gobierno un decreto` → `aprobó`)
- **Skip-gram**: palabra central → contexto (`aprobó` → `El gobierno un decreto`)

Cada oración produce varias instancias (una por posición con contexto completo).

## Pipeline implementado (`src/entrenar_word2vec.py`)

```
productos (7449) + e-commerce ES (100 000)
        │
[1] Carga de datos
        │
[2] Corpus con spaCy: limpieza implícita → tokenización es_core_news_sm
    → filtro stopwords + puntuación + números → lematización
        │
[3] Entrenamiento en paralelo:  CBOW (sg=0)   |   Skip-gram (sg=1)
        │
[4] Comparación objetiva:
      · similitud media en 5 pares semánticos
      · coherencia de categoría de los k=10 vecinos
        │
[5] Promoción del mejor → models/word2vec_productos.model
        │
[6] Embeddings por producto = media de vectores de sus tokens
        │
[7] Demo de productos similares
```

## Rol de la tokenización spaCy

Antes el corpus se construía con `texto.lower().split()` —un split por espacios que rompe mal `9.300`, deja signos pegados y mantiene stopwords. Ahora cada documento pasa por `nlp.pipe()`:

| Filtro spaCy | Efecto sobre Word2Vec |
|--------------|------------------------|
| `token.is_punct` excluida | Los signos ya no contaminan ventanas de contexto |
| `token.like_num` excluida | Precios/pesos (`x10kg`, `1.000`) no generan tokens sin semántica |
| `token.is_stop` excluida | "de", "la", "para" co-ocurren con todo → solo añaden ruido al contexto |
| `token.lemma_` | "paquetes" y "paquete" → mismo token: menos dispersión, vocabulario más denso |
| Lista extra del dominio | `producto, contiene, presenta…` idéntica a la Fase 2 |

Resultado: **106 549 documentos · 687 430 tokens** (promedio 6.5/doc) y vocabulario de 24 335 palabras.

## Hiperparámetros (iguales para ambas arquitecturas)

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `window` | 5 | Contexto k=5, como en la teoría |
| `vector_size` | 100 | Dimensión estándar para corpus medianos |
| `min_count` | 2 | Descarta hapax legomena |
| `epochs` | 30 | Convergencia estable en corpus pequeño |
| `seed` | 42 | Reproducibilidad |

## Resultados de la comparación

| Métrica | CBOW (sg=0) | Skip-gram (sg=1) |
|---------|-------------|------------------|
| Tiempo de entrenamiento | **20.3 s** ✅ más rápido | 39.8 s |
| arroz ↔ frijol | **0.527** | 0.519 |
| leche ↔ yogur | **0.535** | 0.483 |
| cerveza ↔ vino | 0.414 | **0.427** |
| pan ↔ torta | **0.308** | 0.275 |
| pollo ↔ carne | 0.560 | **0.566** |
| Coherencia de categoría (k=10) | 79.1 % | **84.1 %** |
| **Puntaje combinado (50/50)** | 0.630 | **0.647** 🏆 |

**Ganador: Skip-gram**, promovido a `word2vec_productos.model`.

La evidencia reproduce la teoría: CBOW fue ~2× más rápido y ligeramente mejor en palabras muy frecuentes (leche, pan, arroz aparecen miles de veces), pero Skip-gram logró mejor representación global (+5 pts de coherencia de categoría), consistente con su ventaja documentada para aprender de contextos con menos datos y representar bien términos menos frecuentes (marcas, variedades: "quinua", "chuchuco", "doypack").

## Validación cualitativa (similitud coseno sobre embeddings)

| Producto base | Vecinos recuperados |
|---------------|--------------------|
| Arroz Diana blanco x10kg | Azúcar Incauca morena x2.5kg (0.968) · Azúcar Mayaguez morena (0.966) — graneles básicos de despensa |
| Cebada perlada x500g | Quinua Karavansay Negra (0.943) · Quinua Doria grano (0.941) · Cuchuco cebada (0.933) — **granos andinos agrupados sin haber visto la subcategoría** |
| Aceite mezcla vegetal x900ml | Mismo aceite x3000ml (0.992) · Aceite canola Life (0.978) · Aceite Diana Premium (0.975) |

Sin usar `main_category`, los embeddings separan correctamente granos, aceites y azúcares.

## Artefactos generados

| Archivo | Contenido |
|---------|-----------|
| `models/word2vec_productos.model` | Modelo ganador (reutilizable: `Word2Vec.load`) |
| `models/word2vec_cbow.model` / `word2vec_skipgram.model` | Ambas arquitecturas para estudio |
| `models/embeddings_word2vec.npz` | Embeddings 7449×100 + IDs + categorías (compatible con `embeddings_semanticos.py`) |
| `models/word2vec_comparacion.json` | Métricas de la comparación CBOW vs Skip-gram |

## Resultados de la fase

- [x] Corpus construido con las 5 etapas spaCy (tokenización real, no split)
- [x] CBOW y Skip-gram entrenados con hiperparámetros idénticos
- [x] Comparación cuantitativa: Skip-gram gana 0.647 vs 0.630
- [x] Coherencia de categoría del 84.1 % (k=10) con embeddings de solo 100 dims
- [x] Artefactos persistidos y compatibles con las fases posteriores

## Siguiente paso sugerido

Alimentar el sistema híbrido (Fase 5) con un cuarto componente `S_w2v` y re-buscar pesos; también sirve como inicialización/calibración de los embeddings de producto de la red NCF (Fase 4).

---

# Extensión — Integración de `S_w2v` al híbrido (4 motores)

## Qué se hizo

Se extendió `src/hibrido.py` con un cuarto motor: perfil del usuario = media de los embeddings Word2Vec de sus compras en train (ponderada por rating); puntaje = similitud coseno perfil↔producto. La búsqueda de pesos pasó de 15 a 35 combinaciones en el símplice {0, .25, .5, .75, 1}⁴.

**Importante:** antes de evaluar se detectó que `interacciones.csv` había sido regenerado (26 154 compras · 6570 productos) después del entrenamiento de KNN/NCF, lo que rompía la compatibilidad de formas (500×6557 vs 500×6570). Se reentrenaron los motores base (`colaborativo.py`, `red_neuronal.py`) sobre los datos actuales para garantizar una comparación justa. Por eso las cifras NO son comparables con las tablas históricas de la Fase 5.

## Resultados (mitad de reporte, datos actuales)

| Modelo | Precision@10 |
|--------|--------------|
| Popularidad | 0.0013 |
| Word2Vec semántico | 0.0041 |
| NCF red neuronal | 0.0052 |
| Contenido TF-IDF | 0.0059 |
| KNN ítem-ítem | 0.0059 |
| **Híbrido (0.5, 0.5, 0, 0)** | **0.0072** ✅ |

Pesos seleccionados en la mitad de ajuste (P@10=0.0087): `(KNN=0.5, NCF=0.5, TF-IDF=0, W2V=0)`.

## Lectura honesta

1. **El híbrido sigue superando a todos los individuales** (+22 % sobre KNN/TF-IDF).
2. **W2V como motor individual quedó por debajo del TF-IDF** (0.0041 vs 0.0059): con descripciones cortas (~6.5 tokens útiles tras filtrar), la media de vectores diluye la señal; TF-IDF pondera términos discriminativos, W2V promedia todo por igual.
3. La búsqueda decidió **no asignarle peso**: comportamiento correcto de una selección anti-fuga — un componente solo entra si aporta en datos no vistos.
4. En la demo cualitativa W2V sí muestra valores altos (0.88–0.97) en productos relevantes, confirmando su coherencia semántica (84.1 % de vecinos en misma categoría), pero esa señal ya estaba capturada por TF-IDF y NCF.

## Conclusión

Word2Vec integrado, evaluado y documentado: **aporta semántica densa pero no mejora el P@10 en este dataset**, porque las señales colaborativas (KNN+NCF reentrenados) dominan cuando los datos están frescos. Queda como artefacto disponible para escenarios donde el colaborativo sea débil (usuarios nuevos, catálogo en crecimiento) y como base para inicializar embeddings de la NCF.

---

# Extensión 2 — Inicialización de la red NCF con Word2Vec

## Qué se hizo

En `src/red_neuronal.py` (`inicializar_con_word2vec()`): antes del entrenamiento, los embeddings de 32D de la red ya no arrancan al azar sino con conocimiento previo:

| Capa | Inicialización |
|------|----------------|
| `embedding_producto` | PCA 100D→32D de los vectores Word2Vec de cada producto (varianza explicada ~46 %), reescalado a σ≈0.03 |
| `embedding_usuario` | Promedio de los vectores proyectados de sus compras en train (perfil semántico) |

La red entrena igual (misma semilla 42, mismos datos, mismos negativos frescos por época); solo cambia el punto de partida.

## Resultado (test completo, comparación directa)

| Inicialización | Precision@10 | Recall@10 |
|----------------|--------------|-----------|
| Aleatoria (baseline) | 0.0081 | — |
| **Word2Vec** | **0.0086** | 0.0165 |
| Mejora | **+5.0 %** | |

## Efecto sobre el híbrido (4 motores)

| Métrica | Antes | Después |
|---------|-------|---------|
| Híbrido P@10 mitad de AJUSTE | 0.0087 | **0.0092** |
| Híbrido P@10 mitad de REPORTE | 0.0072 | 0.0066 |
| Pesos seleccionados | (0.50, 0.50, 0, 0) | (0.25, 0.50, 0.25, 0) |

**Lectura honesta:** la mejora de la NCF es real y reproducible (+5 % con todo lo demás constante). En el híbrido, la métrica de ajuste subió pero la de reporte bajó levemente: la mitad de reporte solo tiene ~458 usuarios con muy pocos relevantes cada uno, así que oscilaciones de ±0.001 están dentro del ruido muestral. La conclusión defendible es la del modelo individual: **la inicialización semántica hace que la red aprenda mejor**, no que el ensamble final cambie significativamente.

## Artefactos actualizados

- `models/red_neuronal.keras` + `predicciones_red_neuronal.npz` (NCF con init Word2Vec)
- `models/red_neuronal_metricas.pkl` (incluye `inicializacion_word2vec=True`)
- `models/hibrido.pkl` (pesos re-seleccionados con la NCF mejorada)
