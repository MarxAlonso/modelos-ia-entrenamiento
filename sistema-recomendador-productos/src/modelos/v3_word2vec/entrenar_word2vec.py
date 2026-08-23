"""Pipeline Word2Vec con tokenizacion spaCy: CBOW vs Skip-gram.

Integra el pipeline NLP de la Fase 2 (limpieza -> tokenizacion -> stopwords ->
lematizacion con es_core_news_sm) para construir un corpus de calidad, entrena
LAS DOS arquitecturas de Word2Vec y las compara con metricas objetivas:

  - CBOW      (sg=0): rapido, mejor para palabras frecuentes.
  - Skip-gram (sg=1): mas lento, aprende bien palabras raras.

La comparacion usa dos criterios:
  1. Similitud semantica en pares de prueba (arroz-frijol, leche-yogur...).
  2. Coherencia de vecinos: % de k-NN que comparten main_category.

Guarda el mejor modelo como models/word2vec_productos.model y los embeddings
de los productos en models/embeddings_word2vec.npz (compatible con
embeddings_semanticos.py).
"""

import sys
from pathlib import Path
_RUTA_MODULO = Path(__file__).resolve()
while _RUTA_MODULO.name != "src" and _RUTA_MODULO != _RUTA_MODULO.parent:
    _RUTA_MODULO = _RUTA_MODULO.parent
sys.path.insert(0, str(_RUTA_MODULO.parent))

from src.rutas import RUTA_RAIZ

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize

warnings.filterwarnings('ignore')

RUTA_BASE = RUTA_RAIZ
RUTA_DATOS = RUTA_BASE / "data"
RUTA_MODELOS = RUTA_BASE / "models"
STOPWORDS_EXTRA = {"producto", "productos", "contiene", "presentado", "presenta",
                   "ademas", "además", "sugiere", "tipo", "forma", "tamaño", "tamano"}

VENTANA = 5       # contexto k=5 (como en la teoria CBOW/skip-gram)
DIMENSION = 100
EPOCAS = 40


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga datasets existentes y nuevos (glob por sufijos hash en parquets)."""
    print("[1/7] Cargando datasets...")

    df_prod = pd.read_csv(RUTA_DATOS / "productos.csv")
    print(f"  - Productos supermercado: {len(df_prod)}")

    textos_extra = []
    for carpeta in ["titulos-ecommerce-es", "clip-ecommerce"]:
        for parquet in sorted((RUTA_DATOS / carpeta / "data").glob("*.parquet")):
            df = pd.read_parquet(parquet)
            columna = "title" if "title" in df.columns else "Product_name"
            textos_extra.append(df[columna].astype(str))
            if "Description" in df.columns:
                textos_extra.append(df["Description"].astype(str))
            print(f"  - {carpeta}: {len(df):,} filas ({columna})")
    df_ecom = pd.concat(textos_extra, ignore_index=True).to_frame("title")
    print(f"  - Corpus externo total: {len(df_ecom):,} titulos")

    return df_prod, df_ecom


def preparar_corpus_spacy(df_prod: pd.DataFrame, df_ecom: pd.DataFrame, nlp) -> list[list[str]]:
    """Construye el corpus aplicando tokenizacion spaCy + stopwords + lematizacion.

    A diferencia de un split() por espacios, el tokenizador de spaCy separa
    correctamente puntuacion, numeros decimales ("9.300") y palabras pegadas;
    el filtro de stopwords elimina palabras funcionales que co-ocurren con
    todo y solo anaden ruido al contexto de CBOW/skip-gram.
    """
    print("[2/7] Preparando corpus (spaCy: tokenizacion + stopwords + lemas)...")

    textos_prod = [
        str(row.get('texto', '') or row['name'])
        for _, row in df_prod.iterrows()
    ]
    textos_ecom = [str(t) for t in df_ecom['title']]

    corpus = []
    total_tokens = 0
    for doc in nlp.pipe(textos_prod + textos_ecom, batch_size=256):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.like_num
            and token.is_alpha
            and len(token.lemma_) > 2
            and token.lemma_.lower() not in STOPWORDS_EXTRA
        ]
        if len(tokens) >= 2:
            corpus.append(tokens)
            total_tokens += len(tokens)

    print(f"  - Documentos: {len(corpus):,}")
    print(f"  - Tokens totales: {total_tokens:,} (promedio {total_tokens/len(corpus):.1f}/doc)")
    return corpus


def entrenar_arquitectura(corpus: list[list[str]], sg: int) -> Word2Vec:
    """Entrena Word2Vec. sg=0 -> CBOW | sg=1 -> Skip-gram."""
    nombre = "CBOW" if sg == 0 else "Skip-gram"
    inicio = time.time()
    modelo = Word2Vec(
        sentences=corpus,
        vector_size=DIMENSION,
        window=VENTANA,
        min_count=2,
        workers=4,
        epochs=EPOCAS,
        sg=sg,
        seed=42,
    )
    print(f"    {nombre}: vocabulario {len(modelo.wv):,} palabras "
          f"({time.time()-inicio:.1f}s)")
    return modelo


def similitud_pares(modelo: Word2Vec) -> float:
    """Promedio de similitud en pares semanticos de prueba."""
    pares = [
        ("arroz", "frijol"),
        ("leche", "yogur"),
        ("cerveza", "vino"),
        ("pan", "torta"),
        ("pollo", "carne"),
    ]
    sims = []
    for w1, w2 in pares:
        if w1 in modelo.wv and w2 in modelo.wv:
            sims.append(modelo.wv.similarity(w1, w2))
            print(f"    {w1} <-> {w2}: {sims[-1]:.3f}")
        else:
            print(f"    {w1} <-> {w2}: [fuera de vocabulario]")
    return float(np.mean(sims)) if sims else -1.0


def coherencia_categoria(emb: np.ndarray, cats: np.ndarray, k: int = 10,
                         muestra: int = 800, seed: int = 42) -> float:
    """% de los k vecinos mas cercanos que comparten la misma categoria."""
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(emb), size=min(muestra, len(emb)), replace=False)
    emb_n = normalize(emb)
    sub = emb_n[idxs]
    sims = sub @ emb_n.T
    aciertos = 0
    total = 0
    for fila, qi in zip(sims, idxs):
        fila[qi] = -np.inf
        vecinos = np.argpartition(fila, -k)[-k:]
        aciertos += sum(cats[v] == cats[qi] for v in vecinos)
        total += k
    return aciertos / total


def embedding_producto(modelo: Word2Vec, texto: str, nlp=None) -> np.ndarray:
    """Embedding promedio de los tokens conocidos de un texto."""
    tokens = str(texto).lower().split()
    vectores = [modelo.wv[t] for t in tokens if t in modelo.wv]
    if vectores:
        return np.mean(vectores, axis=0)
    return np.zeros(modelo.vector_size)


def evaluar_y_comparar(cbow: Word2Vec, skipgram: Word2Vec,
                       df_prod: pd.DataFrame) -> tuple[Word2Vec, dict]:
    """Compara ambas arquitecturas y devuelve la ganadora."""
    print("[4/7] Evaluacion CBOW vs Skip-gram")
    resultados = {}

    for nombre, modelo in [("cbow", cbow), ("skipgram", skipgram)]:
        print(f"\n  --- {nombre.upper()} ---")
        s_pares = similitud_pares(modelo)

        embeddings = np.array([
            embedding_producto(modelo,
                               row.get('texto_lematizado', '') or row.get('texto', '') or row['name'])
            for _, row in df_prod.iterrows()
        ])
        cats = df_prod['main_category'].values
        s_coher = coherencia_categoria(embeddings, cats)
        puntaje = 0.5 * s_pares + 0.5 * s_coher
        resultados[nombre] = {"pares": s_pares, "coherencia": s_coher, "puntaje": puntaje}
        print(f"    Coherencia de categoria (k=10): {s_coher:.1%}")
        print(f"    Puntaje combinado: {puntaje:.3f}")

    mejor_nombre = max(resultados, key=lambda n: resultados[n]["puntaje"])
    mejor = cbow if mejor_nombre == "cbow" else skipgram
    print(f"\n[5/7] Ganador: {mejor_nombre.upper()} "
          f"(CBOW {resultados['cbow']['puntaje']:.3f} vs "
          f"Skip-gram {resultados['skipgram']['puntaje']:.3f})")
    return mejor, resultados


def guardar_artefactos(mejor: Word2Vec, cbow: Word2Vec, skipgram: Word2Vec,
                       df_prod: pd.DataFrame, resultados: dict) -> np.ndarray:
    """Guarda ambos modelos, el ganador bajo el nombre canonico y embeddings."""
    print("[6/7] Guardando artefactos...")

    mejor.save(str(RUTA_MODELOS / "word2vec_productos.model"))
    cbow.save(str(RUTA_MODELOS / "word2vec_cbow.model"))
    skipgram.save(str(RUTA_MODELOS / "word2vec_skipgram.model"))
    print("  - Modelos: word2vec_productos.model (+ cbow / skipgram)")

    embeddings = np.array([
        embedding_producto(mejor,
                           row.get('texto_lematizado', '') or row.get('texto', '') or row['name'])
        for _, row in df_prod.iterrows()
    ])
    np.savez_compressed(
        RUTA_MODELOS / "embeddings_word2vec.npz",
        embeddings=embeddings,
        product_ids=df_prod['product_id'].values,
        categories=df_prod['main_category'].values,
    )
    print(f"  - Embeddings: embeddings_word2vec.npz shape={embeddings.shape}")

    import json
    (RUTA_MODELOS / "word2vec_comparacion.json").write_text(
        json.dumps({
            "ventana": VENTANA, "dimension": DIMENSION, "epocas": EPOCAS,
            "ganador": max(resultados, key=lambda n: resultados[n]["puntaje"]),
            **{k: {m: round(v, 4) for m, v in d.items()} for k, d in resultados.items()},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("  - Comparacion: word2vec_comparacion.json")
    return embeddings


def demostrar_similares(modelo: Word2Vec, df_prod: pd.DataFrame, embeddings: np.ndarray):
    """Muestra productos similares segun el modelo ganador."""
    print("[7/7] Demostracion de productos similares...")
    ejemplos = ["p0000", "p0100", "p0200"]
    for prod_id in ejemplos:
        match = df_prod.index[df_prod['product_id'] == prod_id]
        if len(match) == 0:
            continue
        idx = match[0]
        emb_query = normalize(embeddings[idx:idx+1])
        sims = (normalize(embeddings) @ emb_query.T).flatten()
        top = np.argsort(sims)[::-1][1:6]
        print(f"\n  Producto: {df_prod.iloc[idx]['name']}")
        for i in top:
            print(f"    - {df_prod.iloc[i]['name'][:55]} (sim={sims[i]:.3f})")


def main():
    print("=" * 60)
    print("WORD2VEC CON TOKENIZACION SPACY: CBOW vs SKIP-GRAM")
    print("=" * 60)
    inicio = time.time()

    df_prod, df_ecom = cargar_datos()

    print("      Cargando es_core_news_sm...")
    nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])

    corpus = preparar_corpus_spacy(df_prod, df_ecom, nlp)

    print("[3/7] Entrenando ambas arquitecturas...")
    cbow = entrenar_arquitectura(corpus, sg=0)
    skipgram = entrenar_arquitectura(corpus, sg=1)

    mejor, resultados = evaluar_y_comparar(cbow, skipgram, df_prod)
    embeddings = guardar_artefactos(mejor, cbow, skipgram, df_prod, resultados)
    demostrar_similares(mejor, df_prod, embeddings)

    print("\n" + "=" * 60)
    print(f"COMPLETADO en {time.time()-inicio:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
