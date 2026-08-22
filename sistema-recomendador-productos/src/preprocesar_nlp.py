import re
import joblib
import numpy as np
import pandas as pd
import spacy
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

RUTA_PRODUCTOS = Path("data/productos.csv")
RUTA_MODELOS = Path("models")
STOPWORDS_EXTRA = {"producto", "productos", "contiene", "presentado", "presenta", "ademas", "además", "sugiere", "sugiere", "tipo", "forma", "tamaño", "tamano"}


def demo_etapas(nlp) -> None:
    ejemplo = "Los Paquetes de Arroz Diana blanco x10kg están corriendo en promoción!!! Compra ya."
    print("=" * 60)
    print("DEMOSTRACION DE LAS 5 ETAPAS NLP")
    print("=" * 60)
    print(f"Texto original : {ejemplo}")

    limpio = limpiar_texto(ejemplo)
    print(f"1. Limpieza    : {limpio}")

    tokens = [t.text for t in nlp(limpio)]
    print(f"2. Tokenizacion: {tokens}")

    sin_stop = [t.text for t in nlp(limpio) if not t.is_stop and t.text not in STOPWORDS_EXTRA]
    print(f"3. Sin stopwords: {sin_stop}")

    lemas = [t.lemma_ for t in nlp(limpio) if not t.is_stop and t.text not in STOPWORDS_EXTRA]
    print(f"4. Lematizacion: {lemas}")
    print("=" * 60)


def limpiar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\.\S+", " ", texto)
    texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def procesar_corpus(textos: list[str], nlp) -> list[str]:
    resultados = []
    for doc in nlp.pipe((limpiar_texto(t) for t in textos), batch_size=128):
        lemas = [
            token.lemma_ for token in doc
            if not token.is_stop
            and token.lemma_ not in STOPWORDS_EXTRA
            and len(token.lemma_) > 2
        ]
        resultados.append(" ".join(lemas))
    return resultados


def main() -> None:
    RUTA_MODELOS.mkdir(exist_ok=True)

    productos = pd.read_csv(RUTA_PRODUCTOS)

    print("[1/5] Cargando modelo spaCy es_core_news_sm...")
    nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])

    demo_etapas(nlp)

    print("[2/5] Procesando 7449 descripciones (limpieza -> tokens -> stopwords -> lematizacion)...")
    textos_procesados = procesar_corpus(productos["texto"].tolist(), nlp)
    productos["texto_lematizado"] = textos_procesados
    productos.to_csv(RUTA_PRODUCTOS, index=False)

    print("[3/5] Vectorizando con TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    matriz_tfidf = vectorizer.fit_transform(productos["texto_lematizado"])
    print(f"      Matriz: {matriz_tfidf.shape[0]} productos x {matriz_tfidf.shape[1]} terminos")

    print("[4/5] Guardando artefactos...")
    joblib.dump(vectorizer, RUTA_MODELOS / "tfidf_vectorizer.pkl")
    sparse.save_npz(RUTA_MODELOS / "tfidf_matrix.npz", matriz_tfidf)

    print("[5/5] DEMO - Productos similares por contenido (cosine similarity):")
    idx = 0
    similitudes = cosine_similarity(matriz_tfidf[idx], matriz_tfidf).flatten()
    mas_similares = np.argsort(similitudes)[::-1][1:6]
    print(f"\nProducto base: {productos.loc[idx, 'name']}\n")
    for i in mas_similares:
        print(f"  {similitudes[i]:.3f} | {productos.loc[i, 'name']}")

    print("\nFase 2 completada.")


if __name__ == "__main__":
    main()
