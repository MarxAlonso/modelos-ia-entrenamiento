
import sys
from pathlib import Path
_RUTA_MODULO = Path(__file__).resolve()
while _RUTA_MODULO.name != "src" and _RUTA_MODULO != _RUTA_MODULO.parent:
    _RUTA_MODULO = _RUTA_MODULO.parent
sys.path.insert(0, str(_RUTA_MODULO.parent))

from src.rutas import RUTA_RAIZ
import re
import joblib
import numpy as np
import pandas as pd
import spacy
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

RUTA_RESENAS = RUTA_RAIZ / "data/amazon/resenas_traducidas.csv"
RUTA_MODELOS = RUTA_RAIZ / "models"

LEXICON_POSITIVO = {
    "excelente", "perfecto", "encantar", "maravilloso", "gran", "bueno", "buena",
    "bonito", "bonita", "hermoso", "cómodo", "comodo", "genial", "recomendar",
    "feliz", "gustar", "calidad", "bonito", "súper", "super", "bien", "mejor",
    "agradable", "suave", "durable", "bonito",
}
LEXICON_NEGATIVO = {
    "malo", "mala", "terrible", "horrible", "barato", "peor", "problema",
    "romperse", "roto", "rota", "decepcionar", "decepcionado", "decepcionada",
    "devolver", "devolución", "devolucion", "desastre", "feo", "fea",
    "incómodo", "incomodo", "pequeño", "pequeno", "caro", "difícil", "dificil",
    "arrepentirse", "basura", "falso", "falsa",
}


def limpiar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\.\S+", " ", texto)
    texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def demo_etapas(nlp, ejemplo: str) -> None:
    print("=" * 60)
    print("5 ETAPAS NLP SOBRE RESENAS EN ESPANOL")
    print("=" * 60)
    print(f"Texto original : {ejemplo[:110]}")
    limpio = limpiar_texto(ejemplo)
    print(f"1. Limpieza    : {limpio[:110]}")
    doc = nlp(limpio)
    tokens = [t.text for t in doc]
    print(f"2. Tokenizacion: {tokens[:14]}")
    sin_stop = [t.text for t in doc if not t.is_stop and len(t.text) > 2]
    print(f"3. Sin stopwords: {sin_stop[:12]}")
    lemas = [t.lemma_ for t in doc if not t.is_stop and len(t.lemma_) > 2]
    print(f"4. Lematizacion: {lemas[:12]}")
    print("5. Vectorizacion -> TF-IDF sobre todo el corpus")


def procesar(textos: list[str], nlp) -> list[str]:
    salida = []
    for doc in nlp.pipe((limpiar_texto(t) for t in textos), batch_size=128):
        lemas = [
            token.lemma_ for token in doc
            if not token.is_stop and len(token.lemma_) > 2
        ]
        salida.append(" ".join(lemas))
    return salida


def sentimiento(lemas_texto: str) -> float:
    palabras = set(lemas_texto.split())
    positivos = len(palabras & LEXICON_POSITIVO)
    negativos = len(palabras & LEXICON_NEGATIVO)
    if positivos + negativos == 0:
        return 0.0
    return (positivos - negativos) / (positivos + negativos)


def main() -> None:
    RUTA_MODELOS.mkdir(exist_ok=True)

    resenas = pd.read_csv(RUTA_RESENAS)
    print(f"[1/6] Cargando {len(resenas)} resenas traducidas al espanol")

    nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])

    ejemplo = resenas.loc[3, "texto_es"]
    demo_etapas(nlp, str(ejemplo))

    print("[2/6] Procesando corpus...")
    resenas["texto_lematizado"] = procesar(resenas["texto_es"].astype(str).tolist(), nlp)

    print("[3/6] Vectorizando TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
    matriz = vectorizer.fit_transform(resenas["texto_lematizado"])
    print(f"      Matriz: {matriz.shape[0]} resenas x {matriz.shape[1]} terminos")

    print("[4/6] Analisis de sentimiento por lexico...")
    resenas["sentimiento"] = resenas["texto_lematizado"].map(sentimiento)
    con_senal = resenas[resenas["sentimiento"] != 0.0]
    r, p = pearsonr(con_senal["sentimiento"], con_senal["rating"])
    print(f"      Correlacion sentimiento-rating: r={r:.3f} (p={p:.2e})")
    print(f"      Resenas con senal lexica: {len(con_senal)}/{len(resenas)}")

    print("[5/6] Vocabulario discriminativo por rating:")
    bajas = " ".join(
        resenas.loc[resenas["rating"] <= 2, "texto_lematizado"]
    ).split()
    altas = " ".join(
        resenas.loc[resenas["rating"] >= 4, "texto_lematizado"]
    ).split()
    vocabulario = set(vectorizer.get_feature_names_out())
    conteo_bajas = pd.Series([w for w in bajas if w in vocabulario]).value_counts()
    conteo_altas = pd.Series([w for w in altas if w in vocabulario]).value_counts()
    print("      Top terminos en resenas NEGATIVAS (1-2 estrellas):")
    for palabra, n in conteo_bajas.head(8).items():
        print(f"         {palabra}: {n}")
    print("      Top terminos en resenas POSITIVAS (4-5 estrellas):")
    for palabra, n in conteo_altas.head(8).items():
        print(f"         {palabra}: {n}")

    print("[6/6] Guardando artefactos...")
    joblib.dump(vectorizer, RUTA_MODELOS / "tfidf_resenas.pkl")
    sparse.save_npz(RUTA_MODELOS / "tfidf_resenas_matrix.npz", matriz)
    resenas.to_csv(RUTA_RESENAS, index=False)

    print("\nEjemplo de pipeline completo:")
    i = 0
    print(f"  Original ES : {resenas.loc[i, 'texto_es'][:100]}")
    print(f"  Lematizado  : {resenas.loc[i, 'texto_lematizado'][:100]}")
    print(f"  Rating={resenas.loc[i, 'rating']} | Sentimiento={resenas.loc[i, 'sentimiento']:.2f}")
    print("\nFase NLP-resenas completada.")


if __name__ == "__main__":
    main()
