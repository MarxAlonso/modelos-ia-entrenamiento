"""Nivel 2: Embeddings semanticos con Transformer (Sentence-BERT).

Codifica titulos+descripciones de productos con un Transformer multilingue
(384D) y los compara contra TF-IDF y Word2Vec mediante coherence de
vecinos por categoria y busqueda semantica.
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

warnings.filterwarnings("ignore")

RUTA_BASE = RUTA_RAIZ
RUTA_DATOS = RUTA_BASE / "data"
RUTA_MODELOS = RUTA_BASE / "models"

MODELO_ST = "paraphrase-multilingual-MiniLM-L12-v2"  # 384D multilingue


def cargar_textos() -> pd.DataFrame:
    df = pd.read_csv(RUTA_DATOS / "productos.csv")
    textos = []
    for _, r in df.iterrows():
        desc = str(r.get("texto", "") or "")[:300]
        lema = str(r.get("texto_lematizado", "") or "")
        texto = f"{r['name']}. {desc}"
        if len(texto) < 60 and lema:
            texto = f"{r['name']}. {lema[:300]}"
        textos.append(texto.strip())
    df["texto_semantico"] = textos
    return df


def codificar_transformer(textos: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    print(f"[1/4] Cargando Transformer '{MODELO_ST}'...")
    modelo = SentenceTransformer(MODELO_ST)
    print(f"[2/4] Codificando {len(textos)} productos (batch, CPU)...")
    inicio = time.time()
    emb = modelo.encode(
        textos, batch_size=64, show_progress_bar=False,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)
    print(f"  - Shape: {emb.shape} | {time.time()-inicio:.1f}s")
    return emb


def vecinos_misma_categoria(emb: np.ndarray, cats: list[str], k: int = 10,
                            muestra: int = 800, seed: int = 42) -> float:
    """Precision de categoria de los k-NN (coherencia semantica)."""
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(emb), size=min(muestra, len(emb)), replace=False)
    emb_n = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    sub = emb_n[idxs]
    sims = sub @ emb_n.T
    np.fill_diagonal(sims[:, idxs], -np.inf)  # no excluirse mal: solo diag real
    aciertos = 0
    total = 0
    for fila, qi in zip(sims, idxs):
        fila[qi] = -np.inf
        vecinos = np.argpartition(fila, -k)[-k:]
        aciertos += sum(cats[v] == cats[qi] for v in vecinos)
        total += k
    return aciertos / total


def demo_busqueda(emb: np.ndarray, df: pd.DataFrame, consultas: list[str]):
    """Busqueda semantica: consulta en lenguaje natural -> productos."""
    from sentence_transformers import SentenceTransformer
    modelo = SentenceTransformer(MODELO_ST)
    q = modelo.encode(consultas, normalize_embeddings=True)
    sims = q @ emb.T
    print("[4/4] Busqueda semantica (consulta libre -> productos):")
    for ci, consulta in enumerate(consultas):
        top = np.argsort(sims[ci])[::-1][:3]
        print(f"\n  '{consulta}'")
        for i in top:
            print(f"    - {df.iloc[i]['name'][:55].strip()} ({df.iloc[i]['main_category']}, {sims[ci][i]:.3f})")


def main():
    print("=" * 60)
    print("EMBEDINGS SEMANTICOS - TRANSFORMER (Sentence-BERT)")
    print("=" * 60)
    t0 = time.time()

    df = cargar_textos()
    print(f"Productos: {len(df)}")

    # Embeddings Transformer
    emb_st = codificar_transformer(df["texto_semantico"].tolist())
    cats = df["main_category"].tolist()

    # Guardar
    np.savez_compressed(
        RUTA_MODELOS / "embeddings_semanticos.npz",
        embeddings=emb_st,
        product_ids=df["product_id"].values,
        categories=np.array(cats),
        model=MODELO_ST,
    )
    print(f"Guardado: models/embeddings_semanticos.npz")

    # Benchmark comparativo
    print("\n[3/4] Benchmark coherencia de vecinos (k=10, misma categoria):")

    res_st = vecinos_misma_categoria(emb_st, cats)
    print(f"  - Transformer ({MODELO_ST.split('-')[-1]}): {res_st:.1%}")

    try:
        w2v = np.load(RUTA_MODELOS / "embeddings_word2vec.npz", allow_pickle=True)["embeddings"]
        res_w2v = vecinos_misma_categoria(w2v, cats)
        print(f"  - Word2Vec (100D):                {res_w2v:.1%}")
    except FileNotFoundError:
        print("  - Word2Vec: [no encontrado]")

    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
    m_tfidf = tfidf.fit_transform(df["texto_semantico"]).toarray().astype(np.float32)
    res_tf = vecinos_misma_categoria(m_tfidf, cats)
    print(f"  - TF-IDF (10K dims):              {res_tf:.1%}")

    # Demo
    demo_busqueda(emb_st, df, [
        "algo dulce para el postre",
        "bebida alcoholica para una fiesta",
        "carne para asar al parrilla",
        "producto saludable ligero",
    ])

    print("\n" + "=" * 60)
    print(f"COMPLETADO en {time.time()-t0:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
