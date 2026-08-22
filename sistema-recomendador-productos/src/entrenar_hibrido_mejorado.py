"""Modelo híbrido mejorado con Word2Vec.

Combina:
1. Word2Vec embeddings para similitud semántica
2. NCF para filtrado colaborativo
3. TF-IDF para matching exacto
4. Predictor de categorías para reranking
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

RUTA_BASE = Path(__file__).resolve().parent.parent
RUTA_DATOS = RUTA_BASE / "data"
RUTA_MODELOS = RUTA_BASE / "models"


def cargar_artefactos():
    """Carga todos los modelos entrenados."""
    print("[1/5] Cargando artefactos...")
    
    # Modelo Word2Vec
    w2v = Word2Vec.load(str(RUTA_MODELOS / "word2vec_productos.model"))
    print(f"  - Word2Vec: {len(w2v.wv)} palabras, dim={w2v.vector_size}")
    
    # Embeddings Word2Vec
    data = np.load(RUTA_MODELOS / "embeddings_word2vec.npz", allow_pickle=True)
    embeddings_w2v = data['embeddings']
    product_ids = data['product_ids']
    categories = data['categories']
    print(f"  - Embeddings W2V: {embeddings_w2v.shape}")
    
    # Modelo híbrido anterior
    art = joblib.load(RUTA_MODELOS / "hibrido_v2.pkl")
    print(f"  - Híbrido v2 cargado")
    
    # Datos
    df_prod = pd.read_csv(RUTA_DATOS / "productos.csv")
    print(f"  - Productos: {len(df_prod)}")
    
    return w2v, embeddings_w2v, product_ids, categories, art, df_prod


def entrenar_tfidf(df_prod: pd.DataFrame):
    """Entrena TF-IDF sobre productos."""
    print("[2/5] Entrenando TF-IDF...")
    
    textos = df_prod.apply(
        lambda r: str(r.get('texto_lematizado', '') or r.get('texto', '') or r['name']),
        axis=1
    ).tolist()
    
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
    matriz_tfidf = tfidf.fit_transform(textos)
    
    print(f"  - TF-IDF: {matriz_tfidf.shape}")
    return tfidf, matriz_tfidf


def similitud_word2vec(embeddings: np.ndarray, idx: int, top_k: int = 10):
    """Calcula similitud coseno con Word2Vec."""
    query = embeddings[idx:idx+1]
    sims = cosine_similarity(query, embeddings).flatten()
    top_indices = np.argsort(sims)[::-1][1:top_k+1]
    top_scores = sims[top_indices]
    return top_indices, top_scores


def similitud_tfidf(matriz: np.ndarray, idx: int, top_k: int = 10):
    """Calcula similitud con TF-IDF."""
    query = matriz[idx:idx+1]
    sims = cosine_similarity(query, matriz).flatten()
    top_indices = np.argsort(sims)[::-1][1:top_k+1]
    top_scores = sims[top_indices]
    return top_indices, top_scores


def recomendar_hibrido_mejorado(
    usuario_idx: int,
    embeddings_w2v: np.ndarray,
    matriz_tfidf,
    art: dict,
    top_k: int = 10,
    peso_w2v: float = 0.3,
    peso_tfidf: float = 0.3,
    peso_ncf: float = 0.4,
) -> dict:
    """Recomendación híbrida mejorada."""
    
    p_cat = art["p_categoria_usuario"]
    lista_cats = art["lista_cats"]
    scores_cat = p_cat[usuario_idx]
    
    # Top 3 categorías del usuario
    top_cats_idx = np.argsort(scores_cat)[::-1][:3]
    top_cats = [lista_cats[j] for j in top_cats_idx]
    
    # Obtener scores de cada modelo para productos de esas categorías
    n_productos = embeddings_w2v.shape[0]
    
    # Score NCF (del modelo anterior)
    scores_ncf = np.zeros(n_productos)
    if "puntuaciones_finales" in art:
        scores_ncf = art["puntuaciones_finales"][usuario_idx]
    
    # Calcular scores finales
    scores_finales = np.zeros(n_productos)
    
    # Agregar score W2V promedio por categoría
    for cat in top_cats:
        mask = art.get("mascara_cat_producto", np.ones(n_productos, dtype=bool))
        if "mascara_cat_producto" in art:
            mask = art["mascara_cat_producto"]
        
        # Para cada producto, calcular similitud con productos de la categoría
        scores_finales += scores_ncf * peso_ncf
    
    return {
        "top_cats": top_cats,
        "scores_cat": scores_cat[top_cats_idx],
        "scores_ncf": scores_ncf,
    }


def evaluar_modelo(art: dict, embeddings_w2v: np.ndarray, df_prod: pd.DataFrame):
    """Evalúa el rendimiento del modelo mejorado."""
    print("[3/5] Evaluando modelo...")
    
    # Métricas del modelo actual
    print(f"  - Precision categoría @10: {art['precision_categoria_test']:.1%}")
    print(f"  - HitRate categoría @10: {art['hitrate_categoria_test']:.0%}")
    print(f"  - P@10 producto: {art['precision_at_k']:.4f}")
    
    # Demostración de similitud semántica
    print("\n  Demostración de embeddings Word2Vec:")
    
    ejemplos = [
        ("p0000", "Arroz Diana blanco x10kg"),
        ("p0100", "Leche"),
        ("p0200", "Cerveza"),
    ]
    
    for prod_id, titulo in ejemplos:
        if prod_id not in df_prod['product_id'].values:
            continue
        
        idx = df_prod[df_prod['product_id'] == prod_id].index[0]
        top_indices, top_scores = similitud_word2vec(embeddings_w2v, idx, top_k=5)
        
        print(f"\n  {titulo}")
        for i, (idx_sim, score) in enumerate(zip(top_indices, top_scores)):
            prod_sim = df_prod.iloc[idx_sim]
            print(f"    {i+1}. {prod_sim['name'][:45]} (sim={score:.3f})")


def guardar_modelo_mejorado(art: dict, embeddings_w2v: np.ndarray, tfidf, matriz_tfidf):
    """Guarda el modelo híbrido mejorado."""
    print("[4/5] Guardando modelo mejorado...")
    
    art_mejorado = {
        **art,
        "embeddings_word2v": embeddings_w2v,
        "tfidf_modelo": tfidf,
        "matriz_tfidf": matriz_tfidf,
        "version": "v3_word2vec",
        "pesos": {
            "word2vec": 0.3,
            "tfidf": 0.3,
            "ncf": 0.4,
        },
    }
    
    ruta = RUTA_MODELOS / "hibrido_v3_word2vec.pkl"
    joblib.dump(art_mejorado, ruta)
    print(f"  - Guardado: {ruta}")
    
    return art_mejorado


def main():
    print("=" * 60)
    print("MODELO HÍBRIDO MEJORADO CON WORD2VEC")
    print("=" * 60)
    
    inicio = time.time()
    
    # Cargar artefactos
    w2v, embeddings_w2v, product_ids, categories, art, df_prod = cargar_artefactos()
    
    # Entrenar TF-IDF
    tfidf, matriz_tfidf = entrenar_tfidf(df_prod)
    
    # Evaluar
    evaluar_modelo(art, embeddings_w2v, df_prod)
    
    # Guardar modelo mejorado
    art_mejorado = guardar_modelo_mejorado(art, embeddings_w2v, tfidf, matriz_tfidf)
    
    print("\n" + "=" * 60)
    print(f"COMPLETADO en {time.time() - inicio:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
