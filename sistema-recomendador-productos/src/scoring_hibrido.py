"""Función de scoring híbrido mejorado con Word2Vec.

Combina scores de:
1. NCF (filtrado colaborativo)
2. Word2Vec (similitud semántica de productos)
3. TF-IDF (matching exacto de texto)
4. Predictor de categorías (reranking)
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


def calcular_scores_word2vec(
    usuario_idx: int,
    embeddings: np.ndarray,
    historial: list[int],
    top_k: int = 100,
) -> np.ndarray:
    """Calcula scores basados en similitud semántica con el historial."""
    n_productos = embeddings.shape[0]
    scores = np.zeros(n_productos)
    
    if not historial:
        return scores
    
    # Embedding promedio del historial del usuario
    emb_historial = embeddings[historial].mean(axis=0)
    
    # Similitud coseno con todos los productos
    sims = cosine_similarity(emb_historial.reshape(1, -1), embeddings).flatten()
    
    return sims


def calcular_scores_tfidf(
    usuario_idx: int,
    matriz_tfidf,
    historial: list[int],
) -> np.ndarray:
    """Calcula scores basados en similitud TF-IDF con el historial."""
    if not historial or matriz_tfidf is None:
        return np.zeros(matriz_tfidf.shape[0])
    
    # TF-IDF promedio del historial
    tfidf_historial = matriz_tfidf[historial].mean(axis=0)
    
    # Similitud con todos los productos
    sims = cosine_similarity(tfidf_historial, matriz_tfidf).flatten()
    
    return sims


def calcular_scores_hibridos(
    usuario_idx: int,
    art: dict,
    embeddings_w2v: np.ndarray = None,
    matriz_tfidf=None,
    historial: list[int] = None,
    pesos: dict = None,
    df_prod: pd.DataFrame = None,
    product_ids: np.ndarray = None,
    embeddings_sem: np.ndarray = None,
) -> np.ndarray:
    """Calcula scores híbridos combinando todos los modelos.
    
    Args:
        usuario_idx: Índice del usuario
        art: Artefactos del modelo (scores NCF, categorías, etc.)
        embeddings_w2v: Embeddings Word2Vec de productos
        matriz_tfidf: Matriz TF-IDF de productos
        historial: Lista de índices de productos comprados
        pesos: Pesos para cada componente {word2vec, afinidad}
        df_prod: DataFrame de productos
        product_ids: Array de product_id en el mismo orden que embeddings_w2v
    
    Returns:
        Array de scores para cada producto
    """
    # Use embeddings size as the number of products
    if embeddings_w2v is not None:
        n_productos = embeddings_w2v.shape[0]
    else:
        n_productos = len(art["idx_producto"])

    if pesos is None:
        pesos = {
            "semantico": 0.45,
            "word2vec": 0.15,
            "afinidad": 0.40,
        }

    # 1. Score Transformer (similitud semantica profunda con historial)
    scores_sem = np.zeros(n_productos)
    if embeddings_sem is not None and historial:
        historial_valido = [h for h in historial if h < embeddings_sem.shape[0]]
        if historial_valido:
            emb_hist = embeddings_sem[historial_valido].mean(axis=0)
            emb_norm = emb_hist / max(np.linalg.norm(emb_hist), 1e-9)
            scores_sem = embeddings_sem @ emb_norm

    # 1b. Score Word2Vec (similitud lexica con historial)
    scores_w2v = np.zeros(n_productos)
    if embeddings_w2v is not None and historial:
        # Filtrar historial válido
        historial_valido = [h for h in historial if h < embeddings_w2v.shape[0]]
        if historial_valido:
            scores_w2v = calcular_scores_word2vec(usuario_idx, embeddings_w2v, historial_valido)
    
    # 2. Score de afinidad por categoría
    scores_afinidad = np.zeros(n_productos)
    p_cat = art["p_categoria_usuario"]
    lista_cats = art["lista_cats"]
    scores_cat = p_cat[usuario_idx]
    
    # Asignar score de categoría a cada producto
    if df_prod is not None and product_ids is not None:
        for i in range(min(n_productos, len(product_ids))):
            pid = product_ids[i]
            if pid in df_prod['product_id'].values:
                cat = df_prod[df_prod['product_id'] == pid]['main_category'].values[0]
                if cat in lista_cats:
                    idx_cat = lista_cats.index(cat)
                    scores_afinidad[i] = scores_cat[idx_cat]
    
    # Combinar scores
    scores_finales = (
        pesos["semantico"] * scores_sem +
        pesos["word2vec"] * scores_w2v +
        pesos["afinidad"] * scores_afinidad
    )

    return scores_finales


def normalizar(scores: np.ndarray) -> np.ndarray:
    """Normaliza scores al rango [0, 1]."""
    min_val = scores.min()
    max_val = scores.max()
    if max_val - min_val < 1e-10:
        return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)


def recomendar_usuario(
    usuario_id: str,
    art: dict,
    embeddings_w2v: np.ndarray = None,
    matriz_tfidf=None,
    df_prod=None,
    top_k: int = 10,
    excluir_comprados: bool = True,
    product_ids: np.ndarray = None,
    embeddings_sem: np.ndarray = None,
) -> list[dict]:
    """Genera recomendaciones para un usuario específico.
    
    Args:
        usuario_id: ID del usuario (ej: "u0007")
        art: Artefactos del modelo
        embeddings_w2v: Embeddings Word2Vec
        matriz_tfidf: Matriz TF-IDF
        df_prod: DataFrame de productos
        top_k: Número de recomendaciones
        excluir_comprados: Si True, excluye productos ya comprados
        product_ids: Array de product_id en el mismo orden que embeddings_w2v
    
    Returns:
        Lista de dicts con recomendaciones
    """
    # Obtener índice del usuario
    if usuario_id not in art["idx_usuario"]:
        return []
    
    usuario_idx = art["idx_usuario"][usuario_id]
    
    # Obtener historial del usuario desde interacciones
    import pandas as pd
    RUTA_BASE = Path(__file__).resolve().parent.parent
    df_interacciones = pd.read_csv(RUTA_BASE / "data" / "interacciones.csv")
    hist_usuario = df_interacciones[df_interacciones['user_id'] == usuario_id]
    productos_comprados = hist_usuario['product_id'].unique()
    
    # Convertir a índices de embeddings
    historial = []
    if product_ids is not None:
        for pid in productos_comprados:
            idx = np.where(product_ids == pid)[0]
            if len(idx) > 0:
                historial.append(idx[0])
    
    # Calcular scores híbridos
    scores = calcular_scores_hibridos(
        usuario_idx, art, embeddings_w2v, matriz_tfidf, historial,
        df_prod=df_prod, product_ids=product_ids, embeddings_sem=embeddings_sem,
    )
    
    # Excluir productos ya comprados
    if excluir_comprados and product_ids is not None:
        for pid in productos_comprados:
            idx = np.where(product_ids == pid)[0]
            if len(idx) > 0:
                scores[idx[0]] = -np.inf
    
    # Obtener top-K
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    # Formatear resultado
    recomendaciones = []
    for idx in top_indices:
        if scores[idx] == -np.inf:
            continue
        
        rec = {
            "producto_idx": int(idx),
            "score": float(scores[idx]),
        }
        
        # Obtener product_id del array de embeddings
        if product_ids is not None and idx < len(product_ids):
            rec["producto_id"] = product_ids[idx]
        
        if df_prod is not None and "producto_id" in rec:
            pid = rec["producto_id"]
            if pid in df_prod['product_id'].values:
                row = df_prod[df_prod['product_id'] == pid].iloc[0]
                rec["nombre"] = row["name"]
                rec["categoria"] = row["main_category"]
        
        recomendaciones.append(rec)
    
    return recomendaciones


def demostrar_recomendaciones():
    """Demuestra el sistema de recomendación mejorado."""
    import joblib
    from pathlib import Path
    from gensim.models import Word2Vec
    import pandas as pd
    
    RUTA_BASE = Path(__file__).resolve().parent.parent
    RUTA_DATOS = RUTA_BASE / "data"
    RUTA_MODELOS = RUTA_BASE / "models"
    
    print("=" * 60)
    print("DEMO: SISTEMA DE RECOMENDACIÓN MEJORADO")
    print("=" * 60)
    
    # Cargar artefactos
    print("\n[1/3] Cargando modelos...")
    art = joblib.load(RUTA_MODELOS / "hibrido_v2.pkl")
    w2v = Word2Vec.load(str(RUTA_MODELOS / "word2vec_productos.model"))
    data = np.load(RUTA_MODELOS / "embeddings_word2vec.npz", allow_pickle=True)
    embeddings_w2v = data['embeddings']
    df_prod = pd.read_csv(RUTA_DATOS / "productos.csv")

    # Embeddings semanticos (Transformer 384D) - Nivel 2
    embeddings_sem = None
    ruta_sem = RUTA_MODELOS / "embeddings_semanticos.npz"
    if ruta_sem.exists():
        sem_data = np.load(ruta_sem, allow_pickle=True)
        embeddings_sem = sem_data["embeddings"]
        print(f"  - Transformer semantico: {embeddings_sem.shape}")
    else:
        print("  - AVISO: ejecuta src/embeddings_semanticos.py")
    
    # Cargar interacciones para obtener historial
    df_interacciones = pd.read_csv(RUTA_DATOS / "interacciones.csv")
    
    print(f"  - Modelos cargados")
    
    # Demostrar para usuarios de ejemplo
    usuarios_demo = ["u0007", "u0100", "u0250"]
    
    print("\n[2/3] Generando recomendaciones...")
    for uid in usuarios_demo:
        if uid not in art["idx_usuario"]:
            continue
        
        print(f"\n{'='*60}")
        print(f"USUARIO: {uid}")
        print(f"{'='*60}")
        
        # Obtener historial del usuario desde interacciones
        hist_usuario = df_interacciones[df_interacciones['user_id'] == uid]
        productos_comprados = hist_usuario['product_id'].unique()
        
        # Convertir a índices
        historial = []
        for pid in productos_comprados:
            if pid in art["idx_producto"]:
                historial.append(art["idx_producto"][pid])
        
        print(f"\nHistorial ({len(historial)} productos):")
        for idx in historial[:5]:
            if idx < len(df_prod):
                print(f"  - {df_prod.iloc[idx]['name'][:50]}")
        if len(historial) > 5:
            print(f"  ... y {len(historial)-5} más")
        
        # Categorías preferidas
        p_cat = art["p_categoria_usuario"]
        lista_cats = art["lista_cats"]
        top_cats_idx = np.argsort(p_cat[art["idx_usuario"][uid]])[::-1][:3]
        print(f"\nCategorías preferidas:")
        for j in top_cats_idx:
            print(f"  - {lista_cats[j]}: {p_cat[art['idx_usuario'][uid], j]:.1%}")
        
        # Recomendaciones
        recs = recomendar_usuario(
            uid, art, embeddings_w2v, None, df_prod, top_k=10,
            product_ids=data['product_ids'], embeddings_sem=embeddings_sem,
        )
        
        print(f"\nTop-10 Recomendaciones:")
        for i, rec in enumerate(recs):
            print(f"  {i+1}. {rec['nombre'][:45]} ({rec['categoria']})")
            print(f"     Score: {rec['score']:.3f}")
    
    print("\n" + "=" * 60)
    print("FIN DE LA DEMOSTRACIÓN")
    print("=" * 60)


if __name__ == "__main__":
    demostrar_recomendaciones()
