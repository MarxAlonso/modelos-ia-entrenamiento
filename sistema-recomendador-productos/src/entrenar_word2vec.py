"""Pipeline de entrenamiento Word2Vec para embeddings de productos.

Entrena un modelo Word2Vec sobre corpus de productos en español
para generar embeddings semánticos de productos.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

warnings.filterwarnings('ignore')

RUTA_BASE = Path(__file__).resolve().parent.parent
RUTA_DATOS = RUTA_BASE / "data"
RUTA_MODELOS = RUTA_BASE / "models"


def limpiar_texto(texto: str) -> str:
    """Limpia y normaliza texto en español."""
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'[^\w\sáéíóúñü]', ' ', texto)
    texto = re.sub(r'\d+', ' NUM ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga datasets existentes y nuevos."""
    print("[1/6] Cargando datasets...")
    
    # Dataset existente (supermercado)
    df_prod = pd.read_csv(RUTA_DATOS / "productos.csv")
    print(f"  - Productos supermercado: {len(df_prod)}")
    
    # Dataset nuevo (e-commerce)
    df_ecom = pd.read_parquet(RUTA_DATOS / "titulos-ecommerce-es/data/train-00000-of-00001.parquet")
    df_ecom_test = pd.read_parquet(RUTA_DATOS / "titulos-ecommerce-es/data/test-00000-of-00001.parquet")
    df_ecom = pd.concat([df_ecom, df_ecom_test], ignore_index=True)
    print(f"  - Productos e-commerce: {len(df_ecom)}")
    
    return df_prod, df_ecom


def preparar_corpus(df_prod: pd.DataFrame, df_ecom: pd.DataFrame) -> list[list[str]]:
    """Prepara corpus tokenizado para Word2Vec."""
    print("[2/6] Preparando corpus...")
    
    corpus = []
    
    # Productos supermercado (usar texto lematizado si existe)
    for _, row in df_prod.iterrows():
        texto = row.get('texto_lematizado', '') or row.get('texto', '') or row['name']
        tokens = limpiar_texto(texto).split()
        if len(tokens) >= 2:
            corpus.append(tokens)
    
    # Productos e-commerce
    for _, row in df_ecom.iterrows():
        tokens = limpiar_texto(str(row['title'])).split()
        if len(tokens) >= 2:
            corpus.append(tokens)
    
    print(f"  - Total documentos: {len(corpus)}")
    print(f"  - Tokens promedio: {np.mean([len(doc) for doc in corpus]):.1f}")
    
    return corpus


def entrenar_word2vec(corpus: list[list[str]], dim: int = 100) -> Word2Vec:
    """Entrena modelo Word2Vec."""
    print(f"[3/6] Entrenando Word2Vec (dim={dim})...")
    inicio = time.time()
    
    modelo = Word2Vec(
        sentences=corpus,
        vector_size=dim,
        window=5,
        min_count=2,
        workers=4,
        epochs=30,
        sg=1,  # Skip-gram (mejor para palabras raras)
        seed=42,
    )
    
    print(f"  - Vocabulario: {len(modelo.wv):,} palabras")
    print(f"  - Tiempo: {time.time() - inicio:.1f}s")
    
    return modelo


def generar_embedding_producto(modelo: Word2Vec, texto: str) -> np.ndarray:
    """Genera embedding promedio de un producto."""
    tokens = limpiar_texto(texto).split()
    vectores = []
    for token in tokens:
        if token in modelo.wv:
            vectores.append(modelo.wv[token])
    if vectores:
        return np.mean(vectores, axis=0)
    return np.zeros(modelo.vector_size)


def evaluar_similitud(modelo: Word2Vec):
    """Evalúa calidad de embeddings con analogías."""
    print("[4/6] Evaluando embeddings...")
    
    # Test de analogías simples
    test_pairs = [
        ("arroz", "frijol"),      # ambos granos
        ("leche", "yogur"),       # lácteos
        ("cerveza", "vino"),      # bebidas alcohólicas
        ("pan", "torta"),         # panadería
    ]
    
    for w1, w2 in test_pairs:
        if w1 in modelo.wv and w2 in modelo.wv:
            sim = modelo.wv.similarity(w1, w2)
            print(f"  - {w1} <-> {w2}: {sim:.3f}")
        else:
            print(f"  - {w1} <-> {w2}: [fuera de vocabulario]")


def guardar_artefactos(modelo: Word2Vec, df_prod: pd.DataFrame, df_ecom: pd.DataFrame):
    """Guarda modelo y embeddings."""
    print("[5/6] Guardando artefactos...")
    
    # Guardar modelo Word2Vec
    ruta_modelo = RUTA_MODELOS / "word2vec_productos.model"
    modelo.save(str(ruta_modelo))
    print(f"  - Modelo guardado: {ruta_modelo}")
    
    # Generar embeddings para productos existentes
    embeddings_prod = []
    for _, row in df_prod.iterrows():
        texto = row.get('texto_lematizado', '') or row.get('texto', '') or row['name']
        emb = generar_embedding_producto(modelo, texto)
        embeddings_prod.append(emb)
    
    embeddings_prod = np.array(embeddings_prod)
    
    # Guardar embeddings
    ruta_embeddings = RUTA_MODELOS / "embeddings_word2vec.npz"
    np.savez_compressed(
        ruta_embeddings,
        embeddings=embeddings_prod,
        product_ids=df_prod['product_id'].values,
        categories=df_prod['main_category'].values,
    )
    print(f"  - Embeddings guardados: {ruta_embeddings}")
    print(f"  - Shape: {embeddings_prod.shape}")
    
    return embeddings_prod


def demostrar_similares(modelo: Word2Vec, df_prod: pd.DataFrame, embeddings: np.ndarray):
    """Muestra productos similares como demostración."""
    print("[6/6] Demostración de productos similares...")
    
    # Seleccionar productos de ejemplo
    ejemplos = [
        ("p0000", "Arroz Diana blanco x10kg"),
        ("p0100", "Leche evaporada"),
        ("p0200", "Cerveza"),
    ]
    
    for prod_id, titulo in ejemplos:
        if prod_id not in df_prod['product_id'].values:
            continue
            
        idx = df_prod[df_prod['product_id'] == prod_id].index[0]
        emb_query = embeddings[idx:idx+1]
        
        # Calcular similitud coseno
        similitudes = np.dot(embeddings, emb_query.T).flatten()
        top_indices = np.argsort(similitudes)[::-1][1:6]  # Top 5 (excluyendo a sí mismo)
        
        print(f"\n  Producto: {titulo}")
        print("  Similares:")
        for i in top_indices:
            prod_sim = df_prod.iloc[i]
            sim = similitudes[i]
            print(f"    - {prod_sim['name'][:50]} (sim={sim:.3f})")


def main():
    print("=" * 60)
    print("ENTRENAMIENTO WORD2VEC PARA PRODUCTOS")
    print("=" * 60)
    
    inicio = time.time()
    
    # Cargar datos
    df_prod, df_ecom = cargar_datos()
    
    # Preparar corpus
    corpus = preparar_corpus(df_prod, df_ecom)
    
    # Entrenar Word2Vec
    modelo = entrenar_word2vec(corpus, dim=100)
    
    # Evaluar
    evaluar_similitud(modelo)
    
    # Guardar
    embeddings = guardar_artefactos(modelo, df_prod, df_ecom)
    
    # Demostración
    demostrar_similares(modelo, df_prod, embeddings)
    
    print("\n" + "=" * 60)
    print(f"COMPLETADO en {time.time() - inicio:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
