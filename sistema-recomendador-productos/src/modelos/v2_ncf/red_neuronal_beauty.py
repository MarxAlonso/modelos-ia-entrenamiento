"""
Fase CRISP-DM 4 - Redes Neuronales con TensorFlow (Beauty)
Material Patricio Peralta p.1: Fundamentos TensorFlow + NumPy
Adaptación de src/modelos/v2_ncf/red_neuronal.py para domain Beauty (ratings-only, sin texto español)
Arquitectura: Embedding(32) + GMF + MLP + sigmoid + BCE + negativos frescos
Integra todo lo construido: supermercado + Fashion + Beauty en pipeline unificado
"""
import sys
from pathlib import Path
_RUTA_MODULO = Path(__file__).resolve()
while _RUTA_MODULO.name != "src" and _RUTA_MODULO != _RUTA_MODULO.parent:
    _RUTA_MODULO = _RUTA_MODULO.parent
sys.path.insert(0, str(_RUTA_MODULO.parent))

from src.rutas import RUTA_RAIZ
import os
import random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import sparse

from src.modelos.v1_colaborativo_svd_knn.colaborativo import construir_indices, evaluar_ranking

SEED = 42
DIM_EMBEDDING = 32
N_NEGATIVOS = 4
EPOCHS = 8
BATCH = 1024
TOP_K = 10
UMBRAL_RELEVANTE = 4.0
# Para CPU, muestrear núcleo para entrenamiento manejable (CRISP-DM: Preparación)
MAX_INTERACCIONES_TRAIN = 40000  # muestra determinística si núcleo > 40k -> evita 140k*59k OOM
MAX_PRODUCTOS_EVAL = 5000  # ranking sobre top populares para evitar 8B matriz

RUTA_DATOS = RUTA_RAIZ / "data/amazon_beauty"
RUTA_MODELOS = RUTA_RAIZ / "models"

def fijar_semillas():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

def construir_modelo(n_usuarios: int, n_productos: int) -> tf.keras.Model:
    # Fundamentos p.1: Importación TensorFlow - framework para IA/ML/DL
    entrada_usuario = tf.keras.Input(shape=(1,), name="usuario", dtype=tf.int32)
    entrada_producto = tf.keras.Input(shape=(1,), name="producto", dtype=tf.int32)
    # Sin contenido español para Beauty (ratings-only) - simplificado pero mismo patrón NCF
    emb_usuario = tf.keras.layers.Flatten()(
        tf.keras.layers.Embedding(n_usuarios + 1, DIM_EMBEDDING, name="embedding_usuario")(entrada_usuario)
    )
    emb_producto = tf.keras.layers.Flatten()(
        tf.keras.layers.Embedding(n_productos + 1, DIM_EMBEDDING, name="embedding_producto")(entrada_producto)
    )
    sesgo_usuario = tf.keras.layers.Flatten()(
        tf.keras.layers.Embedding(n_usuarios + 1, 1, embeddings_initializer="zeros", name="sesgo_usuario")(entrada_usuario)
    )
    sesgo_producto = tf.keras.layers.Flatten()(
        tf.keras.layers.Embedding(n_productos + 1, 1, embeddings_initializer="zeros", name="sesgo_producto")(entrada_producto)
    )
    gmf = tf.keras.layers.Multiply(name="interaccion_gmf")([emb_usuario, emb_producto])
    rama_mlp = tf.keras.layers.Concatenate(name="concatenar")([emb_usuario, emb_producto])
    rama_mlp = tf.keras.layers.Dense(64, activation="relu", name="oculta_1")(rama_mlp)
    rama_mlp = tf.keras.layers.Dropout(0.2, name="dropout")(rama_mlp)
    rama_mlp = tf.keras.layers.Dense(32, activation="relu", name="oculta_2")(rama_mlp)
    fusion = tf.keras.layers.Concatenate(name="fusion")([gmf, rama_mlp])
    fusion = tf.keras.layers.Dense(16, activation="relu", name="fusion_oculta")(fusion)
    logit = tf.keras.layers.Dense(1, name="logit", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))(fusion)
    salida = tf.keras.layers.Add(name="sumar_logit")([logit, sesgo_usuario, sesgo_producto])
    salida = tf.keras.layers.Activation("sigmoid", name="probabilidad_compra")(salida)
    modelo = tf.keras.Model(inputs=[entrada_usuario, entrada_producto], outputs=salida, name="recomendador_beauty_ncf")
    modelo.compile(optimizer=tf.keras.optimizers.Adam(0.002), loss="binary_crossentropy")
    return modelo

def generar_pares(x_usuario, x_producto, compradas_train, n_productos, rng):
    usuarios_pos, productos_pos = [], []
    for u, p in zip(x_usuario, x_producto):
        negativos = []
        while len(negativos) < N_NEGATIVOS:
            cand = int(rng.integers(n_productos))
            if cand not in compradas_train[u] and cand not in negativos:
                negativos.append(cand)
        usuarios_pos.extend([u] * (1 + len(negativos)))
        productos_pos.append(p)
        productos_pos.extend(negativos)
    usuarios_totales = np.array(usuarios_pos, dtype=np.int32)
    productos_totales = np.array(productos_pos, dtype=np.int32)
    etiquetas = np.zeros(len(usuarios_totales), dtype=np.float32)
    etiquetas[np.arange(0, len(usuarios_totales), 1 + N_NEGATIVOS)] = 1.0
    orden = rng.permutation(len(usuarios_totales))
    return usuarios_totales[orden], productos_totales[orden], etiquetas[orden]

def main():
    RUTA_MODELOS.mkdir(exist_ok=True)
    fijar_semillas()
    print("[1/8] Cargando Beauty interacciones (1M nucleo)...")
    interacciones = pd.read_csv(RUTA_DATOS / "interacciones.csv")
    # Muestreo para CPU si es muy grande (CRISP-DM: Preparación)
    if len(interacciones) > MAX_INTERACCIONES_TRAIN * 1.25:
        print(f"      Muestreo {MAX_INTERACCIONES_TRAIN} para entrenamiento manejable (seed 42)")
        interacciones = interacciones.sample(n=MAX_INTERACCIONES_TRAIN, random_state=SEED).reset_index(drop=True)
    usuarios, productos_unicos, idx_usuario, idx_producto = construir_indices(interacciones)
    forma = (len(usuarios), len(productos_unicos))
    print(f"      {len(interacciones)} interacciones | {forma[0]} usuarios | {forma[1]} productos")

    print("[2/8] Division 80/20 (misma semilla Fase 3)...")
    rng = np.random.default_rng(SEED)
    mascara = rng.random(len(interacciones)) < 0.8
    train = interacciones[mascara].reset_index(drop=True)
    test = interacciones[~mascara].reset_index(drop=True)
    print(f"      train={len(train)} | test={len(test)}")
    x_usuario = train["user_id"].map(idx_usuario).to_numpy(dtype=np.int32)
    x_producto = train["product_id"].map(idx_producto).to_numpy(dtype=np.int32)
    compradas_train = [set() for _ in range(forma[0])]
    for u, p in zip(x_usuario, x_producto):
        compradas_train[u].add(p)

    print(f"[3/8] Construyendo NCF Beauty (Embedding {DIM_EMBEDDING}d, GMF+MLP+sesgos)...")
    modelo = construir_modelo(forma[0], forma[1])
    print(f"      Modelo: {modelo.count_params()} params, n_users {forma[0]}, n_products {forma[1]}")

    print(f"[4/8] Entrenando {EPOCHS} epocas BCE + negativos frescos x{N_NEGATIVOS}...")
    for epoca in range(EPOCHS):
        e_u, e_p, e_y = generar_pares(x_usuario, x_producto, compradas_train, forma[1], rng)
        h = modelo.fit([e_u, e_p], e_y, validation_split=0.1, epochs=1, batch_size=BATCH, verbose=0)
        print(f"      Epoca {epoca+1:>2}/{EPOCHS} | loss={h.history['loss'][0]:.4f} | val={h.history['val_loss'][0]:.4f}")

    print(f"[5/8] Generando puntajes evaluados sobre top {MAX_PRODUCTOS_EVAL} populares...")
    f_te = test["user_id"].map(idx_usuario).to_numpy(dtype=np.int32)
    c_te = test["product_id"].map(idx_producto).to_numpy(dtype=np.int32)
    reales = test["rating"].to_numpy()
    pop_counts = np.bincount(x_producto, minlength=forma[1])
    top_prod_idx = np.argsort(pop_counts)[::-1][:min(MAX_PRODUCTOS_EVAL, forma[1])]
    inv_top = {orig: new for new, orig in enumerate(top_prod_idx)}
    forma_eval = (forma[0], len(top_prod_idx))
    print(f"      forma original {forma} -> forma eval {forma_eval} (scoring bajo demanda)")
    # Solo puntajes para usuarios con relevantes + demo, no para 36k usuarios completos
    # Se generarán bajo demanda en [6/8]

    print("[6/8] Metricas ranking (sobre subset populares, scoring bajo demanda)...")
    relevantes_test = {}
    for u, p, r in zip(f_te, c_te, reales):
        if r >= UMBRAL_RELEVANTE and p in inv_top:
            relevantes_test.setdefault(u, set()).add(inv_top[p])
    compradas_train_eval = []
    for s in compradas_train:
        compradas_train_eval.append(set(inv_top[p] for p in s if p in inv_top))
    usuarios_eval = sorted(relevantes_test.keys())
    if len(usuarios_eval) > 3000:
        usuarios_eval = sorted(np.random.default_rng(SEED).choice(usuarios_eval, 3000, replace=False).tolist())
        eval_dict = {u: relevantes_test[u] for u in usuarios_eval}
    else:
        eval_dict = relevantes_test
    # Scoring bajo demanda solo para usuarios_eval + demo
    todos_p_eval = top_prod_idx.astype(np.int32)
    puntajes_eval = {}
    pop_eval_dict = {}
    for idx, u in enumerate(usuarios_eval):
        usuarios_lote = np.full(len(todos_p_eval), u, dtype=np.int32)
        puntajes_eval[u] = modelo.predict([usuarios_lote, todos_p_eval], batch_size=8192, verbose=0).ravel()
        if (idx+1) % 500 == 0:
            print(f"      scored {idx+1}/{len(usuarios_eval)} usuarios eval")
    # Construir matrices densas solo para eval_dict para usar evaluar_ranking
    # Alternativa: evaluar manualmente sin construir matriz completa
    precisions, recalls = [], []
    precisions_pop = []
    pop_vec = pop_counts[top_prod_idx].astype(np.float32)
    for u in usuarios_eval:
        fila = puntajes_eval[u].copy()
        fila[list(compradas_train_eval[u])] = -np.inf
        top = np.argpartition(fila, -TOP_K)[-TOP_K:]
        aciertos = len(eval_dict[u].intersection(top))
        precisions.append(aciertos / TOP_K)
        recalls.append(aciertos / len(eval_dict[u]) if eval_dict[u] else 0)
        # popularidad
        fila_pop = pop_vec.copy()
        fila_pop[list(compradas_train_eval[u])] = -np.inf
        top_pop = np.argpartition(fila_pop, -TOP_K)[-TOP_K:]
        precisions_pop.append(len(eval_dict[u].intersection(top_pop)) / TOP_K)
    precision_at_k = float(np.mean(precisions)) if precisions else 0.0
    recall_at_k = float(np.mean(recalls)) if recalls else 0.0
    precision_pop = float(np.mean(precisions_pop)) if precisions_pop else 0.0
    # Para compatibilidad, puntajes = matriz eval (solo usuarios_eval)
    puntajes = np.zeros((len(usuarios_eval), len(todos_p_eval)), dtype=np.float32)
    for i, u in enumerate(usuarios_eval):
        puntajes[i] = puntajes_eval[u]
    print(f"      Precision@{TOP_K}: {precision_at_k:.4f} (pop {precision_pop:.4f})")
    print(f"      Recall@{TOP_K}: {recall_at_k:.4f}")

    try:
        knn_beauty = joblib.load(RUTA_MODELOS / "colaborativo_beauty.pkl")
        tabla = pd.DataFrame({
            "Modelo": ["Popularidad", "KNN Beauty", "NCF Beauty (TF)"],
            f"Precision@{TOP_K}": [round(precision_pop,4), round(float(knn_beauty["precision_at_k"]),4), round(precision_at_k,4)]
        })
        print(tabla.to_string(index=False))
    except Exception as e:
        print(f"      comparativa skip: {e}")

    print("[7/8] Demo usuario activo Beauty...")
    conteo = train["user_id"].value_counts()
    usuario_demo = conteo.index[0]
    # Demo usa scoring directo (no de puntajes eval si demo no está en eval)
    if usuario_demo in idx_usuario:
        u_demo = idx_usuario[usuario_demo]
        if u_demo in puntajes_eval:
            fila_demo = puntajes_eval[u_demo].copy()
        else:
            fila_demo = modelo.predict([np.full(len(todos_p_eval), u_demo, dtype=np.int32), todos_p_eval], batch_size=8192, verbose=0).ravel()
            # compradas para demo
            comp_demo = set(inv_top[p] for p in compradas_train[u_demo] if p in inv_top)
            fila_demo[list(comp_demo)] = -np.inf
            top_demo = np.argsort(fila_demo)[::-1][:5]
            productos_eval_demo = [productos_unicos[i] for i in top_prod_idx]
            print(f"\nUsuario {usuario_demo} ({conteo.iloc[0]} compras):")
            for rank, col in enumerate(top_demo, start=1):
                print(f"  {rank}. [{fila_demo[col]:.3f}] {productos_eval_demo[col]}")
        # Si demo estaba en eval
        if usuario_demo in idx_usuario and u_demo in puntajes_eval:
            fila_demo = puntajes_eval[u_demo].copy()
            fila_demo[list(compradas_train_eval[u_demo])] = -np.inf
            top = np.argsort(fila_demo)[::-1][:5]
            productos_eval = [productos_unicos[i] for i in top_prod_idx]
            print(f"\nUsuario {usuario_demo} ({conteo.iloc[0]} compras):")
            for rank, col in enumerate(top, start=1):
                print(f"  {rank}. [{fila_demo[col]:.3f}] {productos_eval[col]}")
    else:
        print(f"Usuario demo {usuario_demo} no en Beauty (esperado si muestreo)")

    print("[8/8] Guardando artefactos...")
    modelo.save(RUTA_MODELOS / "red_beauty.keras")
    # Guardar solo muestra evaluada + índices para hibrido futuro
    productos_eval = [productos_unicos[i] for i in top_prod_idx]
    np.savez_compressed(RUTA_MODELOS / "predicciones_beauty_ncf.npz", matriz=puntajes, usuarios=np.array([usuarios[i] for i in usuarios_eval]), productos=np.array(productos_eval), top_prod_idx=top_prod_idx, forma=np.array(forma))
    joblib.dump({"idx_usuario": idx_usuario, "idx_producto": idx_producto, "dim_embedding": DIM_EMBEDDING, "n_negativos": N_NEGATIVOS, "epocas": EPOCHS, "precision_at_k": precision_at_k, "recall_at_k": recall_at_k, "precision_pop": precision_pop, "forma": forma, "forma_eval": forma_eval, "top_prod_idx": top_prod_idx, "usuarios_eval": usuarios_eval}, RUTA_MODELOS / "red_beauty_metricas.pkl")
    print("Artefactos: models/red_beauty.keras, predicciones_beauty_ncf.npz, red_beauty_metricas.pkl")
    print("Fase 4 Beauty completada - une TensorFlow + todo lo construido.")

if __name__ == "__main__":
    main()
