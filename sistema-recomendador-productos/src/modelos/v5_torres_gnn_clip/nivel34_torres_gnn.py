"""Niveles 3 y 4: Two-Tower (Dual Encoder) + LightGCN sobre el grafo usuario-producto.

Two-Tower:
  USER TOWER : embedding usuario + perfil de categorias + stats de historial
  ITEM TOWER : embedding producto + categoria one-hot + Word2Vec (100D)
  Similitud por producto punto, entrenada con BCE y muestreo negativo.

LightGCN:
  Propagacion sobre grafo bipartito usuario-producto (3 capas, sin pesos),
  perdida BPR con muestreo negativo.

Todo se evalua con el mismo split (SEED=42) y se compara contra los modelos
de las fases anteriores.
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
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SEED = 42
RUTA_BASE = RUTA_RAIZ
RUTA_DATOS = RUTA_BASE / "data"
RUTA_MODELOS = RUTA_BASE / "models"


# ---------------------------------------------------------------- datos base
def cargar_base():
    print("[1/6] Cargando datos y artefactos previos...")
    art = joblib.load(RUTA_MODELOS / "hibrido_v2.pkl")
    df_int = pd.read_csv(RUTA_DATOS / "interacciones.csv")

    u_idx = {u: i for i, u in enumerate(sorted(df_int["user_id"].unique()))}
    p_idx = dict(art["idx_producto"])          # pid -> col (espacio 6557)
    n_u, n_p = len(u_idx), len(p_idx)

    filas = df_int["user_id"].map(u_idx).values
    cols = df_int["product_id"].map(p_idx)
    valido = ~cols.isna()
    filas, cols = filas[valido], cols[valido].astype(int)
    ratings = df_int.loc[valido, "rating"].values

    # split 80/20 por usuario (SEED=42)
    rng = np.random.default_rng(SEED)
    train_mask = np.zeros(len(filas), dtype=bool)
    for u in range(n_u):
        idx_u = np.where(filas == u)[0]
        sel = rng.random(len(idx_u)) < 0.8
        train_mask[idx_u[sel]] = True

    train_pairs = set(zip(filas[train_mask], cols[train_mask]))
    test_pairs = list(zip(filas[~train_mask], cols[~train_mask]))

    hist_train = {}
    for (u, p) in train_pairs:
        hist_train.setdefault(u, []).append(p)

    # features de fases anteriores
    df_prod = pd.read_csv(RUTA_DATOS / "productos.csv").set_index("product_id")
    cats_unicas = sorted(df_prod["main_category"].dropna().unique())
    cat_a_j = {c: j for j, c in enumerate(cats_unicas)}

    w2v = np.load(RUTA_MODELOS / "embeddings_word2vec.npz", allow_pickle=True)["embeddings"]
    sem = np.load(RUTA_MODELOS / "embeddings_semanticos.npz", allow_pickle=True)["embeddings"]

    # alinear embeddings (7449 orden csv) -> cols 6557
    col_a_fila = np.zeros(n_p, dtype=int)
    for pid, c in p_idx.items():
        col_a_fila[c] = int(pid[1:])
    W2V = w2v[col_a_fila]
    SEM = sem[col_a_fila]

    CATOH = np.zeros((n_p, len(cats_unicas)), dtype=np.float32)
    for pid, c in p_idx.items():
        cat = df_prod.loc[pid, "main_category"]
        if cat in cat_a_j:
            CATOH[c, cat_a_j[cat]] = 1.0

    print(f"  usuarios={n_u} productos={n_p} train={len(train_pairs)} test={len(test_pairs)}")
    return dict(art=art, n_u=n_u, n_p=n_p, filas=filas, cols=cols,
                train_mask=train_mask, train_pairs=train_pairs,
                test_pairs=test_pairs, hist_train=hist_train,
                cats_unicas=cats_unicas, cat_a_j=cat_a_j,
                W2V=W2V, SEM=SEM, CATOH=CATOH)


def perfil_usuario(D):
    """Perfil categorias + stats por usuario desde train (fase afinidad)."""
    n_u, n_p = D["n_u"], D["n_p"]
    n_c = len(D["cats_unicas"])
    perfil = np.zeros((n_u, n_c), dtype=np.float32)
    stats = np.zeros((n_u, 2), dtype=np.float32)
    cat_prod = D["CATOH"].argmax(axis=1)
    for u, items in D["hist_train"].items():
        for p in items:
            perfil[u, cat_prod[p]] += 1
        stats[u, 0] = len(items)
    perfil /= np.maximum(perfil.sum(axis=1, keepdims=True), 1e-9)
    stats[:, 0] /= max(stats[:, 0].max(), 1)
    return perfil, stats


# ---------------------------------------------------------------- two tower
def entrenar_two_tower(D):
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    tf.get_logger().setLevel("ERROR")

    print("[2/6] Entrenando Two-Tower (Dual Encoder)...")
    t0 = time.time()
    n_u, n_p, n_c = D["n_u"], D["n_p"], len(D["cats_unicas"])
    PERFIL, STATS = perfil_usuario(D)

    DIM_EMB, DIM_H, DIM_OUT, NEG = 16, 64, 32, 4
    rng = np.random.default_rng(SEED)

    # ---- entradas user tower
    inp_u = layers.Input(shape=(1,), dtype=tf.int32, name="usuario")
    inp_perfil = layers.Input(shape=(n_c,), name="perfil_cats")
    inp_stats = layers.Input(shape=(2,), name="stats_historial")
    emb_u = layers.Embedding(n_u, DIM_EMB, name="emb_usuario")(inp_u)
    emb_u = layers.Flatten()(emb_u)
    x_u = layers.Concatenate()([emb_u, inp_perfil, inp_stats])
    x_u = layers.Dense(DIM_H, activation="relu", name="torre_user_d1")(x_u)
    vec_u = layers.Dense(DIM_OUT, name="torre_user_d2")(x_u)      # user vector
    vec_u = layers.Lambda(lambda v: tf.nn.l2_normalize(v, axis=1))(vec_u)

    # ---- entradas item tower
    inp_p = layers.Input(shape=(1,), dtype=tf.int32, name="producto")
    inp_cat = layers.Input(shape=(n_c,), name="cat_onehot")
    inp_w2v = layers.Input(shape=(100,), name="word2vec")
    emb_p = layers.Embedding(n_p, DIM_EMB, name="emb_producto")(inp_p)
    emb_p = layers.Flatten()(emb_p)
    x_i = layers.Concatenate()([emb_p, inp_cat, inp_w2v])
    x_i = layers.Dense(DIM_H, activation="relu", name="torre_item_d1")(x_i)
    vec_i = layers.Dense(DIM_OUT, name="torre_item_d2")(x_i)      # item vector
    vec_i = layers.Lambda(lambda v: tf.nn.l2_normalize(v, axis=1))(vec_i)

    score = layers.Dot(axes=1)([vec_u, vec_i])
    modelo = Model([inp_u, inp_perfil, inp_stats, inp_p, inp_cat, inp_w2v], score)
    modelo.compile(optimizer="adam", loss="binary_crossentropy",
                   metrics=[tf.keras.metrics.AUC(name="auc")])

    # ---- generador de lotes positivos + negativos
    pairs = np.array(list(D["train_pairs"]))
    u_arr, p_arr = pairs[:, 0], pairs[:, 1]

    def gen():
        while True:
            bs = 256
            i = rng.choice(len(pairs), size=bs)
            ub, pb = u_arr[i], p_arr[i]
            neg_matrix = rng.choice(n_p, size=(bs, NEG))
            uu = np.repeat(ub, NEG + 1)
            pp = np.concatenate([pb[:, None], neg_matrix], axis=1).ravel()
            yb = np.tile([1.0] + [0.0] * NEG, bs)
            yield ((uu[:, None].astype(np.int32), PERFIL[uu].astype(np.float32),
                    STATS[uu].astype(np.float32), pp[:, None].astype(np.int32),
                    D["CATOH"][pp].astype(np.float32), D["W2V"][pp].astype(np.float32)),
                   yb.reshape(-1, 1).astype(np.float32))

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            (tf.TensorSpec((None, 1), tf.int32),
             tf.TensorSpec((None, n_c), tf.float32),
             tf.TensorSpec((None, 2), tf.float32),
             tf.TensorSpec((None, 1), tf.int32),
             tf.TensorSpec((None, n_c), tf.float32),
             tf.TensorSpec((None, 100), tf.float32)),
            tf.TensorSpec((None, 1), tf.float32),
        ),
    )
    modelo.fit(ds, steps_per_epoch=250, epochs=6, verbose=2)

    # ---- vectores finales
    todos_u = np.arange(n_u)
    U = Model(modelo.inputs[:3], modelo.get_layer("torre_user_d2").output)(
        [todos_u, PERFIL, STATS]).numpy()
    U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-9)
    todos_p = np.arange(n_p)
    V = Model([modelo.inputs[3], modelo.inputs[4], modelo.inputs[5]],
              modelo.get_layer("torre_item_d2").output)(
        [todos_p, D["CATOH"], D["W2V"]]).numpy()
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)

    modelo.save(RUTA_MODELOS / "torre_doble.keras")
    np.savez_compressed(RUTA_MODELOS / "torre_doble_vectores.npz",
                        usuarios_vec=U, productos_vec=V)
    print(f"  listo en {time.time()-t0:.1f}s | U{U.shape} V{V.shape} -> models/torre_doble.keras")

    # extraer aristas para visualizacion
    cfg = modelo.get_config()
    validos = {str(c.get("name")) for c in cfg["layers"]}
    def _refs(obj, out):
        if isinstance(obj, dict):
            if obj.get("class_name") == "__keras_tensor__":
                h = obj.get("config", {}).get("keras_history")
                if isinstance(h, (list, tuple)) and h:
                    out.append(str(h[0]))
                return
            for v in obj.values():
                _refs(v, out)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _refs(v, out)
    aristas = []
    for cc in cfg["layers"]:
        nombre, refs = str(cc.get("name")), []
        _refs(cc.get("inbound_nodes") or [], refs)
        for r in refs:
            if r != nombre and r in validos and (r, nombre) not in aristas:
                aristas.append((r, nombre))
    nodos = [{"name": str(c.get("name")), "type": c.get("class_name", ""),
              "torre": ("user" if "user" in str(c.get('name')) or str(c.get('name')) in
                        ("usuario", "perfil_cats", "stats_historial") else
                        ("item" if "item" in str(c.get('name')) or str(c.get('name')) in
                         ("producto", "cat_onehot", "word2vec") else "fusion"))}
             for c in cfg["layers"]]
    (RUTA_MODELOS / "torre_doble_arquitectura.json").write_text(
        json.dumps({"nodes": nodos,
                    "links": [{"source": a, "target": b} for a, b in aristas]},
                   ensure_ascii=False), encoding="utf-8")
    return U @ V.T


# ---------------------------------------------------------------- lightgcn
def entrenar_lightgcn(D):
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    print("[3/6] Entrenando LightGCN en grafo usuario-producto...")
    t0 = time.time()
    n_u, n_p, DIM, CAPAS = D["n_u"], D["n_p"], 32, 3
    n_total = n_u + n_p

    R = np.zeros((n_u, n_p), dtype=np.float32)
    for (u, p) in D["train_pairs"]:
        R[u, p] = 1.0
    A = np.zeros((n_total, n_total), dtype=np.float32)
    A[:n_u, n_u:] = R
    deg = A.sum(axis=1)
    dinv = 1.0 / np.maximum(np.sqrt(deg), 1e-9)
    A_norm = A * dinv[:, None] * dinv[None, :]
    filas_a, cols_a = np.nonzero(A_norm)
    A_sp = tf.sparse.reorder(tf.SparseTensor(
        np.stack([filas_a, cols_a], axis=1),
        A_norm[filas_a, cols_a],
        (n_total, n_total),
    ))

    E0 = tf.Variable(tf.random.normal([n_total, DIM], stddev=0.01, seed=SEED))
    opt = tf.optimizers.Adam(learning_rate=5e-4)

    pairs = np.array(list(D["train_pairs"]))
    u_arr, i_arr = pairs[:, 0], pairs[:, 1] + n_u
    rng = np.random.default_rng(SEED)

    @tf.function
    def propagar():
        Es = [E0]
        E = E0
        for _ in range(CAPAS):
            E = tf.sparse.sparse_dense_matmul(A_sp, E)
            Es.append(E)
        return tf.add_n(Es) / (CAPAS + 1)

    def paso_bpr(bs=2048):
        idx = rng.choice(len(pairs), size=min(bs, len(pairs)), replace=False)
        ub = tf.constant(u_arr[idx])
        ib = tf.constant(i_arr[idx])
        nb = tf.constant(n_u + rng.choice(n_p, size=len(idx)))   # negativos: solo items
        with tf.GradientTape() as tape:
            E = propagar()
            s_pos = tf.reduce_sum(tf.gather(E, ub) * tf.gather(E, ib), axis=1)
            s_neg = tf.reduce_sum(tf.gather(E, ub) * tf.gather(E, nb), axis=1)
            perdida = tf.reduce_mean(tf.nn.softplus(s_neg - s_pos)) \
                + 1e-6 * tf.nn.l2_loss(E0)
        grads = tape.gradient(perdida, [E0])
        opt.apply_gradients(zip(grads, [E0]))
        return float(perdida)

    for ep in range(1000):
        p = paso_bpr()
        if ep % 250 == 0:
            print(f"    epoca {ep}: bpr={p:.4f}")

    E_final = propagar().numpy()
    Eu, Ei = E_final[:n_u], E_final[n_u:]
    Ei = Ei / np.maximum(np.linalg.norm(Ei, axis=1, keepdims=True), 1e-9)
    Eu = Eu / np.maximum(np.linalg.norm(Eu, axis=1, keepdims=True), 1e-9)
    np.savez_compressed(RUTA_MODELOS / "lightgcn_embeddings.npz",
                        usuarios=Eu, productos=Ei)
    print(f"  listo en {time.time()-t0:.1f}s | dim={DIM}, capas={CAPAS} -> models/lightgcn_embeddings.npz")
    return Eu @ Ei.T


# ---------------------------------------------------------------- evaluacion
def evaluar(D, matrices: dict[str, np.ndarray]):
    print("[4/6] Evaluacion unificada (test implicito, excluyendo train)...")
    n_u, n_p = D["n_u"], D["n_p"]
    cat_prod = [D["cats_unicas"][j] for j in D["CATOH"].argmax(axis=1)]

    test_por_u = {}
    for (u, p) in D["test_pairs"]:
        test_por_u.setdefault(u, set()).add((p, cat_prod[p]))

    resultados = {}
    for nombre, S in matrices.items():
        aciertos = hits_cat = total = 0
        for u in range(n_u):
            s = S[u].copy()
            for p in D["hist_train"].get(u, []):
                s[p] = -np.inf
            top = np.argpartition(s, -10)[-10:]
            test = test_por_u.get(u, set())
            if not test:
                continue
            prods_test = {p for p, _ in test}
            cats_test = {c for _, c in test}
            aciertos += sum(int(t) in prods_test for t in top)
            hits_cat += sum(cat_prod[t] in cats_test for t in top)
            total += len(top)
        resultados[nombre] = {
            "precision_producto": round(aciertos / total, 4),
            "precision_categoria": round(hits_cat / total, 4),
        }
        r = resultados[nombre]
        print(f"  {nombre:<28} P@10={r['precision_producto']:.4f}  CatPrec@10={r['precision_categoria']:.1%}")
    return resultados


def main():
    print("=" * 62)
    print(" NIVEL 3 (TWO-TOWER) + NIVEL 4 (LIGHTGCN) + EVALUACION UNIDA")
    print("=" * 62)
    t0 = time.time()

    D = cargar_base()

    S_tt = entrenar_two_tower(D)
    S_gnn = entrenar_lightgcn(D)

    # scores por historial con embeddings semanticos/w2v (fases previas)
    n_u, n_p = D["n_u"], D["n_p"]
    S_sem = np.zeros((n_u, n_p), dtype=np.float32)
    S_w2v = np.zeros((n_u, n_p), dtype=np.float32)
    SEM_n = D["SEM"] / np.maximum(np.linalg.norm(D["SEM"], axis=1, keepdims=True), 1e-9)
    W2V_n = D["W2V"] / np.maximum(np.linalg.norm(D["W2V"], axis=1, keepdims=True), 1e-9)
    for u, items in D["hist_train"].items():
        q_sem = SEM_n[items].mean(axis=0); q_sem /= max(np.linalg.norm(q_sem), 1e-9)
        q_w2v = W2V_n[items].mean(axis=0); q_w2v /= max(np.linalg.norm(q_w2v), 1e-9)
        S_sem[u] = SEM_n @ q_sem
        S_w2v[u] = W2V_n @ q_w2v

    pop = np.zeros((n_u, n_p), dtype=np.float32)
    for (_, p) in D["train_pairs"]:
        pop[:, p] += 1

    otros = {}
    for f, k in [("predicciones_svd.npz", "SVD"), ("predicciones_red_neuronal.npz", "NCF red neuronal"),
                 ("puntajes_hibrido_v2.npz", "Hibrido v2 (fase 6)")]:
        otros[k] = np.load(RUTA_MODELOS / f)["matriz"]

    matrices = {"Popularidad": pop, **otros,
                "Word2Vec historial": S_w2v, "Semantico Transformer": S_sem,
                "Two-Tower": S_tt, "LightGCN": S_gnn}
    res = evaluar(D, matrices)

    (RUTA_MODELOS / "nivel34_resultados.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[5/6] Artefactos guardados: torre_doble.keras, torre_doble_vectores.npz,")
    print("      lightgcn_embeddings.npz, nivel34_resultados.json")
    print(f"[6/6] COMPLETADO en {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
