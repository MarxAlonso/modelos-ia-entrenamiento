"""FASE 8: Catalogo ampliado CLIP + comportamiento de compra + version v5.

Etapas:
  A. Traduce los 1913 productos CLIP al espanol (Helsinki opus-mt-en-es)
     y extrae genero/color/talla/categoria.
  B. Simula compras realistas de nuestros 500 usuarios sobre el nuevo
     catalogo: actividad desigual (ley de potencias), reservas que luego
     se compran en cantidad, atributos color/talla/genero por interaccion.
  C. Reentrena Two-Tower + LightGCN + Hibrido V5 sobre el catalogo unido
     (6557 antiguos + nuevos), evalua contra V4 y VERSIONA artefactos.
  D. Regenera recomendaciones.json del app con NOMBRES de usuarios.

Uso: venv\\Scripts\\python.exe src/fase8_clip.py [--sin-traducir] [--max N]
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import argparse
import json
import re
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SEED = 42
RUTA_BASE = Path(__file__).resolve().parent.parent
RUTA_DATOS = RUTA_BASE / "data"
RUTA_MODELOS = RUTA_BASE / "models"
RUTA_APP = RUTA_BASE / "ver_grafos" / "app" / "src" / "data"
RUTA_CLIP = RUTA_DATOS / "clip-ecommerce" / "data" / \
    "train-00000-of-00001-1f042f20fd269c32.parquet"

PESOS_V5 = {"two_tower": 0.28, "lightgcn": 0.10, "ncf_parcial": 0.17,
            "knn": 0.15, "nlp_resenas": 0.30}

# ------------------------------------------------------------------ nombres
_NOMBRES_M = ["Alejandro","Mateo","Sebastián","Diego","Daniel","Santiago","Nicolás","Manuel",
              "Benjamín","Joaquín","Emiliano","Martín","Andrés","Lucas","Felipe","Carlos",
              "Javier","Miguel","Rafael","Ignacio"]
_NOMBRES_F = ["María","Valentina","Camila","Sofía","Luciana","Isabella","Daniela","Paula",
              "Antonella","Renata","Gabriela","Julia","Emma","Marta","Elena","Laura",
              "Andrea","Carla","Claudia","Beatriz"]
_APELLIDOS = ["García","Rodríguez","Martínez","López","González","Pérez","Sánchez","Ramírez",
              "Torres","Flores","Rivera","Gómez","Díaz","Reyes","Cruz","Morales","Ortiz",
              "Gutiérrez","Chávez","Vargas","Rojas","Castro","Mendoza","Aguilar","Paredes"]


def nombres_usuarios() -> dict:
    rng = np.random.default_rng(7)
    df_u = pd.read_csv(RUTA_DATOS / "usuarios.csv")
    salida = {}
    usados = set()
    for i, uid in enumerate(df_u["user_id"]):
        while True:
            genero = rng.random() < 0.5
            n = (rng.choice(_NOMBRES_F) if genero else rng.choice(_NOMBRES_M))
            completo = f"{n} {rng.choice(_APELLIDOS)} {rng.choice(_APELLIDOS)}"
            if completo not in usados:
                usados.add(completo)
                salida[uid] = completo
                break
    return salida


# ------------------------------------------------------- A. preparar catalogo
def preparar_catalogo(max_n: int, sin_traducir: bool) -> pd.DataFrame:
    cache = RUTA_DATOS / "productos_clip.csv"
    if cache.exists():
        print("[A] Catalogo CLIP ya preparado (cache)")
        return pd.read_csv(cache)

    df = pd.read_parquet(RUTA_CLIP, columns=[
        "Product_name", "Price", "colors", "Pattern", "Other Details"])
    df = df.head(max_n).reset_index(drop=True)
    print(f"[A] {len(df)} productos CLIP cargados")

    titulos = df["Product_name"].astype(str).tolist()
    if sin_traducir:
        nombres_es = [re.sub(r"\s+", " ", t.strip())[:70] for t in titulos]
    else:
        print("[A] Cargando traductor Helsinki opus-mt-en-es...")
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
        mod = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")
        mod.eval()
        nombres_es, lote = [], 64
        for i in range(0, len(titulos), lote):
            enc = tok(titulos[i:i + lote], return_tensors="pt", padding=True,
                      truncation=True, max_length=64)
            with torch.no_grad():
                gen = mod.generate(**enc, max_new_tokens=48, num_beams=1)
            nombres_es += [t.strip()[:70] for t in
                           tok.batch_decode(gen, skip_special_tokens=True)]
            if (i // lote) % 6 == 0:
                print(f"    traducidos {min(i + lote, len(titulos))}/{len(titulos)}")

    PATRON_GEN = [(r"\b(men|mens|male|hombre)\b", "hombre"),
                  (r"\b(women|womens|female|mujer|ladies)\b", "mujer"),
                  (r"\b(girls?|ninas?)\b", "niña"), (r"\b(boys?|ninos|kids)\b", "niño")]
    COLORES_ES = {"black":"negro","white":"blanco","red":"rojo","blue":"azul","green":"verde",
                  "yellow":"amarillo","brown":"café","gray":"gris","grey":"gris","pink":"rosa",
                  "purple":"morado","orange":"naranja","beige":"beige","silver":"plateado",
                  "gold":"dorado","navy":"azul marino","multi":"multicolor"}

    filas = []
    for i, fila in df.iterrows():
        texto = (str(fila["Product_name"]) + " " + str(fila.get("Other Details", ""))).lower()
        genero = "unisex"
        for pat, g in PATRON_GEN:
            if re.search(pat, texto):
                genero = g
                break
        crudo = str(fila.get("colors", "")).lower()
        color = "varios"
        for en, es in COLORES_ES.items():
            if en in crudo or en in str(fila["Product_name"]).lower():
                color = es
                break
        m = re.search(r"\b(XXS|XS|S|M|L|XL|XXL|\d{2})\b",
                      str(fila.get("Other Details", "")) + " " + titulos[i])
        talla = m.group(1) if m else np.random.default_rng(SEED + i).choice(
            ["S", "M", "L", "XL"], p=[0.2, 0.4, 0.3, 0.1])
        cat = "moda-" + ("calzado" if re.search(r"shoe|sneaker|boot|sandal", texto)
                         else "accesorio" if re.search(r"watch|bag|belt|jewel|wallet", texto)
                         else "ropa")
        try:
            precio = float(re.sub(r"[^\d.]", "", str(fila["Price"])) or 0) or \
                round(float(np.random.default_rng(i).uniform(15, 120)), 2)
        except Exception:
            precio = 49.9
        filas.append({
            "product_id": f"c{i:04d}", "nombre_es": nombres_es[i],
            "titulo_original": titulos[i][:60], "categoria_moda": cat,
            "genero": genero, "color": color, "talla": str(talla),
            "precio_usd": precio,
        })
    out = pd.DataFrame(filas)
    out.to_csv(cache, index=False)
    print(f"[A] Guardado {cache.name}: {len(out)} productos traducidos/enriquecidos")
    return out


# --------------------------------------------- B. simulacion de comportamiento
def simular_interacciones(df_clip: pd.DataFrame, nombres: dict) -> pd.DataFrame:
    print("[B] Simulando compras con comportamiento realista...")
    rng = np.random.default_rng(SEED)
    uids = sorted(nombres)
    # actividad desigual: ley de potencias -> pocos compran muchísimo
    actividad = (np.random.default_rng(11).pareto(1.6, len(uids)) + 1)
    actividad = actividad / actividad.max()

    # preferencia: cada usuario se inclina a un genero y 2 categorias moda
    pref_gen = rng.choice(["hombre", "mujer", "unisex"], size=len(uids),
                          p=[0.42, 0.42, 0.16])
    cats_posibles = ["moda-ropa", "moda-calzado", "moda-accesorio"]

    ids_clip = df_clip["product_id"].values
    idx_por_gen = {g: df_clip.index[df_clip["genero"].isin([g, "unisex"])].values
                   for g in ["hombre", "mujer", "unisex"]}
    idx_por_cat = {c: df_clip.index[df_clip["categoria_moda"] == c].values
                   for c in cats_posibles}

    filas = []
    for k, uid in enumerate(uids):
        n_compras = int(np.clip(3 + actividad[k] * 45, 3, 50))
        g_pref = pref_gen[k]
        mis_cats = [cats_posibles[j] for j in
                    rng.choice(len(cats_posibles), size=2, replace=False)]
        rating_base = rng.uniform(3.6, 4.6)
        for _ in range(n_compras):
            if rng.random() < 0.55:
                pool = idx_por_gen[g_pref]
            else:
                pool = idx_por_cat[rng.choice(mis_cats)]
            if len(pool) == 0:
                pool = df_clip.index.values
            j = int(rng.choice(pool))
            prod = df_clip.iloc[j]
            reservado = rng.random() < 0.22          # reserva previa
            cantidad = int(rng.choice([1, 2, 3], p=[0.55, 0.3, 0.15]))
            if reservado:
                cantidad += int(rng.integers(2, 5))  # compra en gran cantidad
            mes = int(rng.integers(1, 13)); dia = int(rng.integers(1, 29))
            rating = float(np.clip(rng.normal(rating_base - (0.6 if reservado else 0), 0.4), 1, 5))
            filas.append({
                "user_id": uid, "product_id": prod["product_id"],
                "categoria_producto": prod["categoria_moda"],
                "rating": round(rating, 1), "cantidad": cantidad,
                "fecha": f"2026-{mes:02d}-{dia:02d}",
                "estado": "reservado->comprado" if reservado else "comprado",
                "color": prod["color"], "talla": prod["talla"],
                "genero": prod["genero"],
            })
    out = pd.DataFrame(filas)
    out.to_csv(RUTA_DATOS / "interacciones_clip.csv", index=False)
    print(f"    {len(out)} interacciones nuevas "
          f"(reservadas: {(out['estado'] != 'comprado').sum()})")
    return out


# ------------------------------------------------------- utilidades entrenamiento
def minmax(S):
    mn, mx = S.min(axis=1, keepdims=True), S.max(axis=1, keepdims=True)
    return (S - mn) / np.maximum(mx - mn, 1e-9)


def torres_v5(n_u, n_p, n_c, PERFIL, STATS, CATOH, W2V, train_pairs):
    """Two-Tower sobre catalogo ampliado (arquitectura identica a v4)."""
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    tf.get_logger().setLevel("ERROR")
    rng = np.random.default_rng(SEED)

    inp_u = layers.Input(shape=(1,), dtype=tf.int32, name="usuario")
    ipf = layers.Input(shape=(n_c,), name="perfil")
    ist = layers.Input(shape=(2,), name="stats")
    eu = layers.Flatten()(layers.Embedding(n_u, 16)(inp_u))
    xu = layers.Dense(64, activation="relu")(layers.Concatenate()([eu, ipf, ist]))
    vu = layers.Lambda(lambda v: tf.nn.l2_normalize(v, axis=1))(layers.Dense(32)(xu))

    inp_p = layers.Input(shape=(1,), dtype=tf.int32, name="producto")
    icat = layers.Input(shape=(n_c,), name="catoh")
    iw2 = layers.Input(shape=(100,), name="w2v")
    ei = layers.Flatten()(layers.Embedding(n_p, 16)(inp_p))
    xi = layers.Dense(64, activation="relu")(layers.Concatenate()([ei, icat, iw2]))
    vi = layers.Lambda(lambda v: tf.nn.l2_normalize(v, axis=1))(layers.Dense(32)(xi))

    score = layers.Dot(axes=1)([vu, vi])
    modelo = Model([inp_u, ipf, ist, inp_p, icat, iw2], score)
    modelo.compile(optimizer="adam", loss="binary_crossentropy")

    pairs = np.array(list(train_pairs))
    ub_all, pb_all = pairs[:, 0], pairs[:, 1]
    NEG = 4

    def gen():
        while True:
            i = rng.choice(len(pairs), 256)
            ub, pb = ub_all[i], pb_all[i]
            negm = rng.choice(n_p, size=(256, NEG))
            uu = np.repeat(ub, NEG + 1)
            pp = np.concatenate([pb[:, None], negm], 1).ravel()
            yb = np.tile([1.] + [0.] * NEG, 256)
            yield ((uu[:, None].astype("int32"), PERFIL[uu], STATS[uu],
                    pp[:, None].astype("int32"), CATOH[pp], W2V[pp]),
                   yb.reshape(-1, 1))

    ds = tf.data.Dataset.from_generator(gen, output_signature=(
        (tf.TensorSpec((None, 1), tf.int32), tf.TensorSpec((None, n_c), tf.float32),
         tf.TensorSpec((None, 2), tf.float32), tf.TensorSpec((None, 1), tf.int32),
         tf.TensorSpec((None, n_c), tf.float32), tf.TensorSpec((None, 100), tf.float32)),
        tf.TensorSpec((None, 1), tf.float32)))
    modelo.fit(ds, steps_per_epoch=250, epochs=6, verbose=0)

    todos_u = np.arange(n_u)
    U = Model(modelo.inputs[:3], vu)([todos_u, PERFIL, STATS]).numpy()
    todos_p = np.arange(n_p)
    V = Model([modelo.inputs[3], modelo.inputs[4], modelo.inputs[5]], vi)(
        [todos_p, CATOH, W2V]).numpy()
    U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-9)
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    return U @ V.T


def lightgcn_v5(n_u, n_p, train_pairs, DIM=32, CAPAS=3, pasos=800):
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    rng = np.random.default_rng(SEED)
    n_t = n_u + n_p
    R = np.zeros((n_u, n_p), dtype=np.float32)
    for (u, p) in train_pairs:
        R[u, p] = 1.0
    A = np.zeros((n_t, n_t), dtype=np.float32); A[:n_u, n_u:] = R
    d = A.sum(1); dinv = 1 / np.maximum(np.sqrt(d), 1e-9)
    An = A * dinv[:, None] * dinv[None, :]
    fi, ci = np.nonzero(An)
    Asp = tf.sparse.reorder(tf.SparseTensor(np.stack([fi, ci], 1),
                                            An[fi, ci], (n_t, n_t)))
    E0 = tf.Variable(tf.random.normal([n_t, DIM], stddev=0.01, seed=SEED))
    opt = tf.optimizers.Adam(5e-4)

    @tf.function
    def prop():
        Es = [E0]; E = E0
        for _ in range(CAPAS):
            E = tf.sparse.sparse_dense_matmul(Asp, E); Es.append(E)
        return tf.add_n(Es) / (CAPAS + 1)

    pairs = np.array(list(train_pairs))
    ua, ia = pairs[:, 0], pairs[:, 1] + n_u
    for ep in range(pasos):
        idx = rng.choice(len(pairs), 2048)
        ub = tf.constant(ua[idx]); ib = tf.constant(ia[idx])
        nb = tf.constant(n_u + rng.choice(n_p, len(idx)))
        with tf.GradientTape() as tape:
            E = prop()
            sp = tf.reduce_sum(tf.gather(E, ub) * tf.gather(E, ib), 1)
            sn = tf.reduce_sum(tf.gather(E, ub) * tf.gather(E, nb), 1)
            perdida = tf.reduce_mean(tf.nn.softplus(sn - sp)) + 1e-6 * tf.nn.l2_loss(E0)
        g = tape.gradient(perdida, [E0]); opt.apply_gradients(zip(g, [E0]))
    Ef = prop().numpy()
    Eu, Ei = Ef[:n_u], Ef[n_u:]
    Eu /= np.maximum(np.linalg.norm(Eu, axis=1, keepdims=True), 1e-9)
    Ei /= np.maximum(np.linalg.norm(Ei, axis=1, keepdims=True), 1e-9)
    return Eu @ Ei.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=2000)
    ap.add_argument("--sin-traducir", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    print("=" * 62)
    print(" FASE 8: CLIP + COMPORTAMIENTO + HIBRIDO V5 VERSIONADO")
    print("=" * 62)

    # ---------- A/B datos ----------
    df_clip = preparar_catalogo(args.max, args.sin_traducir)
    nombres = nombres_usuarios()
    inter_clip = simular_interacciones(df_clip, nombres)

    # actualizar usuarios.csv con nombre
    df_u = pd.read_csv(RUTA_DATOS / "usuarios.csv")
    if "nombre" not in df_u.columns:
        df_u.insert(1, "nombre", df_u["user_id"].map(nombres))
        df_u.to_csv(RUTA_DATOS / "usuarios.csv", index=False)
        print("[B] usuarios.csv ahora incluye columna 'nombre'")

    # ---------- C. catalogo unido ----------
    art = joblib.load(RUTA_MODELOS / "hibrido_v2.pkl")
    p_idx_old = dict(art["idx_producto"])
    n_u, n_old = 500, len(p_idx_old)
    lista_clip = df_clip["product_id"].tolist()
    p_idx_new = {pid: n_old + j for j, pid in enumerate(lista_clip)}
    n_p = n_old + len(lista_clip)
    print(f"[C] Catalogo total: {n_old} antiguos + {len(lista_clip)} nuevos = {n_p}")

    df_int = pd.read_csv(RUTA_DATOS / "interacciones.csv")
    todo = pd.concat([df_int, inter_clip], ignore_index=True)
    u_idx = {u: i for i, u in enumerate(sorted(todo["user_id"].unique()))}
    mapa = {**p_idx_old, **p_idx_new}
    filas = todo["user_id"].map(u_idx).values
    cols = todo["product_id"].map(mapa)
    val = ~cols.isna()
    filas, cols = filas[val], cols[val].astype(int)
    ratings = todo.loc[val, "rating"].values

    rng = np.random.default_rng(SEED)
    tm = np.zeros(len(filas), bool)
    for u in range(n_u):
        iu = np.where(filas == u)[0]
        tm[iu[rng.random(len(iu)) < 0.8]] = True
    train_pairs = set(zip(filas[tm], cols[tm]))
    test_pairs = list(zip(filas[~tm], cols[~tm]))
    hist = {}
    for (u, p) in train_pairs:
        hist.setdefault(u, []).append(p)

    cats_new = ["moda-ropa", "moda-calzado", "moda-accesorio"]
    lista_cats = list(art["lista_cats"]) + cats_new
    n_c = len(lista_cats)
    cat_j = {c: j for j, c in enumerate(lista_cats)}
    df_prod_old = pd.read_csv(RUTA_DATOS / "productos.csv").set_index("product_id")

    CATOH = np.zeros((n_p, n_c), dtype=np.float32)
    for pid, c in p_idx_old.items():
        cat = df_prod_old.loc[pid, "main_category"]
        CATOH[c, cat_j[cat]] = 1
    for _, r in df_clip.iterrows():
        CATOH[p_idx_new[r["product_id"]], cat_j[r["categoria_moda"]]] = 1

    # W2V para nuevos items usando el modelo existente (titulos ya en español)
    print("[C] Infiriendo Word2Vec de los nuevos titulos...")
    from gensim.models import Word2Vec
    w2v = Word2Vec.load(str(RUTA_MODELOS / "word2vec_productos.model"))
    W2Vnew = np.zeros((len(lista_clip), 100), dtype=np.float32)
    for j, t in enumerate(df_clip["nombre_es"]):
        vecs = [w2v.wv[w] for w in re.sub(r"[^\w\sáéíóúñ]", " ", str(t).lower()).split()
                if w in w2v.wv]
        if vecs:
            W2Vnew[j] = np.mean(vecs, 0)
    w2v_old_full = np.load(RUTA_MODELOS / "embeddings_word2vec.npz", allow_pickle=True)["embeddings"]
    col_fila_old = np.zeros(n_old, int)
    for pid, c in p_idx_old.items():
        col_fila_old[c] = int(pid[1:])
    W2V = np.vstack([w2v_old_full[col_fila_old], W2Vnew])

    print("[C] Codificando nuevos productos con Sentence-BERT...")
    sem_new_path = RUTA_DATOS / "sem_clip.npy"
    if sem_new_path.exists():
        SEMnew = np.load(sem_new_path)
    else:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        SEMnew = st.encode((df_clip["nombre_es"] + ". " + df_clip["categoria_moda"]).tolist(),
                           batch_size=96, normalize_embeddings=True,
                           convert_to_numpy=True).astype(np.float32)
        np.save(sem_new_path, SEMnew)
    sem_old = np.load(RUTA_MODELOS / "embeddings_semanticos.npz", allow_pickle=True)["embeddings"]
    SEM = np.vstack([sem_old[col_fila_old], SEMnew])
    SEMn = SEM / np.maximum(np.linalg.norm(SEM, axis=1, keepdims=True), 1e-9)

    # perfil/stats usuario sobre catalogo ampliado
    PERFIL = np.zeros((n_u, n_c), dtype=np.float32)
    STATS = np.zeros((n_u, 2), dtype=np.float32)
    cat_prod_idx = CATOH.argmax(1)
    for u, items in hist.items():
        for p in items:
            PERFIL[u, cat_prod_idx[p]] += 1
        STATS[u, 0] = len(items)
    PERFIL /= np.maximum(PERFIL.sum(1, keepdims=True), 1e-9)
    STATS[:, 0] /= max(STATS[:, 0].max(), 1)

    # ---------- entrenar ----------
    print(f"[C] Two-Tower v5 sobre {n_p} productos...")
    S_tt = torres_v5(n_u, n_p, n_c, PERFIL, STATS, CATOH, W2V, train_pairs)
    print("[C] LightGCN v5 sobre grafo ampliado...")
    S_gnn = lightgcn_v5(n_u, n_p, train_pairs)

    # KNN + NLP
    R = np.zeros((n_u, n_p), dtype=np.float32)
    for (u, p) in train_pairs:
        R[u, p] = 1
    In_ = R.T / np.maximum(np.linalg.norm(R.T, axis=1, keepdims=True), 1e-9)
    S_knn = (R @ In_) @ In_.T
    S_nlp = np.zeros((n_u, n_p), dtype=np.float32)
    for u, items in hist.items():
        pos = [p for p in items]
        if pos:
            q = SEMn[pos].mean(0); q /= max(np.linalg.norm(q), 1e-9)
            S_nlp[u] = SEMn @ q

    # NCF solo cubre antiguos: componente parcial
    S_ncf_old = np.load(RUTA_MODELOS / "predicciones_red_neuronal.npz")["matriz"].astype(np.float32)
    S_ncf = np.full((n_u, n_p), -1.0, dtype=np.float32)
    S_ncf[:, :n_old] = S_ncf_old

    def mezclar(w):
        comp = {"two_tower": minmax(S_tt), "lightgcn": minmax(S_gnn),
                "knn": minmax(S_knn), "nlp_resenas": minmax(S_nlp)}
        acc = sum(w[k] * comp[k] for k in comp)
        if w.get("ncf_parcial"):
            mascara = S_ncf >= 0
            ncf_limpio = np.where(mascara, minmax(np.where(mascara, S_ncf, 0)), 0.0)
            pesos_eff = (w["ncf_parcial"] * mascara +
                         (w["two_tower"] + w["lightgcn"] + w["knn"] + w["nlp_resenas"]))
            acc = (w["two_tower"] * comp["two_tower"] + w["lightgcn"] * comp["lightgcn"] +
                   w["knn"] * comp["knn"] + w["nlp_resenas"] * comp["nlp_resenas"] +
                   w["ncf_parcial"] * ncf_limpio) / pesos_eff
        return acc

    S_v5 = mezclar(PESOS_V5)

    # ---------- evaluacion ----------
    print("\n[C] Evaluación (test implícito, excluyendo train):")
    cat_prod = [lista_cats[j] for j in cat_prod_idx]
    test_por_u = {}
    for (u, p) in test_pairs:
        test_por_u.setdefault(u, set()).add((p, cat_prod[p]))

    def evaluar(nombre, S):
        ac = hc = tot = 0
        for u in range(n_u):
            s = S[u].copy()
            for p in hist.get(u, []):
                s[p] = -np.inf
            top = np.argpartition(s, -10)[-10:]
            te = test_por_u.get(u, set())
            if not te:
                continue
            prods = {q for q, _ in te}; cats = {c for _, c in te}
            ac += sum(int(t) in prods for t in top)
            hc += sum(cat_prod[t] in cats for t in top)
            tot += len(top)
        r = {"precision_producto": round(ac / tot, 4),
             "precision_categoria": round(hc / tot, 4)}
        print(f"  {nombre:<24} P@10={r['precision_producto']:.4f}  CatPrec@10={r['precision_categoria']:.1%}")
        return r

    resultados = {
        "Two-Tower v5": evaluar("Two-Tower v5", S_tt),
        "LightGCN v5": evaluar("LightGCN v5", S_gnn),
        "KNN v5": evaluar("KNN v5", S_knn),
        "NLP reseñas v5": evaluar("NLP reseñas v5", S_nlp),
        "HIBRIDO V5": evaluar("HIBRIDO V5", S_v5),
    }

    # ---------- versionado ----------
    marca = time.strftime("%Y%m%d-%H%M%S")
    carpeta = RUTA_MODELOS / "versiones" / f"v5_{marca}"
    carpeta.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(carpeta / "puntajes_hibrido_v5.npz", matriz=S_v5)
    np.savez_compressed(carpeta / "matriz_two_tower_v5.npz", matriz=S_tt)
    joblib.dump({"pesos": PESOS_V5, "metricas": resultados,
                 "catalogo": {"antiguos": n_old, "nuevos": len(lista_clip)},
                 "interacciones_clip": len(inter_clip)}, carpeta / "hibrido_v5.pkl")
    # copia promovida a raiz
    shutil.copy2(carpeta / "puntajes_hibrido_v5.npz", RUTA_MODELOS / "puntajes_hibrido_v5.npz")
    shutil.copy2(carpeta / "hibrido_v5.pkl", RUTA_MODELOS / "hibrido_v5_final.pkl")

    estado_ruta = RUTA_MODELOS / "estado_entrenamiento.json"
    estado = json.loads(estado_ruta.read_text(encoding="utf-8")) if estado_ruta.exists() else {}
    estado.setdefault("ciclos", []).append(
        {"marca": f"v5_{marca}", "estado": "promocionado-manual",
         "detalle": "fase8 catalogo CLIP ampliado", "p10_nuevo": resultados["HIBRIDO V5"]["precision_producto"]})
    estado["p10_vigente"] = resultados["HIBRIDO V5"]["precision_producto"]
    estado_ruta.write_text(json.dumps(estado, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- D. recomendaciones.json con nombres ----------
    print("\n[D] Regenerando recomendaciones.json con nombres...")
    comp_norm = {"two_tower": minmax(S_tt), "lightgcn": minmax(S_gnn),
                 "ncf_parcial": minmax(np.where(S_ncf >= 0, S_ncf, 0)),
                 "knn": minmax(S_knn), "nlp_resenas": minmax(S_nlp)}
    salida = {}
    inv_pid = {c: pid for pid, c in mapa.items()}
    for uid, u in u_idx.items():
        s = S_v5[u].copy()
        for p in hist.get(u, []):
            s[p] = -np.inf
        top = np.argsort(s)[::-1][:10]
        recs = []
        for p in top:
            pid = inv_pid[p]
            if pid.startswith("c"):
                fila_c = df_clip[df_clip["product_id"] == pid].iloc[0]
                nom, cat = fila_c["nombre_es"], fila_c["categoria_moda"]
                extra = f"{fila_c['color']} · T:{fila_c['talla']}"
            else:
                nom = str(df_prod_old.loc[pid, "name"])[:52]
                cat = str(df_prod_old.loc[pid, "main_category"])
                extra = ""
            recs.append({"producto": nom, "categoria": cat, "extra": extra,
                         "score": round(float(s[p]), 3),
                         "contribucion": {k: round(float(comp_norm[k][u, p]), 3)
                                          for k in comp_norm}})
        compras_u = todo[(todo["user_id"] == uid)]
        perfil = compras_u["categoria_producto"].value_counts().head(3).to_dict()

        # historial: ultimas 5 compras (antiguas y de moda CLIP)
        recientes = compras_u.sort_values("fecha", ascending=False).head(5)
        historial = []
        for _, ci in recientes.iterrows():
            pid_h = ci["product_id"]
            if pid_h.startswith("c"):
                fc = df_clip[df_clip["product_id"] == pid_h]
                if len(fc):
                    fc = fc.iloc[0]
                    historial.append({
                        "nombre": str(fc["nombre_es"])[:48],
                        "categoria": str(fc["categoria_moda"]),
                        "rating": float(ci["rating"]),
                        "extra": f"{fc['color']} · T:{fc['talla']}"
                                 f"{' · reservado' if ci.get('estado') != 'comprado' else ''}",
                    })
            elif pid_h in df_prod_old.index:
                historial.append({
                    "nombre": str(df_prod_old.loc[pid_h, "name"])[:48],
                    "categoria": str(df_prod_old.loc[pid_h, "main_category"]),
                    "rating": float(ci["rating"]),
                })

        salida[uid] = {
            "nombre": nombres[uid],
            "perfil": perfil,
            "total_compras": int(len(hist.get(u, []))),
            "historial": historial,
            "recomendaciones": recs,
        }
    RUTA_APP.mkdir(parents=True, exist_ok=True)
    (RUTA_APP / "recomendaciones.json").write_text(json.dumps(
        {"pesos": PESOS_V5, "generado": time.strftime("%Y-%m-%d %H:%M"),
         "version": "v5", "usuarios": salida}, ensure_ascii=False), encoding="utf-8")

    print(f"\nCOMPLETADO {round(time.time()-t0,1)}s | version models/versiones/v5_{marca}")
    print(f"P@10 HIBRIDO V5 = {resultados['HIBRIDO V5']['precision_producto']}")


if __name__ == "__main__":
    main()
