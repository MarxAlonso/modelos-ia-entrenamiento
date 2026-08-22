"""Genera recomendaciones consultables por usuario para la app React.

Para cada usuario produce: perfil, historial reciente y Top-10 del HIBRIDO V4
con la contribucion de cada componente (Two-Tower, LightGCN, NCF, KNN, NLP).
Salida: ver_grafos/app/src/data/recomendaciones.json
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RUTA_BASE = Path(__file__).resolve().parent.parent
RUTA_DATOS = RUTA_BASE / "data"
RUTA_MODELOS = RUTA_MODELOS = RUTA_BASE / "models"
RUTA_APP_DATA = RUTA_BASE / "ver_grafos" / "app" / "src" / "data"
PESOS = {"two_tower": 0.30, "lightgcn": 0.10, "ncf": 0.25,
         "knn": 0.15, "nlp_resenas": 0.20}


def minmax(S):
    mn, mx = S.min(axis=1, keepdims=True), S.max(axis=1, keepdims=True)
    return (S - mn) / np.maximum(mx - mn, 1e-9)


def main():
    print("Generando recomendaciones consultables...")
    t0 = time.time()

    art = joblib.load(RUTA_MODELOS / "hibrido_v2.pkl")
    # mapa col -> product_id en orden de columnas
    pid_de = [None] * len(art["idx_producto"])
    for pid, c in art["idx_producto"].items():
        pid_de[c] = pid

    df_prod = pd.read_csv(RUTA_DATOS / "productos.csv").set_index("product_id")
    df_int = pd.read_csv(RUTA_DATOS / "interacciones.csv")
    u_idx = {u: i for i, u in enumerate(sorted(df_int["user_id"].unique()))}
    usuarios_orden = sorted(u_idx)

    # componentes ya entrenados
    tt = np.load(RUTA_MODELOS / "torre_doble_vectores.npz")
    S_tt = tt["usuarios_vec"] @ tt["productos_vec"].T
    gnn = np.load(RUTA_MODELOS / "lightgcn_embeddings.npz")
    S_gnn = gnn["usuarios"] @ gnn["productos"].T
    S_ncf = np.load(RUTA_MODELOS / "predicciones_red_neuronal.npz")["matriz"].astype(np.float32)
    S_v4 = np.load(RUTA_MODELOS / "puntajes_hibrido_v4.npz")["matriz"]

    sem = np.load(RUTA_MODELOS / "embeddings_semanticos.npz", allow_pickle=True)
    SEM = sem["embeddings"][np.array([int(p[1:]) for p in pid_de])]
    SEMn = SEM / np.maximum(np.linalg.norm(SEM, axis=1, keepdims=True), 1e-9)

    # KNN + NLP por usuario (mismas formulas del pipeline final)
    rng_split = np.random.default_rng(42)
    p_idx = dict(art["idx_producto"])
    filas = df_int["user_id"].map(u_idx).values
    cols = df_int["product_id"].map(p_idx)
    val = ~cols.isna()
    filas, cols = filas[val], cols[val].astype(int)
    ratings = df_int.loc[val, "rating"].values
    train_mask = np.zeros(len(filas), dtype=bool)
    for u in range(len(u_idx)):
        iu = np.where(filas == u)[0]
        train_mask[iu[rng_split.random(len(iu)) < 0.8]] = True
    f_tr, c_tr, r_tr = filas[train_mask], cols[train_mask], ratings[train_mask]

    n_u, n_p = len(u_idx), len(p_idx)
    R = np.zeros((n_u, n_p), dtype=np.float32)
    R[f_tr, c_tr] = 1.0
    In = R.T / np.maximum(np.linalg.norm(R.T, axis=1, keepdims=True), 1e-9)
    S_knn = (R @ In) @ In.T

    S_nlp = np.zeros((n_u, n_p), dtype=np.float32)
    for u in range(n_u):
        pos = c_tr[(f_tr == u) & (r_tr >= 4)]
        if len(pos):
            q = SEMn[pos].mean(axis=0)
            q /= max(np.linalg.norm(q), 1e-9)
            S_nlp[u] = SEMn @ q

    COMP = {"two_tower": minmax(S_tt), "lightgcn": minmax(S_gnn),
            "ncf": minmax(S_ncf), "knn": minmax(S_knn), "nlp_resenas": minmax(S_nlp)}

    hist = {}
    for u, p in zip(f_tr, c_tr):
        hist.setdefault(u, []).append(int(p))

    salida = {}
    for uid in usuarios_orden:
        u = u_idx[uid]
        s = S_v4[u].copy()
        comprados = set(hist.get(u, []))
        for p in comprados:
            s[p] = -np.inf
        top = np.argsort(s)[::-1][:10]

        cats_hist = pd.Series([df_prod.loc[pid_de[col], "main_category"]
                               for col in hist.get(u, [])[:40]]
                              if u in hist else [])
        perfil = (cats_hist.value_counts().head(3).to_dict()
                  if len(cats_hist) else {})

        historial = []
        for col in list(hist.get(u, []))[-5:][::-1]:
            pid = pid_de[col]
            if pid in df_prod.index:
                historial.append({"nombre": str(df_prod.loc[pid, "name"])[:48],
                                  "categoria": str(df_prod.loc[pid, "main_category"]),
                                  "rating": float(ratings[(filas == u) & (cols == col)][0])})

        recs = []
        for col in top:
            pid = pid_de[col]
            if pid not in df_prod.index:
                continue
            contrib = {k: round(float(COMP[k][u, col]), 3) for k in PESOS}
            recs.append({
                "producto": str(df_prod.loc[pid, "name"])[:52],
                "categoria": str(df_prod.loc[pid, "main_category"]),
                "score": round(float(s[col]), 3),
                "contribucion": contrib,
            })

        salida[uid] = {
            "perfil": perfil,
            "total_compras": int(train_mask[(filas == u)].sum()),
            "historial": historial,
            "recomendaciones": recs,
        }

    RUTA_APP_DATA.mkdir(parents=True, exist_ok=True)
    (RUTA_APP_DATA / "recomendaciones.json").write_text(
        json.dumps({"pesos": PESOS, "generado": time.strftime("%Y-%m-%d %H:%M"),
                    "usuarios": salida}, ensure_ascii=False),
        encoding="utf-8")
    print(f"OK {len(salida)} usuarios -> recomendaciones.json ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
