"""FASE FINAL: Arquitectura unificada del sistema recomendador.

    DATOS ──► Usuarios / Productos / Reseñas
       Productos ─► NLP (Sentence-BERT 384D)
       Reseñas   ─► Text Embedding (media de items rating>=4)
    ──► NCF + LightGCN + Two-Tower ──► HÍBRIDO V4 (NCF+KNN+NLP) ──► TOP-N

Evalua el hibrido v4 contra cada componente en el mismo split SEED=42.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import json
import time
from pathlib import Path

import joblib
import numpy as np

SEED = 42
RUTA_BASE = Path(__file__).resolve().parent.parent
RUTA_MODELOS = RUTA_BASE / "models"
PESOS_V4 = {"two_tower": 0.30, "lightgcn": 0.10, "ncf": 0.25,
            "knn": 0.15, "nlp_resenas": 0.20}


def minmax_por_usuario(S):
    mn = S.min(axis=1, keepdims=True)
    mx = S.max(axis=1, keepdims=True)
    return (S - mn) / np.maximum(mx - mn, 1e-9)


def main():
    print("=" * 62)
    print(" HIBRIDO V4 FINAL  (NCF + LightGCN + Two-Tower + KNN + NLP)")
    print("=" * 62)
    t0 = time.time()

    # ---------- datos base (mismo split del nivel 3/4) ----------
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from nivel34_torres_gnn import cargar_base, SEED as _
    D = cargar_base()
    n_u, n_p = D["n_u"], D["n_p"]
    cat_prod = [D["cats_unicas"][j] for j in D["CATOH"].argmax(axis=1)]

    test_por_u = {}
    for (u, p) in D["test_pairs"]:
        test_por_u.setdefault(u, set()).add((p, cat_prod[p]))

    def evaluar(nombre, S):
        ac = hc = tot = 0
        for u in range(n_u):
            s = S[u].copy()
            for p in D["hist_train"].get(u, []):
                s[p] = -np.inf
            top = np.argpartition(s, -10)[-10:]
            te = test_por_u.get(u, set())
            if not te:
                continue
            prods = {p for p, _ in te}
            cats = {c for _, c in te}
            ac += sum(int(t) in prods for t in top)
            hc += sum(cat_prod[t] in cats for t in top)
            tot += len(top)
        r = {"precision_producto": round(ac / tot, 4),
             "precision_categoria": round(hc / tot, 4)}
        print(f"  {nombre:<22} P@10={r['precision_producto']:.4f}  CatPrec@10={r['precision_categoria']:.1%}")
        return r

    # ---------- componentes ----------
    print("\n[1/5] Cargando componentes entrenados...")
    tt = np.load(RUTA_MODELOS / "torre_doble_vectores.npz")
    S_tt = tt["usuarios_vec"] @ tt["productos_vec"].T
    gnn = np.load(RUTA_MODELOS / "lightgcn_embeddings.npz")
    S_gnn = gnn["usuarios"] @ gnn["productos"].T
    S_ncf = np.load(RUTA_MODELOS / "predicciones_red_neuronal.npz")["matriz"]

    SEM = D["SEM"]
    SEMn = SEM / np.maximum(np.linalg.norm(SEM, axis=1, keepdims=True), 1e-9)

    # ---------- NLP resenas: embedding de resenas positivas por usuario ----------
    print("[2/5] Text Embedding de resenas positivas (rating>=4)...")
    import pandas as pd
    df_int = pd.read_csv(D.get("_ruta", RUTA_BASE / "data" / "interacciones.csv"))
    u_idx = {u: i for i, u in enumerate(sorted(df_int["user_id"].unique()))}
    filas_all = df_int["user_id"].map(u_idx).values
    cols_all = df_int["product_id"].map(dict(D["art"]["idx_producto"]))
    val = ~cols_all.isna()
    rat_all = df_int.loc[val, "rating"].values
    f_tr, c_tr, r_tr = filas_all[val][D["train_mask"]], cols_all[val][D["train_mask"]].astype(int), rat_all[D["train_mask"]]

    S_nlp = np.zeros((n_u, n_p), dtype=np.float32)
    for u in range(n_u):
        pos = c_tr[(f_tr == u) & (r_tr >= 4)]
        if len(pos):
            q = SEMn[pos].mean(axis=0)
            q /= max(np.linalg.norm(q), 1e-9)
            S_nlp[u] = SEMn @ q

    # ---------- KNN item-item ----------
    print("[3/5] KNN item-item (coseno sobre matriz train)...")
    R = np.zeros((n_u, n_p), dtype=np.float32)
    R[f_tr, c_tr] = 1.0
    In = R.T / np.maximum(np.linalg.norm(R.T, axis=1, keepdims=True), 1e-9)  # (n_p,n_u)
    # item-item coseno por asociatividad: evita materializar n_p x n_p
    V = R @ In          # (n_u,n_u): perfil del usuario en espacio de items
    S_knn = V @ In.T    # (n_u,n_p): similitud con cada item

    # ---------- blend v4 ----------
    print(f"[4/5] Hibrido V4 con pesos {PESOS_V4}...")
    S_v4 = (PESOS_V4["two_tower"] * minmax_por_usuario(S_tt) +
            PESOS_V4["lightgcn"] * minmax_por_usuario(S_gnn) +
            PESOS_V4["ncf"] * minmax_por_usuario(S_ncf.astype(np.float32)) +
            PESOS_V4["knn"] * minmax_por_usuario(S_knn) +
            PESOS_V4["nlp_resenas"] * minmax_por_usuario(S_nlp))

    # ---------- evaluacion ----------
    print("\n[5/5] Evaluacion final (test implicito):")
    resultados = {
        "Two-Tower": evaluar("Two-Tower", S_tt),
        "LightGCN": evaluar("LightGCN", S_gnn),
        "NLP resenas": evaluar("NLP resenas", S_nlp),
        "KNN item-item": evaluar("KNN item-item", S_knn),
        "HIBRIDO V4": evaluar("HIBRIDO V4", S_v4),
    }

    ruta_res = RUTA_MODELOS / "nivel34_resultados.json"
    previo = json.loads(ruta_res.read_text(encoding="utf-8")) if ruta_res.exists() else {}
    previo.update(resultados)
    ruta_res.write_text(json.dumps(previo, ensure_ascii=False, indent=2), encoding="utf-8")

    np.savez_compressed(RUTA_MODELOS / "puntajes_hibrido_v4.npz", matriz=S_v4)
    joblib.dump({"pesos": PESOS_V4, "metricas": resultados,
                 "componentes": ["torre_doble", "lightgcn", "red_neuronal",
                                 "knn_itemitem", "semantico_resenas"]},
                RUTA_MODELOS / "hibrido_v4_final.pkl")

    mejor = max(resultados, key=lambda k: resultados[k]["precision_producto"])
    print(f"\nMejor P@10: {mejor} | CatPrec: "
          f"{resultados[mejor]['precision_categoria']:.1%}")
    print(f"COMPLETADO en {time.time()-t0:.1f}s -> models/hibrido_v4_final.pkl")


if __name__ == "__main__":
    main()
