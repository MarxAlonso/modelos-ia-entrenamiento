import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error

from colaborativo import (
    calcular_sesgos,
    construir_indices,
    evaluar_ranking,
    matriz_binaria,
    matriz_residuos,
    similitud_item_item,
)

SEED = 42
CANDIDATOS_K = [2, 4, 8]
TOP_K = 10
UMBRAL_RELEVANTE = 4.0

RUTA_DATOS = Path("data/amazon")
RUTA_MODELOS = Path("models")
RUTA_SINTETICO = RUTA_MODELOS / "colaborativo_svd.pkl"


def main() -> None:
    RUTA_MODELOS.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("[1/8] Cargando interacciones REALES de Amazon...")
    interacciones = pd.read_csv(RUTA_DATOS / "interacciones.csv")
    productos_info = pd.read_csv(
        RUTA_DATOS / "productos.csv",
        usecols=["product_id", "titulo"],
    ).set_index("product_id")["titulo"]

    usuarios, productos_unicos, idx_usuario, idx_producto = construir_indices(interacciones)
    forma = (len(usuarios), len(productos_unicos))
    print(f"      {len(interacciones)} interacciones reales | {forma[0]} usuarios | {forma[1]} productos")

    print("[2/8] Division train/test 80/20...")
    mascara = rng.random(len(interacciones)) < 0.8
    train = interacciones[mascara].reset_index(drop=True)
    test = interacciones[~mascara].reset_index(drop=True)
    print(f"      train={len(train)} | test={len(test)}")

    print("[3/8] Seleccion de k por validacion (85/15 dentro de train)...")
    m_val = rng.random(len(train)) < 0.85
    fit_df = train[m_val].reset_index(drop=True)
    val_df = train[~m_val].reset_index(drop=True)

    mu_fit, b_u_fit, b_i_fit = calcular_sesgos(fit_df, idx_usuario, idx_producto, *forma)
    mr_fit = matriz_residuos(fit_df, idx_usuario, idx_producto, forma, mu_fit, b_u_fit, b_i_fit)
    f_val = val_df["user_id"].map(idx_usuario).to_numpy()
    c_val = val_df["product_id"].map(idx_producto).to_numpy()

    mejor_k, mejor_rmse_val = None, np.inf
    for k in CANDIDATOS_K:
        svd_k = TruncatedSVD(n_components=k, random_state=SEED)
        lat = svd_k.fit_transform(mr_fit)
        rec = lat @ svd_k.components_
        est = np.clip(mu_fit + b_u_fit[f_val] + b_i_fit[c_val] + rec[f_val, c_val], 1.0, 5.0)
        rmse_val = mean_squared_error(val_df["rating"].to_numpy(), est) ** 0.5
        marca = ""
        if rmse_val < mejor_rmse_val:
            mejor_rmse_val, mejor_k = rmse_val, k
            marca = "  <-- mejor"
        print(f"      k={k:>2} | RMSE validacion={rmse_val:.4f}{marca}")

    print(f"[4/8] Entrenando SVD final con datos reales (k={mejor_k})...")
    mu, b_u, b_i = calcular_sesgos(train, idx_usuario, idx_producto, *forma)
    residuos = matriz_residuos(train, idx_usuario, idx_producto, forma, mu, b_u, b_i)
    svd_final = TruncatedSVD(n_components=mejor_k, random_state=SEED)
    latente = svd_final.fit_transform(residuos)

    f_te = test["user_id"].map(idx_usuario).to_numpy()
    c_te = test["product_id"].map(idx_producto).to_numpy()
    reales = test["rating"].to_numpy()
    estimados = np.clip(mu + b_u[f_te] + b_i[c_te] + (latente @ svd_final.components_)[f_te, c_te], 1.0, 5.0)
    rmse_modelo = mean_squared_error(reales, estimados) ** 0.5
    rmse_base = mean_squared_error(reales, np.full_like(reales, mu)) ** 0.5
    print(f"      RMSE linea base (media global): {rmse_base:.4f}")
    print(f"      RMSE modelo sesgos+SVD(k={mejor_k}): {rmse_modelo:.4f}")

    print("[5/8] Construyendo ranking KNN item-item sobre compras reales...")
    binaria = matriz_binaria(train, idx_usuario, idx_producto, forma)
    sim = similitud_item_item(binaria)

    compradas_train = [set() for _ in range(forma[0])]
    coo = binaria.tocoo()
    for u, p in zip(coo.row, coo.col):
        compradas_train[u].add(p)

    relevantes_test: dict[int, set[int]] = {}
    for u, p, r in zip(f_te, c_te, reales):
        if r >= UMBRAL_RELEVANTE:
            relevantes_test.setdefault(u, set()).add(p)

    print(f"[6/8] Metricas de ranking (relevante = rating>={UMBRAL_RELEVANTE})...")
    precisions, recalls = [], []
    usuarios_evaluables = sorted(relevantes_test.keys())
    for u in usuarios_evaluables:
        if not compradas_train[u]:
            continue
        fila = np.asarray((binaria[u] @ sim).todense()).ravel()
        fila[list(compradas_train[u])] = -np.inf
        top = np.argpartition(fila, -TOP_K)[-TOP_K:]
        aciertos = len(relevantes_test[u].intersection(top))
        precisions.append(aciertos / TOP_K)
        recalls.append(aciertos / len(relevantes_test[u]))
    precision_at_k = float(np.mean(precisions))
    recall_at_k = float(np.mean(recalls))
    print(f"      Precision@{TOP_K}: {precision_at_k:.4f}")
    print(f"      Recall@{TOP_K}:    {recall_at_k:.4f}")
    print(f"      Usuarios evaluables: {len(precisions)}/{forma[0]}")

    pop_scores = None
    pop = np.tile(np.asarray(binaria.sum(axis=0)).ravel(), (1, 1)).astype(np.float32)[0]
    precisions_pop = []
    for u in usuarios_evaluables:
        if not compradas_train[u]:
            continue
        fila = pop.copy().astype(np.float64)
        fila[list(compradas_train[u])] = -np.inf
        top = np.argpartition(fila, -TOP_K)[-TOP_K:]
        precisions_pop.append(len(relevantes_test[u].intersection(top)) / TOP_K)
    precision_pop = float(np.mean(precisions_pop))
    print(f"      (linea base popularidad: Precision@{TOP_K}={precision_pop:.4f})")

    print("[7/8] Comparacion con el dataset sintetico:")
    sintetico = joblib.load(RUTA_SINTETICO)
    comparacion = pd.DataFrame(
        {
            "Metrica": ["RMSE", "Precision@10", "Recall@10", "Densidad matriz"],
            "Sintetico": [
                round(sintetico["rmse"], 4),
                round(sintetico["precision_at_k"], 4),
                round(sintetico["recall_at_k"], 4),
                "0.62%",
            ],
            "Amazon real": [
                round(rmse_modelo, 4),
                round(precision_at_k, 4),
                round(recall_at_k, 4),
                "0.02%",
            ],
        }
    )
    print(comparacion.to_string(index=False))

    print("[8/8] DEMO - Top-5 para un usuario real activo:")
    conteo = train["user_id"].value_counts()
    usuario_demo = conteo.index[0]
    u_demo = idx_usuario[usuario_demo]
    fila = np.asarray((binaria[u_demo] @ sim).todense()).ravel()
    fila[list(compradas_train[u_demo])] = -np.inf
    top = np.argsort(fila)[::-1][:5]
    inv_prod = {v: k for k, v in idx_producto.items()}
    print(f"\nUsuario real {usuario_demo} ({conteo.iloc[0]} compras en train):\n")
    for rank, col in enumerate(top, start=1):
        pid = inv_prod[col]
        titulo = productos_info.loc[pid]
        print(f"  {rank}. [{fila[col]:.3f}] {str(titulo)[:70]}")

    joblib.dump(
        {
            "svd": svd_final,
            "idx_usuario": idx_usuario,
            "idx_producto": idx_producto,
            "mu": mu,
            "b_u": b_u,
            "b_i": b_i,
            "similitud_item_item": sim,
            "n_componentes": mejor_k,
            "rmse": rmse_modelo,
            "rmse_base": rmse_base,
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "precision_popularidad": precision_pop,
        },
        RUTA_MODELOS / "colaborativo_amazon.pkl",
    )
    print("\nArtefacto guardado: models/colaborativo_amazon.pkl")
    print("Fase 3b completada.")


if __name__ == "__main__":
    main()
