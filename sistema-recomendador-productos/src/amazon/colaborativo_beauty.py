"""
Fase CRISP-DM 4-5: Modelado y Evaluación
Dataset: amazon_beauty (Beauty, 1,021,887 núcleo)
Fundamentos IA/ML: Filtrado colaborativo (SVD + KNN item-item), RMSE, Precision@K, Recall@K
Adaptado de colaborativo_amazon.py para Beauty (prefijo bt_, escala 1-5)
"""
import sys
from pathlib import Path
_RUTA_MODULO = Path(__file__).resolve()
while _RUTA_MODULO.name != "src" and _RUTA_MODULO != _RUTA_MODULO.parent:
    _RUTA_MODULO = _RUTA_MODULO.parent
sys.path.insert(0, str(_RUTA_MODULO.parent))

from src.rutas import RUTA_RAIZ
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error

from src.modelos.v1_colaborativo_svd_knn.colaborativo import (
    calcular_sesgos,
    construir_indices,
    matriz_binaria,
    matriz_residuos,
    similitud_item_item,
)

SEED = 42
CANDIDATOS_K = [2, 4, 8, 16]
TOP_K = 10
UMBRAL_RELEVANTE = 4.0

RUTA_DATOS = RUTA_RAIZ / "data/amazon_beauty"
RUTA_MODELOS = RUTA_RAIZ / "models"


def main() -> None:
    RUTA_MODELOS.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("[1/8] Cargando interacciones Beauty (núcleo 1M)...")
    interacciones = pd.read_csv(RUTA_DATOS / "interacciones.csv")
    productos_info = pd.read_csv(
        RUTA_DATOS / "productos.csv",
        usecols=["product_id", "titulo"],
    ).set_index("product_id")["titulo"]

    # Para 1M interacciones, muestra evaluación más eficiente: muestreo estratificado si es necesario
    # Pero intentamos completo primero (optimizado con sparse)
    usuarios, productos_unicos, idx_usuario, idx_producto = construir_indices(interacciones)
    forma = (len(usuarios), len(productos_unicos))
    print(f"      {len(interacciones)} interacciones | {forma[0]} usuarios | {forma[1]} productos")
    print(f"      Densidad: {len(interacciones)/(forma[0]*forma[1])*100:.4f}%")

    print("[2/8] División train/test 80/20 (CRISP-DM: Preparación)...")
    mascara = rng.random(len(interacciones)) < 0.8
    train = interacciones[mascara].reset_index(drop=True)
    test = interacciones[~mascara].reset_index(drop=True)
    print(f"      train={len(train)} | test={len(test)}")

    print("[3/8] Selección k por validación (85/15 dentro de train)...")
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
        comp_T = svd_k.components_.T  # (n_productos, k)
        # Evitar matriz densa 318k*99k -> calcular solo para val pairs
        mask = np.isfinite(f_val) & np.isfinite(c_val)
        if mask.sum() == 0:
            continue
        f_m = f_val[mask].astype(int)
        c_m = c_val[mask].astype(int)
        # dot producto por par: lat[f] · comp_T[c]
        rec_vals = np.sum(lat[f_m] * comp_T[c_m], axis=1)
        est = np.clip(mu_fit + b_u_fit[f_m] + b_i_fit[c_m] + rec_vals, 1.0, 5.0)
        rmse_val = mean_squared_error(val_df["rating"].to_numpy()[mask], est) ** 0.5
        marca = ""
        if rmse_val < mejor_rmse_val:
            mejor_rmse_val, mejor_k = rmse_val, k
            marca = "  <-- mejor"
        print(f"      k={k:>2} | RMSE validación={rmse_val:.4f}{marca}")

    print(f"[4/8] Entrenando SVD final (k={mejor_k})...")
    mu, b_u, b_i = calcular_sesgos(train, idx_usuario, idx_producto, *forma)
    residuos = matriz_residuos(train, idx_usuario, idx_producto, forma, mu, b_u, b_i)
    svd_final = TruncatedSVD(n_components=mejor_k, random_state=SEED)
    latente = svd_final.fit_transform(residuos)

    f_te = test["user_id"].map(idx_usuario).to_numpy()
    c_te = test["product_id"].map(idx_producto).to_numpy()
    reales = test["rating"].to_numpy()
    mask_te = np.isfinite(f_te) & np.isfinite(c_te)
    f_te_f = f_te[mask_te].astype(int)
    c_te_f = c_te[mask_te].astype(int)
    reales_f = reales[mask_te]
    comp_T_final = svd_final.components_.T
    rec_test = np.sum(latente[f_te_f] * comp_T_final[c_te_f], axis=1)
    estimados = np.clip(mu + b_u[f_te_f] + b_i[c_te_f] + rec_test, 1.0, 5.0)
    rmse_modelo = mean_squared_error(reales_f, estimados) ** 0.5
    rmse_base = mean_squared_error(reales_f, np.full_like(reales_f, mu)) ** 0.5
    print(f"      RMSE línea base (media global {mu:.2f}): {rmse_base:.4f}")
    print(f"      RMSE modelo sesgos+SVD(k={mejor_k}): {rmse_modelo:.4f} ({'supera base' if rmse_modelo < rmse_base else 'no supera base'})")

    print("[5/8] Ranking KNN item-item (coseno) - muestreo si matriz > 20k productos...")
    binaria = matriz_binaria(train, idx_usuario, idx_producto, forma)
    # Si matriz muy grande, limitar ranking a top productos populares para evitar O(n^2)
    MAX_PRODUCTOS_RANKING = 15000
    if forma[1] > MAX_PRODUCTOS_RANKING:
        top_prod_idx = np.argsort(np.asarray(binaria.sum(axis=0)).ravel())[::-1][:MAX_PRODUCTOS_RANKING]
        print(f"      Productos {forma[1]} > {MAX_PRODUCTOS_RANKING}, usando top {MAX_PRODUCTOS_RANKING} populares para ranking")
        # Filtrar binaria a esos productos para sim manejable
        binaria_rank = binaria[:, top_prod_idx]
        sim = similitud_item_item(binaria_rank)
        # Mapear sim de vuelta a espacio original via índices
        # Para ranking, trabajamos en espacio reducido
        usar_subset = True
        subset_idx = top_prod_idx
        inv_subset = {orig: new for new, orig in enumerate(subset_idx)}
    else:
        sim = similitud_item_item(binaria)
        usar_subset = False

    # Preparar estructuras para ranking, manejando subset si existe
    if usar_subset:
        # Trabajar en espacio reducido para sim (15k productos)
        compradas_train = [set() for _ in range(forma[0])]
        coo = binaria_rank.tocoo()
        for u, p in zip(coo.row, coo.col):
            compradas_train[u].add(p)
        # relevantes solo si producto está en subset
        relevantes_test: dict[int, set[int]] = {}
        for u, p_orig, r in zip(f_te_f, c_te_f, reales_f):
            if r >= UMBRAL_RELEVANTE and p_orig in inv_subset:
                relevantes_test.setdefault(u, set()).add(inv_subset[p_orig])
        binaria_eff = binaria_rank
    else:
        compradas_train = [set() for _ in range(forma[0])]
        coo = binaria.tocoo()
        for u, p in zip(coo.row, coo.col):
            compradas_train[u].add(p)
        relevantes_test: dict[int, set[int]] = {}
        for u, p, r in zip(f_te_f, c_te_f, reales_f):
            if r >= UMBRAL_RELEVANTE:
                relevantes_test.setdefault(u, set()).add(p)
        binaria_eff = binaria

    print(f"[6/8] Métricas ranking (relevante >= {UMBRAL_RELEVANTE})...")
    precisions, recalls = [], []
    usuarios_evaluables = sorted(relevantes_test.keys())
    max_eval = 5000
    if len(usuarios_evaluables) > max_eval:
        rng_eval = np.random.default_rng(SEED)
        usuarios_evaluables = sorted(rng_eval.choice(usuarios_evaluables, size=max_eval, replace=False).tolist())
        print(f"      Muestreo evaluables: {max_eval}/{len(relevantes_test)} para acelerar")
    for u in usuarios_evaluables:
        if not compradas_train[u]:
            continue
        fila = np.asarray((binaria_eff[u] @ sim).todense()).ravel()
        # Excluir compradas
        compradas = list(compradas_train[u])
        if compradas:
            fila[compradas] = -np.inf
        top = np.argpartition(fila, -TOP_K)[-TOP_K:]
        aciertos = len(relevantes_test[u].intersection(top))
        precisions.append(aciertos / TOP_K)
        recalls.append(aciertos / len(relevantes_test[u]) if len(relevantes_test[u]) else 0)
    precision_at_k = float(np.mean(precisions)) if precisions else 0.0
    recall_at_k = float(np.mean(recalls)) if recalls else 0.0
    print(f"      Precision@{TOP_K}: {precision_at_k:.4f}")
    print(f"      Recall@{TOP_K}:    {recall_at_k:.4f}")
    print(f"      Usuarios evaluables: {len(precisions)}/{forma[0]} (subset={usar_subset})")

    pop = np.asarray(binaria_eff.sum(axis=0)).ravel()
    precisions_pop = []
    for u in usuarios_evaluables:
        if not compradas_train[u]:
            continue
        fila = pop.copy().astype(np.float64)
        fila[list(compradas_train[u])] = -np.inf
        top = np.argpartition(fila, -TOP_K)[-TOP_K:]
        precisions_pop.append(len(relevantes_test[u].intersection(top)) / TOP_K)
    precision_pop = float(np.mean(precisions_pop)) if precisions_pop else 0.0
    print(f"      (baseline popularidad: Precision@{TOP_K}={precision_pop:.4f})")

    print("[7/8] Comparación Beauty vs Amazon Fashion vs Sintético:")
    try:
        sintetico = joblib.load(RUTA_MODELOS / "colaborativo_svd.pkl")
        amazon = joblib.load(RUTA_MODELOS / "colaborativo_amazon.pkl")
        comp = pd.DataFrame({
            "Métrica": ["RMSE", "Precision@10", "Recall@10"],
            "Sintético (supermercado)": [round(sintetico["rmse"],4), round(sintetico["precision_at_k"],4), round(sintetico["recall_at_k"],4)],
            "Amazon Fashion": [round(amazon["rmse"],4), round(amazon["precision_at_k"],4), round(amazon["recall_at_k"],4)],
            "Beauty (Kaggle)": [round(rmse_modelo,4), round(precision_at_k,4), round(recall_at_k,4)],
        })
        print(comp.to_string(index=False))
    except Exception as e:
        print(f"      (no se pudo cargar comparativa: {e})")

    print("[8/8] DEMO Top-5 usuario Beauty activo:")
    conteo = train["user_id"].value_counts()
    usuario_demo = conteo.index[0]
    u_demo = idx_usuario[usuario_demo]
    fila = np.asarray((binaria_eff[u_demo] @ sim).todense()).ravel()
    fila[list(compradas_train[u_demo])] = -np.inf
    top = np.argsort(fila)[::-1][:5]
    inv_prod = {v: k for k, v in idx_producto.items()}
    if usar_subset:
        # mapear índices subset -> originales
        inv_prod_eff = {new: productos_unicos[orig] for new, orig in enumerate(subset_idx)}
        print(f"\nUsuario {usuario_demo} ({conteo.iloc[0]} compras train) - subset top-productos:\n")
        for rank, col in enumerate(top, start=1):
            pid = inv_prod_eff[col]
            print(f"  {rank}. [{fila[col]:.3f}] {pid}")
    else:
        print(f"\nUsuario {usuario_demo} ({conteo.iloc[0]} compras train):\n")
        for rank, col in enumerate(top, start=1):
            pid = inv_prod[col]
            print(f"  {rank}. [{fila[col]:.3f}] {pid}")

    joblib.dump({
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
    }, RUTA_MODELOS / "colaborativo_beauty.pkl")
    print("\nArtefacto guardado: models/colaborativo_beauty.pkl")
    print("Fase Beauty completada (CRISP-DM: Evaluación).")


if __name__ == "__main__":
    main()
