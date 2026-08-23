
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
from pathlib import Path
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error

SEED = 42
CANDIDATOS_K = [2, 4, 8, 16]
TOP_K = 10
UMBRAL_RELEVANTE = 4.0

RUTA_DATOS = RUTA_RAIZ / "data"
RUTA_MODELOS = RUTA_RAIZ / "models"


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    interacciones = pd.read_csv(RUTA_DATOS / "interacciones.csv")
    productos = pd.read_csv(RUTA_DATOS / "productos.csv", usecols=["product_id", "name"])
    return interacciones, productos


def construir_indices(
    interacciones: pd.DataFrame,
) -> tuple[list[str], list[str], dict[str, int], dict[str, int]]:
    usuarios = sorted(interacciones["user_id"].unique())
    productos_unicos = sorted(interacciones["product_id"].unique())
    idx_usuario = {u: i for i, u in enumerate(usuarios)}
    idx_producto = {p: i for i, p in enumerate(productos_unicos)}
    return usuarios, productos_unicos, idx_usuario, idx_producto


def calcular_sesgos(
    df: pd.DataFrame,
    idx_usuario: dict[str, int],
    idx_producto: dict[str, int],
    n_usuarios: int,
    n_productos: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    mu = float(df["rating"].mean())
    b_u = np.zeros(n_usuarios)
    stats_u = df.groupby("user_id")["rating"].agg(["sum", "count"])
    for usuario, fila in stats_u.iterrows():
        b_u[idx_usuario[usuario]] = (fila["sum"] - mu * fila["count"]) / (fila["count"] + 10)
    residuo = df["rating"].to_numpy() - mu - b_u[df["user_id"].map(idx_usuario).to_numpy()]
    tmp = df[["product_id"]].copy()
    tmp["residuo"] = residuo
    b_i = np.zeros(n_productos)
    stats_i = tmp.groupby("product_id")["residuo"].agg(["sum", "count"])
    for producto, fila in stats_i.iterrows():
        b_i[idx_producto[producto]] = fila["sum"] / (fila["count"] + 10)
    return mu, b_u, b_i


def matriz_residuos(
    df: pd.DataFrame,
    idx_usuario: dict[str, int],
    idx_producto: dict[str, int],
    forma: tuple[int, int],
    mu: float,
    b_u: np.ndarray,
    b_i: np.ndarray,
) -> sparse.csr_matrix:
    filas = df["user_id"].map(idx_usuario).to_numpy()
    columnas = df["product_id"].map(idx_producto).to_numpy()
    valores = (
        df["rating"].to_numpy(dtype=np.float32) - mu - b_u[filas] - b_i[columnas]
    ).astype(np.float32)
    return sparse.csr_matrix((valores, (filas, columnas)), shape=forma)


def matriz_binaria(
    df: pd.DataFrame,
    idx_usuario: dict[str, int],
    idx_producto: dict[str, int],
    forma: tuple[int, int],
) -> sparse.csr_matrix:
    filas = df["user_id"].map(idx_usuario).to_numpy()
    columnas = df["product_id"].map(idx_producto).to_numpy()
    unos = np.ones(len(df), dtype=np.float32)
    return sparse.csr_matrix((unos, (filas, columnas)), shape=forma)


def similitud_item_item(matriz_binaria_usuarios_items: sparse.csr_matrix) -> sparse.csr_matrix:
    normas = np.sqrt(np.asarray(matriz_binaria_usuarios_items.multiply(
        matriz_binaria_usuarios_items
    ).sum(axis=0)).ravel())
    normas[normas == 0] = 1.0
    dinv = sparse.diags(1.0 / normas)
    return (dinv @ (matriz_binaria_usuarios_items.T @ matriz_binaria_usuarios_items) @ dinv).tocsr()


def evaluar_ranking(
    puntajes: np.ndarray,
    compradas_train: list[set[int]],
    relevantes_test: dict[int, set[int]],
    top_k: int = TOP_K,
) -> tuple[float, float]:
    precisions, recalls = [], []
    for u, relevantes in relevantes_test.items():
        if not relevantes:
            continue
        fila = puntajes[u].copy()
        fila[list(compradas_train[u])] = -np.inf
        top = np.argpartition(fila, -top_k)[-top_k:]
        aciertos = len(relevantes.intersection(top))
        precisions.append(aciertos / top_k)
        recalls.append(aciertos / len(relevantes))
    return float(np.mean(precisions)), float(np.mean(recalls))


def main() -> None:
    RUTA_MODELOS.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("[1/8] Cargando datos...")
    interacciones, nombres_productos = cargar_datos()
    usuarios, productos_unicos, idx_usuario, idx_producto = construir_indices(interacciones)
    forma = (len(usuarios), len(productos_unicos))
    print(f"      {len(interacciones)} compras | {forma[0]} usuarios | {forma[1]} productos")

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
    print(f"      k seleccionado: {mejor_k}")

    print(f"[4/8] Entrenando SVD final sobre todo el train (k={mejor_k})...")
    mu, b_u, b_i = calcular_sesgos(train, idx_usuario, idx_producto, *forma)
    residuos = matriz_residuos(train, idx_usuario, idx_producto, forma, mu, b_u, b_i)
    svd_final = TruncatedSVD(n_components=mejor_k, random_state=SEED)
    latente = svd_final.fit_transform(residuos)
    reconstruccion = latente @ svd_final.components_

    f_te = test["user_id"].map(idx_usuario).to_numpy()
    c_te = test["product_id"].map(idx_producto).to_numpy()
    reales = test["rating"].to_numpy()
    estimados = np.clip(mu + b_u[f_te] + b_i[c_te] + reconstruccion[f_te, c_te], 1.0, 5.0)
    rmse_modelo = mean_squared_error(reales, estimados) ** 0.5
    rmse_base = mean_squared_error(reales, np.full_like(reales, mu)) ** 0.5
    print(f"      RMSE linea base (predecir media global): {rmse_base:.4f}")
    print(f"      RMSE modelo sesgos+SVD(k={mejor_k}):      {rmse_modelo:.4f}")

    print("[5/8] Construyendo ranking KNN item-item (coseno binario)...")
    binaria = matriz_binaria(train, idx_usuario, idx_producto, forma)
    sim = similitud_item_item(binaria)
    puntajes_ranking = np.asarray((binaria @ sim).todense(), dtype=np.float32)

    compradas_train = [set() for _ in range(forma[0])]
    for u, p in zip(binaria.tocoo().row, binaria.tocoo().col):
        compradas_train[u].add(p)

    relevantes_test: dict[int, set[int]] = {}
    for u, p, r in zip(f_te, c_te, reales):
        if r >= UMBRAL_RELEVANTE:
            relevantes_test.setdefault(u, set()).add(p)

    print(f"[6/8] Metricas de ranking (relevante = rating>={UMBRAL_RELEVANTE})...")
    precision_at_k, recall_at_k = evaluar_ranking(puntajes_ranking, compradas_train, relevantes_test)
    pop = np.tile(np.asarray(binaria.sum(axis=0)).ravel(), (forma[0], 1)).astype(np.float32)
    precision_pop, _ = evaluar_ranking(pop, compradas_train, relevantes_test)
    print(f"      Precision@{TOP_K}: {precision_at_k:.4f}")
    print(f"      Recall@{TOP_K}:    {recall_at_k:.4f}")
    print(f"      (linea base popularidad: Precision@{TOP_K}={precision_pop:.4f})")

    print(f"[7/8] DEMO - Top-5 para u0007:")
    u_demo = idx_usuario["u0007"]
    fila_demo = puntajes_ranking[u_demo].copy()
    fila_demo[list(compradas_train[u_demo])] = -np.inf
    top = np.argsort(fila_demo)[::-1][:5]
    inv_prod = {v: k for k, v in idx_producto.items()}
    nombres = nombres_productos.set_index("product_id")["name"]
    categorias = pd.read_csv(RUTA_DATOS / "productos.csv", usecols=["product_id", "main_category"])
    cats = categorias.set_index("product_id")["main_category"]
    print("\n")
    for rank, col in enumerate(top, start=1):
        pid = inv_prod[col]
        print(f"  {rank}. [{fila_demo[col]:.3f}] ({cats.loc[pid]}) {nombres.loc[pid]}")

    print("[8/8] Guardando artefactos...")
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
        RUTA_MODELOS / "colaborativo_svd.pkl",
    )
    np.savez_compressed(
        RUTA_MODELOS / "predicciones_svd.npz",
        matriz=puntajes_ranking,
        usuarios=np.array(usuarios),
        productos=np.array(productos_unicos),
    )
    print("\nArtefactos: models/colaborativo_svd.pkl, models/predicciones_svd.npz")
    print("Fase 3 completada.")


if __name__ == "__main__":
    main()
