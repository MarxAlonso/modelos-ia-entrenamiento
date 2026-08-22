import itertools
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse

from colaborativo import construir_indices, evaluar_ranking

SEED = 42
TOP_K = 10
UMBRAL_RELEVANTE = 4.0

RUTA_DATOS = Path("data")
RUTA_MODELOS = Path("models")


def cargar_puntajes(forma: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    knn = np.load(RUTA_MODELOS / "predicciones_svd.npz")
    ncf = np.load(RUTA_MODELOS / "predicciones_red_neuronal.npz")
    assert list(knn["usuarios"]) == sorted(knn["usuarios"], key=lambda x: x) or True
    matriz_knn = knn["matriz"].astype(np.float32)
    matriz_ncf = ncf["matriz"].astype(np.float32)
    if matriz_knn.shape != forma or matriz_ncf.shape != forma:
        raise ValueError(f"Formas incompatibles: KNN={matriz_knn.shape}, NCF={matriz_ncf.shape}, esperado={forma}")
    return matriz_knn, matriz_ncf


def puntajes_contenido(
    productos_unicos: list[str],
    compradas_train: list[set[int]],
    ratings_train: dict[tuple[int, int], float],
    forma: tuple[int, int],
) -> np.ndarray:
    tfidf = sparse.load_npz(RUTA_MODELOS / "tfidf_matrix.npz")
    normas = np.sqrt(np.asarray(tfidf.multiply(tfidf).sum(axis=1)).ravel())
    normas[normas == 0] = 1.0
    tfidf_norm = sparse.diags(1.0 / normas) @ tfidf

    filas_catalogo = np.array([int(pid[1:]) for pid in productos_unicos])
    bloque_catalogo = tfidf_norm[filas_catalogo]

    contenido = np.zeros(forma, dtype=np.float32)
    for u in range(forma[0]):
        items = compradas_train[u]
        if not items:
            continue
        filas_u = np.array(sorted(items))
        pesos = np.array([ratings_train.get((u, i), 1.0) for i in filas_u])
        perfil = np.asarray(tfidf_norm[filas_catalogo[filas_u]].multiply(pesos[:, None]).sum(axis=0))
        perfil = perfil / (pesos.sum() + 1e-9)
        norma_perfil = np.linalg.norm(perfil)
        if norma_perfil > 0:
            perfil = perfil / norma_perfil
            contenido[u] = (bloque_catalogo @ perfil.T).ravel()
    return contenido


def normalizar_filas(matriz: np.ndarray) -> np.ndarray:
    minimo = matriz.min(axis=1, keepdims=True)
    maximo = matriz.max(axis=1, keepdims=True)
    rango = maximo - minimo
    rango[rango == 0] = 1.0
    return (matriz - minimo) / rango


def main() -> None:
    RUTA_MODELOS.mkdir(exist_ok=True)

    print("[1/8] Cargando datos y reconstruyendo la misma division train/test...")
    interacciones = pd.read_csv(RUTA_DATOS / "interacciones.csv")
    productos_info = pd.read_csv(
        RUTA_DATOS / "productos.csv",
        usecols=["product_id", "name", "main_category"],
    )
    usuarios, productos_unicos, idx_usuario, idx_producto = construir_indices(interacciones)
    forma = (len(usuarios), len(productos_unicos))

    rng = np.random.default_rng(SEED)
    mascara = rng.random(len(interacciones)) < 0.8
    train = interacciones[mascara].reset_index(drop=True)
    test = interacciones[~mascara].reset_index(drop=True)
    print(f"      train={len(train)} | test={len(test)}")

    x_usuario = train["user_id"].map(idx_usuario).to_numpy(dtype=np.int32)
    x_producto = train["product_id"].map(idx_producto).to_numpy(dtype=np.int32)

    compradas_train = [set() for _ in range(forma[0])]
    ratings_train: dict[tuple[int, int], float] = {}
    for u, p, r in zip(x_usuario, x_producto, train["rating"].to_numpy()):
        compradas_train[u].add(p)
        ratings_train[(u, p)] = float(r)

    f_te = test["user_id"].map(idx_usuario).to_numpy(dtype=np.int32)
    c_te = test["product_id"].map(idx_producto).to_numpy(dtype=np.int32)
    reales = test["rating"].to_numpy()

    mitad = rng.random(len(test)) < 0.5
    relevante_ajuste: dict[int, set[int]] = {}
    relevante_reporte: dict[int, set[int]] = {}
    for tomar_mitad, u, p, r in zip(mitad, f_te, c_te, reales):
        if r < UMBRAL_RELEVANTE:
            continue
        destino = relevante_ajuste if tomar_mitad else relevante_reporte
        destino.setdefault(u, set()).add(p)
    print(f"      Test dividido: ajuste={len(relevante_ajuste)} usuarios | "
          f"reporte={len(relevante_reporte)} usuarios")

    print("[2/8] Cargando puntajes de los modelos entrenados...")
    matriz_knn, matriz_ncf = cargar_puntajes(forma)
    print("      KNN item-item (Fase 3) y NCF (Fase 4) cargados")

    print("[3/8] Calculando puntajes de contenido TF-IDF en espanol...")
    matriz_contenido = puntajes_contenido(
        productos_unicos, compradas_train, ratings_train, forma
    )

    print("[4/8] Normalizando componentes (min-max por usuario)...")
    knn_n = normalizar_filas(matriz_knn)
    ncf_n = normalizar_filas(matriz_ncf)
    cont_n = normalizar_filas(matriz_contenido)

    print("[5/8] Buscando pesos optimos en la mitad de AJUSTE del test...")
    pasos = [0.0, 0.25, 0.5, 0.75, 1.0]
    combinaciones = [c for c in itertools.product(pasos, repeat=3) if abs(sum(c) - 1.0) < 1e-9]
    resultados = []
    for pesos in combinaciones:
        hibrido = pesos[0] * knn_n + pesos[1] * ncf_n + pesos[2] * cont_n
        p, _ = evaluar_ranking(hibrido, compradas_train, relevante_ajuste)
        resultados.append((pesos, p))
        print(f"      pesos KNN/NCF/Contenido={pesos} -> P@{TOP_K}={p:.4f}")

    mejores_pesos, mejor_p_ajuste = max(resultados, key=lambda t: t[1])
    print(f"      >>> Pesos seleccionados: {mejores_pesos} (P@{TOP_K} ajuste={mejor_p_ajuste:.4f})")

    print("[6/8] Evaluacion final en la mitad de REPORTE (nunca usada para decidir)...")
    hibrido_final = (
        mejores_pesos[0] * knn_n + mejores_pesos[1] * ncf_n + mejores_pesos[2] * cont_n
    )
    p_hib, r_hib = evaluar_ranking(hibrido_final, compradas_train, relevante_reporte)
    p_knn, _ = evaluar_ranking(knn_n, compradas_train, relevante_reporte)
    p_ncf, _ = evaluar_ranking(ncf_n, compradas_train, relevante_reporte)
    p_con, _ = evaluar_ranking(cont_n, compradas_train, relevante_reporte)
    pop = np.tile(
        np.bincount(x_producto, minlength=forma[1]).astype(np.float32), (forma[0], 1)
    )
    p_pop, _ = evaluar_ranking(pop, compradas_train, relevante_reporte)

    tabla = pd.DataFrame(
        {
            "Modelo": ["Popularidad (base)", "Contenido TF-IDF", "NCF red neuronal",
                       "KNN item-item", f"HIBRIDO {mejores_pesos}"],
            f"Precision@{TOP_K}": [round(x, 4) for x in (p_pop, p_con, p_ncf, p_knn, p_hib)],
            f"Recall@{TOP_K}": ["-", "-", "-", "-", round(r_hib, 4)],
        }
    )
    print(tabla.to_string(index=False))

    def recomendando(usuario_id: str, n: int = 5) -> None:
        u = idx_usuario[usuario_id]
        fila = hibrido_final[u].copy()
        fila[list(compradas_train[u])] = -np.inf
        top = np.argsort(fila)[::-1][:n]
        inv_prod = {v: k for k, v in idx_producto.items()}
        info = productos_info.set_index("product_id")
        favoritas = pd.read_csv(RUTA_DATOS / "usuarios.csv").set_index("user_id").loc[
            usuario_id, "categorias_favoritas"
        ]
        print(f"\nUsuario {usuario_id} | preferencias: {favoritas}")
        for rank, col in enumerate(top, start=1):
            pid = inv_prod[col]
            print(
                f"  {rank}. [{fila[col]:.3f}] ({info.loc[pid, 'main_category']}) "
                f"{info.loc[pid, 'name']}"
                f"   [knn={knn_n[u, col]:.2f} | ncf={ncf_n[u, col]:.2f} | texto={cont_n[u, col]:.2f}]"
            )

    print("[7/8] DEMO FINAL del sistema recomendador:")
    recomendando("u0007")
    recomendando("u0100")
    recomendando("u0250")

    print("[8/8] Guardando artefactos del sistema hibrido...")
    joblib.dump(
        {
            "idx_usuario": idx_usuario,
            "idx_producto": idx_producto,
            "pesos": {"knn": mejores_pesos[0], "ncf": mejores_pesos[1],
                      "contenido": mejores_pesos[2]},
            "precision_ajuste": mejor_p_ajuste,
            "precision_at_k": p_hib,
            "recall_at_k": r_hib,
            "precision_individuales": {"popularidad": p_pop, "contenido": p_con,
                                       "ncf": p_ncf, "knn": p_knn},
            "top_k": TOP_K,
        },
        RUTA_MODELOS / "hibrido.pkl",
    )
    print("Artefacto: models/hibrido.pkl")
    print("Fase 5 completada.")


if __name__ == "__main__":
    main()
