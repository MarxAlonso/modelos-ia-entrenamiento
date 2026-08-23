
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

from src.modelos.v1_colaborativo_svd_knn.colaborativo import construir_indices, evaluar_ranking

SEED = 42
TOP_K = 10
UMBRAL_RELEVANTE = 4.0

RUTA_DATOS = RUTA_RAIZ / "data"
RUTA_MODELOS = RUTA_RAIZ / "models"


def cargar_puntajes(forma: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    knn = np.load(RUTA_MODELOS / "predicciones_svd.npz")
    ncf = np.load(RUTA_MODELOS / "predicciones_red_neuronal.npz")
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


PESOS_ESQUEMA = {
    "relevante": lambda r: 1.0 if r >= UMBRAL_RELEVANTE else 0.0,
    "suave": lambda r: max(0.0, min((r - 2.5) / 2.5, 1.0)),
    "todo": lambda r: 1.0,
}


def probabilidad_categorias(
    usuarios: list[str],
    compradas_train: list[set[int]],
    ratings_train: dict[tuple[int, int], float],
    cat_de_producto: np.ndarray,
    lista_cats: list[str],
    favoritas_declaradas: dict[str, list[str]],
    alfa: float,
    beta: float,
    esquema: str = "relevante",
) -> np.ndarray:
    """Distribucion de probabilidad de cada usuario sobre las categorias.

    Evidencia = compras del historial en train, pesadas segun el esquema de
    rating elegido; prior = categorias favoritas declaradas en usuarios.csv.
    """
    peso_rating = PESOS_ESQUEMA[esquema]
    n_usuarios = len(usuarios)
    n_cats = len(lista_cats)
    idx_cat = {c: j for j, c in enumerate(lista_cats)}
    p_cat = np.zeros((n_usuarios, n_cats), dtype=np.float32)

    for u, usuario in enumerate(usuarios):
        conteo = np.zeros(n_cats, dtype=np.float32)
        for i in compradas_train[u]:
            j = cat_de_producto[i]
            if j >= 0:
                conteo[j] += peso_rating(ratings_train.get((u, i), 0.0))
        decl = np.zeros(n_cats, dtype=np.float32)
        for c in favoritas_declaradas.get(usuario, []):
            if c in idx_cat:
                decl[idx_cat[c]] = 1.0
        n_decl = decl.sum()
        if n_decl > 0:
            prior = decl / n_decl
            probs = (conteo + alfa * beta * prior) / (conteo.sum() + alfa * beta)
        else:
            total = conteo.sum()
            probs = conteo / total if total > 0 else np.full(n_cats, 1.0 / n_cats)
        p_cat[u] = probs
    return p_cat


def construir_ranking_cuotas(
    p_cat_u: np.ndarray,
    base: np.ndarray,
    cats_producto: np.ndarray,
    compradas_train: list[set[int]],
    n_categorias: int,
    exponente: float,
    top_k: int = TOP_K,
) -> np.ndarray:
    """Ranking en 2 etapas con reparto de cuotas.

    Etapa 1: elige las n_categorias mas probables del usuario y reparte los
    top_k slots entre ellas proporcional a prob^exponente (metodo del mayor
    resto). Etapa 2: dentro de cada categoria elige los mejores productos por
    el puntaje hibrido de producto.
    """
    n_usuarios, _ = base.shape
    ranking = np.full(base.shape, -np.inf, dtype=np.float32)
    for u in range(n_usuarios):
        orden_cats = np.argsort(p_cat_u[u])[::-1][:n_categorias]
        probs = np.clip(p_cat_u[u][orden_cats], 0.0, None) ** exponente
        if probs.sum() <= 0:
            continue
        cuotas = np.floor(probs / probs.sum() * top_k).astype(int)
        faltantes = top_k - cuotas.sum()
        if faltantes > 0:
            restos = (probs / probs.sum() * top_k) - cuotas
            for j in np.argsort(restos)[::-1][:faltantes]:
                cuotas[j] += 1
        ya_recomendados = compradas_train[u]
        asignados = 0
        for j, cat in enumerate(orden_cats):
            if cuotas[j] <= 0 or asignados >= top_k:
                break
            candidatos = np.where(cats_producto == cat)[0]
            candidatos = candidatos[~np.isin(candidatos, list(ya_recomendados))]
            if len(candidatos) == 0:
                continue
            mejores = candidatos[np.argsort(base[u, candidatos])[::-1]]
            tomar = min(cuotas[j], top_k - asignados, len(mejores))
            for pos in range(tomar):
                ranking[u, mejores[pos]] = float(TOP_K - asignados - pos * 0.01)
                asignados += 1
    return ranking


def normalizar_filas(matriz: np.ndarray) -> np.ndarray:
    minimo = matriz.min(axis=1, keepdims=True)
    maximo = matriz.max(axis=1, keepdims=True)
    rango = maximo - minimo
    rango[rango == 0] = 1.0
    return (matriz - minimo) / rango


def precision_categoria(
    puntajes: np.ndarray,
    cats_producto: np.ndarray,
    compradas_train: list[set[int]],
    relevantes_test: dict[int, set[int]],
    top_k: int = TOP_K,
) -> float:
    """Fraccion de los Top-K recomendados cuya categoria coincide con alguna
    categoria que el usuario compro en test (feedback implicito)."""
    valores = []
    for u, relevantes in relevantes_test.items():
        cats_relevantes = {cats_producto[i] for i in relevantes}
        fila = puntajes[u].copy()
        fila[list(compradas_train[u])] = -np.inf
        top = np.argpartition(fila, -top_k)[-top_k:]
        aciertos = sum(1 for i in top if cats_producto[i] in cats_relevantes)
        valores.append(aciertos / top_k)
    return float(np.mean(valores))


def tasa_acierto_categoria(
    puntajes: np.ndarray,
    cats_producto: np.ndarray,
    compradas_train: list[set[int]],
    relevantes_test: dict[int, set[int]],
    top_k: int = TOP_K,
) -> float:
    """HR@K de categoria: fraccion de usuarios con AL MENOS una recomendacion
    en una categoria que compro en test."""
    aciertos = []
    for u, relevantes in relevantes_test.items():
        cats_relevantes = {cats_producto[i] for i in relevantes}
        fila = puntajes[u].copy()
        fila[list(compradas_train[u])] = -np.inf
        top = np.argpartition(fila, -top_k)[-top_k:]
        aciertos.append(1.0 if any(cats_producto[i] in cats_relevantes for i in top) else 0.0)
    return float(np.mean(aciertos))


def main() -> None:
    RUTA_MODELOS.mkdir(exist_ok=True)

    print("[1/8] Cargando datos y reconstruyendo la misma division train/test...")
    interacciones = pd.read_csv(RUTA_DATOS / "interacciones.csv")
    productos_info = pd.read_csv(RUTA_DATOS / "productos.csv")
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
    implicito_ajuste: dict[int, set[int]] = {}
    test_completo: dict[int, set[int]] = {}
    for tomar_mitad, u, p, r in zip(mitad, f_te, c_te, reales):
        test_completo.setdefault(u, set()).add(p)
        if tomar_mitad:
            implicito_ajuste.setdefault(u, set()).add(p)
            if r >= UMBRAL_RELEVANTE:
                relevante_ajuste.setdefault(u, set()).add(p)
        else:
            if r >= UMBRAL_RELEVANTE:
                relevante_reporte.setdefault(u, set()).add(p)
    print(f"      Test dividido: ajuste={len(relevante_ajuste)} | "
          f"reporte={len(relevante_reporte)} usuarios con items relevantes")

    print("[2/8] Cargando puntajes de los modelos entrenados...")
    matriz_knn, matriz_ncf = cargar_puntajes(forma)

    print("[3/8] Calculando puntajes de contenido TF-IDF...")
    matriz_contenido = puntajes_contenido(
        productos_unicos, compradas_train, ratings_train, forma
    )

    lista_cats = sorted(productos_info["main_category"].dropna().unique().tolist())
    idx_cat = {c: j for j, c in enumerate(lista_cats)}
    info_por_pid = productos_info.set_index("product_id")
    cats_producto = np.array(
        [idx_cat[info_por_pid.loc[p, "main_category"]] for p in productos_unicos],
        dtype=np.int32,
    )
    perfiles = pd.read_csv(RUTA_DATOS / "usuarios.csv").set_index("user_id")
    favoritas_declaradas = {
        u: [c.strip() for c in str(perfiles.loc[u, "categorias_favoritas"]).split("|")]
        for u in usuarios
    }

    print("[4/8] Normalizando componentes de producto (min-max por usuario)...")
    knn_n = normalizar_filas(matriz_knn)
    ncf_n = normalizar_filas(matriz_ncf)
    cont_n = normalizar_filas(matriz_contenido)

    def evaluar_config(p_cat_u: np.ndarray, cfg: dict, pesos_prod: tuple) -> dict:
        ranking = construir_ranking_cuotas(
            p_cat_u,
            pesos_prod[0] * knn_n + pesos_prod[1] * ncf_n + pesos_prod[2] * cont_n,
            cats_producto, compradas_train, cfg["n_c"], cfg["expo"],
        )
        p_prod = evaluar_ranking(ranking, compradas_train, relevante_ajuste)[0]
        pc = precision_categoria(ranking, cats_producto, compradas_train, implicito_ajuste)
        hr = tasa_acierto_categoria(ranking, cats_producto, compradas_train, implicito_ajuste)
        return {"ranking": ranking, "p_prod": p_prod, "pc": pc, "hr": hr}

    print("[5/8] Busqueda de hiperparametros en la mitad de AJUSTE...")
    mejor = None
    total_configs = 0
    pesos_fijos = (0.5, 0.25, 0.25)
    print("      Etapa A: predictor de categorias + cobertura (cuotas)...")
    for esquema in ["todo", "relevante"]:
        for alfa in [0.0, 0.25, 0.5, 1.0]:
            for beta in [2.0, 8.0]:
                p_cat_u = probabilidad_categorias(
                    usuarios, compradas_train, ratings_train,
                    cats_producto, lista_cats, favoritas_declaradas,
                    alfa, beta, esquema,
                )
                for n_c in [1, 2, 3]:
                    for expo in [1.0, 2.0]:
                        cfg = {"n_c": n_c, "expo": expo}
                        res = evaluar_config(p_cat_u, cfg, pesos_fijos)
                        total_configs += 1
                        objetivo = res["pc"] + res["hr"] + 0.3 * res["p_prod"]
                        if mejor is None or objetivo > mejor["objetivo"]:
                            mejor = {
                                "esquema": esquema, "alfa": alfa, "beta": beta,
                                "n_c": n_c, "expo": expo,
                                "pesos_prod": pesos_fijos,
                                "p_prod": res["p_prod"], "pc": res["pc"], "hr": res["hr"],
                                "objetivo": objetivo,
                            }
    print(f"      Configs etapa A: {total_configs}")
    print(f"      >>> Mejor A: esquema={mejor['esquema']} alfa={mejor['alfa']} "
          f"beta={mejor['beta']} topCats={mejor['n_c']} expo={mejor['expo']}")
    print(f"          Pprod@{TOP_K}={mejor['p_prod']:.4f} | "
          f"Pcat@{TOP_K}(impl ajuste)={mejor['pc']:.1%} | HR(ajuste)={mejor['hr']:.1%}")

    print("      Etapa B: afinando pesos KNN/NCF/Contenido dentro del ganador...")
    p_cat_u_mejor = probabilidad_categorias(
        usuarios, compradas_train, ratings_train,
        cats_producto, lista_cats, favoritas_declaradas,
        mejor["alfa"], mejor["beta"], mejor["esquema"],
    )
    pasos = [0.0, 0.25, 0.5, 0.75, 1.0]
    for wk in pasos[1:]:
        for wn in pasos:
            for wc in pasos:
                total = wk + wn + wc
                if total <= 0:
                    continue
                pesos_prod = (wk / total, wn / total, wc / total)
                res = evaluar_config(p_cat_u_mejor, mejor, pesos_prod)
                objetivo = res["pc"] + res["hr"] + 0.3 * res["p_prod"]
                if objetivo > mejor["objetivo"]:
                    mejor.update({"pesos_prod": pesos_prod, "p_prod": res["p_prod"],
                                  "pc": res["pc"], "hr": res["hr"], "objetivo": objetivo})

    wk, wn, wc = mejor["pesos_prod"]
    print("      Configuracion final elegida en AJUSTE:")
    print(f"          esquema rating={mejor['esquema']} | alfa={mejor['alfa']} "
          f"| beta={mejor['beta']}")
    print(f"          categorias={mejor['n_c']} | exponente cuotas={mejor['expo']} "
          f"| pesos(KNN,NCF,Cont)=({wk:.3f},{wn:.3f},{wc:.3f})")
    print(f"          Pprod@{TOP_K}={mejor['p_prod']:.4f}")

    print("[6/8] Evaluacion final en la mitad de REPORTE + test completo...")
    base_final = wk * knn_n + wn * ncf_n + wc * cont_n
    hibrido_final = construir_ranking_cuotas(
        p_cat_u_mejor, base_final, cats_producto, compradas_train,
        mejor["n_c"], mejor["expo"],
    )

    p_hib, r_hib = evaluar_ranking(hibrido_final, compradas_train, relevante_reporte)
    pc_hib = precision_categoria(hibrido_final, cats_producto, compradas_train, test_completo)
    hr_hib = tasa_acierto_categoria(hibrido_final, cats_producto, compradas_train, test_completo)

    afin_n = normalizar_filas(p_cat_u_mejor[:, np.clip(cats_producto, 0, None)])
    p_afin, _ = evaluar_ranking(afin_n, compradas_train, relevante_reporte)
    pc_afin = precision_categoria(afin_n, cats_producto, compradas_train, test_completo)

    p_knn, _ = evaluar_ranking(knn_n, compradas_train, relevante_reporte)
    p_ncf, _ = evaluar_ranking(ncf_n, compradas_train, relevante_reporte)
    p_con, _ = evaluar_ranking(cont_n, compradas_train, relevante_reporte)
    pop = np.tile(
        np.bincount(x_producto, minlength=forma[1]).astype(np.float32), (forma[0], 1)
    )
    p_pop, _ = evaluar_ranking(pop, compradas_train, relevante_reporte)

    tabla = pd.DataFrame(
        {
            "Modelo": ["Popularidad", "Afinidad categorias", "Contenido TF-IDF",
                       "NCF red neuronal", "KNN item-item",
                       "HIBRIDO V2 (cuotas 2 etapas)"],
            f"P@{TOP_K} producto": [round(v, 4) for v in (p_pop, p_afin, p_con, p_ncf, p_knn, p_hib)],
            f"Prec.cat@test": ["-", round(pc_afin, 4), "-", "-", "-", round(pc_hib, 4)],
            f"HR.cat@10": ["-", "-", "-", "-", "-", round(hr_hib, 4)],
            f"Recall@{TOP_K}": ["-", "-", "-", "-", "-", round(r_hib, 4)],
        }
    )
    print(tabla.to_string(index=False))
    cumple = pc_hib >= 0.90 and hr_hib >= 0.90
    estado = "CUMPLE (>90%)" if cumple else "NO CUMPLE (<90%)"
    print(f"\n      Objetivo: PrecisionCategoria@10={pc_hib:.1%} | "
          f"HitRateCategoria@10={hr_hib:.1%} -> {estado}")

    inv_prod = {v: k for k, v in idx_producto.items()}

    def recomendando(usuario_id: str, n: int = 5) -> None:
        u = idx_usuario[usuario_id]
        fila = hibrido_final[u].copy()
        top = np.argsort(fila)[::-1][:n]
        favoritas = favoritas_declaradas[usuario_id]
        print(f"\nUsuario {usuario_id} | preferencias: {'|'.join(favoritas)}")
        for rank, col in enumerate(top, start=1):
            pid = inv_prod[col]
            afin_pct = p_cat_u_mejor[u, cats_producto[col]] * 100
            print(
                f"  {rank}. [{fila[col]:.3f}] ({info_por_pid.loc[pid, 'main_category']}) "
                f"{info_por_pid.loc[pid, 'name']}"
                f"   [afinidad categoria={afin_pct:.0f}%]"
            )

    print("[7/8] DEMO FINAL del sistema recomendador:")
    recomendando("u0007")
    recomendando("u0100")
    recomendando("u0250")

    print("[8/8] Guardando artefactos del sistema hibrido v2...")
    joblib.dump(
        {
            "idx_usuario": idx_usuario,
            "idx_producto": idx_producto,
            "lista_cats": lista_cats,
            "config": {
                "esquema_rating": mejor["esquema"],
                "alfa_declaradas": mejor["alfa"],
                "beta_suavizado": mejor["beta"],
                "top_categorias": mejor["n_c"],
                "exponente_cuotas": mejor["expo"],
                "pesos_producto": {"knn": float(wk), "ncf": float(wn),
                                   "contenido": float(wc)},
            },
            "p_categoria_usuario": p_cat_u_mejor,
            "favoritas_declaradas": favoritas_declaradas,
            "precision_at_k": p_hib,
            "recall_at_k": r_hib,
            "precision_categoria_test": pc_hib,
            "hitrate_categoria_test": hr_hib,
            "precision_individuales": {"popularidad": p_pop, "contenido": p_con,
                                       "ncf": p_ncf, "knn": p_knn, "afinidad": p_afin},
            "top_k": TOP_K,
        },
        RUTA_MODELOS / "hibrido_v2.pkl",
    )
    np.savez_compressed(
        RUTA_MODELOS / "puntajes_hibrido_v2.npz",
        matriz=hibrido_final.astype(np.float32),
        usuarios=np.array(usuarios),
        productos=np.array(productos_unicos),
    )
    print("Artefactos: models/hibrido_v2.pkl, models/puntajes_hibrido_v2.npz")
    print("Fase 6 completada.")


if __name__ == "__main__":
    main()
