"""
Re-ranker v2: LightGBM (lambdarank) + modelo de confianza calibrado + prediccion selectiva.

Por que cambia el enfoque (src/reranker.py era un ranker lineal listwise):
  - El lineal solo ordenaba. Aqui el problema se parte en dos:
      ETAPA A  ranking    : LightGBM con objetivo lambdarank (optimiza NDCG@K) sobre el
                            pool real de cada usuario. Un grupo por usuario.
      ETAPA B  confianza  : clasificador binario "P(el producto real esta en el top-10 de la
                            etapa A)" a partir de rasgos AGREGADOS del pool (margen entre el
                            primero y el segundo, entropia, parecido del primero al historial,
                            cuantos candidatos hay de la categoria que el cliente compra...).
                            Su salida se calibra con isotonia (PAV): "confianza 0.9" significa
                            "acierta el 90 % de las veces", no "el promedio del softmax da 0.9".
  - Con esa confianza el sistema puede ABSTENER: cuando no sabe, no recomienda. Es la
    prediccion selectiva (Geifman & El-Yaniv, 2019) y la via legitima para garantizar un
    acierto del 90 % en una fraccion de las consultas.

Tres detalles de metodo que hay que tener presentes (ver src/evaluacion.py):
  1. EMPATES: hr_ndcg rompe los empates con un epsilon aleatorio. Sin eso, un modelo que
     devuelve la misma puntuacion a miles de productos se lleva el 100 % de aciertos.
     Se reporta ademas la version pesimista (el empate cuenta como fallo).
  2. FUGA: rating_promedio y numero_resenas del catalogo se recalculan SOLO con el split
     train (el catalogo los trae calculados con todas las interacciones, incluida la del
     test). La diferencia medida es de 0.026 en HR@10: se reporta la version sin fuga.
  3. POOL: en el servicio el producto real solo esta en el pool el 25 % de las veces. Para
     que el ranker aprenda a colocar bien ese producto se entrena con el pool AUMENTADO
     (los 300 populares + los 300 similares + el producto real, que en servicio puede faltar).
     La evaluacion se hace sobre el pool REAL, sin aumentar.

Protocolo: mismos usuarios de evaluacion E que src/sasrec.py; cross-fitting en 5 pliegues
por usuario; calibracion y punto de operacion cross-fitted.

Uso:   python src/reranker_gbdt.py [--sasrec vNNN_...] [--pool 600] [--pliegues 5]
Salida: models/reranker/vNNN_<fecha>/
    pesos.json, modelo_gbdt.txt, modelo_gbdt_1p99.txt, confianza_gbdt.txt,
    calibrador.json, metricas.json
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from comun import DATA_SPLITS, K, MODELOS, SEMILLA, hr_ndcg
from rasgos import RASGOS, rasgos
from sasrec import SASRec, particion_eval
from two_tower import nueva_carpeta, ultima_version

POOL_SERVICIO = 600
PLIEGUES = 5
RNG = np.random.default_rng(SEMILLA)

RASGOS_CONF = ["top1", "top2", "margen", "media10", "desv10", "entropia", "n_hist", "n_cats_hist",
               "sim_perfil_top1", "sim_ultima_top1", "cat_ultima_top1", "cat_dom_top1", "cat_dom_cand",
               "log_pop_top1", "puesto_pop_top1", "sim_max_pool", "cand_misma_cat", "pop_max_pool",
               "p1_prob", "p1_es_popular"]


# --------------------------------------------------------------------- metricas
def hr_pesimista(puntajes, k=K):
    """Los empates cuentan como fallo: el producto real debe estar ESTRICTAMENTE entre los k."""
    rango = (puntajes[:, 1:] >= puntajes[:, :1]).sum(1)
    return float((rango < k).mean())


def ece(conf, acierto, bins=10):
    b = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum(abs(conf[b == j].mean() - acierto[b == j].mean()) * (b == j).mean()
                     for j in range(bins) if (b == j).any()))


def punto_selectivo(conf, acierto, objetivo=0.90):
    """El umbral MAS BAJO (menos restrictivo) cuya precision todavia alcanza el objetivo."""
    orden = np.argsort(-conf)
    c, ok = conf[orden], acierto[orden]
    acum = np.cumsum(ok) / np.arange(1, len(ok) + 1)
    bueno = np.flatnonzero(acum >= objetivo)
    if not len(bueno):
        return {"precision": None, "cobertura": 0.0, "umbral": None}
    i = bueno[-1]
    return {"precision": round(float(acum[i]), 4), "cobertura": round(float((i + 1) / len(ok)), 4),
            "umbral": round(float(c[i]), 4), "n_seleccionados": int(i + 1)}


def curva(conf, acierto, puntos=40):
    orden = np.argsort(-conf)
    c, ok = conf[orden], acierto[orden]
    acum = np.cumsum(ok) / np.arange(1, len(ok) + 1)
    idx = np.unique(np.linspace(0, len(ok) - 1, puntos).astype(int))
    return [{"cobertura": round(float((i + 1) / len(ok)), 4), "precision": round(float(acum[i]), 4),
             "confianza": round(float(c[i]), 4)} for i in idx]


# --------------------------------------------------------------------- calibracion isotonica
def pava(x, y):
    """Regresion isotonica por PAV (promedios de bloques: la solucion optima)."""
    o = np.argsort(x, kind="stable")
    xs, ys = x[o], y[o]
    vals, wts, ns = [], [], []
    for _, yi in zip(xs, ys):
        vals.append(float(yi)); wts.append(1.0); ns.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, n2 = vals.pop(), wts.pop(), ns.pop()
            v1, w1, n1 = vals.pop(), wts.pop(), ns.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2)); wts.append(w1 + w2); ns.append(n1 + n2)
    return xs, np.repeat(np.array(vals), ns)


def ajustar_isotonico(x, y):
    xs, ys = pava(x, y)
    unicos = np.unique(xs)
    niveles = ys if len(unicos) == len(xs) else np.array([ys[xs == u].mean() for u in unicos])
    return {"x": unicos.tolist(), "y": niveles.tolist()}


def aplicar_calibrador(cal, v):
    return np.interp(v, cal["x"], cal["y"])


# --------------------------------------------------------------------- scorers
def tabla_sasrec(modelo, historias, cand):
    """Puntaje de SASRec sobre el pool con la tabla de productos precomputada (proyectar
    2M de candidatos uno a uno es lentisimo)."""
    from sasrec import LARGO
    L = LARGO
    seqs = np.zeros((len(historias), L), dtype=np.int64)
    for i, h in enumerate(historias):
        s = np.asarray(h[-L:], dtype=np.int64) + 1
        if len(s):
            seqs[i, L - len(s):] = s
    with torch.no_grad():
        tabla = modelo.repr_items().numpy()
        sesgo = modelo.sesgo().numpy()
        pos = modelo.pos(torch.arange(L)).numpy()
        x = torch.tensor(tabla[seqs]) + torch.tensor(pos)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
        h = modelo.norma(modelo.transformer(x, mask=causal))[:, -1].numpy()
    C = cand + 1
    return np.einsum("nd,nkd->nk", h, tabla[C]) + sesgo[C]


def top_similares(items, historias, P, trozo=256):
    """Los P productos mas parecidos al perfil de cada usuario (mismo criterio que el backend)."""
    Q = np.stack([items[h].mean(0) if len(h) else np.zeros(items.shape[1], np.float32)
                  for h in historias])
    Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8)
    out = []
    for s in range(0, len(Q), trozo):
        sim = Q[s:s + trozo] @ items.T
        for j, h in enumerate(historias[s:s + trozo]):
            if len(h):
                sim[j, h] = -np.inf
        idx = np.argpartition(-sim, P, axis=1)[:, :P]
        for j in range(len(idx)):
            out.append(idx[j][np.argsort(-sim[j, idx[j]])])
    return out


def pool_servicio(items, historias, popularidad, top_pop, P=POOL_SERVICIO, inyectar=None):
    """El pool que arma el backend: los P/2 mas parecidos + los P/2 mas populares, excluyendo
    lo que el cliente ya compro. Todos los pools tienen el mismo tamano.
    inyectar: item_idx que se anade (pool aumentado para entrenar)."""
    similares = top_similares(items, historias, P // 2)
    pools = []
    for i, (h, sims) in enumerate(zip(historias, similares)):
        ya = set(h.tolist())
        populares = [j for j in top_pop if j not in ya][:P // 2]
        out = list(dict.fromkeys(populares + sims.tolist()))
        if inyectar is not None:
            v = int(inyectar[i])
            if v >= 0 and v not in out:
                out = out[:-1] + [v]  # entra el producto real a cambio del ultimo popular
        if len(out) < P:
            out += [j for j in top_pop if j not in set(out)][:P - len(out)]
        pools.append(np.array(out[:P], dtype=np.int64))
    return np.stack(pools)


# --------------------------------------------------------------------- rasgos de confianza
def rasgos_confianza(prob, pools, historias, items, popularidad, catalogo):
    """Rasgos AGREGADOS del pool (uno por usuario) para el modelo de confianza."""
    cat = catalogo["categoria"].to_numpy()
    puesto = np.empty(len(popularidad), dtype=np.int64)
    puesto[np.argsort(-popularidad, kind="stable")] = np.arange(len(popularidad))
    n = len(pools)
    out = np.zeros((n, len(RASGOS_CONF)), dtype=np.float32)
    orden = np.argsort(-prob, axis=1)
    for i in range(n):
        p = prob[i]
        o = orden[i]
        s = np.sort(p)[::-1]
        top = pools[i][o[0]]
        h = historias[i]
        z = np.exp(p - p.max()); z /= z.sum()
        ent = float(-(z * np.log(z + 1e-12)).sum())
        sims = items[pools[i]] @ items[h].T if len(h) else np.zeros((len(pools[i]), 1))
        cats_h = cat[h] if len(h) else np.array([], dtype=object)
        dom = None
        if len(cats_h):
            vals, cnt = np.unique(cats_h, return_counts=True)
            dom = vals[int(np.argmax(cnt))]
        out[i] = [
            s[0], s[1], s[0] - s[1], s[:10].mean(), s[:10].std(), ent,
            np.log1p(len(h)), len(set(cats_h.tolist())),
            float(sims[o[0]].max()) if len(h) else 0.0,
            float(sims[o[0], -1]) if len(h) else 0.0,
            float(cat[top] == cats_h[-1]) if len(h) else 0.0,
            float(cat[top] == dom) if dom is not None else 0.0,
            float(np.mean(cat[pools[i]] == dom)) if dom is not None else 0.0,
            np.log1p(popularidad[top]), float(puesto[top]),
            float(sims.max()) if len(h) else 0.0,
            float((cat[pools[i]] == dom).sum()) if dom is not None else 0.0,
            float(popularidad[pools[i]].max()),
            float(p[o[0]]), float(puesto[top] < 1000),
        ]
    return out


# --------------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sasrec")
    ap.add_argument("--pool", type=int, default=POOL_SERVICIO)
    ap.add_argument("--pliegues", type=int, default=PLIEGUES)
    ap.add_argument("--hojas", type=int, default=63)
    ap.add_argument("--iteraciones", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    t0 = time.time()
    import lightgbm as lgb
    from lightgbm import Dataset

    if args.sasrec:
        ruta_sas = os.path.join(MODELOS, "sasrec", args.sasrec)
    else:
        pub = os.path.join(MODELOS, "servir", "manifiesto.json")
        publicado = json.load(open(pub, encoding="utf-8")).get("sasrec") if os.path.exists(pub) else None
        ruta_sas = os.path.join(MODELOS, "sasrec", publicado) if publicado else ultima_version("sasrec")
    ck = torch.load(os.path.join(ruta_sas, "sasrec.pt"), map_location="cpu", weights_only=False)
    a = ck["args"]
    items = np.load(os.path.join(ck["two_tower"], "items.npy")).astype(np.float32)
    train, val, test = (pd.read_parquet(os.path.join(DATA_SPLITS, f"{n}.parquet")) for n in ("train", "val", "test"))
    catalogo = pd.read_parquet(os.path.join(DATA_SPLITS, "catalogo.parquet")).sort_values("item_idx")
    popularidad = np.bincount(train["item_idx"], minlength=len(items)).astype(np.float32)

    # --- sin fuga: los agregados de producto se recalculan con SOLO el split train
    agg = train.groupby("item_idx").agg(rating_promedio=("rating", "mean"), numero_resenas=("rating", "count"))
    catalogo = catalogo.copy()
    catalogo["rating_promedio"] = catalogo["item_idx"].map(agg["rating_promedio"]).fillna(0.0).astype(float)
    catalogo["numero_resenas"] = catalogo["item_idx"].map(agg["numero_resenas"]).fillna(0).astype(float)

    seq, es_eval, E, prev, _ = particion_eval(train, val, test, a.get("frac_eval", 0.3), a.get("min_compras", 2))
    filas = test[es_eval]
    historias = [np.asarray(prev.get(u, []), dtype=np.int64) for u in filas["user_id"]]
    real = filas["item_idx"].to_numpy()
    n_usuarios = len(historias)
    # estrellas del historial (rasgo disponible en el servicio: historial.npz trae los ratings)
    hist_previo = pd.concat([train, val]).sort_values(["user_id", "fecha"])
    mapa_est = {}
    for u, g in hist_previo.groupby("user_id"):
        mapa_est[u] = g["rating"].to_numpy(np.float32)
    estrellas = [mapa_est.get(u, np.array([], np.float32)) for u in filas["user_id"]]
    print(f"SASRec {os.path.basename(ruta_sas)}  usuarios E={n_usuarios:,}  pool={args.pool}")

    modelo = SASRec(items, np.log1p(popularidad), dim=a.get("dim", 64))
    modelo.load_state_dict(ck["state_dict"])
    modelo.eval()
    top_pop = np.argsort(-popularidad)[:max(args.pool, 2000)]
    pliegue = RNG.integers(args.pliegues, size=n_usuarios)
    METRICAS = {}

    def entrenar_ranker(F, y, etiqueta, iteraciones=None):
        """LightGBM lambdarank: un grupo por usuario; solo entran los usuarios con el
        producto real entre los candidatos (los demas no tienen senal de ranking)."""
        prob = np.zeros(F.shape[:2], dtype=np.float32)
        modelos = []
        for p in range(args.pliegues):
            m_tr, m_te = pliegue != p, pliegue == p
            con_pos = y[m_tr].any(1)
            X = F[m_tr].reshape(-1, F.shape[-1])[con_pos.repeat(F.shape[1])]
            yy = y[m_tr].reshape(-1)[con_pos.repeat(F.shape[1])]
            ds = Dataset(X, label=yy, group=np.full(con_pos.sum(), F.shape[1]), feature_name=RASGOS)
            m = lgb.train({"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [K],
                           "learning_rate": args.lr, "num_leaves": args.hojas, "min_data_in_leaf": 50,
                           "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
                           "lambda_l2": 5.0, "lambdarank_truncation_level": 30, "verbose": -1,
                           "num_threads": max(1, (os.cpu_count() or 4) - 1)},
                          ds, num_boost_round=iteraciones or args.iteraciones)
            prob[m_te] = m.predict(F[m_te].reshape(-1, F.shape[-1])).reshape(F[m_te].shape[:2])
            modelos.append(m)
            print(f"    {etiqueta} pliegue {p+1}/{args.pliegues} ({time.time()-t0:.0f}s)")
        return prob, modelos

    def evaluar_confianza(prob, cand, hit, etiqueta):
        """Etapa B: P(hit@10) a partir de rasgos agregados; luego isotonia cross-fitted."""
        GC = rasgos_confianza(prob, cand, historias, items, popularidad, catalogo)
        prob_conf = np.zeros(n_usuarios, dtype=np.float32)
        for p in range(args.pliegues):
            m_tr, m_te = pliegue != p, pliegue == p
            ds = Dataset(GC[m_tr], label=hit[m_tr].astype(int), feature_name=RASGOS_CONF)
            m = lgb.train({"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
                           "num_leaves": 15, "min_data_in_leaf": 40, "feature_fraction": 0.9,
                           "lambda_l2": 5.0, "verbose": -1,
                           "num_threads": max(1, (os.cpu_count() or 4) - 1)},
                          ds, num_boost_round=200)
            prob_conf[m_te] = m.predict(GC[m_te])
        mitad = RNG.random(n_usuarios) < 0.5
        conf = np.zeros(n_usuarios, dtype=np.float32)
        cal = None
        for a_, b_ in [(mitad, ~mitad), (~mitad, mitad)]:
            c = ajustar_isotonico(prob_conf[a_], hit[a_].astype(float))
            conf[b_] = aplicar_calibrador(c, prob_conf[b_])
            if a_ is mitad:
                cal = c
        return conf, cal, prob_conf, GC

    # ============================================================ PROTOCOLO 1+99 (informe)
    cand99 = np.load(os.path.join(DATA_SPLITS, "candidatos_test.npy"))[es_eval]
    s99 = tabla_sasrec(modelo, historias, cand99)
    F99 = rasgos(historias, cand99, s99, items, popularidad, catalogo, estrellas)
    y99 = np.zeros(cand99.shape, dtype=np.int8); y99[:, 0] = 1
    print(f"  1+99: rasgos {F99.shape}")
    prob99, m99 = entrenar_ranker(F99, y99, "1+99")
    hr, nd = hr_ndcg(prob99)
    hr_pop, nd_pop = hr_ndcg(popularidad[cand99] + RNG.random(cand99.shape) * 1e-6)
    cat = catalogo["categoria"].to_numpy()
    orden99 = np.argsort(-prob99, axis=1)
    top10_99 = np.take_along_axis(cand99, orden99[:, :K], axis=1)
    hit99 = (top10_99 == real[:, None]).any(1)
    top1_99 = (cand99[np.arange(n_usuarios), orden99[:, 0]] == real)
    cat_top1 = float(np.mean([cat[cand99[i, orden99[i, 0]]] == cat[cand99[i, 0]] for i in range(n_usuarios)]))
    conf99, cal99, pc99, GC99 = evaluar_confianza(prob99, cand99, hit99, "1+99")
    METRICAS["protocolo_1_99"] = {
        "n": int(n_usuarios), f"hr@{K}": round(hr, 4), f"ndcg@{K}": round(nd, 4),
        f"hr@{K}_pesimista": round(hr_pesimista(prob99), 4),
        "hr@1": round(float(top1_99.mean()), 4),
        "popularidad": {f"hr@{K}": round(hr_pop, 4), f"ndcg@{K}": round(nd_pop, 4)},
        "categoria@1": round(cat_top1, 4),
        "confianza": {"ece_top10": round(ece(conf99, hit99), 4),
                      "media": round(float(conf99.mean()), 4),
                      "selectivo_90_top10": punto_selectivo(conf99, hit99, 0.90),
                      "selectivo_90_top1": punto_selectivo(conf99, top1_99, 0.90),
                      "curva": curva(conf99, hit99)},
    }
    print(json.dumps({k: v for k, v in METRICAS["protocolo_1_99"].items() if k != "confianza"} |
                      {"confianza": {k: v for k, v in METRICAS["protocolo_1_99"]["confianza"].items()
                                     if k != "curva"}}, indent=2))

    # ============================================================ POOL DEL SERVICIO
    pools = pool_servicio(items, historias, popularidad, top_pop, args.pool)
    pos = np.array([int(np.flatnonzero(p == y)[0]) if (p == y).any() else -1 for p, y in zip(pools, real)])
    sp = tabla_sasrec(modelo, historias, pools)
    Fp = rasgos(historias, pools, sp, items, popularidad, catalogo, estrellas)
    yp = np.zeros(pools.shape, dtype=np.int8)
    tiene = pos >= 0
    yp[tiene, pos[tiene]] = 1
    # pool AUMENTADO para entrenar (el producto real puede no estar en el pool del servicio)
    pools_aum = pool_servicio(items, historias, popularidad, top_pop, args.pool, inyectar=real)
    spa = tabla_sasrec(modelo, historias, pools_aum)
    Fp_aum = rasgos(historias, pools_aum, spa, items, popularidad, catalogo, estrellas)
    print(f"  pool: real dentro {tiene.mean():.4f}  ({time.time()-t0:.0f}s)")

    probP, mP = entrenar_ranker(Fp_aum, yp, "pool(aumentado)")
    ordenP = np.argsort(-probP, axis=1)
    top10P = np.take_along_axis(pools, ordenP[:, :K], axis=1)
    hitP = (top10P == real[:, None]).any(1)
    top1P = (pools[np.arange(n_usuarios), ordenP[:, 0]] == real)
    hr_pool = float(hitP.mean())
    s_pop_pool = popularidad[pools] + RNG.random(pools.shape) * 1e-6
    hr_pop_pool = float(np.mean([real[i] in pools[i][np.argsort(-s_pop_pool[i])[:K]].tolist()
                                 if real[i] in set(pools[i].tolist()) else False
                                 for i in range(n_usuarios)]))
    confP, calP, pcP, GCP = evaluar_confianza(probP, pools, hitP, "pool")
    METRICAS["pool_servicio"] = {
        "n": int(n_usuarios), "pool": int(pools.shape[1]),
        "real_en_pool": round(float(tiene.mean()), 4),
        f"hr@{K}": round(hr_pool, 4), "hr@1": round(float(top1P.mean()), 4),
        "popularidad": {f"hr@{K}": round(hr_pop_pool, 4)},
        "confianza": {"ece_top10": round(ece(confP, hitP), 4),
                      "media": round(float(confP.mean()), 4),
                      "selectivo_90_top10": punto_selectivo(confP, hitP, 0.90),
                      "selectivo_90_top1": punto_selectivo(confP, top1P, 0.90),
                      "curva": curva(confP, hitP)},
    }
    print(json.dumps({k: v for k, v in METRICAS["pool_servicio"].items() if k != "confianza"} |
                      {"confianza": {k: v for k, v in METRICAS["pool_servicio"]["confianza"].items()
                                     if k != "curva"}}, indent=2))

    # ============================================================ guardar
    carpeta = nueva_carpeta("reranker")
    mP[0].save_model(os.path.join(carpeta, "modelo_gbdt.txt"))
    m99[0].save_model(os.path.join(carpeta, "modelo_gbdt_1p99.txt"))
    with open(os.path.join(carpeta, "calibrador.json"), "w", encoding="utf-8") as f:
        json.dump({"pool": {"x": calP["x"], "y": calP["y"]},
                   "1p99": {"x": cal99["x"], "y": cal99["y"]}}, f)
    with open(os.path.join(carpeta, "pesos.json"), "w", encoding="utf-8") as f:
        json.dump({"formato": "gbdt", "rasgos": RASGOS, "rasgos_confianza": RASGOS_CONF,
                   "modelo": "modelo_gbdt.txt", "modelo_1p99": "modelo_gbdt_1p99.txt",
                   "calibrador": "calibrador.json", "pool": int(pools.shape[1]),
                   "sasrec": os.path.basename(ruta_sas), "two_tower": os.path.basename(ck["two_tower"])}, f, indent=2)
    with open(os.path.join(carpeta, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(METRICAS, f, indent=2)
    print(f"OK -> {carpeta}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
