"""
Paso 4: re-ranker aprendido + metricas de confiabilidad.

La fusion RRF two-tower + SASRec empeoraba (0.36 vs 0.47): RRF pesa igual a un modelo debil
y a uno fuerte. Aqui se aprende cuanto pesa cada senal con un ranker lineal listwise
(softmax sobre los 100 candidatos, el producto real es la clase correcta):

  sasrec      puntaje del Transformer secuencial (ya incluye su sesgo de popularidad)
  log_pop     log(1 + compras del producto)
  tt_media    coseno con el promedio de lo que compro el cliente (two-tower)
  tt_max      coseno con la compra mas parecida   tt_ultima  coseno con la ultima compra
  cat_frac    % del historial en la categoria del candidato   cat_ultima  misma categoria que la ultima
  pub_frac    % del historial con el mismo publico   tienda   ya le compro a esa tienda
  rating, log_resenas

Evaluacion honesta: cross-fitting en 5 partes sobre los usuarios E de src/sasrec.py (los
mismos candidatos de candidatos_test.npy): cada usuario se puntua con un ranker que no lo vio.

Metricas (todas fuera de muestra, con sus lineas base):
  hr@10 / ndcg@10        acierto exacto del producto
  categoria@1 / @3       el top-1 / top-3 incluye la categoria de lo que compro
  confianza              P(acierto en top-10) = masa del softmax en el top-10. Se reporta la
                         calibracion (ECE) y el modo selectivo: umbral elegido en una mitad de
                         usuarios para que la precision sea >= --precision-objetivo, medido en
                         la otra mitad, con su cobertura (que % de clientes recibe esa garantia).

Catalogo completo: lo anterior se repite con los candidatos reales del backend (300 populares
+ los mas parecidos, 600 en total, contra los 71k productos). Ese ranker -con una clase
"ninguno" para cuando el producto real no esta en el pool- es el que se publica.

Uso:    python src/reranker.py [--sasrec vNNN_...]
Salida: models/reranker/vNNN_<fecha>/  pesos.json (lo lee el backend), metricas.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from comun import DATA_SPLITS, K, MODELOS, SEMILLA, hr_ndcg
from sasrec import SASRec, particion_eval, puntuar
from two_tower import nueva_carpeta, ultima_version

RASGOS = ["sasrec", "log_pop", "tt_media", "tt_max", "tt_ultima", "cat_frac", "cat_ultima",
          "pub_frac", "tienda", "rating", "log_resenas"]
PLIEGUES = 5
POOL_SERVICIO = 600


def rasgos(historias, cand, s_sas, items, popularidad, catalogo):
    """[n, 100, n_rasgos] para cada (usuario, candidato)."""
    cat = catalogo["categoria"].to_numpy()
    pub = catalogo["publico"].to_numpy()
    tienda = catalogo["tienda"].fillna("").to_numpy()
    rating = catalogo["rating_promedio"].fillna(0).to_numpy(np.float32)
    resenas = np.log1p(catalogo["numero_resenas"].fillna(0).to_numpy(np.float32))
    F = np.zeros(cand.shape + (len(RASGOS),), dtype=np.float32)
    for i, h in enumerate(historias):
        c = cand[i]
        ec, eh = items[c], items[h]
        sims = ec @ eh.T                                  # [100, n_hist]
        media = eh.mean(0); media /= max(np.linalg.norm(media), 1e-8)
        F[i, :, 2] = ec @ media
        F[i, :, 3] = sims.max(1)
        F[i, :, 4] = sims[:, -1]
        F[i, :, 5] = (cat[c][:, None] == cat[h][None]).mean(1)
        F[i, :, 6] = cat[c] == cat[h[-1]]
        F[i, :, 7] = (pub[c][:, None] == pub[h][None]).mean(1)
        F[i, :, 8] = np.isin(tienda[c], [t for t in tienda[h] if t])
    F[:, :, 0] = s_sas
    F[:, :, 1] = np.log1p(popularidad[cand])
    F[:, :, 9] = rating[cand]
    F[:, :, 10] = resenas[cand]
    return F


def entrenar_lineal(F, y=None, epocas=300, lr=0.05, l2=1e-3):
    """Ranker lineal listwise: softmax(F @ w). y = columna del producto real (por defecto 0).
    Con y dado, y = -1 significa "no esta entre los candidatos": se agrega una clase "ninguno"
    con puntaje b aprendido, asi la confianza descuenta la probabilidad de que no este."""
    media, desv = F.reshape(-1, F.shape[-1]).mean(0), F.reshape(-1, F.shape[-1]).std(0) + 1e-6
    X = torch.as_tensor((F - media) / desv)
    w = torch.zeros(F.shape[-1], requires_grad=True)
    b = torch.zeros(1, requires_grad=y is not None)
    nulo = y is not None
    y = torch.zeros(len(X), dtype=torch.long) if y is None else torch.as_tensor(np.where(y < 0, F.shape[1], y))
    opt = torch.optim.Adam([w, b] if nulo else [w], lr=lr)
    for _ in range(epocas):
        logits = X @ w
        if nulo:
            logits = torch.cat([logits, b.expand(len(X), 1)], 1)
        loss = torch.nn.functional.cross_entropy(logits, y) + l2 * (w ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return {"media": media, "desv": desv, "w": w.detach().numpy(), "b": float(b.item()) if nulo else None}


def aplicar(m, F):
    return ((F - m["media"]) / m["desv"]) @ m["w"]


def prob_top(s, k=K, b=None):
    """P(el producto real esta en el top-k) = masa del softmax en el top-k. Con b (clase
    "ninguno"), la masa se reparte tambien con la posibilidad de que no este entre los candidatos."""
    z = s if b is None else np.concatenate([s, np.full((len(s), 1), b, dtype=s.dtype)], 1)
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    return np.sort(p[:, :s.shape[1]], 1)[:, ::-1][:, :k].sum(1)


def pool_servicio(h, items, popularidad, top_pop, P=POOL_SERVICIO):
    """Los mismos candidatos que arma el backend en modo usuario: los 300 mas populares que no
    compro + los mas parecidos (coseno con su perfil) hasta completar P."""
    q = items[h].mean(0); q /= max(np.linalg.norm(q), 1e-8)
    sim = items @ q
    sim[h] = -np.inf
    parecidos = np.argpartition(-sim, P)[:P]
    parecidos = parecidos[np.argsort(-sim[parecidos])]
    ya = set(h.tolist())
    populares = [i for i in top_pop if i not in ya][:P // 2]
    return np.array(list(dict.fromkeys(populares + parecidos.tolist()))[:P], dtype=np.int64)


def acierto_en_pool(s, pos, k=K):
    """pos = columna del producto real en el pool, -1 si quedo fuera (cuenta como fallo)."""
    fila = np.arange(len(s))
    return (pos >= 0) & ((s > s[fila, np.maximum(pos, 0)][:, None]).sum(1) < k)


def acierto_topk(s, k=K):
    return (s[:, 1:] > s[:, :1]).sum(1) < k


def categoria_en_top(s, cand, cat, k):
    top = np.argsort(-s, 1)[:, :k]
    real = cat[cand[:, 0]]
    return float(np.mean([real[i] in set(cat[cand[i, top[i]]]) for i in range(len(s))]))


def ece(conf, acierto, bins=10):
    b = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum(abs(conf[b == j].mean() - acierto[b == j].mean()) * (b == j).mean()
                     for j in range(bins) if (b == j).any()))


def selectivo(conf, acierto, mitad, objetivo):
    """Umbral elegido en una mitad (el menor que da precision >= objetivo), medido en la otra."""
    res = []
    for a, b in [(mitad, ~mitad), (~mitad, mitad)]:
        orden = np.sort(np.unique(conf[a]))
        umbral = next((u for u in orden if acierto[a][conf[a] >= u].mean() >= objetivo), None)
        if umbral is None:
            res.append((np.nan, 0.0, np.nan)); continue
        sel = conf[b] >= umbral
        res.append((acierto[b][sel].mean() if sel.any() else np.nan, sel.mean(), umbral))
    prec, cob, umb = (np.nanmean([r[j] for r in res]) for j in range(3))
    return {"precision": round(float(prec), 4), "cobertura": round(float(cob), 4), "umbral": round(float(umb), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sasrec", help="carpeta vNNN_... (por defecto la ultima)")
    ap.add_argument("--precision-objetivo", type=float, default=0.9)
    args = ap.parse_args()

    ruta_sas = os.path.join(MODELOS, "sasrec", args.sasrec) if args.sasrec else ultima_version("sasrec")
    ck = torch.load(os.path.join(ruta_sas, "sasrec.pt"), map_location="cpu", weights_only=False)
    a = ck["args"]
    items = np.load(os.path.join(ck["two_tower"], "items.npy")).astype(np.float32)
    train, val, test = (pd.read_parquet(os.path.join(DATA_SPLITS, f"{n}.parquet")) for n in ("train", "val", "test"))
    catalogo = pd.read_parquet(os.path.join(DATA_SPLITS, "catalogo.parquet")).sort_values("item_idx")
    popularidad = np.bincount(train["item_idx"], minlength=len(items)).astype(np.float32)
    _, es_eval, _, prev, _ = particion_eval(train, val, test, a.get("frac_eval", 0.3), a.get("min_compras", 2))

    modelo = SASRec(items, np.log1p(popularidad), dim=a.get("dim", 64))
    modelo.load_state_dict(ck["state_dict"])
    filas = test[es_eval]
    cand = np.load(os.path.join(DATA_SPLITS, "candidatos_test.npy"))[es_eval]
    historias = [np.array(prev.get(u, []), dtype=np.int64) for u in filas["user_id"]]
    import sasrec as _s
    _s.DISPOSITIVO = "cpu"
    s_sas = puntuar(modelo, [list(h) for h in historias], cand)
    F = rasgos(historias, cand, s_sas, items, popularidad, catalogo)
    print(f"SASRec {os.path.basename(ruta_sas)}  usuarios E={len(F):,}  rasgos={F.shape}")

    # ----------------------------------------------------- cross-fitting (fuera de muestra)
    rng = np.random.default_rng(SEMILLA)
    pliegue = rng.integers(PLIEGUES, size=len(F))
    s_rr = np.zeros(cand.shape, dtype=np.float32)
    for p in range(PLIEGUES):
        m = entrenar_lineal(F[pliegue != p])
        s_rr[pliegue == p] = aplicar(m, F[pliegue == p])
    final = entrenar_lineal(F)  # el que se publica: entrenado con todo E

    cat = catalogo["categoria"].to_numpy()
    s_pop = popularidad[cand] + np.random.default_rng(0).random(cand.shape) * 1e-6
    s_azar = np.random.default_rng(1).random(cand.shape)
    metricas = {"sasrec_version": os.path.basename(ruta_sas), "n_usuarios_eval": int(len(F)),
                "protocolo": f"1 positivo + 99 negativos, @{K}; cross-fitting {PLIEGUES} pliegues"}
    for nombre, s in [("azar", s_azar), ("popularidad", s_pop), ("sasrec", s_sas), ("reranker", s_rr)]:
        hr, nd = hr_ndcg(s)
        metricas[nombre] = {f"hr@{K}": round(hr, 4), f"ndcg@{K}": round(nd, 4),
                            "categoria@1": round(categoria_en_top(s, cand, cat, 1), 4),
                            "categoria@3": round(categoria_en_top(s, cand, cat, 3), 4)}

    # ----------------------------------------------------- confiabilidad del reranker
    conf, ok = prob_top(s_rr), acierto_topk(s_rr)
    mitad = np.random.default_rng(2).random(len(F)) < 0.5
    d = np.digitize(conf, np.quantile(conf, np.linspace(0, 1, 11)[1:-1]))  # decil de confianza
    largo = np.array([len(h) for h in historias])
    metricas["confianza"] = {
        "ece_top10": round(ece(conf, ok), 4),
        "por_decil": [{"conf_media": round(float(conf[d == j].mean()), 3),
                       "acierto_real": round(float(ok[d == j].mean()), 3), "n": int((d == j).sum())}
                      for j in range(10) if (d == j).any()],
        **{f"selectivo_{int(o * 100)}": selectivo(conf, ok, mitad, o) for o in (0.7, 0.8, args.precision_objetivo)},
    }
    metricas["por_largo_historial"] = {
        f"{lo}-{hi}" if hi < 99 else f">={lo}": {"n": int(sel.sum()), "hr@10": round(float(ok[sel].mean()), 4)}
        for lo, hi in [(1, 1), (2, 2), (3, 4), (5, 99)] if (sel := (largo >= lo) & (largo <= hi)).any()}
    metricas["pesos"] = dict(zip(RASGOS, [round(float(x), 4) for x in final["w"]]))
    print(json.dumps({k: v for k, v in metricas.items() if k != "confianza"} | {"confianza": {
        k: v for k, v in metricas["confianza"].items() if k != "por_decil"}}, indent=2))

    # ------------------------------------------- catalogo completo (lo que vive el backend)
    # Los 99 negativos al azar son casi todos productos sin ventas: facilitan el ranking.
    # Aqui el producto real compite contra los 71k, a traves del pool del backend.
    top_pop = np.argsort(-popularidad)[:POOL_SERVICIO]
    pools = np.stack([pool_servicio(h, items, popularidad, top_pop) for h in historias])
    pos = np.array([int(np.flatnonzero(pl == y)[0]) if (pl == y).any() else -1
                    for pl, y in zip(pools, filas["item_idx"])])
    Fp = rasgos(historias, pools, puntuar(modelo, [list(h) for h in historias], pools), items, popularidad, catalogo)
    sp_rr = np.zeros(pools.shape, dtype=np.float32)
    conf_p = np.zeros(len(Fp), dtype=np.float32)
    for p in range(PLIEGUES):
        m = entrenar_lineal(Fp[pliegue != p], pos[pliegue != p])
        sp_rr[pliegue == p] = aplicar(m, Fp[pliegue == p])
        conf_p[pliegue == p] = prob_top(sp_rr[pliegue == p], b=m["b"])
    final_pool = entrenar_lineal(Fp, pos)
    ok_p = acierto_en_pool(sp_rr, pos)
    s_pop_p = Fp[:, :, 1] + np.random.default_rng(0).random(pools.shape) * 1e-6
    metricas["catalogo_completo"] = {
        "protocolo": f"recall@{K} contra los {len(items):,} productos, via el pool del backend ({POOL_SERVICIO})",
        "real_dentro_del_pool": round(float((pos >= 0).mean()), 4),
        "popularidad": round(float(acierto_en_pool(s_pop_p, pos).mean()), 4),
        "sasrec": round(float(acierto_en_pool(Fp[:, :, 0], pos).mean()), 4),
        "rrf_similitud+sasrec (backend actual)": round(float(acierto_en_pool(_s.fusion_rrf(Fp[:, :, 2], Fp[:, :, 0]), pos).mean()), 4),
        "reranker": round(float(ok_p.mean()), 4),
        "ece_top10": round(ece(conf_p, ok_p), 4),
        **{f"selectivo_{int(o * 100)}": selectivo(conf_p, ok_p, mitad, o) for o in (0.2, 0.3, 0.5)},
        "pesos": dict(zip(RASGOS, [round(float(x), 4) for x in final_pool["w"]])) | {"ninguno": round(final_pool["b"], 4)},
    }
    print(json.dumps(metricas["catalogo_completo"], indent=2))

    carpeta = nueva_carpeta("reranker")
    with open(os.path.join(carpeta, "pesos.json"), "w", encoding="utf-8") as f:
        # el backend usa el entrenado sobre su propio pool (no el de los 99 negativos al azar)
        json.dump({"rasgos": RASGOS, "media": final_pool["media"].tolist(), "desv": final_pool["desv"].tolist(),
                   "w": final_pool["w"].tolist(), "b_ninguno": final_pool["b"], "pool": POOL_SERVICIO,
                   "sasrec": os.path.basename(ruta_sas), "two_tower": os.path.basename(ck["two_tower"])},
                  f, indent=2)
    with open(os.path.join(carpeta, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2)
    print(f"OK -> {carpeta}")


if __name__ == "__main__":
    main()
