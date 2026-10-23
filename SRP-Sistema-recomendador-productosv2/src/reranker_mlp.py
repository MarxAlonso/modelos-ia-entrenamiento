"""
Re-ranker v3: red neuronal con objetivo listwise (softmax sobre el pool) y clase "ninguno".

Por que una red y no un GBDT para el pool del servicio: el pool real (600 candidatos =
300 populares + 300 parecidos) es un problema MUCHO mas dificil que el protocolo 1+99, donde
99 de los 100 negativos son productos casi sin ventas (ahi la popularidad separa sola). En el
pool las senales hay que combinarlas de verdad. El softmax listwise sobre todo el pool, con
una clase "ninguno" que representa "el producto que busca NO esta en este pool", si lo hace:

  - el softmax es una DISTRIBUCION sobre los candidatos: la masa del top-10 es directamente
    P(el producto real esta en el top-10) y esa magnitud se puede calibrar;
  - la clase "ninguno" se aprende de los usuarios cuyo producto real no esta en el pool
    (el 75 %): el modelo aprende a repartir menos masa cuando no tiene nada que ofrecer.

Estructura: MLP sobre los rasgos (36) -> puntaje por candidato. El "ninguno" es un sesgo
aprendido (no un candidato mas), igual que en el ranker lineal de src/reranker.py.

Protocolo: mismos usuarios E que src/sasrec.py; cross-fitting en 5 pliegues por usuario;
calibracion isotonica cross-fitted de la confianza.

Uso:   python src/reranker_mlp.py [--sasrec vNNN_...] [--pool 600] [--epocas 40]
                               [--barrido-pool]  (prueba variantes de pool)
Salida: models/reranker_mlp/vNNN_<fecha>/
    pesos.json, red.pt, red.onnx, calibrador.json, metricas.json
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from comun import DATA_SPLITS, K, MODELOS, SEMILLA, hr_ndcg
from rasgos import RASGOS, rasgos
from reranker_gbdt import (aplicar_calibrador, ajustar_isotonico, curva, ece, hr_pesimista,
                            pool_servicio, punto_selectivo, tabla_sasrec)
from sasrec import SASRec, particion_eval
from two_tower import nueva_carpeta, ultima_version

PLIEGUES = 5
RNG = np.random.default_rng(SEMILLA)


class Ranker(torch.nn.Module):
    """Puntaje por candidato a partir de sus rasgos + un sesgo para la clase 'ninguno'."""

    def __init__(self, n_rasgos, ocultos=64, capas=2, dropout=0.1):
        super().__init__()
        dims = [n_rasgos] + [ocultos] * capas
        self.capas = torch.nn.ModuleList(torch.nn.Linear(dims[i], dims[i + 1]) for i in range(capas))
        self.salida = torch.nn.Linear(dims[-1], 1)
        self.normas = torch.nn.ModuleList(torch.nn.LayerNorm(dims[i + 1]) for i in range(capas))
        self.drop = torch.nn.Dropout(dropout)
        self.ninguno = torch.nn.Parameter(torch.zeros(1))
        for c in self.capas:
            torch.nn.init.xavier_uniform_(c.weight); torch.nn.init.zeros_(c.bias)

    def forward(self, x):
        """x: [n, m, rasgos] -> [n, m] puntajes."""
        h = x
        for capa, norma in zip(self.capas, self.normas):
            h = self.drop(F.gelu(capa(h)))
            h = norma(h)
        return self.salida(h).squeeze(-1)


def softmax_confianza(s, b, k=K, temperatura=1.0):
    """P(el producto real esta en el top-k) = masa del softmax en el top-k, con la clase
    'ninguno' compitiendo (si el modelo cree que no esta en el pool, la masa baja).
    b: sesgo 'ninguno' por usuario (escalar o vector)."""
    b = np.broadcast_to(np.asarray(b, dtype=s.dtype).reshape(-1), (len(s),))
    z = np.concatenate([s / temperatura, b.reshape(-1, 1)], 1)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(1, keepdims=True)
    orden = np.argsort(-s, axis=1)
    return np.take_along_axis(p[:, :-1], orden[:, :k], axis=1).sum(1)


def calibrar(masa, hit, rng):
    """Isotonia cross-fitted: la masa del softmax -> P(acierto). Se calibra en una mitad
    de los usuarios y se mide en la otra."""
    mitad = rng.random(len(masa)) < 0.5
    conf = np.zeros(len(masa), dtype=np.float32)
    cal = None
    for a_, b_ in [(mitad, ~mitad), (~mitad, mitad)]:
        c = ajustar_isotonico(masa[a_], hit[a_].astype(float))
        conf[b_] = aplicar_calibrador(c, masa[b_])
        if a_ is mitad:
            cal = c
    return conf, cal


class RankerONNX(torch.nn.Module):
    """Envoltorio para exportar: normaliza los rasgos y devuelve los puntajes.
    El backend solo le pasa la matriz de rasgos crudos."""

    def __init__(self, red, media, desv):
        super().__init__()
        self.red = red
        self.register_buffer("media", torch.tensor(media, dtype=torch.float32))
        self.register_buffer("desv", torch.tensor(desv, dtype=torch.float32))

    def forward(self, x):
        return self.red((x - self.media) / self.desv)


def exportar_onnx(red, media, desv, ruta):
    import onnxruntime as ort
    modelo = RankerONNX(red, media, desv).float().cpu().eval()
    ej = torch.rand(2, 7, len(media))
    torch.onnx.export(modelo, ej, ruta, dynamo=False, opset_version=17,
                      input_names=["rasgos"], output_names=["puntajes"],
                      dynamic_axes={"rasgos": {0: "lote", 1: "candidatos"},
                                    "puntajes": {0: "lote", 1: "candidatos"}})
    sess = ort.InferenceSession(ruta, providers=["CPUExecutionProvider"])
    o = sess.run(None, {"rasgos": ej.numpy()})[0]
    with torch.no_grad():
        t = modelo(ej).numpy()
    diff = float(np.abs(o - t).max())
    print(f"  paridad ONNX vs PyTorch: {diff:.2e}")
    assert diff < 1e-3
    return diff


def hr_ndcg_en_pool(s, cand, real, k=K, semilla=SEMILLA):
    """HR@k y NDCG@k cuando el producto real NO esta en una columna fija (el pool del
    servicio lo tiene en una posicion distinta en cada usuario).
    Devuelve (hr, ndcg, hr_pesimista, hit_top1)."""
    rng = np.random.default_rng(semilla)
    n = len(s)
    fila = np.arange(n)
    pos = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        hit = np.flatnonzero(cand[i] == real[i])
        if len(hit):
            pos[i] = hit[0]
    dentro = pos >= 0
    s_real = np.where(dentro, s[fila, np.maximum(pos, 0)], -np.inf)
    # pesimista: cuentan como por delante los que empatan o superan
    rango_pes = (s >= s_real[:, None]).sum(1) - dentro
    # optimista: los empates se rompen al azar (como la linea base de popularidad)
    eps = rng.random(s.shape) * 1e-9
    rango = ((s + eps) > (s_real[:, None] + eps)).sum(1)
    hit = dentro & (rango < k)
    hr = float(hit.mean())
    ndcg = float(np.where(hit, 1.0 / np.log2(rango + 2), 0.0).mean())
    hr_pes = float((dentro & (rango_pes < k)).mean())
    top1 = dentro & (rango == 0)
    return hr, ndcg, hr_pes, float(top1.mean())


def modelo_confianza(s, b, cand, hit, etiqueta, historias, items, popularidad, catalogo):
    """ETAPA B: P(el producto real esta en el top-10) a partir de rasgos AGREGADOS del pool.

    La masa del softmax del ranker sola no separa bien (los usuarios con historial mas
    denso tienen mas candidatos parecidos y una masa parecida). Lo que si separa es el
    CONTEXTO: cuanto margen hay entre el primero y el segundo, cuantos candidatos hay de
    la categoria que el cliente compra, cuan parecido es el primero a su ultima compra.
    Un LightGBM con esos rasgos predice el acierto; la isotonia lo calibra.
    """
    import lightgbm as lgb
    from lightgbm import Dataset
    from reranker_gbdt import RASGOS_CONF, rasgos_confianza
    b = np.broadcast_to(np.asarray(b, dtype=s.dtype).reshape(-1), (len(s),))
    z = np.concatenate([s / 5.0, b.reshape(-1, 1)], 1)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(1, keepdims=True)
    GC = rasgos_confianza(p[:, :-1], cand, historias, items, popularidad, catalogo)
    pliegue = RNG.integers(PLIEGUES, size=len(s))
    prob_conf = np.zeros(len(s), dtype=np.float32)
    for q in range(PLIEGUES):
        m_tr, m_te = pliegue != q, pliegue == q
        ds = Dataset(GC[m_tr], label=hit[m_tr].astype(int), feature_name=RASGOS_CONF)
        m = lgb.train({"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
                       "num_leaves": 15, "min_data_in_leaf": 40, "feature_fraction": 0.9,
                       "lambda_l2": 5.0, "verbose": -1, "num_threads": 8},
                      ds, num_boost_round=200)
        prob_conf[m_te] = m.predict(GC[m_te])
    return prob_conf, GC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sasrec")
    ap.add_argument("--pool", type=int, default=600)
    ap.add_argument("--pliegues", type=int, default=PLIEGUES)
    ap.add_argument("--epocas", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--ocultos", type=int, default=64)
    ap.add_argument("--capas", type=int, default=2)
    ap.add_argument("--barrido-pool", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
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
    # sin fuga: los agregados de producto se recalculan SOLO con el split train
    agg = train.groupby("item_idx").agg(rating_promedio=("rating", "mean"), numero_resenas=("rating", "count"))
    catalogo = catalogo.copy()
    catalogo["rating_promedio"] = catalogo["item_idx"].map(agg["rating_promedio"]).fillna(0.0).astype(float)
    catalogo["numero_resenas"] = catalogo["item_idx"].map(agg["numero_resenas"]).fillna(0).astype(float)

    seq, es_eval, E, prev, _ = particion_eval(train, val, test, a.get("frac_eval", 0.3), a.get("min_compras", 2))
    filas = test[es_eval]
    historias = [np.asarray(prev.get(u, []), dtype=np.int64) for u in filas["user_id"]]
    real = filas["item_idx"].to_numpy()
    n_usuarios = len(historias)
    hist_previo = pd.concat([train, val]).sort_values(["user_id", "fecha"])
    mapa_est = {u: g["rating"].to_numpy(np.float32) for u, g in hist_previo.groupby("user_id")}
    estrellas = [mapa_est.get(u, np.array([], np.float32)) for u in filas["user_id"]]
    print(f"SASRec {os.path.basename(ruta_sas)}  usuarios E={n_usuarios:,}")

    modelo_sas = SASRec(items, np.log1p(popularidad), dim=a.get("dim", 64))
    modelo_sas.load_state_dict(ck["state_dict"])
    modelo_sas.eval()
    top_pop = np.argsort(-popularidad)[:6000]
    pliegue = RNG.integers(args.pliegues, size=n_usuarios)
    METRICAS = {}

    # ------------------------------------------------------------------ utilidades de pool
    categoria = catalogo["categoria"].to_numpy()
    pop_por_cat = {}
    for c in np.unique(categoria):
        idx = np.flatnonzero(categoria == c)
        pop_por_cat[c] = idx[np.argsort(-popularidad[idx])]

    def pool_por_categoria(h, n_cat=600, n_pop=300, n_sim=300):
        """Populares de las categorias que el cliente compra + populares globales + parecidos."""
        ya = set(h.tolist())
        cats = np.unique([categoria[i] for i in h]) if len(h) else np.array([], dtype=object)
        out = []
        for c in cats:
            for j in pop_por_cat.get(c, [])[:max(1, n_cat // max(len(cats), 1))]:
                if j not in ya:
                    out.append(j)
        out += [j for j in top_pop if j not in ya][:n_pop]
        q = items[h].mean(0) if len(h) else items[0]
        q = q / max(np.linalg.norm(q), 1e-8)
        sim = items @ q
        if len(h):
            sim[h] = -np.inf
        out += [j for j in np.argpartition(-sim, n_sim)[:n_sim] if j not in ya]
        out = list(dict.fromkeys(out))
        out += [j for j in top_pop if j not in set(out)]
        return np.array(out, dtype=np.int64)

    # ------------------------------------------------------------------ entrenamiento
    def entrenar_red(Xtr, ytr, epocas, lr, ocultos, capas):
        """Softmax listwise: la clase correcta es la columna del producto real; si no esta
        entre los candidatos, la clase correcta es 'ninguno' (su sesgo aprende a competir)."""
        red = Ranker(Xtr.shape[-1], ocultos, capas)
        opt = torch.optim.AdamW(red.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epocas)
        X = Xtr if torch.is_tensor(Xtr) else torch.tensor(Xtr, dtype=torch.float32)
        y = ytr if torch.is_tensor(ytr) else torch.tensor(ytr, dtype=torch.long)
        for ep in range(epocas):
            red.train()
            perm = torch.randperm(len(X))
            for s in range(0, len(perm), 256):
                b = perm[s:s + 256]
                z = torch.cat([red(X[b]), red.ninguno.expand(len(b), 1)], 1)
                loss = F.cross_entropy(z, y[b])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(red.parameters(), 1.0)
                opt.step()
            sched.step()
        red.eval()
        return red

    def normalizar(F):
        media = F.reshape(-1, F.shape[-1]).mean(0)
        desv = F.reshape(-1, F.shape[-1]).std(0) + 1e-6
        return (F - media) / desv, media, desv

    def cross_fit(F, etiqueta_pos, con_ninguno, etiqueta):
        """Cross-fitting en 5 pliegues: cada usuario lo puntua una red que no lo vio.
        etiqueta_pos = columna del producto real; -1 = no esta en el pool (clase 'ninguno')."""
        X, media, desv = normalizar(F)
        Xt = torch.tensor(X, dtype=torch.float32)
        m = F.shape[1]
        # la clase 'ninguno' es la ultima columna del logits (el sesgo 'ninguno')
        y_idx = torch.tensor(np.where(etiqueta_pos >= 0, etiqueta_pos, m), dtype=torch.long)
        s = np.zeros(F.shape[:2], dtype=np.float32)
        b_ning = np.zeros(len(F), dtype=np.float32)
        redes = []
        for p in range(args.pliegues):
            m_tr, m_te = pliegue != p, pliegue == p
            red = entrenar_red(Xt[m_tr], y_idx[m_tr], args.epocas, args.lr, args.ocultos, args.capas)
            with torch.no_grad():
                s[m_te] = red(Xt[m_te]).numpy()
                b_ning[m_te] = float(red.ninguno.item())
            redes.append(red)
            print(f"    {etiqueta} pliegue {p+1}/{args.pliegues} ({time.time()-t0:.0f}s)")
        return s, b_ning, {"media": media, "desv": desv}, redes

    def metricas_de(s, cand, etiqueta_pos, hit, top1, conf, nombre, real_=None):
        """Metricas del ranking. En el pool la posicion del producto real cambia por usuario,
        asi que el HR se calcula con hr_ndcg_en_pool (no con la convencion de columna 0)."""
        if real_ is not None:
            hr, nd, hr_pes, hr1 = hr_ndcg_en_pool(s, cand, real_)
        else:
            hr, nd = hr_ndcg(s)
            hr_pes = hr_pesimista(s)
            hr1 = float(top1.mean())
        cat = catalogo["categoria"].to_numpy()
        cat_top1 = float(np.mean([cat[cand[i, np.argmax(s[i])]] == cat[cand[i, etiqueta_pos[i]]]
                                  for i in range(len(cand))]))
        return {"n": int(len(s)), f"hr@{K}": round(hr, 4), f"ndcg@{K}": round(nd, 4),
                f"hr@{K}_pesimista": round(hr_pes, 4), "hr@1": round(hr1, 4),
                "categoria@1": round(cat_top1, 4),
                "confianza": {"ece_top10": round(ece(conf, hit), 4),
                              "media": round(float(conf.mean()), 4),
                              "selectivo_90_top10": punto_selectivo(conf, hit, 0.90),
                              "selectivo_90_top1": punto_selectivo(conf, top1, 0.90),
                              "curva": curva(conf, hit)}}

    # ======================================================== protocolo 1+99
    cand99 = np.load(os.path.join(DATA_SPLITS, "candidatos_test.npy"))[es_eval]
    s99 = tabla_sasrec(modelo_sas, historias, cand99)
    F99 = rasgos(historias, cand99, s99, items, popularidad, catalogo, estrellas)
    pos99 = np.zeros(n_usuarios, dtype=np.int64)
    s, b, norm, redes99 = cross_fit(F99, pos99, pos99, "1+99")
    orden = np.argsort(-s, axis=1)
    hit = (np.take_along_axis(cand99, orden[:, :K], axis=1) == real[:, None]).any(1)
    top1 = cand99[np.arange(n_usuarios), orden[:, 0]] == real
    # en 1+99 el producto real SIEMPRE esta: no hay clase "ninguno". La confianza sale del
    # modelo de contexto (etapa B) y se calibra con isotonia: "confianza 0.9" significa
    # "acierta el 90 % de las veces".
    pc99, GC99 = modelo_confianza(s, np.zeros(n_usuarios), cand99, hit, "1+99",
                                   historias, items, popularidad, catalogo)
    conf, cal99 = calibrar(pc99, hit, RNG)
    METRICAS["protocolo_1_99"] = metricas_de(s, cand99, pos99, hit, top1, conf, "1+99")
    print(json.dumps({k: v for k, v in METRICAS["protocolo_1_99"].items() if k != "confianza"} |
                      {"confianza": {k: v for k, v in METRICAS["protocolo_1_99"]["confianza"].items()
                                     if k != "curva"}}, indent=2))

    # ======================================================== pool del servicio
    variantes = [("600 = 300 pop + 300 sim (actual)", None)]
    if args.barrido_pool:
        variantes += [("cat 1000 + pop 300 + sim 300", (1000, 300, 300))]
    for nombre, params in variantes:
        if params is None:
            pools = pool_servicio(items, historias, popularidad, top_pop, args.pool)
        else:
            try:
                pools = np.stack([pool_por_categoria(h, *params) for h in historias])
            except Exception as e:
                print(f"  (variante de pool '{nombre}' omitida: {e})")
                continue
        pos = np.array([int(np.flatnonzero(p == y)[0]) if (p == y).any() else -1
                        for p, y in zip(pools, real)])
        sp = tabla_sasrec(modelo_sas, historias, pools)
        Fp = rasgos(historias, pools, sp, items, popularidad, catalogo, estrellas)
        ypos = np.where(pos >= 0, pos, 0)
        sP, bP, normP, redesP = cross_fit(Fp, ypos, pos, "pool")
        ordenP = np.argsort(-sP, axis=1)
        hitP = (np.take_along_axis(pools, ordenP[:, :K], axis=1) == real[:, None]).any(1)
        top1P = pools[np.arange(n_usuarios), ordenP[:, 0]] == real
        confP = softmax_confianza(sP, bP, temperatura=5.0)
        pcP, GCP = modelo_confianza(sP, bP, pools, hitP, "pool",
                                    historias, items, popularidad, catalogo)
        confP, calP = calibrar(pcP, hitP, RNG)
        s_pop_pool = popularidad[pools] + RNG.random(pools.shape) * 1e-6
        hr_pop_pool = float(np.mean([real[i] in pools[i][np.argsort(-s_pop_pool[i])[:K]].tolist()
                                     if real[i] in set(pools[i].tolist()) else False
                                     for i in range(n_usuarios)]))
        m = metricas_de(sP, pools, ypos, hitP, top1P, confP, "pool", real_=real)
        m.update({"pool": int(pools.shape[1]), "real_en_pool": round(float((pos >= 0).mean()), 4),
                  "popularidad": {f"hr@{K}": round(hr_pop_pool, 4)}})
        METRICAS.setdefault("pool_servicio", {})[nombre] = m
        # se guardan los puntajes: recalcular metricas no obliga a reentrenar
        os.makedirs(os.path.join(MODELOS, "reranker_mlp"), exist_ok=True)
        np.savez_compressed(os.path.join(MODELOS, "reranker_mlp", "puntajes_pool.npz"),
                            s=sP, pools=pools, real=real, pos=pos)
        print(f"\n  POOL: {nombre}  tam={pools.shape[1]}  real_en_pool={m['real_en_pool']}")
        print(json.dumps({k: v for k, v in m.items() if k != "confianza"} |
                          {"confianza": {k: v for k, v in m["confianza"].items() if k != "curva"}},
                         indent=2))

    # ======================================================== guardar
    carpeta = nueva_carpeta("reranker_mlp")
    # el modelo que se PUBLICA (red.pt) es el del POOL: es el que usa el backend. El del
    # protocolo 1+99 se guarda aparte para las metricas del informe.
    torch.save({"state_dict": redesP[0].state_dict(), "media": normP["media"], "desv": normP["desv"],
                "rasgos": RASGOS, "n_rasgos": len(RASGOS), "ocultos": args.ocultos,
                "capas": args.capas, "b_ninguno": float(redesP[0].ninguno.item())},
               os.path.join(carpeta, "red.pt"))
    torch.save({"state_dict": redes99[0].state_dict(), "media": norm["media"], "desv": norm["desv"],
                "rasgos": RASGOS, "n_rasgos": len(RASGOS), "ocultos": args.ocultos,
                "capas": args.capas, "b_ninguno": float(redes99[0].ninguno.item())},
               os.path.join(carpeta, "red_1p99.pt"))
    try:
        exportar_onnx(redesP[0], normP["media"], normP["desv"], os.path.join(carpeta, "red.onnx"))
    except Exception as e:
        print(f"  (no se pudo exportar a ONNX: {e})")
    with open(os.path.join(carpeta, "calibrador.json"), "w", encoding="utf-8") as f:
        json.dump({"1p99": cal99, "pool": calP}, f)
    with open(os.path.join(carpeta, "pesos.json"), "w", encoding="utf-8") as f:
        json.dump({"formato": "mlp", "rasgos": RASGOS, "red": "red.pt", "onnx": "red.onnx",
                   "red_1p99": "red_1p99.pt", "calibrador": "calibrador.json",
                   "b_ninguno": float(redesP[0].ninguno.item()), "pool": args.pool,
                   "sasrec": os.path.basename(ruta_sas),
                   "two_tower": os.path.basename(ck["two_tower"])}, f, indent=2)
    with open(os.path.join(carpeta, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(METRICAS, f, indent=2)
    print(f"OK -> {carpeta}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
