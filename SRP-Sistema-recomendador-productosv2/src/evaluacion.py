"""
Evaluacion centralizada: TODAS las metricas del informe en un solo lugar.

Este modulo es la fuente de verdad. Re-calcula, con el mismo protocolo y sin fuga, lo que
se reporta en el documento del curso, y lo deja en:
    models/evaluacion/evaluacion.json   (datos crudos)
    models/evaluacion/resumen.md        (tablas listas para pegar en el informe)

Tres cuidados que este modulo aplica SIEMPRE (y que conviene revisar si se toca el codigo):

1. EMPATES. hr_ndcg (src/comun.py) rompe los empates con un epsilon aleatorio. Sin eso, un
   modelo que devuelve la misma puntuacion a cientos de productos se llevaria el 100 % de
   aciertos. Se reporta ademas la version pesimista (el empate cuenta como fallo), que es
   la que se usa para las afirmaciones del informe.

2. FUGA EN LOS AGREGADOS DE PRODUCTO. rating_promedio y numero_resenas del catalogo estan
   calculados con TODAS las interacciones (incluida la del test). Aqui se recalculan con el
   split train solamente. Medido: la diferencia es de 0.026 en HR@10.

3. USUARIOS DE EVALUACION. Los mismos que en src/sasrec.py (leave-last-out: el producto que
   se evalua es el ultimo del usuario y el modelo no lo vio).

Uso:  python src/evaluacion.py
"""
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from comun import DATA_SPLITS, K, MODELOS, SEMILLA, hr_ndcg
from rasgos import RASGOS, rasgos
from reranker_gbdt import (aplicar_calibrador, ajustar_isotonico, curva, ece, hr_pesimista,
                            pool_servicio, punto_selectivo, tabla_sasrec)
from sasrec import SASRec, particion_eval

RNG = np.random.default_rng(SEMILLA)
BOOT = 1000


def ic95(muestra):
    """Intervalo de confianza al 95 % por bootstrap (percentiles 2.5 y 97.5)."""
    return [round(float(np.percentile(muestra, 2.5)), 4), round(float(np.percentile(muestra, 97.5)), 4)]


def bootstrap(aciertos, n=BOOT, semilla=SEMILLA):
    """IC al 95 % del acierto medio por bootstrap sobre los usuarios."""
    rng = np.random.default_rng(semilla)
    a = aciertos.astype(np.float64)
    m = rng.integers(0, len(a), size=(n, len(a)))
    return ic95(a[m].mean(1))


def main():
    t0 = time.time()
    OUT = os.path.join(MODELOS, "evaluacion")
    os.makedirs(OUT, exist_ok=True)
    train, val, test = (pd.read_parquet(os.path.join(DATA_SPLITS, f"{n}.parquet")) for n in ("train", "val", "test"))
    catalogo = pd.read_parquet(os.path.join(DATA_SPLITS, "catalogo.parquet")).sort_values("item_idx")
    popularidad = np.bincount(train["item_idx"], minlength=len(catalogo)).astype(np.float32)
    # sin fuga: agregados de producto solo con el split train
    agg = train.groupby("item_idx").agg(rating_promedio=("rating", "mean"), numero_resenas=("rating", "count"))
    catalogo = catalogo.copy()
    catalogo["rating_promedio"] = catalogo["item_idx"].map(agg["rating_promedio"]).fillna(0.0).astype(float)
    catalogo["numero_resenas"] = catalogo["item_idx"].map(agg["numero_resenas"]).fillna(0).astype(float)

    # ---------------------------------------------------------- que modelos se evaluan
    pub = json.load(open(os.path.join(MODELOS, "servir", "manifiesto.json"), encoding="utf-8"))
    ruta_sas = os.path.join(MODELOS, "sasrec", pub["sasrec"])
    ck = torch.load(os.path.join(ruta_sas, "sasrec.pt"), map_location="cpu", weights_only=False)
    a = ck["args"]
    items = np.load(os.path.join(ck["two_tower"], "items.npy")).astype(np.float32)
    seq, es_eval, E, prev, _ = particion_eval(train, val, test, a.get("frac_eval", 0.3), a.get("min_compras", 2))
    filas = test[es_eval]
    historias = [np.asarray(prev.get(u, []), dtype=np.int64) for u in filas["user_id"]]
    real = filas["item_idx"].to_numpy()
    n = len(historias)
    hist_previo = pd.concat([train, val]).sort_values(["user_id", "fecha"])
    mapa_est = {u: g["rating"].to_numpy(np.float32) for u, g in hist_previo.groupby("user_id")}
    estrellas = [mapa_est.get(u, np.array([], np.float32)) for u in filas["user_id"]]
    print(f"usuarios de evaluacion E = {n:,}   SASRec = {pub['sasrec']}")

    modelo_sas = SASRec(items, np.log1p(popularidad), dim=a.get("dim", 64))
    modelo_sas.load_state_dict(ck["state_dict"])
    modelo_sas.eval()
    cat = catalogo["categoria"].to_numpy()

    # ---------------------------------------------------------- protocolo 1+99
    cand = np.load(os.path.join(DATA_SPLITS, "candidatos_test.npy"))[es_eval]
    s_sas = tabla_sasrec(modelo_sas, historias, cand)
    F = rasgos(historias, cand, s_sas, items, popularidad, catalogo, estrellas)
    q = np.stack([items[h].mean(0) for h in historias])
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    s_tt = np.einsum("nd,nkd->nk", q, items[cand])

    modelos = {}
    modelos["popularidad"] = popularidad[cand] + RNG.random(cand.shape) * 1e-6
    modelos["sasrec"] = s_sas
    modelos["two_tower_item_to_item"] = s_tt

    # re-ranker lineal (src/reranker.py) si esta disponible
    base_rr = os.path.join(MODELOS, "reranker")
    v_lineal = None
    for d in sorted(os.listdir(base_rr), reverse=True):
        p = os.path.join(base_rr, d, "pesos.json")
        if os.path.exists(p):
            w = json.load(open(p, encoding="utf-8"))
            if w.get("formato") is None and "w" in w:
                v_lineal = (d, w); break
    if v_lineal:
        w = v_lineal[1]
        Fw = F[:, :, :len(w["w"])]
        modelos["reranker_lineal"] = ((Fw - np.array(w["media"])) / np.array(w["desv"])) @ np.array(w["w"])

    # re-ranker GBDT (lambdarank) sobre el protocolo 1+99
    v_gbdt = None
    for d in sorted(os.listdir(base_rr), reverse=True):
        p = os.path.join(base_rr, d, "pesos.json")
        if os.path.exists(p):
            w = json.load(open(p, encoding="utf-8"))
            if w.get("formato") == "gbdt" and os.path.exists(os.path.join(base_rr, d, "modelo_gbdt_1p99.txt")):
                v_gbdt = (d, w); break
    if v_gbdt:
        import lightgbm as lgb
        m = lgb.Booster(model_file=os.path.join(base_rr, v_gbdt[0], "modelo_gbdt_1p99.txt"))
        modelos["reranker_gbdt"] = m.predict(F.reshape(-1, F.shape[-1])).reshape(F.shape[:2])

    # re-ranker MLP (listwise) sobre el protocolo 1+99
    base_mlp = os.path.join(MODELOS, "reranker_mlp")
    v_mlp = None
    if os.path.isdir(base_mlp):
        for d in sorted(os.listdir(base_mlp), reverse=True):
            if os.path.exists(os.path.join(base_mlp, d, "red_1p99.pt")):
                v_mlp = (d, os.path.join(base_mlp, d)); break
    if v_mlp:
        from reranker_mlp import Ranker
        ck_mlp = torch.load(os.path.join(v_mlp[1], "red_1p99.pt"), map_location="cpu", weights_only=False)
        red = Ranker(ck_mlp["n_rasgos"], ck_mlp["ocultos"], ck_mlp["capas"])
        red.load_state_dict(ck_mlp["state_dict"]); red.eval()
        X = (F - ck_mlp["media"]) / ck_mlp["desv"]
        with torch.no_grad():
            modelos["reranker_mlp"] = red(torch.tensor(X, dtype=torch.float32)).numpy()

    print(f"  modelos evaluados: {list(modelos)}  ({time.time()-t0:.0f}s)")

    # ---------------------------------------------------------- metricas 1+99
    tabla = {}
    aciertos_por_modelo = {}
    for nombre, s in modelos.items():
        hr, nd = hr_ndcg(s)
        rango = (s[:, 1:] >= s[:, :1]).sum(1)
        hit = rango < K
        top1 = (cand[np.arange(n), np.argmax(s, axis=1)] == real)
        cat_top1 = float(np.mean([cat[cand[i, np.argmax(s[i])]] == cat[cand[i, 0]] for i in range(n)]))
        aciertos_por_modelo[nombre] = hit
        tabla[nombre] = {
            f"hr@{K}": round(hr, 4), f"hr@{K}_pesimista": round(hr_pesimista(s), 4),
            f"ic95_hr@{K}": bootstrap(hit),
            f"ndcg@{K}": round(nd, 4), "hr@1": round(float(top1.mean()), 4),
            "categoria@1": round(cat_top1, 4),
        }
    print(json.dumps(tabla, indent=2))

    # ---------------------------------------------------------- pool del servicio
    top_pop = np.argsort(-popularidad)[:6000]
    pools = pool_servicio(items, historias, popularidad, top_pop, 600)
    pos = np.array([int(np.flatnonzero(p == y)[0]) if (p == y).any() else -1 for p, y in zip(pools, real)])
    s_pop_pool = popularidad[pools] + RNG.random(pools.shape) * 1e-6
    hr_pop_pool = float(np.mean([real[i] in pools[i][np.argsort(-s_pop_pool[i])[:K]].tolist()
                                 if real[i] in set(pools[i].tolist()) else False for i in range(n)]))
    pool_tab = {"protocolo": f"pool del servicio (600 = 300 populares + 300 similares), recall@{K} "
                            f"contra los {len(catalogo):,} productos del catalogo",
                "real_en_pool": round(float((pos >= 0).mean()), 4),
                "popularidad": {f"hr@{K}": round(hr_pop_pool, 4)}}
    if v_mlp:
        from reranker_mlp import Ranker, softmax_confianza
        sp = tabla_sasrec(modelo_sas, historias, pools)
        Fp = rasgos(historias, pools, sp, items, popularidad, catalogo, estrellas)
        ck_mlp = torch.load(os.path.join(v_mlp[1], "red.pt"), map_location="cpu", weights_only=False)
        red = Ranker(ck_mlp["n_rasgos"], ck_mlp["ocultos"], ck_mlp["capas"])
        red.load_state_dict(ck_mlp["state_dict"]); red.eval()
        Xp = (Fp - ck_mlp["media"]) / ck_mlp["desv"]
        with torch.no_grad():
            sP = red(torch.tensor(Xp, dtype=torch.float32)).numpy()
            bP = float(ck_mlp.get("b_ninguno", red.ninguno.item()))
        from reranker_mlp import hr_ndcg_en_pool
        hr_p, nd_p, hr_pes_p, hr1_p = hr_ndcg_en_pool(sP, pools, real)
        ordenP = np.argsort(-sP, axis=1)
        hitP = (np.take_along_axis(pools, ordenP[:, :K], axis=1) == real[:, None]).any(1)
        confP = softmax_confianza(sP, bP)
        pool_tab["reranker_mlp"] = {f"hr@{K}": round(hr_p, 4), f"hr@{K}_pesimista": round(hr_pes_p, 4),
                                    f"ndcg@{K}": round(nd_p, 4),
                                    "hr@1": round(hr1_p, 4),
                                    "confianza_media": round(float(confP.mean()), 4),
                                    "ece": round(ece(confP, hitP), 4),
                                    "selectivo_90": punto_selectivo(confP, hitP, 0.90)}
    print(json.dumps(pool_tab, indent=2))

    # ---------------------------------------------------------- confiabilidad del GBDT (1+99)
    conf_tab = {}
    if v_gbdt:
        cal_f = os.path.join(base_rr, v_gbdt[0], "calibrador.json")
        if os.path.exists(cal_f):
            cal = json.load(open(cal_f, encoding="utf-8"))["1p99"]
            s = modelos["reranker_gbdt"]
            orden = np.argsort(-s, axis=1)
            hit = (np.take_along_axis(cand, orden[:, :K], axis=1) == real[:, None]).any(1)
            top1 = (cand[np.arange(n), orden[:, 0]] == real)
            # el GBDT no trae confianza propia: se usa la masa de probabilidad del softmax
            # sobre los puntajes normalizados, calibrada con la isotonia guardada
            z = s - s.max(1, keepdims=True)
            p = np.exp(z); p /= p.sum(1, keepdims=True)
            masa = np.take_along_axis(p, orden[:, :K], axis=1).sum(1)
            conf = aplicar_calibrador(cal, masa)
            conf_tab = {"ece_top10": round(ece(conf, hit), 4),
                        "selectivo_90_top10": punto_selectivo(conf, hit, 0.90),
                        "selectivo_90_top1": punto_selectivo(conf, top1, 0.90),
                        "curva": curva(conf, hit)}

    # ---------------------------------------------------------- calidad de la resena
    base_cal = os.path.join(MODELOS, "calidad")
    cal_tab = None
    if os.path.isdir(base_cal):
        vs = sorted(d for d in os.listdir(base_cal) if d[:1] == "v")
        if vs:
            mc = json.load(open(os.path.join(base_cal, vs[-1], "metricas.json"), encoding="utf-8"))
            cal_tab = {"version": vs[-1], **mc["test"]}

    # ---------------------------------------------------------- guardar
    evaluacion = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "notas": {
            "empates": "hr_ndcg rompe los empates con un epsilon aleatorio; la version "
                       "'pesimista' cuenta el empate como fallo.",
            "fuga": "rating_promedio y numero_resenas recalculados con el split train solamente.",
            "protocolo": f"1 positivo + 99 negativos al azar, @{K}, sobre {n:,} usuarios de "
                         f"evaluacion (leave-last-out, los mismos que en src/sasrec.py).",
        },
        "modelos_1_99": tabla,
        "pool_servicio": pool_tab,
        "confiabilidad": conf_tab,
        "calidad_resena": cal_tab,
    }
    with open(os.path.join(OUT, "evaluacion.json"), "w", encoding="utf-8") as f:
        json.dump(evaluacion, f, indent=2)

    # tabla markdown para el informe
    L = []
    L.append(f"# Resultados (generado el {evaluacion['generado'][:10]})\n")
    L.append(f"Protocolo: 1 positivo + 99 negativos al azar, @{K}, {n:,} usuarios de evaluacion.")
    L.append("Los intervalos son IC al 95 % por bootstrap sobre los usuarios.\n")
    L.append("| Modelo | HR@10 | IC 95 % | HR@10 pesimista | NDCG@10 | HR@1 | categoria@1 |")
    L.append("|---|---|---|---|---|---|---|")
    for nombre, v in tabla.items():
        ic = v[f"ic95_hr@{K}"]
        L.append(f"| {nombre} | {v[f'hr@{K}']:.4f} | [{ic[0]:.3f}, {ic[1]:.3f}] | "
                 f"{v[f'hr@{K}_pesimista']:.4f} | {v[f'ndcg@{K}']:.4f} | {v['hr@1']:.4f} | {v['categoria@1']:.4f} |")
    L.append("\n## Catalogo completo (pool del servicio)\n")
    L.append("| Metrica | Valor |")
    L.append("|---|---|")
    for k, v in pool_tab.items():
        L.append(f"| {k} | {v if not isinstance(v, dict) else json.dumps(v, ensure_ascii=False)} |")
    if conf_tab:
        L.append("\n## Confiabilidad (prediccion selectiva)\n")
        L.append("| Metrica | Valor |")
        L.append("|---|---|")
        for k, v in conf_tab.items():
            if k != "curva":
                L.append(f"| {k} | {json.dumps(v, ensure_ascii=False)} |")
    if cal_tab:
        L.append("\n## Calidad de la resena (modelo de satisfaccion)\n")
        L.append("| Metrica | Valor |")
        L.append("|---|---|")
        for grupo, v in cal_tab.items():
            if isinstance(v, dict):
                for k, val in v.items():
                    L.append(f"| {grupo}.{k} | {val} |")
    with open(os.path.join(OUT, "resumen.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\nOK -> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
