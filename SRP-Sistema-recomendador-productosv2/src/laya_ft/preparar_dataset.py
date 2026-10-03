"""
Paso 4a: dataset para afinar Laya en las dos tareas que le pide el backend.

  workflow "resenas_moda"      resena -> satisfaccion / recomendaria (gold del rating real)
  workflow "consultas_compra"  consulta del cliente -> categoria / publico / es_regalo /
                               sensibilidad_precio (gold de consultas sinteticas, ver
                               consultas_sinteticas.py)

Laya aprende de distribuciones "gold" por pregunta (RLCD). Aqui el gold sale del rating
real que dio cada cliente, asi que no hace falta etiquetar nada a mano:
  satisfaccion (score 0-4)  80 % de probabilidad en el nivel del rating, 10 % a cada vecino
  recomendaria (noul)       p(si) segun el rating: 1->0.05 2->0.15 3->0.45 4->0.85 5->0.95

Formato de salida = el de LocalLLaMA/typed-decisions, que es lo que lee el notebook oficial
de fine-tuning: cada fila trae id, workflow, y state/questions/gold como strings JSON.

Uso:   python src/laya_ft/preparar_dataset.py [--por-rating 1500] [--consultas 4000]
Salida: data/laya/train.jsonl, data/laya/test.jsonl
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comun import DATA_LAYA, DATA_RAW, DATA_SPLITS, RAIZ, SEMILLA  # noqa: E402
from laya_ft import consultas_sinteticas  # noqa: E402

# Las MISMAS preguntas que usa el backend (fuente unica): el checkpoint afinado reconoce
# estas instrucciones y criterios exactos.
with open(os.path.join(RAIZ, "backend", "app", "preguntas_laya.json"), encoding="utf-8") as _f:
    _P = json.load(_f)
PREGUNTAS = _P["resena"]
PREGUNTAS_CONSULTA = _P["consulta"]
P_RECOMIENDA = {1: 0.05, 2: 0.15, 3: 0.45, 4: 0.85, 5: 0.95}


def gold(rating: int):
    nivel = int(rating) - 1
    p = np.zeros(5)
    p[nivel] = 0.8
    for v in (nivel - 1, nivel + 1):
        if 0 <= v < 5:
            p[v] = 0.1
    p /= p.sum()
    pr = P_RECOMIENDA[int(rating)]
    return {
        "satisfaccion": {"probabilities": {str(i): round(float(x), 4) for i, x in enumerate(p)},
                         "label": nivel, "score": round(float((p * np.arange(5)).sum()), 4)},
        "recomendaria": {"probabilities": {"true": pr, "false": round(1 - pr, 4)},
                         "label": "true" if pr >= 0.5 else "false", "noul": pr},
    }


def fila(i, prefijo, titulo, resena, rating, idioma):
    return {"id": f"{prefijo}-{i}", "workflow": "resenas_moda", "idioma": idioma, "rating": int(rating),
            "state": json.dumps({"producto": titulo, "resena": resena}, ensure_ascii=False),
            "questions": json.dumps(PREGUNTAS, ensure_ascii=False),
            "gold": json.dumps(gold(rating), ensure_ascii=False)}


def fila_consulta(i, prefijo, texto, gold_c):
    return {"id": f"{prefijo}-{i}", "workflow": "consultas_compra", "idioma": "es", "rating": None,
            "state": json.dumps(texto, ensure_ascii=False),
            "questions": json.dumps(PREGUNTAS_CONSULTA, ensure_ascii=False),
            "gold": json.dumps(gold_c, ensure_ascii=False)}


def balancear(df, n, rng):
    """Mismo numero de casos por rating: con 60 % de 5 estrellas Laya aprenderia a decir
    'muy satisfecho' a todo."""
    return pd.concat([g.sample(min(n, len(g)), random_state=rng) for _, g in df.groupby("rating")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--por-rating", type=int, default=1500, help="casos en ingles por rating para train")
    ap.add_argument("--test-por-rating", type=int, default=150)
    ap.add_argument("--consultas", type=int, default=4000, help="consultas sinteticas para train (test = 10 %%)")
    args = ap.parse_args()
    os.makedirs(DATA_LAYA, exist_ok=True)
    cat = pd.read_parquet(os.path.join(DATA_SPLITS, "catalogo.parquet"))[["product_id", "item_idx", "titulo"]]
    leer = lambda n: pd.read_parquet(os.path.join(DATA_SPLITS, f"{n}.parquet"))
    train = leer("train").merge(cat[["item_idx", "titulo"]], on="item_idx")
    evalu = pd.concat([leer("test"), leer("test_frio")]).merge(cat[["item_idx", "titulo"]], on="item_idx")
    for df in (train, evalu):
        # Amazon pone titulos automaticos "Five Stars" / "One Star": delatarian el rating.
        df["resena"] = df["resena"].str.replace(r"^(one|two|three|four|five) stars?\.?\s*", "", case=False, regex=True)
        df.drop(df[df["resena"].str.len() < 20].index, inplace=True)
        df["resena"] = df["resena"].str.slice(0, 1500)

    # Resenas traducidas al espanol: las que pertenecen a test van a test; del resto, 15 %.
    tr = pd.read_csv(os.path.join(DATA_RAW, "resenas_traducidas.csv"), usecols=["user_id", "product_id", "rating", "texto_es"])
    tr = tr.merge(cat, on="product_id").dropna(subset=["texto_es"])
    claves_eval = set(zip(evalu["user_id"], evalu["product_id"]))
    en_eval = tr.apply(lambda f: (f.user_id, f.product_id) in claves_eval, axis=1)
    rng = np.random.default_rng(SEMILLA)
    tr_test = tr[en_eval | (rng.random(len(tr)) < 0.15)]
    tr_train = tr.drop(tr_test.index)

    filas_train = ([fila(i, "en", f.titulo, f.resena, f.rating, "en")
                    for i, f in enumerate(balancear(train, args.por_rating, SEMILLA).itertuples())]
                   + [fila(i, "es", f.titulo, f.texto_es, f.rating, "es") for i, f in enumerate(tr_train.itertuples())])
    filas_test = ([fila(i, "en-test", f.titulo, f.resena, f.rating, "en")
                   for i, f in enumerate(balancear(evalu, args.test_por_rating, SEMILLA).itertuples())]
                  + [fila(i, "es-test", f.titulo, f.texto_es, f.rating, "es") for i, f in enumerate(tr_test.itertuples())])
    rr = random.Random(SEMILLA)
    filas_train += [fila_consulta(i, "q", t, g) for i, (t, g) in enumerate(
        consultas_sinteticas.generar(args.consultas, consultas_sinteticas.PLANTILLAS_TRAIN, rr))]
    filas_test += [fila_consulta(i, "q-test", t, g) for i, (t, g) in enumerate(
        consultas_sinteticas.generar(args.consultas // 10, consultas_sinteticas.PLANTILLAS_TEST, rr))]
    np.random.default_rng(SEMILLA).shuffle(filas_train)

    for nombre, filas in [("train", filas_train), ("test", filas_test)]:
        with open(os.path.join(DATA_LAYA, f"{nombre}.jsonl"), "w", encoding="utf-8") as f:
            for r in filas:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        d = pd.DataFrame(filas)
        decisiones = sum(len(json.loads(r["questions"])) for r in filas)
        print(f"  {nombre}: {len(filas):,} casos ({decisiones:,} decisiones) | workflow={d.workflow.value_counts().to_dict()} "
              f"| idioma={d.idioma.value_counts().to_dict()}")
    viejo = os.path.join(DATA_LAYA, "preguntas_resena.json")
    if os.path.exists(viejo):
        os.remove(viejo)  # reemplazado por backend/app/preguntas_laya.json
    print(f"OK -> {DATA_LAYA}")


if __name__ == "__main__":
    main()
