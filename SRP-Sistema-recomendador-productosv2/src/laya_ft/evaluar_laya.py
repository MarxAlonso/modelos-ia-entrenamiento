"""
Paso 4c: evalua un checkpoint de Laya sobre data/laya/test.jsonl.

    python src/laya_ft/evaluar_laya.py                                   # base zero-shot
    python src/laya_ft/evaluar_laya.py --modelo models/laya_srp  # afinado en Kaggle

Metricas por workflow (resenas_moda / consultas_compra), por idioma y por pregunta:
  choice y noul: accuracy (choice ademas: precision y cobertura por umbral de confianza)
  score: accuracy del nivel, MAE y % a un nivel o menos
Guarda el reporte en data/laya/reporte_<nombre>.json para comparar antes/despues.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comun import DATA_LAYA  # noqa: E402


UMBRALES = (0.35, 0.5, 0.7, 0.9)


def metricas(filas):
    """Por pregunta: choice/noul -> accuracy; score -> accuracy del nivel, MAE y % a <= 1 nivel."""
    por_q = {}
    for f in filas:
        for qid, q in f["questions"].items():
            g, p, d = f["gold"][qid], f["pred"][qid], por_q.setdefault(qid, {"ok": [], "err": [], "uno": [], "conf": []})
            if q["type"] == "choice":
                d["ok"].append(p["choice"] == str(g["label"]))
                d["conf"].append(float(p.get("answer_confidence", 0.0)))
            elif q["type"] == "noul":
                d["ok"].append((float(p["noul"]) >= 0.5) == (str(g["label"]).lower() == "true"))
            else:
                probs = [p["probabilities"].get(str(i), 0.0) for i in range(len(q["criteria"]))]
                d["ok"].append(int(np.argmax(probs)) == int(g["label"]))
                e = abs(float(p["score"]) - int(g["label"]))
                d["err"].append(e); d["uno"].append(e <= 1.0)
    salida = {"n": len(filas)}
    for qid, d in por_q.items():
        salida[qid] = {"accuracy": round(float(np.mean(d["ok"])), 4)}
        if d["conf"]:
            # el backend solo aplica la respuesta si su confianza alcanza LAYA_UMBRAL: la
            # precision que importa es la de las respuestas usadas, junto con cuantas se usan
            ok, conf = np.array(d["ok"]), np.array(d["conf"])
            salida[qid]["selectivo"] = {
                str(u): {"precision": round(float(ok[conf >= u].mean()), 4) if (conf >= u).any() else None,
                         "cobertura": round(float((conf >= u).mean()), 4)}
                for u in UMBRALES}
        if d["err"]:
            salida[qid].update(mae=round(float(np.mean(d["err"])), 4), dentro_de_1=round(float(np.mean(d["uno"])), 4))
    return salida


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="convaiinnovations/laya-multilingual")
    ap.add_argument("--n", type=int, default=0, help="limitar a n casos (0 = todos)")
    ap.add_argument("--dispositivo", default=None)
    ap.add_argument("--workflow", default=None, help="resenas_moda o consultas_compra (por defecto ambos)")
    args = ap.parse_args()
    import laya

    casos = [json.loads(l) for l in open(os.path.join(DATA_LAYA, "test.jsonl"), encoding="utf-8")]
    if args.workflow:
        casos = [c for c in casos if c["workflow"] == args.workflow]
    if args.n:
        casos = casos[:: max(1, len(casos) // args.n)][: args.n]
    agente = laya.load(args.modelo, device=args.dispositivo)
    filas, t0 = [], time.time()
    for k, c in enumerate(casos):
        r = agente.predict(json.loads(c["state"]), json.loads(c["questions"]))
        filas.append({"workflow": c["workflow"], "idioma": c["idioma"], "questions": json.loads(c["questions"]),
                      "gold": json.loads(c["gold"]), "pred": r["answers"]})
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(casos)} {time.time() - t0:.0f}s", flush=True)
    reporte = {"modelo": args.modelo, "seg_por_caso": round((time.time() - t0) / len(casos), 3)}
    for wf in sorted({f["workflow"] for f in filas}):
        del_wf = [f for f in filas if f["workflow"] == wf]
        reporte[wf] = {"todos": metricas(del_wf)}
        for idioma in sorted({f["idioma"] for f in del_wf}):
            reporte[wf][idioma] = metricas([f for f in del_wf if f["idioma"] == idioma])
    print(json.dumps(reporte, indent=2, ensure_ascii=False))
    nombre = os.path.basename(args.modelo.rstrip("/\\")).replace("/", "_")
    sufijo = f"_{args.workflow}" if args.workflow else ""
    with open(os.path.join(DATA_LAYA, f"reporte_{nombre}{sufijo}.json"), "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
