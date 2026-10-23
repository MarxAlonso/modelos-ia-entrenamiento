"""
Paso 5: arma models/servir/ (lo unico que lee el backend) con la ultima version de cada
modelo y deja constancia en models/registro_versiones.json.

Uso:  python src/publicar.py [--two-tower vNNN_...] [--sasrec vNNN_...]

models/servir/
  encoder.onnx + tokenizer.json ...   torre de consulta del two-tower (Transformer multilingue)
  items.npy                           embeddings de los 71k productos
  sasrec.onnx                         Transformer secuencial para re-rankear
  reranker.json                       pesos del re-ranker lineal (src/reranker.py) que combina
                                      SASRec, similitud, categoria, tienda y popularidad
  catalogo.csv                        productos con categoria/publico (filtros de Laya)
  historial.npz + usuarios.json       compras de cada usuario, en orden (con estrellas y fecha)
  resenas.parquet                     texto de la resena de cada compra (mismo orden que historial)
  manifiesto.json                     versiones, metricas y sha256 de cada archivo
"""
import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from comun import DATA_SPLITS, MODELOS, SERVIR
from two_tower import ultima_version

ARCHIVOS_TOKENIZER = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def reemplazar(tmp, destino, intentos=10):
    """En Windows el antivirus o un backend en marcha pueden bloquear un momento los archivos
    recien escritos (el .onnx pesa ~470 MB): se reintenta y, si no, se copia."""
    import time
    for _ in range(intentos):
        try:
            if os.path.isdir(destino):
                shutil.rmtree(destino)
            os.replace(tmp, destino)
            return
        except PermissionError:
            time.sleep(2)
    shutil.copytree(tmp, destino, dirs_exist_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--two-tower")
    ap.add_argument("--sasrec")
    ap.add_argument("--reranker", help="vNNN_... (por defecto el ultimo entrenado sobre ese SASRec; "
                                       "'ninguno' = ordenar con RRF como antes)")
    ap.add_argument("--calidad", help="carpeta vNNN_... de models/calidad (modelo de resenas)")
    args = ap.parse_args()
    tt = os.path.join(MODELOS, "two_tower", args.two_tower) if args.two_tower else ultima_version("two_tower")
    sr = os.path.join(MODELOS, "sasrec", args.sasrec) if args.sasrec else ultima_version("sasrec")
    m_sr = json.load(open(os.path.join(sr, "metricas.json"), encoding="utf-8"))
    origen_tt = m_sr.get("two_tower_version", m_sr.get("two_tower"))  # v001-v002 lo guardaban en "two_tower"
    if not isinstance(origen_tt, str) or os.path.normpath(origen_tt) != os.path.normpath(tt):
        raise SystemExit(f"El SASRec {sr} se entreno sobre los embeddings de {origen_tt}, no de {tt}.")
    m_tt = json.load(open(os.path.join(tt, "metricas.json"), encoding="utf-8"))

    tmp = SERVIR + "_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    shutil.copy(os.path.join(tt, "encoder.onnx"), tmp)
    shutil.copy(os.path.join(tt, "items.npy"), tmp)
    for a in ARCHIVOS_TOKENIZER:
        if os.path.exists(os.path.join(tt, "hf", a)):
            shutil.copy(os.path.join(tt, "hf", a), tmp)
    shutil.copy(os.path.join(sr, "sasrec.onnx"), tmp)
    rr, m_rr = None, None
    if args.reranker != "ninguno":
        # hay tres formatos de re-ranker: lineal (src/reranker.py), GBDT (src/reranker_gbdt.py)
        # y MLP en ONNX (src/reranker_mlp.py). Se publica el que diga --reranker.
        base_rr = os.path.join(MODELOS, "reranker")
        base_mlp = os.path.join(MODELOS, "reranker_mlp")
        candidatas = []
        if args.reranker:
            candidatas = [args.reranker]
        else:
            for base in (base_mlp, base_rr):
                if os.path.isdir(base):
                    candidatas += [os.path.join(base, d) for d in sorted(os.listdir(base), reverse=True)
                                   if d[:1] == "v"]
        for v in candidatas:
            vdir = v if os.path.isabs(v) else (os.path.join(base_mlp, v) if os.path.exists(os.path.join(base_mlp, v))
                                               else os.path.join(base_rr, v))
            if not os.path.isdir(vdir):
                continue
            p = os.path.join(vdir, "pesos.json")
            if not os.path.exists(p):
                continue
            w = json.load(open(p, encoding="utf-8"))
            if w.get("sasrec") != os.path.basename(sr):
                continue
            if w.get("formato") in (None, "lineal") and "b_ninguno" not in w:
                continue
            rr = vdir
            break
        if args.reranker and rr is None:
            raise SystemExit(f"El re-ranker {args.reranker} no se entreno sobre el SASRec {os.path.basename(sr)}.")
        if rr:
            pesos = json.load(open(os.path.join(rr, "pesos.json"), encoding="utf-8"))
            # el backend usa el re-ranker con los rasgos de src/rasgos.py (copiado aqui)
            shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rasgos.py"),
                        os.path.join(tmp, "rasgos.py"))
            for aux in ("modelo_gbdt.txt", "modelo_gbdt_1p99.txt", "red.pt", "red.onnx",
                        "calibrador.json"):
                if os.path.exists(os.path.join(rr, aux)):
                    shutil.copy(os.path.join(rr, aux), os.path.join(tmp, aux))
            # el calibrador (isotonia) viaja dentro del pesos.json para que el backend no
            # tenga que saber de donde salio
            if os.path.exists(os.path.join(rr, "calibrador.json")):
                cal = json.load(open(os.path.join(rr, "calibrador.json"), encoding="utf-8"))
                if pesos.get("formato") == "gbdt" and "1p99" in cal:
                    pesos["calibrador_valores"] = cal["1p99"]
            if pesos.get("formato") == "mlp":
                ck_red = torch.load(os.path.join(rr, "red.pt"), map_location="cpu", weights_only=False)
                pesos["b_ninguno"] = float(ck_red.get("b_ninguno", 0.0))
            shutil.copy(os.path.join(rr, "pesos.json"), os.path.join(tmp, "reranker.json"))
            m_rr = json.load(open(os.path.join(rr, "metricas.json"), encoding="utf-8"))

    cat = pd.read_parquet(os.path.join(DATA_SPLITS, "catalogo.parquet"))
    # SIN FUGA: el catalogo trae rating_promedio y numero_resenas calculados con TODAS las
    # interacciones (incluida la del test). Aqui se recalculan con el split train
    # solamente, que es lo que un sistema en linea sabria del producto.
    tr = pd.read_parquet(os.path.join(DATA_SPLITS, "train.parquet"))
    agg = tr.groupby("item_idx").agg(rating_promedio=("rating", "mean"), numero_resenas=("rating", "count"))
    cat["rating_promedio"] = cat["item_idx"].map(agg["rating_promedio"]).fillna(0.0).astype(float)
    cat["numero_resenas"] = cat["item_idx"].map(agg["numero_resenas"]).fillna(0).astype(int)
    cat.drop(columns=["texto"]).to_csv(os.path.join(tmp, "catalogo.csv"), index=False)
    todo = pd.concat([pd.read_parquet(os.path.join(DATA_SPLITS, f"{n}.parquet"))
                      for n in ("train", "val", "test", "test_frio")]).sort_values(["user_id", "fecha"])
    usuarios = todo["user_id"].unique()  # ya ordenados por el sort
    fila = {u: k for k, u in enumerate(usuarios)}
    u_idx = todo["user_id"].map(fila).to_numpy()
    np.savez_compressed(os.path.join(tmp, "historial.npz"),
                        ptr=np.searchsorted(u_idx, np.arange(len(usuarios) + 1)).astype(np.int64),
                        items=todo["item_idx"].to_numpy(dtype=np.int64),
                        popularidad=np.bincount(todo["item_idx"], minlength=len(cat)).astype(np.int64),
                        ratings=todo["rating"].to_numpy(dtype=np.int8),
                        fechas=pd.to_datetime(todo["fecha"]).dt.strftime("%Y%m%d").astype(np.int32).to_numpy())
    pd.DataFrame({"resena": todo["resena"].fillna("").str.slice(0, 280).to_numpy()}).to_parquet(
        os.path.join(tmp, "resenas.parquet"), index=False)
    with open(os.path.join(tmp, "usuarios.json"), "w", encoding="utf-8") as f:
        json.dump(fila, f)

    # ------------------------------------------------ modelo de calidad de la resena
    m_cal = None
    base_cal = os.path.join(MODELOS, "calidad")
    v_cal = args.calidad
    if v_cal is None and os.path.isdir(base_cal):
        vs = sorted(d for d in os.listdir(base_cal) if d[:1] == "v")
        v_cal = vs[-1] if vs else None
    if v_cal and os.path.exists(os.path.join(base_cal, v_cal, "calidad.onnx")):
        shutil.copy(os.path.join(base_cal, v_cal, "calidad.onnx"), os.path.join(tmp, "calidad.onnx"))
        m_cal = json.load(open(os.path.join(base_cal, v_cal, "metricas.json"), encoding="utf-8"))

    version = f"srp-{datetime.now():%Y%m%d-%H%M}"
    evaluacion = os.path.join(MODELOS, "evaluacion", "evaluacion.json")
    manifiesto = {
        "version": version, "creado": datetime.now().isoformat(timespec="seconds"),
        "two_tower": os.path.basename(tt), "sasrec": os.path.basename(sr),
        "reranker": os.path.basename(rr) if rr else None,
        "reranker_formato": (json.load(open(os.path.join(rr, "pesos.json"), encoding="utf-8")).get("formato")
                             if rr else None),
        "calidad": v_cal,
        "metricas": {"two_tower_test": {k: v for k, v in m_tt["test"].items()},
                     "comparacion_usuarios_eval": {k: v for k, v in (m_rr or m_sr).items()
                                                   if isinstance(v, dict) and "hr@10" in v},
                     **({"catalogo_completo": m_rr["catalogo_completo"], "confianza": m_rr["confianza"]}
                        if m_rr and "catalogo_completo" in m_rr else {}),
                     **({"pool_servicio": m_rr["pool_servicio"], "protocolo_1_99": m_rr["protocolo_1_99"]}
                        if m_rr and "pool_servicio" in m_rr else {}),
                     **({"calidad_resena": m_cal["test"]} if m_cal else {}),
                     **({"evaluacion": json.load(open(evaluacion, encoding="utf-8"))}
                        if os.path.exists(evaluacion) else {})},
        "archivos": {},
    }
    for a in sorted(os.listdir(tmp)):
        ruta = os.path.join(tmp, a)
        manifiesto["archivos"][a] = {"sha256": sha256(ruta), "mb": round(os.path.getsize(ruta) / 1e6, 2)}
    with open(os.path.join(tmp, "manifiesto.json"), "w", encoding="utf-8") as f:
        json.dump(manifiesto, f, indent=2)
    reemplazar(tmp, SERVIR)

    reg = os.path.join(MODELOS, "registro_versiones.json")
    registro = json.load(open(reg, encoding="utf-8")) if os.path.exists(reg) else []
    registro.append({k: manifiesto[k] for k in ("version", "creado", "two_tower", "sasrec", "reranker", "metricas")})
    with open(reg, "w", encoding="utf-8") as f:
        json.dump(registro, f, indent=2)
    for a, info in manifiesto["archivos"].items():
        print(f"  {a:<26} {info['mb']:>8.2f} MB")
    print(f"OK -> {SERVIR} ({version})")


if __name__ == "__main__":
    main()
