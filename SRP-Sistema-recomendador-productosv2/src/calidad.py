"""
Calidad de la resena: rating (RMSE) y decision "recomendaria" (accuracy).

Por que hace falta: el re-ranker ordena productos, pero el cliente tambien pregunta
"¿esto me va a gustar?". El informe pide RMSE y este modelo lo responde con un
Transformers afinado sobre las resenas reales y sus estrellas:

    salida 1  satisfaccion  1..5 estrellas  -> RMSE, MAE, exactitud, exactitud +-1 estrella
    salida 2  recomendaria  rating>=4       -> accuracy, F1, AUC, Brier

El encoder es el mismo multilingual MiniLM que usa el two-tower (las torres y el
clasificador comparten la misma base: lo que el producto significa y lo que la gente
opina de el se learnt en el mismo espacio).

Fine-tune en la GTX 1650 (4 GB): vocabulary de embeddings congelada (es el 80 % de los
parametros y casi todos son tokens que nunca aparecen aqui) y lotes de 16.

Uso:   python src/calidad.py [--epocas 2] [--max-datos 60000] [--lote 16]
Salida: models/calidad/vNNN_<fecha>/
        hf/            pesos del clasificador (formato Hugging Face)
        calidad.onnx   el mismo clasificador en ONNX
        metricas.json
"""
import argparse
import json
import math
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from comun import DATA_SPLITS, ENCODER_BASE, MODELOS, SEMILLA

DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DISPOSITIVO == "cuda" and os.environ.get("SRP_FP16") == "1"
MAX_LEN = 96


class Clasificador(torch.nn.Module):
    """MiniLM + dos cabezas: satisfaccion (5 clases) y recomendaria (binaria)."""

    def __init__(self, ruta):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(ruta)
        h = self.transformer.config.hidden_size
        self.cabeza_estrellas = torch.nn.Linear(h, 5)
        self.cabeza_recomienda = torch.nn.Linear(h, 2)

    def forward(self, input_ids, attention_mask):
        h = self.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        emb = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)  # mean pooling
        return self.cabeza_estrellas(emb), self.cabeza_recomienda(emb)


def nueva_carpeta(tipo):
    base = os.path.join(MODELOS, tipo)
    os.makedirs(base, exist_ok=True)
    n = max([int(d[1:4]) for d in os.listdir(base) if d[:1] == "v" and d[1:4].isdigit()], default=0) + 1
    ruta = os.path.join(base, f"v{n:03d}_{datetime.now():%Y%m%d-%H%M}")
    os.makedirs(ruta)
    return ruta


def preparar(df, tok, lote_token=1024):
    """Tokeniza una vez y guarda los ids: tokenizar en cada epoca cuesta mas que entrenar."""
    ids, mask = [], []
    textos = df["resena"].fillna("").astype(str).tolist()
    for s in range(0, len(textos), lote_token):
        b = tok(textos[s:s + lote_token], padding="max_length", truncation=True, max_length=MAX_LEN)
        ids.append(np.asarray(b["input_ids"], dtype=np.int32))
        mask.append(np.asarray(b["attention_mask"], dtype=np.int8))
    return np.concatenate(ids), np.concatenate(mask)


def lote(i, ids, mask, y_est, y_rec, idx):
    return (torch.tensor(ids[idx].astype(np.int64)), torch.tensor(mask[idx].astype(np.int64)),
            torch.tensor(y_est[idx]), torch.tensor(y_rec[idx]))


@torch.no_grad()
def evaluar(modelo, ids, mask, y_est, y_rec, lote_eval=64):
    modelo.eval()
    n = len(ids)
    out_est = np.zeros(n, dtype=np.float32)
    p_rec = np.zeros(n, dtype=np.float32)
    for s in range(0, n, lote_eval):
        b = slice(s, min(s + lote_eval, n))
        bi = torch.tensor(ids[b].astype(np.int64)).to(DISPOSITIVO)
        bm = torch.tensor(mask[b].astype(np.int64)).to(DISPOSITIVO)
        with torch.autocast(DISPOSITIVO, dtype=torch.float16, enabled=AMP):
            ce, cr = modelo(bi, bm)
        out_est[b] = torch.log_softmax(ce.float(), -1).argmax(-1).cpu().numpy()
        p_rec[b] = torch.softmax(cr.float(), -1)[:, 1].cpu().numpy()
    return out_est, p_rec


def metricas_binarias(y_true, p, pos=1):
    y = (y_true == pos).astype(int)
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    acc = (tp + tn) / max(len(y), 1)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    # AUC por el metodo del rango (sin sklearn)
    ord_ = np.argsort(p, kind="stable")
    rang = np.empty(len(p), dtype=np.float64); rang[ord_] = np.arange(1, len(p) + 1)
    sp = np.where(np.diff(p[ord_]) > 0)[0]
    auc = float((np.sum(rang[y == 1]) - len(p[y == 1]) * (len(p[y == 1]) + 1) / 2) /
                max(len(p[y == 1]) * len(p[y == 0]), 1))
    brier = float(np.mean((p - y) ** 2))
    return {"accuracy": round(acc, 4), "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "auc": round(auc, 4), "brier": round(brier, 4),
            "n": int(len(y)), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "base_mayoritaria": round(float(max(y.mean(), 1 - y.mean())), 4)}


def exportar_onnx(carpeta):
    """Exporta el clasificador a ONNX (mismo formato que el encoder del two-tower)."""
    import onnxruntime as ort
    modelo = Clasificador(os.path.join(carpeta, "hf"))
    cabezas = torch.load(os.path.join(carpeta, "cabezas.pt"), map_location="cpu", weights_only=False)
    modelo.cabeza_estrellas.load_state_dict(cabezas["cabeza_estrellas"])
    modelo.cabeza_recomienda.load_state_dict(cabezas["cabeza_recomienda"])
    modelo = modelo.float().cpu().eval()
    tok = AutoTokenizer.from_pretrained(os.path.join(carpeta, "hf"))
    b = tok(["excelente producto", "no me gusto nada"], padding=True, truncation=True,
            max_length=MAX_LEN, return_tensors="pt")
    ruta = os.path.join(carpeta, "calidad.onnx")
    torch.onnx.export(modelo, (b["input_ids"], b["attention_mask"]), ruta, dynamo=False,
                      opset_version=17, input_names=["input_ids", "attention_mask"],
                      output_names=["estrellas", "recomienda"], dynamic_axes={
                          "input_ids": {0: "lote", 1: "secuencia"},
                          "attention_mask": {0: "lote", 1: "secuencia"},
                          "estrellas": {0: "lote"}, "recomienda": {0: "lote"}})
    sess = ort.InferenceSession(ruta, providers=["CPUExecutionProvider"])
    o = sess.run(None, {"input_ids": b["input_ids"].numpy().astype(np.int64),
                        "attention_mask": b["attention_mask"].numpy().astype(np.int64)})
    with torch.no_grad():
        t = modelo(b["input_ids"], b["attention_mask"])
    diff = max(float(np.abs(o[0] - t[0].numpy()).max()),
               float(np.abs(o[1] - t[1].numpy()).max()))
    print(f"  paridad ONNX vs PyTorch: {diff:.2e}")
    assert diff < 1e-3
    return diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=2)
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-datos", type=int, default=60_000)
    ap.add_argument("--continuar", action="store_true")
    ap.add_argument("--solo-exportar", action="store_true",
                    help="exporta a ONNX la version ya entrenada (no vuelve a entrenar)")
    args = ap.parse_args()
    random_seed = SEMILLA
    torch.manual_seed(random_seed); np.random.seed(random_seed)

    if args.solo_exportar:
        base = os.path.join(MODELOS, "calidad")
        vs = sorted(d for d in os.listdir(base) if d[:1] == "v")
        carpeta = os.path.join(base, vs[-1])
        print(f"Exportando {carpeta}")
        exportar_onnx(carpeta)
        print("OK")
        return

    print(f"Dispositivo: {DISPOSITIVO}" + (f" ({torch.cuda.get_device_name(0)})" if DISPOSITIVO == "cuda" else ""))
    train = pd.read_parquet(os.path.join(DATA_SPLITS, "train.parquet"))
    val = pd.read_parquet(os.path.join(DATA_SPLITS, "val.parquet"))
    test = pd.read_parquet(os.path.join(DATA_SPLITS, "test.parquet"))
    train = train[train["resena"].notna() & (train["resena"].str.len() >= 3)]
    if len(train) > args.max_datos:
        train = train.sample(args.max_datos, random_state=SEMILLA)
    val = val[val["resena"].notna() & (val["resena"].str.len() >= 3)]

    # etiquetas: estrellas 1..5 y binario "recomendaria" (>=4 -> 1, <=2 -> 0, 3 -> neutral)
    def etiqueta(r):
        e = int(r)
        return e - 1, (1 if e >= 4 else (0 if e <= 2 else -1))

    for df in (train, val, test):
        par = [etiqueta(int(e)) for e in df["rating"]]
        df["y_est"] = [p[0] for p in par]
        df["y_rec"] = [p[1] for p in par]

    tok = AutoTokenizer.from_pretrained(ENCODER_BASE)
    print("  tokenizando...")
    ids_tr, mask_tr = preparar(train, tok)
    ids_va, mask_va = preparar(val, tok)
    ids_te, mask_te = preparar(test, tok)
    y_tr_est = train["y_est"].to_numpy(np.int64); y_tr_rec = train["y_rec"].to_numpy(np.int64)
    y_va_est = val["y_est"].to_numpy(np.int64); y_va_rec = val["y_rec"].to_numpy(np.int64)
    y_te_est = test["y_est"].to_numpy(np.int64); y_te_rec = test["y_rec"].to_numpy(np.int64)
    print(f"  train={len(train):,} val={len(val):,} test={len(test):,}  "
          f"distribucion estrellas test: {np.bincount(y_te_est + 1, minlength=6)[1:].tolist()}")

    modelo = Clasificador(ENCODER_BASE).to(DISPOSITIVO)
    modelo.transformer.embeddings.word_embeddings.requires_grad_(False)
    entrenables = [p for p in modelo.parameters() if p.requires_grad]
    print(f"  parametros entrenables: {sum(p.numel() for p in entrenables) / 1e6:.1f}M")
    opt = torch.optim.AdamW(entrenables, lr=args.lr, weight_decay=0.01)
    pasos = math.ceil(len(train) / args.lote) * args.epocas
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * pasos), pasos)
    scaler = torch.amp.GradScaler(enabled=AMP)

    def perdida(ce, cr, ye, yr):
        l1 = F.cross_entropy(ce.float(), ye)
        mask = yr >= 0
        l2 = F.cross_entropy(cr.float()[mask], yr[mask]) if mask.any() else torch.zeros((), device=ce.device)
        return l1 + 0.5 * l2

    mejor, mejor_ep, mejor_estado, historial, t0 = -1.0, 0, None, [], time.time()
    for ep in range(1, args.epocas + 1):
        modelo.train()
        orden = np.random.permutation(len(train))
        acum, nb = 0.0, 0
        for s in range(0, len(orden) - args.lote + 1, args.lote):
            idx = orden[s:s + args.lote]
            bi, bm, be, br = lote(s, ids_tr, mask_tr, y_tr_est, y_tr_rec, idx)
            bi, bm, be, br = bi.to(DISPOSITIVO), bm.to(DISPOSITIVO), be.to(DISPOSITIVO), br.to(DISPOSITIVO)
            with torch.autocast(DISPOSITIVO, dtype=torch.float16, enabled=AMP):
                ce, cr = modelo(bi, bm)
            loss = perdida(ce, cr, be, br)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(entrenables, 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            acum += float(loss.item()); nb += 1
            if nb % 400 == 0:
                print(f"  epoca {ep} paso {nb}/{pasos} loss={acum/nb:.4f} {time.time()-t0:.0f}s", flush=True)
        est_va, rec_va = evaluar(modelo, ids_va, mask_va, y_va_est, y_va_rec)
        rmse = float(np.sqrt(np.mean((est_va + 1 - (y_va_est + 1)) ** 2)))
        sel_va = y_va_rec >= 0
        bin_va = metricas_binarias(y_va_rec[sel_va], rec_va[sel_va], pos=1)
        historial.append({"epoca": ep, "train_loss": acum / max(nb, 1), "val_rmse": rmse,
                          "val_accuracy_binaria": bin_va["accuracy"]})
        print(f"  epoca {ep}: loss={acum/max(nb,1):.4f} val_rmse={rmse:.4f} "
              f"val_acc_binaria={bin_va['accuracy']:.4f}  ({time.time()-t0:.0f}s)")
        if bin_va["accuracy"] > mejor:
            mejor, mejor_ep = bin_va["accuracy"], ep
            mejor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
    modelo.load_state_dict(mejor_estado)

    # --------------------------------------------------------------- evaluacion final en test
    est_te, rec_te = evaluar(modelo, ids_te, mask_te, y_te_est, y_te_rec)
    real = y_te_est + 1
    pred = est_te + 1
    err = pred - real
    m_est = {
        "n": int(len(real)),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "mae": round(float(np.mean(np.abs(err))), 4),
        "accuracy_exacta": round(float((err == 0).mean()), 4),
        "accuracy_dentro_1": round(float((np.abs(err) <= 1).mean()), 4),
        "accuracy_dentro_2": round(float((np.abs(err) <= 2).mean()), 4),
        "r2": round(float(1 - np.sum(err ** 2) / np.sum((real - real.mean()) ** 2)), 4),
        "distribucion_predicha": np.bincount(pred.astype(int), minlength=6)[1:].tolist(),
        "distribucion_real": np.bincount(real.astype(int), minlength=6)[1:].tolist(),
    }
    # binario: se separa el caso neutro (3 estrellas) para no premiar a la clase mayoritaria
    sel = y_te_rec >= 0
    m_pos = metricas_binarias(y_te_rec[sel], rec_te[sel], pos=1)
    m_neg = metricas_binarias(y_te_rec[sel], rec_te[sel], pos=0)
    m_todo = metricas_binarias(y_te_rec, rec_te, pos=1)  # 3 estrellas como negativo
    print(json.dumps({"satisfaccion": m_est, "recomendaria_(sin_neutros)": m_pos,
                      "recomendaria_(3 estrellas = no)": m_todo}, indent=2))

    carpeta = nueva_carpeta("calidad")
    modelo.transformer.save_pretrained(os.path.join(carpeta, "hf"))
    tok.save_pretrained(os.path.join(carpeta, "hf"))
    torch.save({"cabeza_estrellas": modelo.cabeza_estrellas.state_dict(),
                "cabeza_recomienda": modelo.cabeza_recomienda.state_dict()},
               os.path.join(carpeta, "cabezas.pt"))
    m = {"test": {"satisfaccion": m_est, "recomendaria_sin_neutros": m_pos,
                  "recomendaria_negativo": m_neg, "recomendaria_3_como_negativo": m_todo},
         "historial": historial, "mejor_epoca": mejor_ep, "args": vars(args),
         "distribucion_3_estrellas_test": int((y_te_rec == -1).sum())}
    with open(os.path.join(carpeta, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    try:
        exportar_onnx(carpeta)
    except Exception as e:  # la exportacion no debe tumbar el entrenamiento ya hecho
        print(f"  (no se pudo exportar a ONNX: {e})")
    print(f"OK -> {carpeta}")


if __name__ == "__main__":
    main()
