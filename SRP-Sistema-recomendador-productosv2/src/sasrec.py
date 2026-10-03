"""
Paso 3: SASRec (Kang & McAuley 2018) - Transformer secuencial para re-rankear.

Lee la secuencia de compras del usuario con atencion causal y predice la siguiente. Cada
producto entra como  proyeccion(embedding del two-tower) + embedding de id + posicion:
el contenido del two-tower le da algo que decir sobre los productos que casi nadie compro
(la mayoria en este dataset), y el id aprende patrones de co-compra.

Sesgo de popularidad: al puntaje se le suma beta * log(1 + compras del producto), con beta
aprendido. En este dataset la popularidad sola es una linea base muy fuerte (HR@10 0.47):
sin el sesgo, el Transformer gasta sus pocos ejemplos (11k) en redescubrirla y sobreajusta
desde la 1.a epoca (v001: HR@10 0.38). Con el sesgo aprende solo lo que se aparta de ella.

Particion: con leave-last-out puro casi no quedan transiciones para entrenar (la mayoria
de usuarios con historial tiene 2 resenas, y la 2.a es su test). Por eso se fija un 30 %
de los usuarios de test como usuarios de evaluacion (E): de ellos solo se entrena con el
historial previo a su test; del resto se usan las secuencias completas. Todas las
metricas finales (SASRec, two-tower, fusion, popularidad) se reportan sobre E con los
mismos candidatos de data/splits/candidatos_test.npy.

Uso:   python src/sasrec.py           (requiere haber corrido src/two_tower.py)
Salida: models/sasrec/vNNN_<fecha>/  sasrec.pt (pesos), sasrec.onnx, metricas.json
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from comun import DATA_SPLITS, K, SEMILLA, hr_ndcg
from two_tower import nueva_carpeta, ultima_version

DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
LARGO = 10       # ultimas compras que mira el modelo
PAD = 0          # los productos se desplazan +1 para reservar el 0 como relleno


class SASRec(nn.Module):
    def __init__(self, items_texto, log_pop, dim=64, capas=2, cabezas=2, dropout=0.3, p_drop_id=0.5):
        super().__init__()
        n, d_txt = items_texto.shape
        self.register_buffer("log_pop", torch.cat([torch.zeros(1), torch.as_tensor(log_pop, dtype=torch.float32)]))
        self.beta = nn.Parameter(torch.tensor(1.0))
        relleno = torch.zeros(1, d_txt)
        self.register_buffer("texto", torch.cat([relleno, torch.as_tensor(items_texto, dtype=torch.float32)]))
        self.proy = nn.Linear(d_txt, dim)
        self.emb_id = nn.Embedding(n + 1, dim, padding_idx=PAD)
        nn.init.normal_(self.emb_id.weight, std=0.02)
        self.pos = nn.Embedding(LARGO, dim)
        capa = nn.TransformerEncoderLayer(dim, cabezas, dim * 2, dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(capa, capas, enable_nested_tensor=False)
        self.norma = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.p_drop_id = p_drop_id

    def repr_items(self, idx=None):
        """Representacion de cada producto (entrada y salida comparten pesos)."""
        t = self.texto if idx is None else self.texto[idx]
        e = self.emb_id.weight if idx is None else self.emb_id(idx)
        if self.training and self.p_drop_id > 0:
            e = e * (torch.rand(e.shape[:-1] + (1,), device=e.device) >= self.p_drop_id)
        return self.proy(t) + e

    def sesgo(self, idx=None):
        return self.beta * (self.log_pop if idx is None else self.log_pop[idx])

    def logits_todos(self, h):
        """[B, dim] -> [B, n_items] puntaje de cada producto (sin el relleno)."""
        return h @ self.repr_items()[1:].T + self.sesgo()[1:]

    def codificar(self, seq):
        """seq [B, L] con relleno a la izquierda -> estado [B, L, dim]."""
        L = seq.shape[1]
        x = self.repr_items(seq) + self.pos(torch.arange(L, device=seq.device))
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=seq.device), 1)
        # Sin src_key_padding_mask a proposito: el relleno (izquierda) entra como un vector
        # constante que el modelo aprende a ignorar, y asi el ONNX calcula exactamente lo mismo.
        h = self.transformer(self.drop(x), mask=causal)
        return self.norma(h)


class SASRecExport(nn.Module):
    """Para ONNX: tabla de productos precalculada y salida = puntaje de cada candidato."""

    def __init__(self, m: SASRec):
        super().__init__()
        m.eval()
        self.m = m
        with torch.no_grad():
            self.register_buffer("tabla", m.repr_items().detach().clone())
            self.register_buffer("sesgo", m.sesgo().detach().clone())

    def forward(self, secuencia, candidatos):
        L = secuencia.shape[1]
        x = self.tabla[secuencia] + self.m.pos(torch.arange(L, device=secuencia.device))
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
        h = self.m.norma(self.m.transformer(x, mask=causal))[:, -1]
        return (self.tabla[candidatos] * h.unsqueeze(1)).sum(-1) + self.sesgo[candidatos]


def a_secuencia(items, largo=LARGO):
    s = [i + 1 for i in items[-largo:]]
    return [PAD] * (largo - len(s)) + s


def particion_eval(train, val, test, frac_eval=0.3, min_compras=2):
    """Usuarios de evaluacion E (fijos por SEMILLA) y secuencias con las que se entrena.
    La usa tambien src/reranker.py para evaluar exactamente sobre los mismos usuarios."""
    rng = np.random.default_rng(SEMILLA)
    todo = pd.concat([train, val, test]).sort_values(["user_id", "fecha"])
    seq = todo.groupby("user_id")["item_idx"].agg(list)
    seq = seq[seq.str.len() >= min_compras]
    es_eval = (rng.random(len(test)) < frac_eval) & test["user_id"].isin(set(seq.index))
    E = set(test["user_id"][es_eval])
    prev = pd.concat([train, val]).sort_values(["user_id", "fecha"]).groupby("user_id")["item_idx"].agg(list)
    entrenar_seqs = ([s for u, s in seq.items() if u not in E]
                     + [prev[u] for u in E if u in prev and len(prev[u]) >= min_compras])
    return seq, es_eval, E, prev, entrenar_seqs


def ejemplos(secuencias):
    """Una secuencia de n compras da n-1 ejemplos (prefijo -> siguiente)."""
    X, Y = [], []
    for items in secuencias:
        for j in range(1, len(items)):
            X.append(a_secuencia(items[:j])); Y.append(items[j] + 1)
    return torch.tensor(X), torch.tensor(Y)


@torch.no_grad()
def puntuar(modelo, secuencias, cand):
    modelo.eval()
    X = torch.tensor([a_secuencia(s) for s in secuencias], device=DISPOSITIVO)
    h = modelo.codificar(X)[:, -1]
    C = torch.as_tensor(cand + 1, device=DISPOSITIVO)
    return ((modelo.repr_items(C) * h.unsqueeze(1)).sum(-1) + modelo.sesgo(C)).cpu().numpy()


def fusion_rrf(*puntajes, k=60):
    """Reciprocal Rank Fusion: suma 1/(k + rango) de cada modelo. No necesita calibrar escalas."""
    total = 0
    for s in puntajes:
        rango = (-s).argsort(1).argsort(1)
        total = total + 1.0 / (k + rango + 1)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=40)
    ap.add_argument("--lote", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--p-drop-id", type=float, default=0.5,
                    help="prob. de apagar el embedding de id (1.0 = solo contenido del two-tower)")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--paciencia", type=int, default=0,
                    help="cortar si la val no mejora en N epocas (0 = correr todas)")
    ap.add_argument("--sobremuestreo-densos", type=int, default=1,
                    help="repetir N veces las secuencias de clientes con >=5 compras")
    ap.add_argument("--frac-eval", type=float, default=0.3)
    ap.add_argument("--min-compras", type=int, default=2,
                    help="solo usuarios con al menos N compras entran a entrenar y a evaluar "
                         "(ej. 5: cohorte densa; ojo: con 5 quedan ~274 usuarios)")
    args = ap.parse_args()
    torch.manual_seed(SEMILLA)

    tt = ultima_version("two_tower")
    items_txt = np.load(os.path.join(tt, "items.npy")).astype(np.float32)
    print(f"Embeddings de producto del two-tower: {tt}")
    train, val, test = (pd.read_parquet(os.path.join(DATA_SPLITS, f"{n}.parquet")) for n in ("train", "val", "test"))
    cand_test = np.load(os.path.join(DATA_SPLITS, "candidatos_test.npy"))
    n_items = len(items_txt)
    popularidad = np.bincount(train["item_idx"], minlength=n_items).astype(np.float32)

    seq, es_eval, E, prev, entrenar_seqs = particion_eval(train, val, test, args.frac_eval, args.min_compras)
    # 10 % de las secuencias (no de los ejemplos) para elegir la mejor epoca: asi los prefijos
    # de un mismo cliente no quedan a ambos lados, y el sobremuestreo no se filtra a la val
    perm = np.random.default_rng(SEMILLA).permutation(len(entrenar_seqs))
    nv = max(50, len(entrenar_seqs) // 10)
    seqs_v = [entrenar_seqs[i] for i in perm[:nv]]
    seqs_t = [entrenar_seqs[i] for i in perm[nv:]]
    if args.sobremuestreo_densos > 1:
        # los clientes con historial largo son pocos y sus ultimas compras las peor modeladas:
        # se repiten en vez de filtrarlos (filtrar dejaba 274 usuarios y empeoraba la metrica)
        seqs_t = seqs_t + [s for s in seqs_t if len(s) >= 5] * (args.sobremuestreo_densos - 1)
    Xt, Yt = ejemplos(seqs_t)
    Xv, Yv = ejemplos(seqs_v)
    print(f"  secuencias de entrenamiento={len(entrenar_seqs):,} ejemplos train={len(Xt):,} val={len(Xv):,} "
          f"usuarios de evaluacion={len(E):,}")

    modelo = SASRec(items_txt, np.log1p(popularidad), dim=args.dim, dropout=args.dropout,
                    p_drop_id=args.p_drop_id).to(DISPOSITIVO)
    opt = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mejor, mejor_estado, historial, t0 = float("inf"), None, [], time.time()
    for ep in range(1, args.epocas + 1):
        modelo.train()
        orden = torch.randperm(len(Xt))
        acum = 0.0
        for s in range(0, len(Xt), args.lote):
            b = orden[s:s + args.lote]
            x, y = Xt[b].to(DISPOSITIVO), Yt[b].to(DISPOSITIVO)
            h = modelo.codificar(x)[:, -1]
            logits = modelo.logits_todos(h)  # softmax sobre los 71k productos
            loss = F.cross_entropy(logits, y - 1)
            opt.zero_grad(); loss.backward(); opt.step()
            acum += loss.item() * len(b)
        modelo.eval()
        with torch.no_grad():
            hv = modelo.codificar(Xv.to(DISPOSITIVO))[:, -1]
            lv = F.cross_entropy(modelo.logits_todos(hv), Yv.to(DISPOSITIVO) - 1).item()
        historial.append({"epoca": ep, "train": acum / len(Xt), "val": lv, "beta": modelo.beta.item()})
        if ep % 5 == 0 or ep == 1:
            print(f"  epoca {ep}: train={acum / len(Xt):.4f} val={lv:.4f} beta={modelo.beta.item():.3f} "
                  f"{time.time() - t0:.0f}s")
        if lv < mejor:
            mejor, mejor_ep, mejor_estado = lv, ep, {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        elif args.paciencia and ep - mejor_ep >= args.paciencia:
            print(f"  parada temprana en la epoca {ep} (mejor: {mejor_ep}, val={mejor:.4f})")
            break
    modelo.load_state_dict(mejor_estado)

    # ------------------------------------------------------------- evaluacion sobre E
    filas = test[es_eval]
    cand = cand_test[es_eval]
    historias = [prev.get(u, []) for u in filas["user_id"]]
    s_sas = puntuar(modelo, historias, cand)
    # two-tower item-to-item: el usuario es el promedio de los embeddings de lo que compro.
    # (Codificar el historial como texto dio HR@10 0.186; esto, 0.255.)
    q = np.stack([items_txt[h].mean(0) for h in historias])
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    s_tt = np.einsum("nd,nkd->nk", q, items_txt[cand])
    s_pop = popularidad[cand] + np.random.default_rng(0).random(cand.shape) * 1e-6
    metricas = {"n_usuarios_eval": int(len(filas)), "protocolo": f"1 positivo + 99 negativos, @{K}",
                "mejor_epoca": mejor_ep}
    for nombre, s in [("sasrec", s_sas), ("two_tower", s_tt), ("fusion_two_tower+sasrec", fusion_rrf(s_tt, s_sas)),
                      ("popularidad", s_pop), ("fusion_two_tower+popularidad", fusion_rrf(s_tt, s_pop))]:
        hr, nd = hr_ndcg(s)
        metricas[nombre] = {f"hr@{K}": round(hr, 4), f"ndcg@{K}": round(nd, 4)}
    print(json.dumps(metricas, indent=2))

    carpeta = nueva_carpeta("sasrec")
    torch.save({"state_dict": modelo.state_dict(), "args": vars(args), "two_tower": tt},
               os.path.join(carpeta, "sasrec.pt"))
    exp = SASRecExport(modelo.cpu()).eval()
    ej_s = torch.tensor([a_secuencia([1, 2, 3]), a_secuencia([5])])
    ej_c = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    ruta = os.path.join(carpeta, "sasrec.onnx")
    torch.onnx.export(exp, (ej_s, ej_c), ruta, dynamo=False, opset_version=17,
                      input_names=["secuencia", "candidatos"], output_names=["puntajes"],
                      dynamic_axes={"secuencia": {0: "lote"}, "candidatos": {0: "lote", 1: "n"}, "puntajes": {0: "lote", 1: "n"}})
    import onnxruntime as ort
    o = ort.InferenceSession(ruta, providers=["CPUExecutionProvider"]).run(
        None, {"secuencia": ej_s.numpy(), "candidatos": ej_c.numpy()})[0]
    with torch.no_grad():
        diff = float(np.abs(o - exp(ej_s, ej_c).numpy()).max())
    print(f"  paridad ONNX vs PyTorch: {diff:.2e}")
    metricas.update({"historial": historial, "paridad_onnx": diff, "two_tower_version": tt, "args": vars(args)})
    with open(os.path.join(carpeta, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2)
    print(f"OK -> {carpeta}")


if __name__ == "__main__":
    main()
