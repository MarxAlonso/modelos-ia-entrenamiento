"""
Paso 2: two-tower con Transformer (recuperacion de candidatos).

Un encoder Transformer multilingue (MiniLM-L12, compartido por las dos torres) convierte
en vectores de 384 dimensiones:
  torre consulta: lo que pide el cliente (texto libre en espanol o ingles, una resena, o
                  los titulos de lo que compro antes)
  torre producto: titulo + tienda + descripcion
y se afina con aprendizaje contrastivo (InfoNCE con negativos dentro del lote): cada
consulta debe quedar mas cerca de SU producto que de los otros 63 del lote.

Pares de entrenamiento (solo del split train):
  1. resena -> producto                 como habla la gente de cada producto
  2. historial -> siguiente compra      personalizacion para usuarios con historial
  3. consulta sintetica en espanol      "busco aretes para mujer" -> producto con esa categoria
  4. resenas traducidas al espanol      alineacion espanol -> catalogo en ingles

Uso:
    python src/two_tower.py                     # afina y exporta
    python src/two_tower.py --solo-evaluar-base # metricas del encoder sin afinar (baseline)
    python src/two_tower.py --continuar         # sigue afinando desde la ultima version

Salida: models/two_tower/vNNN_<fecha>/ con
    hf/                  model.safetensors + config + tokenizer (formato Hugging Face)
    encoder.onnx         el mismo encoder para inferencia con onnxruntime
    items.npy            embeddings de los 71k productos (float16)
    metricas.json
"""
import argparse
import json
import math
import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from comun import (DATA_RAW, DATA_SPLITS, ENCODER_BASE, K, MAX_LEN_CONSULTA, MAX_LEN_PRODUCTO,
                   MODELOS, PREFIJO_CONSULTA, PREFIJO_PRODUCTO, SEMILLA, hr_ndcg)

DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
# fp16 solo si la GPU tiene tensor cores: en la GTX 1650 (TU117) fp16 resulto 4x MAS LENTO
# que fp32 (78 vs 326 textos/s). Activar con SRP_FP16=1 en una T4/RTX.
AMP = DISPOSITIVO == "cuda" and os.environ.get("SRP_FP16") == "1"
HIST_MAX = 5

CAT_ES = {"ropa": ["ropa", "un vestido", "una blusa", "un polo", "un pantalón", "una casaca"],
          "calzado": ["zapatos", "zapatillas", "botas", "sandalias"],
          "joyeria": ["joyas", "aretes", "un collar", "una pulsera", "un anillo"],
          "relojes": ["un reloj", "una correa de reloj"], "lentes": ["lentes de sol", "gafas"],
          "bolsos": ["una cartera", "un bolso", "una mochila", "una billetera"],
          "ropa_interior": ["medias", "ropa interior", "un pijama"],
          "accesorios": ["un cinturón", "un gorro", "una bufanda", "guantes"]}
PUB_ES = {"mujer": ["para mujer", "para mi esposa", "para dama", "para mi mamá"],
          "hombre": ["para hombre", "para mi papá", "para caballero", "para mi esposo"],
          "ninos": ["para niños", "para mi hija", "para bebé", "para mi hijo"], "unisex": [""]}
COLORES = {"black": "negro", "white": "blanco", "red": "rojo", "blue": "azul", "green": "verde",
           "pink": "rosado", "gray": "gris", "grey": "gris", "brown": "marrón", "purple": "morado",
           "yellow": "amarillo", "silver": "plateado", "gold": "dorado", "beige": "beige"}
PLANTILLAS = ["busco {cat} {pub} {color}", "quiero comprar {cat} {color} {pub}",
              "necesito {cat} {pub}", "{cat} {color} {pub}", "regalo: {cat} {pub}",
              "me recomiendas {cat} {color}?"]


# ----------------------------------------------------------------------------- modelo
class Encoder(torch.nn.Module):
    """Transformer + mean pooling + L2. Es lo que se exporta a ONNX."""

    def __init__(self, ruta):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(ruta)

    def forward(self, input_ids, attention_mask):
        h = self.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        emb = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return F.normalize(emb, dim=-1)


@torch.no_grad()
def codificar(encoder, tok, textos, max_len, lote=64):
    # Lotes chicos y cache liberada: con 4 GB de VRAM, si el pico no cabe Windows desborda a
    # RAM compartida y la evaluacion pasa de minutos a casi una hora.
    if DISPOSITIVO == "cuda":
        torch.cuda.empty_cache()
    encoder.eval()
    # Lotes de textos de largo parecido: casi no hay relleno y la GPU no trabaja en vano.
    orden = np.argsort([len(t) for t in textos])
    salida = []
    for s in range(0, len(textos), lote):
        b = tok([textos[i] for i in orden[s:s + lote]], padding=True, truncation=True, max_length=max_len,
                return_tensors="pt").to(DISPOSITIVO)
        with torch.autocast(DISPOSITIVO, dtype=torch.float16, enabled=AMP):
            salida.append(encoder(b["input_ids"], b["attention_mask"]).float().cpu())
    emb = torch.cat(salida).numpy()
    res = np.empty_like(emb)
    res[orden] = emb
    return res


# ----------------------------------------------------------------------------- pares
def texto_historial(titulos):
    return PREFIJO_CONSULTA + "compró antes: " + " | ".join(t[:80] for t in titulos[-HIST_MAX:])


def consulta_sintetica(fila, rng):
    if fila.categoria not in CAT_ES:
        return None
    color = next((es for en, es in COLORES.items() if f" {en}" in f" {fila.titulo.lower()}"), "")
    txt = rng.choice(PLANTILLAS).format(cat=rng.choice(CAT_ES[fila.categoria]),
                                        pub=rng.choice(PUB_ES[fila.publico]), color=color)
    return PREFIJO_CONSULTA + " ".join(txt.split())


def construir_pares(train, cat, test_claves, max_resenas, rng):
    titulos = cat["titulo"].tolist()
    pares = []  # (texto_consulta, item_idx)
    # 1. resena -> producto
    res = train[train["resena"].str.len() >= 15]
    res = res.sample(min(max_resenas, len(res)), random_state=SEMILLA)
    pares += [(PREFIJO_CONSULTA + t[:400], i) for t, i in zip(res["resena"], res["item_idx"])]
    n1 = len(pares)
    # 2. historial -> siguiente (x2: es la senal de personalizacion y hay poca)
    for _, g in train.groupby("user_id"):
        if len(g) < 2:
            continue
        items = g["item_idx"].tolist()
        for j in range(1, len(items)):
            q = texto_historial([titulos[i] for i in items[:j]])
            pares += [(q, items[j])] * 2
    n2 = len(pares) - n1
    # 3. consultas sinteticas en espanol
    muestra = cat.sample(min(40_000, len(cat)), random_state=SEMILLA)
    for fila in muestra.itertuples():
        q = consulta_sintetica(fila, rng)
        if q:
            pares.append((q, fila.item_idx))
    n3 = len(pares) - n1 - n2
    # 4. resenas traducidas (excluye las que caen en test para no filtrar la respuesta)
    tr = pd.read_csv(os.path.join(DATA_RAW, "resenas_traducidas.csv"), usecols=["user_id", "product_id", "texto_es"])
    tr = tr.merge(cat[["product_id", "item_idx"]], on="product_id")
    tr = tr[~tr.apply(lambda f: (f.user_id, f.item_idx) in test_claves, axis=1)]
    pares += [(PREFIJO_CONSULTA + t[:400], i) for t, i in zip(tr["texto_es"], tr["item_idx"])]
    n4 = len(pares) - n1 - n2 - n3
    print(f"  pares: resena={n1:,} historial={n2:,} sinteticas_es={n3:,} traducidas_es={n4:,}")
    random.Random(SEMILLA).shuffle(pares)
    return pares


# ----------------------------------------------------------------------------- evaluacion
def historiales(*dfs):
    h = pd.concat(dfs).sort_values(["user_id", "fecha"])
    return h.groupby("user_id")["item_idx"].agg(list).to_dict()


def evaluar(encoder, tok, cat, filas, cand, hist, popularidad, frio=None):
    titulos = cat["titulo"].tolist()
    items = codificar(encoder, tok, (PREFIJO_PRODUCTO + cat["texto"]).tolist(), MAX_LEN_PRODUCTO)
    q = codificar(encoder, tok, [texto_historial([titulos[i] for i in hist[u]]) for u in filas["user_id"]],
                  MAX_LEN_CONSULTA)
    s = np.einsum("nd,nkd->nk", q, items[cand])
    hr, ndcg = hr_ndcg(s)
    hr_p, ndcg_p = hr_ndcg(popularidad[cand] + np.random.default_rng(0).random(cand.shape) * 1e-6)
    # recall sobre TODO el catalogo (mas exigente que 1+99), excluyendo lo ya comprado
    excluir = [hist[u] for u in filas["user_id"]]
    rec100 = recall_catalogo(q, items, filas["item_idx"].to_numpy(), 100, excluir)
    m = {"n": int(len(filas)), f"hr@{K}": hr, f"ndcg@{K}": ndcg, "recall@100_catalogo": rec100,
         "popularidad": {f"hr@{K}": hr_p, f"ndcg@{K}": ndcg_p}}
    if frio is not None:  # usuario nuevo: su resena como si fuera una busqueda
        qf = codificar(encoder, tok, (PREFIJO_CONSULTA + frio["resena"].str.slice(0, 400)).tolist(), MAX_LEN_CONSULTA)
        m["busqueda_texto_usuario_nuevo"] = {
            "n": int(len(frio)), "recall@10_catalogo": recall_catalogo(qf, items, frio["item_idx"].to_numpy(), 10)}
    return m, items


def recall_catalogo(q, items, pos, k, excluir=None, trozo=1024):
    """Fraccion de filas cuyo producto real queda en el top-k de los 71k (por trozos: la
    matriz completa ocuparia varios GB)."""
    aciertos = 0
    for s in range(0, len(q), trozo):
        sc = q[s:s + trozo] @ items.T
        if excluir is not None:
            for j, h in enumerate(excluir[s:s + trozo]):
                sc[j, h] = -np.inf
        top = np.argpartition(-sc, k, axis=1)[:, :k]
        aciertos += int((top == pos[s:s + trozo, None]).any(1).sum())
    return aciertos / len(q)


# ----------------------------------------------------------------------------- exportacion
def exportar_onnx(encoder, tok, ruta):
    encoder = encoder.float().cpu().eval()
    b = tok(["consulta: ejemplo de texto", "producto: otro"], padding=True, return_tensors="pt")
    torch.onnx.export(encoder, (b["input_ids"], b["attention_mask"]), ruta, dynamo=False, opset_version=17,
                      input_names=["input_ids", "attention_mask"], output_names=["embedding"],
                      dynamic_axes={"input_ids": {0: "lote", 1: "secuencia"},
                                    "attention_mask": {0: "lote", 1: "secuencia"}, "embedding": {0: "lote"}})
    import onnxruntime as ort
    sess = ort.InferenceSession(ruta, providers=["CPUExecutionProvider"])
    b = tok(["busco aretes de plata para mi mamá", "Women's Casual Summer Dress"], padding=True, return_tensors="np")
    o = sess.run(None, {"input_ids": b["input_ids"].astype(np.int64), "attention_mask": b["attention_mask"].astype(np.int64)})[0]
    with torch.no_grad():
        t = encoder(torch.from_numpy(b["input_ids"]).long(), torch.from_numpy(b["attention_mask"]).long()).numpy()
    diff = float(np.abs(o - t).max())
    print(f"  paridad ONNX vs PyTorch: {diff:.2e}")
    assert diff < 1e-3
    return diff


def nueva_carpeta(tipo):
    base = os.path.join(MODELOS, tipo)
    os.makedirs(base, exist_ok=True)
    n = max([int(d[1:4]) for d in os.listdir(base) if d[:1] == "v" and d[1:4].isdigit()], default=0) + 1
    ruta = os.path.join(base, f"v{n:03d}_{datetime.now():%Y%m%d-%H%M}")
    os.makedirs(ruta)
    return ruta


@torch.no_grad()
def minar_negativos_duros(encoder, tok, pares, textos_prod, K, trozo=512):
    """Para cada par (consulta, producto) busca los K productos que el encoder actual
    confunde con el correcto (los mas parecidos que NO son el correcto).

    Con solo negativos dentro del lote, el encoder nunca ve un confuso real: el lote de 64
    es facil de separar y el modelo no aprende a distinguir el producto correcto de uno
    que se le parece de verdad (misma categoria, mismo estilo). Estos son los negativos
    que separan al modelo de verdad.
    """
    emb_items = codificar(encoder, tok, textos_prod, MAX_LEN_PRODUCTO)
    rng = np.random.default_rng(SEMILLA)
    negs = np.zeros((len(pares), K), dtype=np.int64)
    textos = [PREFIJO_CONSULTA + q[:400] for q, _ in pares]
    for s in range(0, len(textos), trozo):
        idx = np.arange(s, min(s + trozo, len(textos)))
        q = codificar(encoder, tok, [textos[i] for i in idx], MAX_LEN_CONSULTA)
        sim = q @ emb_items.T
        for j, i in enumerate(idx):
            sim[j, pares[i][1]] = -np.inf          # el correcto no es un negativo
            top = np.argpartition(-sim[j], K)[:K]
            negs[i] = top[rng.permutation(K)]
        del sim
    print(f"  negativos duros minados: {K} por par ({len(pares):,} pares)")
    return negs


def ultima_version(tipo):
    base = os.path.join(MODELOS, tipo)
    vs = sorted(d for d in os.listdir(base) if d[:1] == "v") if os.path.isdir(base) else []
    if not vs:
        raise SystemExit(f"No hay versiones en {base}")
    return os.path.join(base, vs[-1])


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=1)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-resenas", type=int, default=120_000)
    ap.add_argument("--escala", type=float, default=20.0, help="1/temperatura de InfoNCE")
    ap.add_argument("--continuar", action="store_true")
    ap.add_argument("--solo-evaluar-base", action="store_true")
    ap.add_argument("--terminar", action="store_true",
                    help="evalua y exporta la ultima version ya entrenada (si se corto la evaluacion)")
    ap.add_argument("--negativos-duros", type=int, default=0,
                    help="K negativos duros por par (0 = solo los del lote)")
    ap.add_argument("--paciencia", type=int, default=0,
                    help="cortar si el recall@10 de busqueda no mejora en N epocas (0 = todas)")
    ap.add_argument("--max-len-producto", type=int, default=0,
                    help="0 = el de comun.py; 128 da mas contexto del producto")
    ap.add_argument("--max-pares", type=int, default=0,
                    help="0 = todos; recorta el numero de pares (para pruebas rapidas)")
    args = ap.parse_args()
    if args.max_len_producto:
        globals()["MAX_LEN_PRODUCTO"] = args.max_len_producto
    random.seed(SEMILLA); np.random.seed(SEMILLA); torch.manual_seed(SEMILLA)
    print(f"Dispositivo: {DISPOSITIVO}" + (f" ({torch.cuda.get_device_name(0)})" if DISPOSITIVO == "cuda" else ""))

    cat = pd.read_parquet(os.path.join(DATA_SPLITS, "catalogo.parquet"))
    train, val, test, frio = (pd.read_parquet(os.path.join(DATA_SPLITS, f"{n}.parquet"))
                              for n in ("train", "val", "test", "test_frio"))
    cand_val = np.load(os.path.join(DATA_SPLITS, "candidatos_val.npy"))
    cand_test = np.load(os.path.join(DATA_SPLITS, "candidatos_test.npy"))
    popularidad = np.bincount(train["item_idx"], minlength=len(cat)).astype(np.float32)
    hist_val, hist_test = historiales(train), historiales(train, val)

    origen = os.path.join(ultima_version("two_tower"), "hf") if (args.continuar or args.terminar) else ENCODER_BASE
    print(f"Encoder inicial: {origen}")
    tok = AutoTokenizer.from_pretrained(origen)
    encoder = Encoder(origen).to(DISPOSITIVO)

    if args.terminar:
        evaluar_y_exportar(encoder, tok, ultima_version("two_tower"), cat, test, cand_test, hist_test, popularidad, frio)
        return
    if args.solo_evaluar_base:
        m, _ = evaluar(encoder, tok, cat, test, cand_test, hist_test, popularidad, frio)
        print(json.dumps(m, indent=2))
        return

    rng = random.Random(SEMILLA)
    test_claves = set(zip(pd.concat([test, val])["user_id"], pd.concat([test, val])["item_idx"]))
    pares = construir_pares(train, cat, test_claves, args.max_resenas, rng)
    if args.max_pares and len(pares) > args.max_pares:
        pares = random.Random(SEMILLA).sample(pares, args.max_pares)
        print(f"  pares recortados a {len(pares):,}")
    textos_prod = (PREFIJO_PRODUCTO + cat["texto"]).tolist()

    # El vocabulario (250k tokens x 384) queda congelado: es el 80 % de los parametros, casi
    # todos tokens que nunca aparecen aqui, y actualizarlo en cada paso es lo mas caro.
    encoder.transformer.embeddings.word_embeddings.requires_grad_(False)
    entrenables = [p for p in encoder.parameters() if p.requires_grad]
    print(f"  parametros entrenables: {sum(p.numel() for p in entrenables) / 1e6:.1f}M")
    opt = torch.optim.AdamW(entrenables, lr=args.lr, weight_decay=0.01)
    pasos = math.ceil(len(pares) / args.lote) * args.epocas
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * pasos), pasos)
    scaler = torch.amp.GradScaler(enabled=AMP)
    historial, paso, t0 = [], 0, time.time()
    m_val, _ = evaluar(encoder, tok, cat, val, cand_val, hist_val, popularidad)
    print(f"  val antes de afinar: hr@{K}={m_val[f'hr@{K}']:.4f} ndcg@{K}={m_val[f'ndcg@{K}']:.4f}")
    historial.append({"paso": 0, **{k: v for k, v in m_val.items() if k != "popularidad"}})
    mejor_recall, mejor_epoca, mejor_estado = -1.0, 0, None

    negs_duros = None
    if args.negativos_duros > 0:
        negs_duros = minar_negativos_duros(encoder, tok, pares, textos_prod, args.negativos_duros)

    for ep in range(args.epocas):
        encoder.train()
        for s in range(0, len(pares) - args.lote + 1, args.lote):
            lote = pares[s:s + args.lote]
            # Un producto repetido en el lote seria un falso negativo: se queda el primero.
            vistos, lote_u = set(), []
            for q, i in lote:
                if i not in vistos:
                    vistos.add(i); lote_u.append((q, i))
            bq = tok([q for q, _ in lote_u], padding=True, truncation=True, max_length=MAX_LEN_CONSULTA, return_tensors="pt").to(DISPOSITIVO)
            bp = tok([textos_prod[i] for _, i in lote_u], padding=True, truncation=True, max_length=MAX_LEN_PRODUCTO, return_tensors="pt").to(DISPOSITIVO)
            with torch.autocast(DISPOSITIVO, dtype=torch.float16, enabled=AMP):
                eq = encoder(bq["input_ids"], bq["attention_mask"])
                ep_ = encoder(bp["input_ids"], bp["attention_mask"])
            if negs_duros is not None:
                # negativos duros: K por consulta, codificados junto a los positivos
                nh = negs_duros[s:s + args.lote]          # [B, K]
                textos_neg = [textos_prod[j] for fila in nh for j in fila]
                bn = tok(textos_neg, padding=True, truncation=True, max_length=MAX_LEN_PRODUCTO, return_tensors="pt").to(DISPOSITIVO)
                with torch.autocast(DISPOSITIVO, dtype=torch.float16, enabled=AMP):
                    en_ = encoder(bn["input_ids"], bn["attention_mask"])
                ep_ = torch.cat([ep_, en_], 0)            # [B + B*K, dim]
            logits = args.escala * eq.float() @ ep_.float().T
            if negs_duros is None:
                etiquetas = torch.arange(len(lote_u), device=DISPOSITIVO)
                # simetrica: consulta->producto y producto->consulta
                loss = (F.cross_entropy(logits, etiquetas) + F.cross_entropy(logits.T, etiquetas)) / 2
            else:
                n_neg = negs_duros.shape[1]
                # cada consulta tiene su positivo en la columna i*(1+n_neg)
                etiquetas = torch.tensor([i * (1 + n_neg) for i in range(len(lote_u))], device=DISPOSITIVO)
                loss = F.cross_entropy(logits, etiquetas)
                # simetrica solo sobre los positivos del lote (producto -> consulta)
                loss = (loss + F.cross_entropy(logits[:, :len(lote_u)].T,
                                               torch.arange(len(lote_u), device=DISPOSITIVO))) / 2
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(entrenables, 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            paso += 1
            if paso % 200 == 0:
                print(f"  epoca {ep + 1} paso {paso}/{pasos} loss={loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        m_val, _ = evaluar(encoder, tok, cat, val, cand_val, hist_val, popularidad)
        print(f"  val epoca {ep + 1}: hr@{K}={m_val[f'hr@{K}']:.4f} ndcg@{K}={m_val[f'ndcg@{K}']:.4f}")
        historial.append({"paso": paso, **{k: v for k, v in m_val.items() if k != "popularidad"}})
        # criterio de parada: el recall@10 de la busqueda de texto (la tarea del two-tower)
        if args.paciencia:
            items_val = codificar(encoder, tok, (PREFIJO_PRODUCTO + cat["texto"]).tolist(), MAX_LEN_PRODUCTO)
            qf = codificar(encoder, tok, (PREFIJO_CONSULTA + val["resena"].str.slice(0, 400)).tolist(),
                           MAX_LEN_CONSULTA)
            r10 = recall_catalogo(qf, items_val, val["item_idx"].to_numpy(), 10)
            print(f"    recall@10 de busqueda (validacion): {r10:.4f}")
            if r10 > mejor_recall:
                mejor_recall, mejor_epoca = r10, ep + 1
                mejor_estado = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
            elif ep + 1 - mejor_epoca >= args.paciencia:
                print(f"  parada temprana en la epoca {ep+1} (mejor: {mejor_epoca}, recall {mejor_recall:.4f})")
                break
    if mejor_estado is not None:
        encoder.load_state_dict(mejor_estado)
        print(f"  se conserva la epoca {mejor_epoca} (recall@10 de busqueda {mejor_recall:.4f})")

    # Se guarda ANTES de evaluar: si el proceso muere en la evaluacion (tarda minutos), el
    # entrenamiento no se pierde y se termina con --terminar.
    carpeta = nueva_carpeta("two_tower")
    encoder.transformer.save_pretrained(os.path.join(carpeta, "hf"))  # -> model.safetensors
    tok.save_pretrained(os.path.join(carpeta, "hf"))
    with open(os.path.join(carpeta, "entrenamiento.json"), "w", encoding="utf-8") as f:
        json.dump({"historial_val": historial, "origen": origen, "args": vars(args), "n_pares": len(pares),
                   "negativos_duros": bool(negs_duros), "mejor_epoca": mejor_epoca,
                   "mejor_recall_busqueda": mejor_recall, "seg": round(time.time() - t0)}, f, indent=2)
    print(f"  pesos guardados en {carpeta}")
    evaluar_y_exportar(encoder, tok, carpeta, cat, test, cand_test, hist_test, popularidad, frio)


def evaluar_y_exportar(encoder, tok, carpeta, cat, test, cand_test, hist_test, popularidad, frio):
    t0 = time.time()
    print("Evaluacion en test")
    m_test, items = evaluar(encoder, tok, cat, test, cand_test, hist_test, popularidad, frio)
    print(json.dumps(m_test, indent=2))
    np.save(os.path.join(carpeta, "items.npy"), items.astype(np.float16))
    diff = exportar_onnx(encoder, tok, os.path.join(carpeta, "encoder.onnx"))
    with open(os.path.join(carpeta, "metricas.json"), "w", encoding="utf-8") as f:
        entreno = json.load(open(os.path.join(carpeta, "entrenamiento.json"), encoding="utf-8"))
        json.dump({"test": m_test, "paridad_onnx": diff, "seg_evaluacion": round(time.time() - t0), **entreno},
                  f, indent=2)
    print(f"OK -> {carpeta}")


if __name__ == "__main__":
    main()
