"""Afinamiento por instrucciones (SFT) del Mini-GPT -> ASISTENTE del proyecto.

Es el mismo salto tecnico que convirtio GPT-3 en ChatGPT, en escala educativa:

  Etapa A (modelo base):   preentrenamiento de siguiente-token sobre el corpus
                           propio (descripciones + titulos e-commerce).
  Etapa B (asistente):     SFT sobre parejas pregunta->respuesta generadas desde
                           las SALIDAS REALES del motor hibrido
                           (data/entrenamiento_qa.jsonl, creado por
                           datos_entrenamiento_qa.py). La perdida se calcula SOLO
                           sobre la respuesta (mascara), para que el modelo aprenda
                           a obedecer instrucciones, no a repetir preguntas.

Arquitectura mejorada frente al modelo base v6: 6 bloques, d=256, 8 cabezas,
FFN 768, contexto 512 (~4M parametros vs ~0.6M) y formato de instruccion con
linea '### Contexto:' (RAG en el prompt): los datos recuperados del hibrido
viajan DENTRO del prompt y el modelo solo tiene que redactarlos.

Artefactos (en models/):
    mini_gpt_chat_vocab.json     vocabulario caracter->id
    mini_gpt_chat_base.weights.h5   checkpoint tras la etapa A
    mini_gpt_chat.weights.h5     pesos finales tras SFT
    mini_gpt_chat_metricas.json  metricas de las dos etapas

Uso:
    venv\\Scripts\\python.exe src/modelos/v6_mini_gpt/afinar_gpt_chat.py
    venv\\Scripts\\python.exe src/modelos/v6_mini_gpt/afinar_gpt_chat.py --usar-base
    venv\\Scripts\\python.exe src/modelos/v6_mini_gpt/afinar_gpt_chat.py \\
        --generar "Que le recomiendas a u0007?"
"""

import sys
from pathlib import Path

_RUTA_MODULO = Path(__file__).resolve()
while _RUTA_MODULO.name != "src" and _RUTA_MODULO != _RUTA_MODULO.parent:
    _RUTA_MODULO = _RUTA_MODULO.parent
sys.path.insert(0, str(_RUTA_MODULO.parent))

from src.rutas import RUTA_RAIZ

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf

# ----------------------------- hiperparametros ------------------------------
SEED = 42
CONTEXTO = 512       # caracteres visibles: pregunta + contexto RAG + respuesta
VENTANA_A = 320      # ventana usada SOLO en el preentrenamiento (etapa A)
D_MODELO = 256       # antes 128
BLOQUES = 6          # antes 3
CABEZAS = 8          # antes 4
FFN = 768            # antes 512
BATCH_BASE = 24      # etapa A (corpus)
BATCH_SFT = 8        # etapa B: secuencias de 512 no caben con lotes grandes (4 GB VRAM)
PASOS_BASE = 900     # etapa A por defecto
PASOS_SFT = 1600     # etapa B por defecto
LR_BASE = 3e-4
LR_SFT = 1e-4        # mas bajo: no olvidar el lenguaje general al afinar

# Formato de instruccion compartido con servidor_chat.py
FMT_PREGUNTA = "### Pregunta: "
FMT_CONTEXTO = "\n### Contexto: "
FMT_RESPUESTA = "\n### Respuesta: "
FMT_FIN = "\n\n"


def armar_prefijo(pregunta: str, contexto: str = "") -> str:
    """Pregunta [+ contexto RAG] antes de la respuesta del asistente."""
    prefijo = FMT_PREGUNTA + pregunta.strip()
    if contexto:
        prefijo += FMT_CONTEXTO + contexto.strip()
    return prefijo + FMT_RESPUESTA

BUCKETS_ANCHO = (256, CONTEXTO)   # lotes SFT agrupados por largo (menos padding)

RUTA_DATOS = RUTA_RAIZ / "data"
RUTA_MODELOS = RUTA_RAIZ / "models"
ARCH_QA = RUTA_DATOS / "entrenamiento_qa.jsonl"
ARCH_VOCAB = RUTA_MODELOS / "mini_gpt_chat_vocab.json"
ARCH_BASE = RUTA_MODELOS / "mini_gpt_chat_base.weights.h5"
ARCH_FINAL = RUTA_MODELOS / "mini_gpt_chat.weights.h5"
ARCH_METRICAS = RUTA_MODELOS / "mini_gpt_chat_metricas.json"


def fijar_semillas() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)


# --------------------------------- corpus ----------------------------------
def cargar_corpus() -> str:
    """Mismo corpus del modelo base v6: descripciones + titulos."""
    productos = pd.read_csv(RUTA_DATOS / "productos.csv")
    partes = [
        f"{r['name']}. {r['texto']}"
        for _, r in productos.iterrows()
        if isinstance(r.get("texto"), str)
    ]
    df_ecom = pd.read_parquet(
        RUTA_DATOS / "titulos-ecommerce-es/data/train-00000-of-00001.parquet")
    partes.extend(str(t) for t in df_ecom["title"])
    return "\n".join(partes)


def cargar_parejas_qa() -> list[tuple[str, str, str]]:
    lineas = ARCH_QA.read_text(encoding="utf-8").splitlines()
    return [(d["instruccion"], d["respuesta"], d.get("contexto", ""))
            for d in map(json.loads, lineas)]


# -------------------------------- vocabulario -------------------------------
def construir_vocabulario(textos: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    caracteres = sorted(set("".join(textos)))
    stoi = {c: i for i, c in enumerate(caracteres)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def codificar(stoi: dict[str, int], texto: str, limite: int | None = None) -> list[int]:
    ids = [stoi[c] for c in texto if c in stoi]
    return ids[:limite] if limite else ids


# --------------------------------- modelo -----------------------------------
def construir_modelo(vocab_size: int) -> tf.keras.Model:
    """Mini-GPT mejorado: vocab_size reservado como PAD (relleno, nunca objetivo).

    La entrada acepta cualquier largo <= CONTEXTO: el entrenamiento SFT usa
    lotes recortados al largo real (menos padding) y la generacion usa la
    ventana completa.
    """
    entrada = tf.keras.Input(shape=(None,), dtype=tf.int32, name="tokens")
    tok = tf.keras.layers.Embedding(vocab_size + 1, D_MODELO, name="emb_token")(entrada)
    indices_pos = tf.keras.layers.Lambda(
        lambda t: tf.broadcast_to(tf.range(tf.shape(t)[1]), tf.shape(t)),
        output_shape=(None,), name="indices_posicion")(entrada)
    pos = tf.keras.layers.Embedding(CONTEXTO, D_MODELO, name="emb_posicion")(indices_pos)
    x = tf.keras.layers.Add(name="sumar_posiciones")([tok, pos])

    for b in range(BLOQUES):
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=CABEZAS, key_dim=D_MODELO // CABEZAS,
            name=f"atencion_{b}",
        )(x, x, use_causal_mask=True)
        x = tf.keras.layers.Add(name=f"residuo_1_{b}")([x, attn])
        x = tf.keras.layers.LayerNormalization(name=f"norma_1_{b}")(x)
        h = tf.keras.layers.Dense(FFN, activation="gelu", name=f"ffn_1_{b}")(x)
        h = tf.keras.layers.Dense(D_MODELO, name=f"ffn_2_{b}")(h)
        x = tf.keras.layers.Add(name=f"residuo_2_{b}")([x, h])
        x = tf.keras.layers.LayerNormalization(name=f"norma_2_{b}")(x)

    logits = tf.keras.layers.Dense(vocab_size + 1, name="cabeza_lm")(x)
    return tf.keras.Model(entrada, logits, name="mini_gpt_chat")


# ------------------------------ bucles de entrenamiento ---------------------
def entrenar_corpus(modelo, corpus: str, stoi, pasos: int,
                    log_cada: int = 50) -> float:
    """Etapa A: ventanas aleatorias del corpus (siguiente caracter)."""
    print("      Codificando corpus...")
    corpus_ids = codificar(stoi, corpus)
    rng = np.random.default_rng(SEED)
    perdida_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizador = tf.keras.optimizers.Adam(LR_BASE, clipnorm=1.0)

    @tf.function(reduce_retracing=True)
    def paso(x, y):
        with tf.GradientTape() as cinta:
            logits = modelo(x, training=True)
            valor = perdida_fn(y, logits)
        gradientes = cinta.gradient(valor, modelo.trainable_variables)
        optimizador.apply_gradients(zip(gradientes, modelo.trainable_variables))
        return valor

    acumulada = []
    for paso_i in range(1, pasos + 1):
        idx = rng.integers(0, len(corpus_ids) - VENTANA_A - 1, size=BATCH_BASE)
        x = np.stack([corpus_ids[i:i + VENTANA_A] for i in idx]).astype(np.int32)
        y = np.stack([corpus_ids[i + 1:i + VENTANA_A + 1] for i in idx]).astype(np.int32)
        valor = paso(x, y).numpy()
        acumulada.append(float(valor))
        if paso_i % log_cada == 0:
            media = np.mean(acumulada[-log_cada:])
            print(f"      A paso {paso_i:>5}/{pasos} | perdida={media:.4f} | "
                  f"perplejidad={np.exp(media):.2f}", flush=True)
    return float(np.mean(acumulada[-log_cada:]))


def lotes_sft(parejas_cod: list[dict], stoi, rng: np.random.Generator):
    """Lotes agrupados por largo (bucketing): menos PAD, formas fijas."""
    pad_id = len(stoi)
    por_bucket: dict[int, list[dict]] = {w: [] for w in BUCKETS_ANCHO}
    for ej in parejas_cod:
        w = next((w for w in BUCKETS_ANCHO if len(ej["tokens"]) <= w),
                 BUCKETS_ANCHO[-1])
        por_bucket[w].append(ej)

    while True:
        mezclados: list[list[dict]] = []
        for w, grupo in por_bucket.items():
            g = list(grupo)
            rng.shuffle(g)
            mezclados += [g[k:k + BATCH_SFT] for k in range(0, len(g), BATCH_SFT)]
        rng.shuffle(mezclados)
        for grupo in mezclados:
            if len(grupo) < 2:
                continue
            w = next(k for k in BUCKETS_ANCHO if len(grupo[0]["tokens"]) <= k)
            b = len(grupo)
            x = np.full((b, w), pad_id, dtype=np.int32)
            y = np.full((b, w), pad_id, dtype=np.int32)
            m = np.zeros((b, w), dtype=np.float32)
            for i, ej in enumerate(grupo):
                t = ej["tokens"]
                ini = ej["inicio_resp"]
                x[i, :len(t) - 1] = t[:-1]
                y[i, :len(t) - 1] = t[1:]
                m[i, ini - 1:len(t) - 1] = 1.0
            yield x, y, m


def entrenar_sft(modelo, stoi, pasos: int, log_cada: int = 50) -> tuple[float, float]:
    """Etapa B: parejas pregunta->respuesta con perdida solo en la respuesta."""
    parejas = cargar_parejas_qa()
    print(f"      {len(parejas)} parejas QA | construyendo lotes...")
    parejas_cod = []
    for q, r, ctx in parejas:
        prefijo = armar_prefijo(q, ctx)
        completo = prefijo + r + FMT_FIN
        tokens = codificar(stoi, completo, limite=CONTEXTO)
        inicio_resp = min(len(codificar(stoi, prefijo)), len(tokens) - 1)
        if len(tokens) < 16 or inicio_resp >= len(tokens):
            continue
        parejas_cod.append({"tokens": tokens, "inicio_resp": inicio_resp})

    rng = np.random.default_rng(SEED + 1)
    gen_lotes = lotes_sft(parejas_cod, stoi, rng)
    perdida_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                               reduction="none")
    optimizador = tf.keras.optimizers.Adam(LR_SFT, clipnorm=1.0)

    @tf.function(reduce_retracing=True)
    def paso(x, y, m):
        with tf.GradientTape() as cinta:
            logits = modelo(x, training=True)
            valores = perdida_fn(y, logits)
            valor = tf.reduce_sum(valores * m) / tf.maximum(tf.reduce_sum(m), 1.0)
        gradientes = cinta.gradient(valor, modelo.trainable_variables)
        optimizador.apply_gradients(zip(gradientes, modelo.trainable_variables))
        return valor

    acumulada = []
    for paso_i in range(1, pasos + 1):
        x, y, m = next(gen_lotes)
        valor = paso(x, y, m).numpy()
        acumulada.append(float(valor))
        if paso_i % log_cada == 0:
            media = np.mean(acumulada[-log_cada:])
            print(f"      B paso {paso_i:>5}/{pasos} | perdida={media:.4f} | "
                  f"perplejidad={np.exp(media):.2f}", flush=True)
    return float(np.mean(acumulada[-log_cada:])), float(np.exp(
        np.mean(acumulada[-log_cada:])))


# ------------------------------- generacion ---------------------------------
def _fabrica_paso_infer(modelo):
    """Compila UNA vez la pasada hacia adelante (clave para inferencia rapida)."""
    @tf.function(reduce_retracing=True)
    def paso(ventana):
        return modelo(ventana, training=False)
    return paso


def generar_respuestas(modelo, itos: dict[int, str], stoi: dict[str, int],
                       pregunta: str, n_max: int = 240,
                       temperatura: float = 0.25, top_k: int = 6,
                       candidatos: int = 1, semilla: int | None = None,
                       contexto: str = "") -> list[str]:
    """Genera VARIOS candidatos en paralelo usando el lote como eje de muestreo.

    En GPU procesar un lote cuesta casi lo mismo que una sola secuencia:
    asi el servidor obtiene varias redacciones por el precio de una y se
    queda con la mejor validada contra los datos del hibrido.
    """
    rng = np.random.default_rng(semilla if semilla is not None
                                else abs(hash(pregunta)) % (2**32))
    pad_id = len(stoi)
    prompt = armar_prefijo(pregunta, contexto)
    tokens = [stoi[c] for c in prompt if c in stoi]
    if not tokens:
        return [""]
    k_total = max(1, candidatos)
    paso = _fabrica_paso_infer(modelo)

    generados: list[list[int]] = [[] for _ in range(k_total)]
    terminados = [False] * k_total
    for _ in range(n_max):
        ventana = np.full((k_total, CONTEXTO), pad_id, dtype=np.int32)
        for k in range(k_total):
            if terminados[k]:
                continue
            seq = (tokens + generados[k])[-CONTEXTO:]
            ventana[k, :len(seq)] = seq
        logits = paso(tf.convert_to_tensor(ventana)).numpy()
        for k in range(k_total):
            if terminados[k]:
                continue
            largo = len(tokens) + len(generados[k])
            fila = logits[k, min(largo, CONTEXTO) - 1, :pad_id].astype(np.float64)
            fila -= fila.max()                    # softmax numericamente estable
            probs = np.exp(fila / temperatura)
            mejores = np.argsort(probs)[::-1][:top_k]
            p = probs[mejores]
            p /= p.sum()
            elegido = int(rng.choice(mejores, p=p))
            generados[k].append(elegido)
            cola = "".join(itos.get(t, "") for t in generados[k][-8:])
            if cola.endswith(FMT_FIN) or cola.endswith("\n###"):
                terminados[k] = True

    salidas = []
    for gen in generados:
        texto = "".join(itos.get(t, "") for t in gen)
        for corte in (FMT_FIN, "\n###"):
            texto = texto.split(corte)[0]
        salidas.append(texto.strip())
    return salidas


def generar_respuesta(modelo, itos: dict[int, str], stoi: dict[str, int],
                      pregunta: str, n_max: int = 240,
                      temperatura: float = 0.25, top_k: int = 6,
                      semilla: int | None = None,
                      contexto: str = "") -> str:
    """Continua FMT_RESPUESTA y corta al empezar otra pregunta (FMT_FIN/###)."""
    return generar_respuestas(modelo, itos, stoi, pregunta, n_max=n_max,
                              temperatura=temperatura, top_k=top_k,
                              candidatos=1, semilla=semilla,
                              contexto=contexto)[0]


def cargar_chat(ruta_pesos: Path = ARCH_FINAL, ruta_vocab: Path = ARCH_VOCAB):
    """Para reuso desde servidor_chat.py: devuelve (modelo, stoi, itos)."""
    stoi_guardado = json.loads(ruta_vocab.read_text(encoding="utf-8"))
    stoi = {c: int(i) for c, i in stoi_guardado.items()}
    itos = {i: c for c, i in stoi.items()}
    modelo = construir_modelo(len(stoi))
    modelo.load_weights(ruta_pesos)
    return modelo, stoi, itos


# ---------------------------------- main ------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Afinamiento SFT del Mini-GPT")
    parser.add_argument("--pasos-base", type=int, default=PASOS_BASE)
    parser.add_argument("--pasos-sft", type=int, default=PASOS_SFT)
    parser.add_argument("--usar-base", action="store_true",
                        help="reutiliza mini_gpt_chat_base.weights.h5 si existe")
    parser.add_argument("--continuar", action="store_true",
                        help="sigue el SFT desde los pesos finales existentes")
    parser.add_argument("--generar", nargs="?", const="", metavar="PREGUNTA",
                        help="solo cargar pesos finales y responder")
    parser.add_argument("--uid", metavar="UID", default="u0007",
                        help="usuario para el contexto RAG en --generar")
    parser.add_argument("--con-contexto", action="store_true",
                        help="en --generar, construir el contexto desde "
                             "recomendaciones.json (RAG)")
    parser.add_argument("--temperatura", type=float, default=0.25)
    args = parser.parse_args()

    fijar_semillas()

    if args.generar is not None:
        modelo, stoi, itos = cargar_chat()
        pregunta = args.generar or "Que me recomiendas?"
        contexto = ""
        if args.con_contexto:
            from src.modelos.v6_mini_gpt.datos_entrenamiento_qa import (
                construir_contexto)
            datos = json.loads(
                (RUTA_RAIZ / "ver_grafos" / "app" / "src" / "data"
                 / "recomendaciones.json").read_text(encoding="utf-8"))
            u = datos["usuarios"].get(args.uid)
            if u is None:
                raise SystemExit(f"usuario {args.uid} no encontrado")
            from src.servicio.servidor_chat import detectar_intencion
            contexto = construir_contexto(detectar_intencion(pregunta),
                                          u, args.uid)
        inicio = time.time()
        respuesta = generar_respuesta(modelo, itos, stoi, pregunta,
                                      temperatura=args.temperatura,
                                      contexto=contexto)
        print(f"P: {pregunta}")
        if contexto:
            print(f"C: {contexto}")
        print(f"R: {respuesta}")
        print(f"({time.time() - inicio:.1f}s)")
        return

    inicio_total = time.time()

    print("[1/7] Cargando corpus y parejas QA...")
    corpus = cargar_corpus()
    parejas_qa = cargar_parejas_qa()
    print(f"      corpus={len(corpus):,} chars | qa={len(parejas_qa)} parejas")

    print("[2/7] Vocabulario de caracteres (corpus U qa)...")
    stoi, itos = construir_vocabulario(
        [corpus] + [q + ctx + r for q, r, ctx in parejas_qa])
    print(f"      vocabulario={len(stoi)} simbolos")

    print(f"[3/7] Construyendo Mini-GPT chat ({BLOQUES} bloques, d={D_MODELO}, "
          f"{CABEZAS} cabezas, FFN={FFN}, contexto={CONTEXTO})...")
    modelo = construir_modelo(len(stoi))
    params = modelo.count_params()
    print(f"      parametros: {params:,}")

    perdida_base = None
    if args.continuar and ARCH_FINAL.exists():
        print("[4/7] Etapa A omitida (--continuar): cargando pesos finales...")
        modelo.load_weights(ARCH_FINAL)
    elif args.usar_base and ARCH_BASE.exists():
        print("[4/7] Etapa A omitida (--usar-base): cargando checkpoint base...")
        modelo.load_weights(ARCH_BASE)
    else:
        print(f"[4/7] ETAPA A - preentrenamiento ({args.pasos_base} pasos)...")
        t0 = time.time()
        perdida_base = entrenar_corpus(modelo, corpus, stoi, args.pasos_base)
        print(f"      perdida final={perdida_base:.4f} "
              f"({time.time() - t0:.0f}s)")
        modelo.save_weights(ARCH_BASE)

    print(f"[5/7] ETAPA B - SFT por instrucciones ({args.pasos_sft} pasos, "
          f"lr={LR_SFT}, solo respuestas)...")
    t0 = time.time()
    perdida_sft, ppl_sft = entrenar_sft(modelo, stoi, args.pasos_sft)
    print(f"      perdida final={perdida_sft:.4f} ({time.time() - t0:.0f}s)")

    print("[6/7] Guardando artefactos...")
    modelo.save_weights(ARCH_FINAL)
    ARCH_VOCAB.write_text(json.dumps(stoi, ensure_ascii=False), encoding="utf-8")
    ARCH_METRICAS.write_text(json.dumps({
        "parametros": int(params),
        "bloques": BLOQUES, "d_modelo": D_MODELO, "cabezas": CABEZAS,
        "ffn": FFN, "contexto": CONTEXTO, "vocabulario": len(stoi),
        "pasos_base": args.pasos_base, "pasos_sft": args.pasos_sft,
        "parejas_qa": len(parejas_qa),
        "perdida_base": round(perdida_base, 4) if perdida_base else None,
        "perdida_sft": round(perdida_sft, 4), "perplejidad_sft": round(ppl_sft, 2),
        "duracion_total_s": round(time.time() - inicio_total, 1),
        "formato": {"pregunta": FMT_PREGUNTA, "contexto": FMT_CONTEXTO,
                    "respuesta": FMT_RESPUESTA},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[7/7] DEMO del asistente afinado:")
    demos = [
        ("Que me recomiendas?", ""),
        ("Que le recomiendas al usuario u0007?",
         "u0007; top: ver recomendaciones.json"),
        ("Cual es mi perfil de compra?", ""),
        ("Quien eres?", ""),
    ]
    for d, ctx in demos:
        r = generar_respuesta(modelo, itos, stoi, d, contexto=ctx)
        print(f"\n  P: {d}\n  R: {r}")

    print(f"\nCOMPLETADO en {time.time() - inicio_total:.0f}s | "
          f"perplejidad SFT={ppl_sft:.2f}")


if __name__ == "__main__":
    main()
