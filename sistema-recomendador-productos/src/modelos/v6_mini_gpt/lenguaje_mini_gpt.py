"""Mini-GPT: modelo de lenguaje Transformer entrenado DESDE CERO en Keras.

La misma arquitectura de GPT (tokenizacion -> embeddings -> bloques de
auto-atencion con mascara causal -> prediccion del siguiente token),
en escala educativa (~600K parametros) y sobre el corpus propio del
proyecto: descripciones del supermercado + titulos e-commerce en espanol.

A diferencia de Word2Vec (ventana fija, una capa), aqui cada token predice
el siguiente atendiendo a TODO el contexto previo mediante self-attention.

Uso:
    venv\\Scripts\\python.exe src/lenguaje_mini_gpt.py            # entrenar + demo
    venv\\Scripts\\python.exe src/lenguaje_mini_gpt.py --generar "Arroz Diana"
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
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
CONTEXTO = 96       # caracteres de contexto que el modelo puede atender
D_MODELO = 128      # dimension del embedding / del modelo
BLOQUES = 3         # capas de transformer apiladas
CABEZAS = 4         # cabezas de atencion por bloque
FFN = 512           # anchura de la red feed-forward interna
BATCH = 64
PASOS = 1500        # pasos por defecto (~7 min de CPU); sube con --pasos para mas calidad
LR = 3e-4

RUTA_BASE = RUTA_RAIZ
RUTA_DATOS = RUTA_BASE / "data"
RUTA_MODELOS = RUTA_BASE / "models"


def fijar_semillas() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)


def cargar_corpus() -> str:
    """Une nombres+descripciones de supermercado y titulos de e-commerce."""
    print("[1/8] Cargando corpus en espanol...")
    productos = pd.read_csv(RUTA_DATOS / "productos.csv")
    partes = [
        f"{r['name']}. {r['texto']}"
        for _, r in productos.iterrows()
        if isinstance(r.get("texto"), str)
    ]
    df_ecom = pd.read_parquet(
        RUTA_DATOS / "titulos-ecommerce-es/data/train-00000-of-00001.parquet")
    partes.extend(str(t) for t in df_ecom["title"])
    corpus = "\n".join(partes)
    print(f"      {len(productos)} descripciones + {len(df_ecom):,} titulos "
          f"= {len(corpus):,} caracteres")
    return corpus


def construir_vocabulario(corpus: str) -> tuple[dict[str, int], dict[int, str]]:
    """Vocabulario a nivel de caracter: sin palabras fuera de vocabulario.

    Los LLM reales usan subpalabras (BPE); a esta escala, el caracter da un
    vocabulario diminuto (~100) y entrenamiento visible en minutos de CPU.
    """
    print("[2/8] Construyendo vocabulario a nivel de caracter...")
    caracteres = sorted(set(corpus))
    stoi = {c: i for i, c in enumerate(caracteres)}
    itos = {i: c for c, i in stoi.items()}
    print(f"      Vocabulario: {len(caracteres)} simbolos")
    return stoi, itos


def construir_modelo(vocab_size: int) -> tf.keras.Model:
    """GPT minimo: embeddings + N bloques de auto-atencion causal + cabeza LM.

    El indice vocab_size se reserva como PAD para rellenar prompts cortos
    durante la generacion (nunca aparece como objetivo en el entrenamiento).
    """
    entrada = tf.keras.Input(shape=(CONTEXTO,), dtype=tf.int32, name="tokens")
    tok = tf.keras.layers.Embedding(vocab_size + 1, D_MODELO, name="emb_token")(entrada)
    posiciones = tf.keras.ops.expand_dims(tf.keras.ops.arange(CONTEXTO), axis=0)
    pos = tf.keras.layers.Embedding(CONTEXTO, D_MODELO, name="emb_posicion")(posiciones)
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
    return tf.keras.Model(entrada, logits, name="mini_gpt")


def lote(rng: np.random.Generator, datos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = rng.integers(0, len(datos) - CONTEXTO - 1, size=BATCH)
    x = np.stack([datos[i:i + CONTEXTO] for i in idx]).astype(np.int32)
    y = np.stack([datos[i + 1:i + CONTEXTO + 1] for i in idx]).astype(np.int32)
    return x, y


def generar(modelo: tf.keras.Model, itos: dict[int, str], stoi: dict[str, int],
            prompt: str, n_caracteres: int = 200, temperatura: float = 0.8) -> str:
    """Generacion autorregresiva con muestreo por temperatura.

    La ventana se rellena a la derecha con PAD y la distribucion del
    siguiente caracter se toma en la posicion del ultimo token real.
    """
    rng = np.random.default_rng(SEED)
    pad_id = len(stoi)
    tokens = [stoi[c] for c in prompt if c in stoi]
    if not tokens:
        return ""
    generados = list(tokens)
    for _ in range(n_caracteres):
        largo = min(len(generados), CONTEXTO)
        ventana = np.full(CONTEXTO, pad_id, dtype=np.int32)
        ventana[:largo] = generados[-largo:]
        logits = modelo(ventana[None, :], training=False)[0, largo - 1].numpy()
        probs = np.exp(logits[:pad_id] / temperatura)
        probs /= probs.sum()
        generados.append(int(rng.choice(len(probs), p=probs)))
    return "".join(itos[t] for t in generados)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini-GPT del proyecto")
    parser.add_argument("--generar", nargs="?", const="", metavar="PROMPT",
                        help="solo cargar pesos y generar desde PROMPT")
    parser.add_argument("--pasos", type=int, default=PASOS, metavar="N",
                        help=f"pasos de entrenamiento (default {PASOS})")
    args = parser.parse_args()
    fijar_semillas()

    if args.generar is not None:
        stoi_guardado = json.loads(
            (RUTA_MODELOS / "mini_gpt_vocab.json").read_text(encoding="utf-8"))
        stoi = {c: int(i) for c, i in stoi_guardado.items()}
        itos = {i: c for c, i in stoi.items()}
        modelo = construir_modelo(len(stoi))
        modelo.load_weights(RUTA_MODELOS / "mini_gpt.weights.h5")
        prompt = args.generar or "Arroz Diana blanco"
        print(generar(modelo, itos, stoi, prompt))
        return

    inicio_total = time.time()
    corpus = cargar_corpus()
    stoi_nuevo, itos_nuevo = construir_vocabulario(corpus)

    print("[3/8] Codificando corpus...")
    datos = np.array([stoi_nuevo[c] for c in corpus], dtype=np.int32)

    print(f"[4/8] Construyendo Mini-GPT ({BLOQUES} bloques, {CABEZAS} cabezas, "
          f"d={D_MODELO}, contexto={CONTEXTO})...")
    modelo = construir_modelo(len(stoi_nuevo))
    params = modelo.count_params()
    modelo.summary()

    perdida = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizador = tf.keras.optimizers.Adam(LR, clipnorm=1.0)

    @tf.function
    def paso(x, y):
        with tf.GradientTape() as cinta:
            logits = modelo(x, training=True)
            valor = perdida(y, logits)
        gradientes = cinta.gradient(valor, modelo.trainable_variables)
        optimizador.apply_gradients(zip(gradientes, modelo.trainable_variables))
        return valor

    print(f"[5/8] Entrenando {args.pasos} pasos (batch={BATCH})...")
    rng = np.random.default_rng(SEED)
    inicio = time.time()
    acumulada = []
    for paso_i in range(1, args.pasos + 1):
        x, y = lote(rng, datos)
        valor = paso(x, y).numpy()
        acumulada.append(float(valor))
        if paso_i % 100 == 0:
            media = np.mean(acumulada[-100:])
            print(f"      Paso {paso_i:>5}/{args.pasos} | perdida={media:.4f} | "
                  f"perplejidad={np.exp(media):.2f}", flush=True)
    perdida_final = float(np.mean(acumulada[-100:]))

    print("[6/8] Guardando artefactos...")
    modelo.save_weights(RUTA_MODELOS / "mini_gpt.weights.h5")
    (RUTA_MODELOS / "mini_gpt_vocab.json").write_text(
        json.dumps(stoi_nuevo, ensure_ascii=False), encoding="utf-8")
    (RUTA_MODELOS / "mini_gpt_metricas.json").write_text(json.dumps({
        "parametros": int(params),
        "pasos": args.pasos,
        "perdida_final": round(perdida_final, 4),
        "perplejidad": round(float(np.exp(perdida_final)), 2),
        "contexto": CONTEXTO,
        "vocabulario": len(stoi_nuevo),
        "caracteres_corpus": len(corpus),
        "duracion_s": round(time.time() - inicio, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[7/8] DEMO - Generacion autorregresiva (temperatura 0.8):")
    for prompt in ["Arroz Diana", "Leche entera", "Bolsa con capacidad para",
                   "Galletas de chocolate", "Vino tinto"]:
        texto = generar(modelo, itos_nuevo, stoi_nuevo, prompt, n_caracteres=180)
        print(f"\n  Prompt: '{prompt}'\n  -> {texto!r}")

    print("[8/8] COMPLETADO "
          f"({time.time()-inicio_total:.1f}s) | perplejidad final="
          f"{np.exp(perdida_final):.2f}")


if __name__ == "__main__":
    main()
