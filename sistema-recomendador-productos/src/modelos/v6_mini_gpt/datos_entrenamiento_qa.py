"""Genera el dataset de instrucciones (pregunta -> respuesta) para afinar
el Mini-GPT como ASISTENTE de recomendaciones.

La fuente de verdad son las salidas reales del motor hibrido
(ver_grafos/app/src/data/recomendaciones.json, generado por
src/servicio/recomendaciones_app.py): para cada uno de los 500 usuarios
tenemos perfil, historial y Top-10 con la contribucion de cada componente.
Asi el modelo aprende a RESPONDER con los mismos datos que muestra la app,
no a inventar.

Salida: data/entrenamiento_qa.jsonl  (una pareja {"instruccion","respuesta"} por linea)

Uso:
    venv\\Scripts\\python.exe src/modelos/v6_mini_gpt/datos_entrenamiento_qa.py
"""

import sys
from pathlib import Path

_RUTA_MODULO = Path(__file__).resolve()
while _RUTA_MODULO.name != "src" and _RUTA_MODULO != _RUTA_MODULO.parent:
    _RUTA_MODULO = _RUTA_MODULO.parent
sys.path.insert(0, str(_RUTA_MODULO.parent))

from src.rutas import RUTA_RAIZ

import json
import random

import pandas as pd

RUTA_DATOS = RUTA_RAIZ / "data"
RUTA_JSON = RUTA_RAIZ / "ver_grafos" / "app" / "src" / "data" / "recomendaciones.json"
RUTA_SALIDA = RUTA_DATOS / "entrenamiento_qa.jsonl"

SEED = 42
MAX_RESPUESTA = 300   # caracteres: mantiene cada pareja dentro del contexto del GPT
SOBREMUESTRA_NO_REC = 3   # equilibra intenciones frente a "recomendar"

NOMBRE_COMPONENTE = {
    "two_tower": "el motor Two-Tower",
    "lightgcn": "la red de grafos LightGCN",
    "ncf": "la red neuronal colaborativa",
    "knn": "la similitud entre usuarios parecidos",
    "nlp_resenas": "el analisis semantico de textos",
}

PREGUNTAS_RECOMENDAR = [
    "Que me recomiendas?",
    "Que productos me recomiendas comprar?",
    "Dame tus recomendaciones.",
    "Recomiendame productos nuevos.",
    "Que puedo comprar hoy?",
    "Cuales son tus sugerencias para mi?",
    "Que le recomiendas al usuario {uid}?",
    "Que productos le recomiendas a {uid}?",
    "Dame recomendaciones para {uid}.",
    "Que le sugieres comprar a {uid}?",
]
PREGUNTAS_PERFIL = [
    "Cual es mi perfil de compra?",
    "Que categorias prefiero?",
    "Cuales son mis categorias favoritas?",
    "Cuales son las categorias favoritas de {uid}?",
    "Que le gusta comprar a {uid}?",
    "Como es el perfil de {uid}?",
]
PREGUNTAS_HISTORIAL = [
    "Que he comprado ultimamente?",
    "Cuales fueron mis ultimas compras?",
    "Que ha comprado {uid}?",
    "Muestra el historial de compras de {uid}.",
]
PREGUNTAS_PORQUE = [
    "Por que me recomiendas {producto}?",
    "Por que aparece {producto} en mi lista?",
    "Por que recomiendas {producto} a {uid}?",
]
PREGUNTAS_TOP1 = [
    "Cual es tu recomendacion numero uno?",
    "Cual es el producto estrella para mi?",
    "Cual es la mejor recomendacion para {uid}?",
]
PREGUNTAS_COMPRAS = [
    "Cuantas compras tengo registradas?",
    "Cuantas compras tiene {uid}?",
    "Cuantas compras hay en mi historial?",
]


def leer_csv(ruta: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(ruta, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(ruta, encoding="latin-1")


def limpiar(texto: str) -> str:
    return " ".join(str(texto).split())


def acortar(nombre: str, n: int) -> str:
    nombre = limpiar(nombre)
    return nombre if len(nombre) <= n else nombre[: n - 1].rstrip() + "..."


def mejor_componente(contribucion: dict) -> str:
    return max(contribucion, key=contribucion.get) if contribucion else "knn"


def respuesta_recomendar(u: dict, uid: str, tercera: bool) -> str:
    recs = u["recomendaciones"][:3]
    partes = [f"{acortar(r['producto'], 40)} ({r['categoria']}, "
              f"{round(r['score'] * 100)}%)" for r in recs]
    lista = f"{partes[0]}, {partes[1]} y {partes[2]}"
    cats = list(u["perfil"].keys())[:2]
    if tercera:
        return (f"Para el usuario {uid} recomiendo: {lista}. Encajan con el "
                f"interes del usuario y el motor hibrido les da los puntajes "
                f"mas altos.")
    razon = (f"Encajan con tu interes en {' y '.join(cats)}, y el motor hibrido "
             f"les da los puntajes mas altos." if cats else
             "Son los puntajes mas altos del motor hibrido.")
    return f"Te recomiendo: {lista}. {razon}"


def respuesta_perfil(u: dict, uid: str, tercera: bool) -> str:
    perfil = list(u["perfil"].items())[:3]
    detalle = ", ".join(f"{cat} con {n} compras" for cat, n in perfil)
    if tercera:
        return (f"Las categorias favoritas de {uid} son {detalle}. "
                f"Ese perfil alimenta la matriz de afinidad del sistema.")
    return (f"Tus categorias favoritas son {detalle}. "
            f"Ese perfil personaliza todas tus recomendaciones.")


def respuesta_historial(u: dict, uid: str, tercera: bool) -> str:
    hist = u["historial"][:3]
    partes = [f"{acortar(h['nombre'], 36)} ({h['categoria']})" for h in hist]
    lista = ", ".join(partes)
    if tercera:
        return (f"Las ultimas compras de {uid} fueron: {lista}. "
                f"Con ese historial el hibrido calcula su afinidad con el catalogo.")
    return (f"Tus ultimas compras fueron: {lista}. "
            f"Con ese historial se calcula tu afinidad con el catalogo.")


def respuesta_porque(rec: dict, u: dict, uid: str, tercera: bool) -> str:
    comp = NOMBRE_COMPONENTE.get(mejor_componente(rec["contribucion"]),
                                 "el motor hibrido")
    pct = round(rec["score"] * 100)
    cat = rec["categoria"]
    if tercera:
        return (f"{acortar(rec['producto'], 44)} obtuvo {pct}% de afinidad para "
                f"{uid}: destaca la aportacion de {comp} y coincide con las "
                f"categorias {cat} que ese usuario suele comprar.")
    return (f"Lo recomiendo porque obtuvo {pct}% de afinidad: destaca {comp} y "
            f"coincide con tu historial en {cat}. Es de lo mejor de tu Top-10.")


def respuesta_top1(u: dict, uid: str, tercera: bool) -> str:
    r = u["recomendaciones"][0]
    comp = NOMBRE_COMPONENTE.get(mejor_componente(r["contribucion"]), "el motor hibrido")
    if tercera:
        return (f"La mejor recomendacion para {uid} es {acortar(r['producto'], 40)} "
                f"de {r['categoria']}, con {round(r['score'] * 100)}% de afinidad; "
                f"la impulsa sobre todo {comp}.")
    return (f"Mi recomendacion numero uno es {acortar(r['producto'], 40)} de "
            f"{r['categoria']}, con {round(r['score'] * 100)}% de afinidad; "
            f"la impulsa sobre todo {comp}.")


def respuesta_compras(u: dict, uid: str) -> str:
    return (f"{uid} tiene {u['total_compras']} compras registradas en el conjunto "
            f"de entrenamiento; con ellas se construye su perfil y sus Top-10.")


def pares_globales(datos: dict, df_usuarios: pd.DataFrame) -> list[dict]:
    n_prod = len(pd.read_csv(RUTA_DATOS / "productos.csv"))
    n_usu = len(datos["usuarios"])
    fijas = [
        ("Hola", "Hola. Soy el asistente del sistema recomendador de supermercado. "
                 "Puedo darte recomendaciones, tu perfil o tu historial. Que deseas saber?"),
        ("Quien eres?", "Soy un mini-GPT entrenado desde cero con Keras y luego afinado "
                        "con instrucciones sobre las salidas reales del motor hibrido "
                        "de recomendaciones."),
        ("Que puedes hacer?", "Puedo contarte que productos te recomienda el motor "
                              "hibrido, explicarte por que, resumir tu perfil de compra "
                              "y tu historial. Preguntame por cualquier usuario (ej: u0007)."),
        ("Como funcionas?", "Recibo tu pregunta, detecto la intencion y el usuario, consulto "
                            "las recomendaciones reales del motor hibido y redacto la "
                            "respuesta con mi modelo de lenguaje afinado."),
        ("Como generas tus respuestas?", "Combino dos piezas: el motor hibido calcula los "
                                         "Top-10 con datos reales y yo, un transformer "
                                         "afinado por instrucciones, redacto la respuesta."),
        ("Que motores forman el hibrido?", "Cinco motores votan: Two-Tower, red neuronal "
                                           "colaborativa (NCF), KNN de usuarios similares, "
                                           "analisis semantico NLP de textos y LightGCN de grafos."),
        ("Que es Two-Tower?", "Es una red neuronal con dos torres de embeddings, una para "
                              "usuarios y otra para productos; su producto punto mide la "
                              "compatibilidad usuario-producto."),
        ("Que es LightGCN?", "Es una red neuronal de grafos que propaga preferencias por el "
                             "grafo usuario-producto: si compradores parecidos gustan de un "
                             "producto, su puntaje sube."),
        ("Que es NCF?", "Neural Collaborative Filtering: una red neuronal que aprende las "
                        "interacciones usuario-producto y predice la afinidad de cada par."),
        ("Que es KNN en este proyecto?", "Encuentra usuarios con historiales parecidos al tuyo "
                                         "y te recomienda lo que a ellos les funciono: "
                                         "filtrado colaborativo por vecinos."),
        ("Cuantos productos tiene el catalogo?", f"El supermercado tiene {n_prod} productos "
                                                 f"cargados, con categoria principal, subcategoria "
                                                 f"y descripcion textual."),
        ("Cuantos usuarios hay?", f"Hay {n_usu} usuarios registrados (de u0000 a u{len(list(datos['usuarios'])) - 1:04d}), "
                                  f"cada uno con perfil, historial y Top-10 calculados."),
        ("Gracias", "Con gusto. Si quieres otra recomendacion, dime el usuario "
                    "(por ejemplo u0007) y seguimos."),
    ]
    fuera = []
    for i in range(0, 60):
        uid_malo = f"u9{i % 10}{(i // 10) % 10}{i % 3}"
        if uid_malo in datos["usuarios"]:
            continue
        fuera.append((
            f"Que le recomiendas al usuario {uid_malo}?",
            f"No encuentro a {uid_malo}. Los usuarios van de u0000 a "
            f"u{len(list(datos['usuarios'])) - 1:04d}; prueba con uno de ellos.",
        ))
        fuera.append((
            f"Dame el perfil de {uid_malo}",
            f"No tengo datos de {uid_malo}. Prueba con un usuario existente "
            f"(ejemplo: u0000).",
        ))
    return [{"instruccion": q, "respuesta": r} for q, r in fijas + fuera]


def construir_contexto(intent: str, u: dict, uid: str,
                       rec: dict | None = None) -> str:
    """Reconstruye la linea '### Contexto:' para el servidor (RAG en vivo)."""
    if intent in ("global", "saludo", "quien", "capacidades", "motores",
                  "gracias"):
        return "general"
    recs = u.get("recomendaciones") or []
    cats = ", ".join(list(u["perfil"].keys())[:3])
    base = f"usuario={uid}; intereses={cats}"

    def ctx_top(lista: list[dict], n: int = 3) -> str:
        return "; ".join(f"{acortar(r['producto'], 34)}|{r['categoria']}|"
                         f"{round(r['score'] * 100)}%" for r in lista[:n])

    if intent == "recomendar":
        return f"{base}; top: {ctx_top(recs)}"
    if intent == "top1":
        return f"{base}; numero1: {ctx_top(recs, 1)}"
    if intent == "perfil":
        detalle = "; ".join(f"{cat}={n}"
                            for cat, n in list(u["perfil"].items())[:3])
        return f"usuario={uid}; categorias: {detalle}"
    if intent == "historial":
        ultimas = "; ".join(f"{acortar(h['nombre'], 30)}|{h['categoria']}"
                            for h in u.get("historial", [])[:3])
        return f"usuario={uid}; ultimas_compras: {ultimas}"
    if intent == "porque" and rec:
        return (f"usuario={uid}; producto={acortar(rec['producto'], 34)}|"
                f"{rec['categoria']}|{round(rec['score'] * 100)}%; "
                f"motor_principal={mejor_componente(rec['contribucion'])}")
    if intent == "compras":
        return f"usuario={uid}; compras_registradas={u['total_compras']}"
    return base


def main() -> None:
    rng = random.Random(SEED)
    datos = json.loads(RUTA_JSON.read_text(encoding="utf-8"))
    df_usuarios = leer_csv(RUTA_DATOS / "usuarios.csv")
    nombres = dict(zip(df_usuarios["user_id"], df_usuarios["nombre"])) \
        if "nombre" in df_usuarios.columns else {}

    pares: list[dict] = []

    def agregar(instruccion: str, respuesta: str, intencion: str,
                contexto: str = "general") -> None:
        pares.append({"instruccion": instruccion,
                      "respuesta": respuesta[:MAX_RESPUESTA],
                      "intencion": intencion,
                      "contexto": contexto})

    def ctx_top(recs: list[dict], n: int = 3) -> str:
        return "; ".join(f"{acortar(r['producto'], 34)}|{r['categoria']}|"
                         f"{round(r['score'] * 100)}%" for r in recs[:n])

    for uid, u in sorted(datos["usuarios"].items()):
        recs = u.get("recomendaciones") or []
        if not recs:
            continue
        nombre = nombres.get(uid)
        cats = ", ".join(list(u["perfil"].keys())[:3])
        base_ctx = f"usuario={uid}; intereses={cats}"

        # 1) recomendaciones (2 variantes por usuario + variante con nombre)
        for k in (0, 6):
            q = PREGUNTAS_RECOMENDAR[(k + hash(uid) % 4) % len(PREGUNTAS_RECOMENDAR)]
            tercera = "{uid}" in q
            agregar(q.format(uid=uid),
                    respuesta_recomendar(u, uid, tercera), "recomendar",
                    f"{base_ctx}; top: {ctx_top(recs)}")
        if nombre and len(nombre.split()) >= 2:
            agregar(f"Que me recomiendas, {nombre.split()[0]}?",
                    respuesta_recomendar(u, uid, False), "recomendar",
                    f"{base_ctx}; top: {ctx_top(recs)}")

        # 2) perfil (2 variantes para equilibrar con recomendar)
        detalle_perfil = "; ".join(f"{cat}={n}"
                                   for cat, n in list(u["perfil"].items())[:3])
        for q in rng.sample(PREGUNTAS_PERFIL, 2):
            tercera = "{uid}" in q
            agregar(q.format(uid=uid),
                    respuesta_perfil(u, uid, tercera), "perfil",
                    f"usuario={uid}; categorias: {detalle_perfil}")

        # 3) historial
        q = PREGUNTAS_HISTORIAL[rng.randrange(len(PREGUNTAS_HISTORIAL))]
        tercera = "{uid}" in q
        ultimas = "; ".join(f"{acortar(h['nombre'], 30)}|{h['categoria']}"
                            for h in u.get("historial", [])[:3])
        agregar(q.format(uid=uid),
                respuesta_historial(u, uid, tercera), "historial",
                f"usuario={uid}; ultimas_compras: {ultimas}")

        # 4) porque (para los 2 primeros del Top-10)
        for rec in recs[:2]:
            q = PREGUNTAS_PORQUE[rng.randrange(len(PREGUNTAS_PORQUE))]
            tercera = "a {uid}" in q
            agregar(q.format(producto=acortar(rec["producto"], 30), uid=uid),
                    respuesta_porque(rec, u, uid, tercera), "porque",
                    f"usuario={uid}; producto={acortar(rec['producto'], 34)}|"
                    f"{rec['categoria']}|{round(rec['score'] * 100)}%; "
                    f"motor_principal={mejor_componente(rec['contribucion'])}")

        # 5) top-1
        q = PREGUNTAS_TOP1[rng.randrange(len(PREGUNTAS_TOP1))]
        tercera = "{uid}" in q
        agregar(q.format(uid=uid), respuesta_top1(u, uid, tercera), "top1",
                f"{base_ctx}; numero1: {ctx_top(recs, 1)}")

        # 6) cuantas compras
        q = PREGUNTAS_COMPRAS[rng.randrange(len(PREGUNTAS_COMPRAS))]
        agregar(q.format(uid=uid), respuesta_compras(u, uid), "compras",
                f"usuario={uid}; compras_registradas={u['total_compras']}")

    for par in pares_globales(datos, df_usuarios):
        agregar(par["instruccion"], par["respuesta"], "global", "general")

    # Equilibrio de intenciones: las que NO son "recomendar" se repiten para
    # que el modelo no responda todo con el formato de recomendaciones.
    equilibrados: list[dict] = []
    for par in pares:
        copias = 1 if par["intencion"] == "recomendar" else SOBREMUESTRA_NO_REC
        equilibrados.extend([par] * copias)
    pares = [{k: v for k, v in par.items() if k != "intencion"}
             for par in equilibrados]

    rng.shuffle(pares)
    RUTA_SALIDA.write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pares),
        encoding="utf-8")

    largos = [len(p["instruccion"]) + len(p.get("contexto", ""))
              + len(p["respuesta"]) for p in pares]
    print(f"OK {len(pares)} parejas pregunta-respuesta -> {RUTA_SALIDA}")
    print(f"   largo combinado medio={sum(largos)/len(largos):.0f} chars | "
          f"maximo={max(largos)} | p95={sorted(largos)[int(len(largos)*0.95)]}")
    print("\nEJEMPLOS:")
    for p in pares[:3]:
        print(f"\n  P: {p['instruccion']}"
              f"\n  C: {p.get('contexto', '')}\n  R: {p['respuesta']}")


if __name__ == "__main__":
    main()
