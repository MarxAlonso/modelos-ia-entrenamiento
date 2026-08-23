"""Servidor local del ASISTENTE de recomendaciones (Mini-GPT afinado + hibrido).

Arquitectura RAG educativa:
    1. Detecta la intencion de la pregunta y el usuario involucrado.
    2. Consulta los datos REALES del motor hibrido (recomendaciones.json,
       el mismo archivo que consume la app React).
    3. Le pide al Mini-GPT afinado que redacte la respuesta con el formato
       exacto con el que fue entrenado (SFT).
    4. Valida que el GPT cite productos reales; si se va a terreno inventado,
       responde con la plantilla basada en datos (nunca alucina numeros).

Endpoints:
    GET  /api/salud     -> estado del servidor y del modelo
    POST /api/chat      -> {"mensaje": "...", "uid": "u0007"} | {"respuesta": "..."}

Ejecucion (mejor con GPU via WSL; tambien funciona en CPU):
    wsl -d Ubuntu-22.04 -- ~/run-tf.sh src/servicio/servidor_chat.py
    venv\\Scripts\\python.exe src/servicio/servidor_chat.py            (CPU)
"""

import sys
from pathlib import Path

_RUTA_MODULO = Path(__file__).resolve()
while _RUTA_MODULO.name != "src" and _RUTA_MODULO != _RUTA_MODULO.parent:
    _RUTA_MODULO = _RUTA_MODULO.parent
sys.path.insert(0, str(_RUTA_MODULO.parent))

from src.rutas import RUTA_RAIZ

import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import tensorflow  # noqa: F401  (necesario antes de cargar el modelo)

from src.modelos.v6_mini_gpt.afinar_gpt_chat import (
    cargar_chat, generar_respuestas)
from src.modelos.v6_mini_gpt.datos_entrenamiento_qa import (
    acortar, limpiar, construir_contexto,
    respuesta_compras, respuesta_historial, respuesta_perfil,
    respuesta_porque, respuesta_recomendar, respuesta_top1)

PUERTO = 8000
ARCH_RECOMENDACIONES = (RUTA_RAIZ / "ver_grafos" / "app" / "src" / "data"
                        / "recomendaciones.json")

TEMPERATURA = 0.2
TOP_K = 4

RESPUESTAS_FIJAS = {
    "saludo": ("Hola",
               "Hola. Soy el asistente del sistema recomendador. Puedo darte "
               "recomendaciones, tu perfil o tu historial. Que deseas saber?"),
    "quien": ("Quien eres?",
              "Soy un mini-GPT entrenado desde cero con Keras y afinado con "
              "instrucciones sobre las salidas reales del motor hibrido de "
              "recomendaciones."),
    "capacidades": ("Que puedes hacer?",
                    "Puedo contarte que productos recomienda el motor hibrido, "
                    "explicarte por que, resumir tu perfil y tu historial de "
                    "compras. Preguntame por cualquier usuario (ej: u0007)."),
    "motores": ("Que motores forman el hibrido?",
                "Cinco motores votan en el hibrido: Two-Tower, red neuronal "
                "colaborativa (NCF), KNN de usuarios similares, analisis "
                "semantico NLP y LightGCN de grafos."),
    "gracias": ("Gracias",
                "Con gusto. Si quieres otra recomendacion, dime el usuario "
                "(por ejemplo u0007) y seguimos."),
}

ESTADO = {"modelo": None, "stoi": None, "itos": None, "datos": None}
_CACHE: dict[tuple, dict] = {}   # (intencion, pregunta canonica) -> respuesta


# --------------------------- deteccion de intencion -------------------------
def detectar_intencion(mensaje: str) -> str:
    m = mensaje.lower()
    if re.search(r"\b(hola|buenas|buenos dias|buenas tardes|hey)\b", m):
        return "saludo"
    if "gracias" in m:
        return "gracias"
    if re.search(r"quien eres|que eres|presentate", m):
        return "quien"
    if re.search(r"puedes hacer|como funcionas|como generas|que sabes hacer", m):
        return "capacidades"
    if re.search(r"motores|two.?tower|lightgcn|light gcn|\bncf\b|\bknn\b|hibrido", m):
        return "motores"
    if "por que" in m or "porque me" in m:
        return "porque"
    if re.search(r"numero uno|número uno|estrella|mejor recomendacion|mejor producto", m):
        return "top1"
    if re.search(r"cuantas compras|cuántas compras|registradas", m):
        return "compras"
    if re.search(r"perfil|categorias favoritas|categorías|que le gusta|que prefiero|prefiero", m):
        return "perfil"
    if re.search(r"historial|ultimas compras|últimas|he comprado|ha comprado", m):
        return "historial"
    if re.search(r"recomienda|recomiendas|sugiere|sugerencias|dame|comprar", m):
        return "recomendar"
    return "recomendar"


def detectar_uid(mensaje: str, uid_base: str) -> tuple[str | None, bool]:
    """Devuelve (uid, explicito). Prioriza el uid escrito en el mensaje."""
    encontrados = re.findall(r"\bu\d{3,5}\b", mensaje.lower())
    for u in encontrados:
        if u in ESTADO["datos"]["usuarios"]:
            return u, True
    if encontrados:
        return None, True          # menciono un usuario que NO existe
    return (uid_base if uid_base in ESTADO["datos"]["usuarios"] else None), False


# ------------------------------ grounding -----------------------------------
def canonica(intent: str, uid: str, primera: bool, rec=None) -> str:
    """Pregunta canonica en el EXACTO formato visto durante el SFT."""
    if intent == "perfil":
        return ("Cual es mi perfil de compra?" if primera else
                f"Cuales son las categorias favoritas de {uid}?")
    if intent == "historial":
        return "Que he comprado ultimamente?" if primera else f"Que ha comprado {uid}?"
    if intent == "porque":
        prod = acortar((rec or {}).get("producto", "un producto"), 30)
        return (f"Por que me recomiendas {prod}?" if primera else
                f"Por que recomiendas {prod} a {uid}?")
    if intent == "top1":
        return ("Cual es tu recomendacion numero uno?" if primera else
                f"Cual es la mejor recomendacion para {uid}?")
    if intent == "compras":
        return "Cuantas compras tengo registradas?" if primera else \
            f"Cuantas compras tiene {uid}?"
    return "Que me recomiendas?" if primera else \
        f"Que le recomiendas al usuario {uid}?"


def plantilla(intent: str, u: dict, uid: str, primera: bool, rec=None) -> str:
    if intent == "perfil":
        return respuesta_perfil(u, uid, not primera)
    if intent == "historial":
        return respuesta_historial(u, uid, not primera)
    if intent == "porque":
        return respuesta_porque(rec or u["recomendaciones"][0], u, uid, not primera)
    if intent == "top1":
        return respuesta_top1(u, uid, not primera)
    if intent == "compras":
        return respuesta_compras(u, uid)
    if intent == "recomendar":
        return respuesta_recomendar(u, uid, not primera)
    return ""


def validar_con_datos(texto: str, nombres_productos: list[str]) -> bool:
    """El GPT debe citar casi todos los productos/categorias esperados."""
    t = limpiar(texto).lower()
    claves = []
    for nombre in nombres_productos:
        palabras = [w for w in limpiar(nombre).lower().split() if len(w) >= 4]
        if palabras:
            claves.append(palabras[0])
    if not claves:
        return len(t) > 30
    aciertos = sum(1 for c in set(claves) if c in t)
    umbral = 1 if len(set(claves)) == 1 else \
        -(-len(set(claves)) * 2 // 3)          # techo de 2/3 de las claves
    return aciertos >= umbral


def responder(mensaje: str, uid_base: str) -> dict:
    inicio = time.time()
    datos = ESTADO["datos"]
    intent = detectar_intencion(mensaje)

    # ---- plan de la respuesta: intencion + usuario + pregunta canonica ----
    rec_objetivo = None
    u_perfil_items: list[tuple[str, str]] = []
    if intent in RESPUESTAS_FIJAS:
        pregunta_canonica, fija = RESPUESTAS_FIJAS[intent]
        uid_final = None
        esperados: list[str] = []
        contexto = "general"
    else:
        uid, explicito = detectar_uid(mensaje, uid_base)
        if uid is None:
            ultimo = sorted(datos["usuarios"])[-1]
            return {"respuesta": f"No encuentro ese usuario. Los usuarios van "
                                 f"de u0000 a {ultimo}; prueba con uno de ellos.",
                    "intencion": intent, "uid": explicito,
                    "fuente": "plantilla",
                    "tiempo_ms": int((time.time() - inicio) * 1000)}
        u = datos["usuarios"][uid]
        uid_final = uid
        u_perfil_items = [(limpiar(str(cat)).lower(), str(n))
                          for cat, n in list(u.get("perfil", {}).items())[:3]]
        primera = bool(re.search(r"\b(mi|me)\b", mensaje.lower())) and not explicito
        recs = u.get("recomendaciones") or []
        if intent == "porque":
            # producto mencionado en el mensaje (si esta en el Top-10 real)
            objetivo = limpiar(mensaje).lower()
            for r in recs:
                clave = " ".join(limpiar(r["producto"]).lower().split()[:3])
                if len(clave) >= 8 and clave[:14] in objetivo:
                    rec_objetivo = r
                    break
            rec_objetivo = rec_objetivo or (recs[0] if recs else None)
        pregunta_canonica = canonica(intent, uid, primera, rec_objetivo)
        contexto = construir_contexto(intent, u, uid, rec_objetivo)
        fija = plantilla(intent, u, uid, primera, rec_objetivo)
        # claves que una respuesta CORRECTA debe mencionar, por intencion
        if intent == "perfil":
            esperados = list(u.get("perfil", {}).keys())[:3]
        elif intent == "compras":
            esperados = [str(u.get("total_compras", ""))]
        elif intent == "historial":
            esperados = [h["nombre"] for h in u.get("historial", [])[:3]]
        elif intent == "porque" and rec_objetivo:
            esperados = [rec_objetivo["producto"]]
        else:
            esperados = [r["producto"] for r in recs[:3]]

    # ------------------------------ cache ---------------------------------
    clave = (intent, pregunta_canonica, contexto)
    if clave in _CACHE:
        return {**_CACHE[clave],
                "cache": True,
                "tiempo_ms": int((time.time() - inicio) * 1000)}

    # ------------- generacion: varios candidatos en paralelo ---------------
    candidatos = generar_respuestas(
        ESTADO["modelo"], ESTADO["itos"], ESTADO["stoi"],
        pregunta_canonica, n_max=240, temperatura=TEMPERATURA, top_k=TOP_K,
        candidatos=3, semilla=int(time.time() * 1000) % 2**31,
        contexto=contexto)

    def aceptable(texto: str) -> bool:
        if not texto or len(texto) <= 30:
            return False
        if uid_final and uid_final in pregunta_canonica and \
                uid_final not in texto:
            return False          # hablo de otro usuario
        if intent in RESPUESTAS_FIJAS:
            # debe solaparse claramente con la respuesta canonica conocida
            ref = {w for w in re.findall(r"[a-z]{5,}", fija.lower())}
            hip = {w for w in re.findall(r"[a-z]{5,}", texto.lower())}
            return len(ref & hip) >= max(3, int(len(ref) * 0.4))
        if intent == "perfil":
            # exige los pares categoria=cantidad tal como estaban en el contexto
            items = list(u_perfil_items)[:3] if u_perfil_items else []
            t_norm = limpiar(texto).lower()
            pares = sum(1 for cat, n in items
                        if f"{cat} con {n}" in t_norm)
            return pares >= len(items)          # copia EXACTA de todo el contexto
        if intent == "compras":
            return f"{uid_final} tiene {esperados[0]}" in limpiar(texto).lower()
        if esperados:
            return validar_con_datos(texto, esperados)
        return True

    final = next((c for c in candidatos if aceptable(c)), "")
    ok = bool(final)
    if not ok:
        final = fija

    resultado = {"respuesta": final, "intencion": intent, "uid": uid_final,
                 "fuente": ("gpt" if uid_final is None else "gpt+hibrido")
                 if ok else ("plantilla" if intent in RESPUESTAS_FIJAS
                             else "hibrido"),
                 "cache": False,
                 "tiempo_ms": int((time.time() - inicio) * 1000)}
    _CACHE[clave] = resultado
    return dict(resultado)


# --------------------------------- servidor ---------------------------------
class Manejador(BaseHTTPRequestHandler):

    def _cabeceras(self, codigo: int = 200, tipo: str = "application/json"):
        self.send_response(codigo)
        self.send_header("Content-Type", f"{tipo}; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, fmt, *args):
        print(f"[chat] {args[0] if args else ''}")

    def do_OPTIONS(self):
        self._cabeceras(204)

    def do_GET(self):
        if self.path.startswith("/api/salud"):
            cuerpo = {"ok": True, "modelo_cargado": ESTADO["modelo"] is not None,
                      "usuarios": len(ESTADO["datos"]["usuarios"])
                      if ESTADO["datos"] else 0}
            self._cabeceras()
            self.wfile.write(json.dumps(cuerpo).encode())
            return
        self._cabeceras()
        self.wfile.write(json.dumps({
            "servicio": "asistente-recomendador",
            "endpoints": ["GET /api/salud", "POST /api/chat"],
        }).encode())

    def do_POST(self):
        if not self.path.startswith("/api/chat"):
            self._cabeceras(404)
            self.wfile.write(b"{}")
            return
        try:
            largo = int(self.headers.get("Content-Length", 0))
            peticion = json.loads(self.rfile.read(largo) or b"{}")
            mensaje = str(peticion.get("mensaje", "")).strip()
            uid_base = str(peticion.get("uid", "u0007"))
            if not mensaje:
                raise ValueError("mensaje vacio")
            resultado = responder(mensaje, uid_base)
            self._cabeceras()
            self.wfile.write(json.dumps(resultado, ensure_ascii=False).encode("utf-8"))
        except Exception as exc:                       # noqa: BLE001
            self._cabeceras(500)
            self.wfile.write(json.dumps({"error": str(exc)},
                                        ensure_ascii=False).encode("utf-8"))


def main() -> None:
    print("[1/2] Cargando modelo afinado y recomendaciones del hibrido...")
    ESTADO["modelo"], ESTADO["stoi"], ESTADO["itos"] = cargar_chat()
    ESTADO["datos"] = json.loads(
        ARCH_RECOMENDACIONES.read_text(encoding="utf-8"))
    print(f"      {len(ESTADO['datos']['usuarios'])} usuarios listos")
    print(f"[2/2] Servidor del asistente en http://localhost:{PUERTO}"
          f"  (POST /api/chat)")
    ThreadingHTTPServer(("0.0.0.0", PUERTO), Manejador).serve_forever()


if __name__ == "__main__":
    main()
