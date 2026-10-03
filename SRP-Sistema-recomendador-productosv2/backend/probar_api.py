"""
Demo de la API (con el backend corriendo en http://localhost:8000):

    ..\\.venv\\Scripts\\python probar_api.py
    ..\\.venv\\Scripts\\python probar_api.py "quiero una cartera de cuero negra"
"""
import json
import sys

import httpx

API = "http://localhost:8000"
CONSULTAS = sys.argv[1:] or [
    "busco unos aretes de plata para regalarle a mi mamá",
    "zapatillas para correr de hombre, algo barato",
    "un vestido elegante para una boda en verano",
]
RESENAS = ["La talla vino muy pequeña y la tela se siente barata, no lo volvería a comprar.",
           "Me encantó, la calidad es excelente y llegó antes de lo esperado."]


def mostrar(recs):
    for r in recs:
        print(f"   - [{r['categoria']}/{r['publico']}] {r['titulo'][:90]}")


with httpx.Client(base_url=API, timeout=300) as c:
    print(json.dumps(c.get("/salud").json(), ensure_ascii=False))
    for texto in CONSULTAS:
        r = c.post("/recomendar/consulta", json={"texto": texto, "k": 5}).json()
        print(f"\n>> {texto}\n   Laya: {r['decision_laya']}  ({r['ms']['laya']} ms Laya, {r['ms']['recomendador']} ms recomendador)")
        if r.get("aviso"):
            print("   aviso:", r["aviso"])
        mostrar(r["recomendaciones"])
    usuario = c.get("/usuarios/ejemplo", params={"n": 1}).json()[0]
    r = c.get(f"/recomendar/usuario/{usuario['user_id']}", params={"k": 5}).json()
    print(f"\n>> usuario {usuario['user_id']} (compro: {[t[:40] for t in r.get('historial', [])]})\n   {r['estrategia']}")
    mostrar(r["recomendaciones"])
    for texto in RESENAS:
        r = c.post("/resenas/analizar", json={"texto": texto}).json()
        print(f"\n>> resena: {texto}\n   rating estimado={r['rating_estimado']} recomendaria={r['recomendaria']} ({r['checkpoint']})")
