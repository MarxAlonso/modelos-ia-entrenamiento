"""Pruebas reales contra el backend en http://localhost:8000. Escribe la salida en archivos .txt."""
import json, sys, os
import httpx

OUT = sys.argv[1]
c = httpx.Client(base_url="http://localhost:8000", timeout=300)

def arreglar(t):
    try:
        return t.encode("cp1252").decode("utf-8")
    except Exception:
        return t

def corto(t, n=66):
    t = arreglar(t)
    return t if len(t) <= n else t[:n - 1] + "…"

def guardar(nombre, lineas):
    open(os.path.join(OUT, nombre), "w", encoding="utf-8").write("\n".join(lineas))
    print("\n".join(lineas)); print("=" * 40)

# A. búsqueda en lenguaje natural
L = ["PS> python pruebas\\prueba_sistema.py --consultas"]
s = c.get("/salud").json()
L.append(f"GET /salud -> 200 OK  modelo={s['modelo']}  productos={s['productos']:,}  usuarios={s['usuarios']:,}")
for q in ["busco unos aretes de plata para regalarle a mi mamá", "reloj deportivo resistente al agua",
          "lentes de sol para la playa"]:
    r = c.post("/recomendar/consulta", json={"texto": q, "k": 4}).json()
    dec = r["decision_laya"] or {}
    L.append("")
    L.append(f">> POST /recomendar/consulta  \"{q}\"")
    L.append(f"   Laya: categoria={dec.get('categoria')}  publico={dec.get('publico')}   "
             f"({r['ms']['recomendador']} ms recomendador)")
    for i, x in enumerate(r["recomendaciones"], 1):
        L.append(f"   {i}. [{x['categoria']}/{x['publico']}] {corto(x['titulo'], 70)}")
guardar("prueba_consultas.txt", L)

# B. prueba retrospectiva: ocultar la última compra
L = ["PS> python pruebas\\prueba_sistema.py --ocultar-ultima-compra"]
a = c.get("/usuarios/aciertos").json()
L.append(f"GET /usuarios/aciertos -> {a['revisados']} clientes revisados, {a['aciertos']} con la compra oculta "
         f"en el top-{a['k']} ({a['tasa']*100:.1f} % de 71 214 productos)")
for cli in a["clientes"]:
    if cli["puesto"] <= 3:
        r = c.get(f"/usuarios/{cli['user_id']}/prueba", params={"k": 12}).json()
        L.append("")
        L.append(f">> GET /usuarios/{cli['user_id'][:12]}…/prueba   ({cli['nombre']}, {len(r['historial_usado'])} compras previas)")
        for h in r["historial_usado"]:
            L.append(f"   compró:  {corto(h['titulo'], 74)}")
        L.append(f"   oculta:  {corto(r['oculta']['titulo'], 74)}")
        L.append(f"   resultado: puesto #{r['puesto']} de {r['catalogo']:,}  acertó top-{r['k']}={r['acerto_top_k']}  ({r['ms']} ms)")
    if len(L) > 14:
        break
guardar("prueba_oculta.txt", L)

# C. reseñas
L = ["PS> python pruebas\\prueba_sistema.py --resenas"]
for t in ["La talla vino muy pequeña y la tela se siente barata, no lo volvería a comprar.",
          "Me encantó, la calidad es excelente y llegó antes de lo esperado.",
          "Está bien por el precio, aunque el color es distinto al de la foto.",
          "Beautiful watch, works perfectly and looks expensive. Love it!"]:
    r = c.post("/resenas/analizar", json={"texto": t}).json()
    L.append("")
    L.append(f">> POST /resenas/analizar  \"{t}\"")
    L.append(f"   estrellas = {r['estrellas']}  (confianza {r['confianza']:.2f})   "
             f"P(recomendaría) = {r['recomendaria']:.3f}   modelo = {r['modelo']}")
guardar("prueba_resenas.txt", L)
print(json.dumps(r, ensure_ascii=False)[:600])

# D. entrada / salida reales para la diapositiva de la Tarea
uid = "AGSTDL3I5TMKBYCEXDKCEC43ETCQ"
r = c.get(f"/recomendar/usuario/{uid}", params={"k": 3}).json()
salida = {
    "user_id": r["user_id"][:14] + "…",
    "historial": [corto(h["titulo"], 38) for h in r["historial"]],
    "recomendaciones": [{"titulo": corto(x["titulo"], 34), "categoria": x["categoria"],
                         "puntaje": round(x.get("puntaje", 0), 3)} for x in r["recomendaciones"]],
    "confianza_top10": r["traza"]["confianza_top10"],
    "ms": r["ms"],
}
open(os.path.join(OUT, "salida_usuario.json"), "w", encoding="utf-8").write(json.dumps(salida, ensure_ascii=False, indent=2))
print(json.dumps(salida, ensure_ascii=False, indent=2))
print(json.dumps(r["recomendaciones"][0], ensure_ascii=False)[:800])
