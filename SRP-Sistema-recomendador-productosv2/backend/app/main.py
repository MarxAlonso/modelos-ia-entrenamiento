"""
API del Sistema Recomendador de Productos v2: Transformers (two-tower + SASRec, en ONNX) + Laya.

    cd backend
    ../.venv/Scripts/python -m uvicorn app.main:app --port 8000
    Documentacion interactiva: http://localhost:8000/docs
"""
import os
import time
from typing import Optional

from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import laya_cliente
from .recomendador import Recomendador

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELOS_DIR = os.environ.get("MODELOS_DIR", os.path.join(RAIZ, "models", "servir"))

app = FastAPI(title="SRP v2 - Transformers + Laya",
              description="Two-tower Transformer para recuperar, SASRec para re-rankear, Laya para entender texto.")
rec = Recomendador(MODELOS_DIR)


def _precalentar():
    """Calcula en segundo plano lo que la demo pide al inicio (escenarios y clientes con
    acierto, ~40 s), para que en la exposicion responda al instante."""
    rec.escenarios()
    rec.clientes_con_acierto()


import threading  # noqa: E402
threading.Thread(target=_precalentar, daemon=True).start()
# Frontend de demo: http://localhost:8000/app
app.mount("/app", StaticFiles(directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "static"),
                              html=True), name="app")


@app.get("/", include_in_schema=False)
def inicio():
    return RedirectResponse("/app/")


class Consulta(BaseModel):
    texto: str = Field(..., min_length=2, examples=["busco unos aretes de plata para regalarle a mi mamá"])
    k: int = Field(10, ge=1, le=50)
    user_id: Optional[str] = Field(None, description="si el cliente inicio sesion, se mezcla su historial")
    historial_ids: Optional[List[str]] = Field(None, description="historial armado a mano (reemplaza al de user_id)")
    usar_laya: bool = True


class Historial(BaseModel):
    product_ids: List[str] = Field(..., description="compras del cliente, de la mas antigua a la mas reciente")
    k: int = Field(10, ge=1, le=50)
    categoria: Optional[str] = None
    publico: Optional[str] = None


class Resena(BaseModel):
    texto: str = Field(..., min_length=3, examples=["La talla vino muy pequeña y la tela se siente barata."])
    producto: Optional[str] = None


def _ms(t0):
    return round((time.perf_counter() - t0) * 1000, 1)


@app.get("/salud")
def salud():
    return {"modelo": rec.version, "productos": len(rec.catalogo), "usuarios": len(rec.usuarios),
            "laya": laya_cliente.estado()}


@app.get("/modelo")
def modelo():
    return rec.manifiesto


@app.get("/escenarios")
def escenarios():
    """Clientes reales elegidos para la demo (uno por categoria + uno variado)."""
    return rec.escenarios()


@app.get("/metricas")
def metricas():
    """Todo lo medido, para la pestana de resultados: two-tower, personalizacion y Laya."""
    import glob
    import json as _json
    m = rec.manifiesto["metricas"]
    sasrec_dir = os.path.join(RAIZ, "models", "sasrec", rec.manifiesto["sasrec"])
    tt_dir = os.path.join(RAIZ, "models", "two_tower", rec.manifiesto["two_tower"])
    leer = lambda ruta: _json.load(open(ruta, encoding="utf-8")) if os.path.exists(ruta) else None
    laya = {os.path.basename(r)[8:-5]: leer(r) for r in glob.glob(os.path.join(RAIZ, "data", "laya", "reporte_*.json"))}
    tt = leer(os.path.join(tt_dir, "metricas.json")) or {}
    return {"version": rec.version, "two_tower_test": m.get("two_tower_test"),
            "two_tower_val": tt.get("historial_val"),
            "comparacion": m.get("comparacion_usuarios_eval"),
            "catalogo_completo": leer(os.path.join(sasrec_dir, "metricas_catalogo_completo.json")),
            "laya": laya, "laya_afinado": laya_cliente.estado()["afinado"]}


@app.get("/usuarios/ejemplo")
def usuarios_ejemplo(n: int = Query(5, ge=1, le=50), minimo: int = Query(3, ge=1, le=50),
                     semilla: Optional[int] = None):
    return rec.usuarios_ejemplo(n, minimo, semilla)


@app.get("/productos/buscar")
def buscar_productos(q: str = Query(..., min_length=2), k: int = Query(10, ge=1, le=50)):
    return rec.buscar(q, k)


@app.post("/recomendar/historial")
def recomendar_historial(h: Historial):
    """Recomienda para un historial armado a mano (simula un cliente nuevo que va comprando)."""
    desconocidos = [p for p in h.product_ids if p not in rec.idx_producto]
    if desconocidos:
        raise HTTPException(404, f"productos no encontrados: {desconocidos[:5]}")
    t0 = time.perf_counter()
    r = rec.para_historial([rec.idx_producto[p] for p in h.product_ids], h.k, h.categoria, h.publico)
    return {**r, "ms": _ms(t0)}


@app.get("/usuarios/aciertos")
def usuarios_aciertos():
    """Clientes reales donde la prueba acierta + con que frecuencia pasa (se calcula una vez)."""
    return rec.clientes_con_acierto()


@app.get("/metricas/densos")
def metricas_densos(minimo: int = Query(5, ge=2, le=20), k: int = Query(12, ge=1, le=50),
                    revisar: int = Query(300, ge=1, le=2000), semilla: int = Query(7)):
    """Tasa de acierto solo en clientes con >= `minimo` compras (cohorte densa)."""
    t0 = time.perf_counter()
    return {**rec.acierto_densos(minimo, k, revisar, semilla), "ms": _ms(t0)}


@app.get("/usuarios/{user_id}/perfil")
def perfil_usuario(user_id: str):
    p = rec.perfil_usuario(user_id)
    if p is None:
        raise HTTPException(404, "cliente no encontrado (o sin compras)")
    return p


@app.get("/usuarios/{user_id}/prueba")
def prueba_usuario(user_id: str, k: int = Query(12, ge=1, le=50)):
    """Oculta la ultima compra real y mira si el sistema la habria recomendado."""
    t0 = time.perf_counter()
    r = rec.prueba(user_id, k)
    if r is None:
        raise HTTPException(404, "hace falta un cliente con al menos 2 compras")
    return {**r, "ms": _ms(t0)}


@app.get("/recomendar/usuario/{user_id}")
def recomendar_usuario(user_id: str, k: int = Query(10, ge=1, le=50),
                       categoria: Optional[str] = None, publico: Optional[str] = None):
    t0 = time.perf_counter()
    return {**rec.para_usuario(user_id, k, categoria, publico), "ms": _ms(t0)}


@app.post("/recomendar/consulta")
async def recomendar_consulta(c: Consulta):
    """Lenguaje natural -> Laya decide categoria/publico/regalo/precio -> el two-tower recupera
    productos parecidos a la consulta -> SASRec re-rankea si el usuario tiene historial."""
    t0 = time.perf_counter()
    decision, respuestas, aviso = {}, None, None
    if c.usar_laya:
        try:
            r = await laya_cliente.preguntar_consulta(c.texto)
            decision = laya_cliente.interpretar_consulta(r)
            respuestas = {q: {kk: v for kk, v in a.items()
                              if kk in ("choice", "score", "noul", "answer_confidence", "probabilities")}
                          for q, a in r["answers"].items()}
        except Exception as e:  # sin Laya igual se recomienda, solo sin filtros
            aviso = f"Laya no disponible ({type(e).__name__}: {e}); se recomienda solo con el two-tower."
    t_laya = _ms(t0)
    t1 = time.perf_counter()
    hist = [rec.idx_producto[p] for p in c.historial_ids if p in rec.idx_producto] if c.historial_ids else None
    r = rec.para_consulta(c.texto, c.k, c.user_id, hist=hist, **decision)
    return {"decision_laya": decision or None, "respuestas_laya": respuestas, "aviso": aviso, **r,
            "ms": {"laya": t_laya, "recomendador": _ms(t1)}}


@app.get("/productos/{product_id}/similares")
def similares(product_id: str, k: int = Query(10, ge=1, le=50)):
    r = rec.similares(product_id, k)
    if r is None:
        raise HTTPException(404, "producto no encontrado")
    return r


@app.post("/resenas/analizar")
async def analizar_resena(r: Resena):
    """Laya (idealmente el checkpoint afinado con las resenas del proyecto) convierte una
    resena sin estrellas en satisfaccion estimada y probabilidad de recomendar."""
    try:
        resp = await laya_cliente.preguntar_resena({"producto": r.producto or "", "resena": r.texto})
    except Exception as e:
        raise HTTPException(503, f"Laya no disponible: {type(e).__name__}: {e}")
    a = resp["answers"]
    return {"rating_estimado": round(1 + float(a["satisfaccion"]["score"]), 2),
            "recomendaria": round(float(a["recomendaria"]["noul"]), 4),
            "checkpoint": laya_cliente.CHECKPOINT_RESENAS, "detalle": a}
