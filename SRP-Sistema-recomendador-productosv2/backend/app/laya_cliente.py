"""
Laya dentro del backend (libreria `laya`, mismo proceso).

Laya responde preguntas tipadas sobre un texto en una sola pasada, sin generar texto:
  choice -> elige una opcion   score -> nivel en una escala   noul -> probabilidad de "si"

Checkpoint: models/laya_srp/ (afinado en Kaggle con notebooks/laya_finetune_srp_kaggle.ipynb
sobre consultas y resenas del proyecto). Si todavia no existe, se usa el publico multilingue
zero-shot, y entonces NO se aplican es_regalo ni sensibilidad_precio: zero-shot acerto 0.37 y
0.33 (data/laya/reporte_laya-multilingual_consultas_compra.json), peor que no usarlos.
Variables de entorno: LAYA_CONSULTAS, LAYA_RESENAS, LAYA_DISPOSITIVO (cpu/cuda), LAYA_UMBRAL.
"""
import asyncio
import json
import os
import threading

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = "convaiinnovations/laya-multilingual"
AFINADO_LOCAL = os.path.join(RAIZ, "models", "laya_srp")
_POR_DEFECTO = AFINADO_LOCAL if os.path.isdir(AFINADO_LOCAL) else BASE
CHECKPOINT_CONSULTAS = os.environ.get("LAYA_CONSULTAS", _POR_DEFECTO)
CHECKPOINT_RESENAS = os.environ.get("LAYA_RESENAS", _POR_DEFECTO)
DISPOSITIVO = os.environ.get("LAYA_DISPOSITIVO") or None  # None = laya elige (cuda si hay)
UMBRAL_CONFIANZA = float(os.environ.get("LAYA_UMBRAL", "0.35"))

# Fuente unica de las preguntas: preguntas_laya.json (tambien la lee
# src/laya_ft/preparar_dataset.py, asi el checkpoint afinado ve exactamente estas
# instrucciones). Las claves de "categoria" y "publico" coinciden con catalogo.csv.
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "preguntas_laya.json"), encoding="utf-8") as _f:
    _PREGUNTAS = json.load(_f)
PREGUNTAS_CONSULTA = _PREGUNTAS["consulta"]
PREGUNTAS_RESENA = _PREGUNTAS["resena"]

_agentes, _candado = {}, threading.Lock()


def _agente(checkpoint):
    with _candado:  # carga perezosa: el primer pedido descarga/carga el checkpoint
        if checkpoint not in _agentes:
            import laya
            _agentes[checkpoint] = laya.load(checkpoint, device=DISPOSITIVO)
        return _agentes[checkpoint]


def _predecir(checkpoint, state, preguntas):
    return _agente(checkpoint).predict(state, preguntas)


async def preguntar_consulta(texto):
    return await asyncio.to_thread(_predecir, CHECKPOINT_CONSULTAS, texto, PREGUNTAS_CONSULTA)


async def preguntar_resena(state):
    return await asyncio.to_thread(_predecir, CHECKPOINT_RESENAS, state, PREGUNTAS_RESENA)


def estado():
    return {"consultas": CHECKPOINT_CONSULTAS, "resenas": CHECKPOINT_RESENAS,
            "afinado": CHECKPOINT_CONSULTAS != BASE, "cargados": list(_agentes)}


def interpretar_consulta(respuesta):
    """Respuestas de Laya -> filtros. Un filtro solo se aplica si la confianza alcanza el umbral."""
    a = respuesta["answers"]
    cat, pub = a["categoria"], a["publico"]
    afinado = CHECKPOINT_CONSULTAS != BASE
    return {
        "categoria": cat["choice"] if cat["choice"] != "otros" and cat["answer_confidence"] >= UMBRAL_CONFIANZA else None,
        "publico": pub["choice"] if pub["choice"] != "unisex" and pub["answer_confidence"] >= UMBRAL_CONFIANZA else None,
        "es_regalo": afinado and a["es_regalo"]["noul"] >= 0.5,
        "sensibilidad_precio": float(a["sensibilidad_precio"]["score"]) / 2.0 if afinado else 0.0,  # 0..1
    }
