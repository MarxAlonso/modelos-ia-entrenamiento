"""
Demo — Servidor de inferencia para la interfaz Astro (/demo)

Recibe una radiografía + dominio, responde clase, probabilidad, latencia y el
overlay Grad-CAM. Reutiliza EXACTAMENTE los artefactos campeones del registro:

    v1 pediátrico -> v003   |  v2 NIH adulto -> v008   |  v3 RSNA adulto -> v009

Uso:
    python src/servidor_demo.py            # http://127.0.0.1:8000
    (en otra terminal) cd viz && pnpm dev  # http://localhost:4321/demo

Herramienta académica: NO usar para diagnóstico real.
"""
import base64
import io
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from cuantizar import InterpreteTFLite
from gradcam import construir_modelo_cam, gradcam
from preprocesamiento import IMG_SIZE, NORMALIZADORES, _decodificar

RAIZ = Path(__file__).parent.parent

CAMPEONES = {
    "v1": {
        "etiqueta": "Pediátrico (Kermany)",
        "carpeta": RAIZ / "models/versiones/v003_20260823-2049_mobilenetv2_fe",
    },
    "v2": {
        "etiqueta": "Adulto NIH",
        "carpeta": RAIZ / "models/versiones/v008_20260825-1626_mobilenetv2_fe",
    },
    "v3": {
        "etiqueta": "Adulto RSNA",
        "carpeta": RAIZ / "models/versiones/v009_20260825-1959_mobilenetv2_fe",
    },
}

app = FastAPI(title="Rayos-X Lab · demo de inferencia", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321", "http://127.0.0.1:4321"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache: dict[str, dict] = {}


def cargar_dominio(dominio: str) -> dict:
    """Carga perezosa por dominio: intérprete TFLite + modelos Keras para Grad-CAM."""
    if dominio not in CAMPEONES:
        raise HTTPException(400, f"dominio inválido: {dominio}")
    if dominio in _cache:
        return _cache[dominio]

    carpeta = CAMPEONES[dominio]["carpeta"]
    tflite = carpeta / "modelo_quant_int8.tflite"
    keras = carpeta / "modelo.keras"
    if not tflite.exists() or not keras.exists():
        raise HTTPException(500, f"artefactos no encontrados en {carpeta.name}")

    config = (carpeta / "config.json").read_text(encoding="utf-8")
    import json

    config = json.loads(config)
    print(f"[demo] cargando dominio {dominio} ({carpeta.name}) ...")
    cam_base, head = construir_modelo_cam(tf.keras.models.load_model(keras))
    _cache[dominio] = {
        "interp": InterpreteTFLite(tflite),
        "cam_base": cam_base,
        "head": head,
        "normalizacion": config.get("normalizacion", "rescale"),
        "recorte": int(config.get("recorte_borde_px", 0)),
        "umbral": float(config.get("umbral_decision", 0.5)),
    }
    return _cache[dominio]


def _png_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _original_base64(x01: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(x01[:, :, 0], cmap="gray")
    ax.axis("off")
    fig.subplots_adjust(pad=0)
    return _png_base64(fig)


def _gradcam_base64(cam_base, head, x_norm: np.ndarray, x01: np.ndarray) -> tuple[str, float]:
    mapa, prob_keras = gradcam(cam_base, head, tf.constant(x_norm)[0])
    calor = tf.image.resize(mapa[..., None], IMG_SIZE, method="bilinear").numpy()[..., 0]
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(x01[:, :, 0], cmap="gray")
    ax.imshow(calor, cmap="jet", alpha=0.4)
    ax.axis("off")
    fig.subplots_adjust(pad=0)
    return _png_base64(fig), prob_keras


@app.get("/api/dominios")
def dominios():
    return [
        {"id": k, "etiqueta": v["etiqueta"], "version": v["carpeta"].name}
        for k, v in CAMPEONES.items()
    ]


@app.post("/api/predict")
async def predict(imagen: UploadFile = File(...), dominio: str = Form("v1")):
    modelo = cargar_dominio(dominio)

    contenido = await imagen.read()
    if len(contenido) > 15 * 1024 * 1024:
        raise HTTPException(413, "imagen demasiado grande (>15 MB)")
    temporal = Path("data") / "_demo_temp"
    temporal.mkdir(parents=True, exist_ok=True)
    ruta = temporal / f"upload_{int(time.time()*1000)}{Path(imagen.filename or 'x.jpg').suffix or '.jpg'}"
    ruta.write_bytes(contenido)

    try:
        x01 = _decodificar(tf.constant(str(ruta)), modelo["recorte"]).numpy()   # (224,224,3)
        x01 = NORMALIZADORES[modelo["normalizacion"]](x01)
        x_lote = x01[None, ...]                                                  # (1,224,224,3)

        t0 = time.perf_counter()
        prob_tflite = float(np.asarray(modelo["interp"](x_lote)).reshape(-1)[0])
        latencia = round((time.perf_counter() - t0) * 1000, 1)

        overlay_b64, prob_keras = _gradcam_base64(
            modelo["cam_base"], modelo["head"], x_lote, x01
        )
    finally:
        ruta.unlink(missing_ok=True)

    umbral = modelo["umbral"]
    p = prob_tflite
    clase = "PNEUMONIA" if p >= umbral else "NORMAL"

    return {
        "dominio": dominio,
        "clase": clase,
        "probabilidad": round(p * 100, 2),
        "probabilidad_gradcam_keras": round(prob_keras * 100, 2),
        "umbral": umbral,
        "latencia_ms": latencia,
        "original_png": _original_base64(x01),
        "gradcam_png": overlay_b64,
        "aviso": "herramienta académica · no constituye un diagnóstico médico",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
