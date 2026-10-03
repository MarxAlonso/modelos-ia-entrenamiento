"""
Demo — Servidor de inferencia para la interfaz Astro (/demo)

Recibe una radiografía, responde clase (NORMAL | PNEUMONIA), probabilidad,
latencia y el overlay Grad-CAM.

Modelo por defecto: el CAMPEÓN UNIVERSAL promovido en models/campeon/, un único
artefacto TensorFlow entrenado sobre la unión de los tres dominios del proyecto
(pediátrico + NIH adulto + RSNA adulto). Esa es la diferencia importante: el
usuario que sube una placa NO tiene por qué saber de qué hospital viene, así que
el modelo que la analiza tampoco puede exigírselo.

Los tres especialistas por dominio siguen disponibles como segunda opinión:

    v1 pediátrico -> v003   |  v2 NIH adulto -> v008   |  v3 RSNA adulto -> v009

El campeón universal se actualiza solo: src/promover_campeon.py reescribe
models/campeon/ y este servidor lo recarga sin tocar una línea de código.

Uso:
    python src/servidor_demo.py            # http://127.0.0.1:8000
    (en otra terminal) cd viz && pnpm dev  # http://localhost:4321/demo

Herramienta académica: NO usar para diagnóstico real.
"""
import base64
import io
import json
import threading
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
MAX_BYTES = 15 * 1024 * 1024

CAMPEONES = {
    "auto": {
        "etiqueta": "Universal",
        "desc": "cualquier radiografía de tórax",
        "carpeta": RAIZ / "models/campeon",
        "universal": True,
    },
    "v1": {
        "etiqueta": "Pediátrico (Kermany)",
        "desc": "especialista en radiografías de niños",
        "carpeta": RAIZ / "models/versiones/v003_20260823-2049_mobilenetv2_fe",
        "universal": False,
    },
    "v2": {
        "etiqueta": "Adulto NIH",
        "desc": "especialista estilo NIH ChestX-ray14",
        "carpeta": RAIZ / "models/versiones/v008_20260825-1626_mobilenetv2_fe",
        "universal": False,
    },
    "v3": {
        "etiqueta": "Adulto RSNA",
        "desc": "especialista estilo RSNA",
        "carpeta": RAIZ / "models/versiones/v009_20260825-1959_mobilenetv2_fe",
        "universal": False,
    },
}
DOMINIO_POR_DEFECTO = "auto"

app = FastAPI(title="Rayos-X Lab · demo de inferencia", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321", "http://127.0.0.1:4321"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache: dict[str, dict] = {}
# Ni el intérprete TFLite ni la carga del caché son seguros entre hilos, y
# /api/predict corre en el pool de hilos de FastAPI: se serializan.
_candado = threading.Lock()


def _firma(carpeta: Path) -> float:
    """Marca de tiempo del .tflite: si promueves un campeón nuevo, cambia y el caché se invalida."""
    return (carpeta / "modelo_quant_int8.tflite").stat().st_mtime


def cargar_dominio(dominio: str) -> dict:
    """Carga perezosa por dominio: intérprete TFLite + modelos Keras para Grad-CAM."""
    if dominio not in CAMPEONES:
        raise HTTPException(400, f"dominio inválido: {dominio}")

    carpeta = CAMPEONES[dominio]["carpeta"]
    tflite = carpeta / "modelo_quant_int8.tflite"
    keras = carpeta / "modelo.keras"
    if not tflite.exists() or not keras.exists():
        if CAMPEONES[dominio]["universal"]:
            raise HTTPException(
                503,
                "todavía no hay campeón universal promovido. Ejecuta: "
                "python src/promover_campeon.py",
            )
        raise HTTPException(500, f"artefactos no encontrados en {carpeta.name}")

    firma = _firma(carpeta)
    if dominio in _cache and _cache[dominio]["firma"] == firma:
        return _cache[dominio]

    config = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
    print(f"[demo] cargando dominio {dominio} ({carpeta.name}, versión {config.get('version')}) ...")
    cam_base, head = construir_modelo_cam(tf.keras.models.load_model(keras))
    _cache[dominio] = {
        "firma": firma,
        "interp": InterpreteTFLite(tflite),
        "cam_base": cam_base,
        "head": head,
        "version": config.get("version", "?"),
        "arquitectura": config.get("arquitectura", "?"),
        "normalizacion": config.get("normalizacion", "rescale"),
        "recorte": int(config.get("recorte_borde_px", 0)),
        "umbral": float(config.get("umbral_decision", 0.5)),
    }
    return _cache[dominio]


# ---------------------------------------------------------------------------
# Entrada: cualquier formato razonable acaba siendo un PNG en escala de grises
# ---------------------------------------------------------------------------
def normalizar_subida(contenido: bytes, nombre: str, destino: Path) -> dict:
    """
    El pipeline de entrenamiento lee ficheros del disco, así que la subida se
    materializa en un PNG con el MISMO aspecto que las imágenes de train.

    Acepta lo que llega de un navegador (jpg/png/webp/bmp/tif) y también DICOM,
    que es el formato real en el que salen las placas de un equipo de rayos X.
    """
    sufijo = Path(nombre or "").suffix.lower()
    aviso = None

    if sufijo in {".dcm", ".dicom"} or contenido[128:132] == b"DICM":
        import pydicom

        ds = pydicom.dcmread(io.BytesIO(contenido), force=True)
        pixeles = ds.pixel_array.astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixeles = pixeles.max() - pixeles  # MONOCHROME1 viene invertido
        rango = float(pixeles.max() - pixeles.min())
        pixeles = (pixeles - pixeles.min()) / rango * 255.0 if rango > 0 else pixeles * 0
        img = Image.fromarray(pixeles.astype(np.uint8), mode="L")
        formato = "DICOM"
    else:
        try:
            img = Image.open(io.BytesIO(contenido))
            img.load()
        except Exception:
            raise HTTPException(400, "no pude leer la imagen: ¿es un archivo de imagen válido?")
        formato = img.format or sufijo.lstrip(".").upper() or "?"
        # Una radiografía es monocroma. Si llega muy saturada, casi seguro que
        # NO es una placa: se avisa, pero no se bloquea (puede ser un pseudocolor).
        if img.mode in {"RGB", "RGBA", "P"}:
            rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
            saturacion = float(np.mean(rgb.max(axis=2) - rgb.min(axis=2)))
            if saturacion > 18:
                aviso = ("la imagen tiene mucho color: este modelo solo ha visto radiografías "
                         "de tórax en escala de grises, así que su respuesta no significa nada aquí")
        img = img.convert("L")

    ancho, alto = img.size
    img.save(destino, format="PNG")
    return {"formato": formato, "resolucion": f"{ancho}x{alto}", "aviso_entrada": aviso}


def _png_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _original_base64(x01: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(x01[:, :, 0], cmap="gray")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return _png_base64(fig)


def _gradcam_base64(cam_base, head, x_norm: np.ndarray, x01: np.ndarray) -> tuple[str, float]:
    mapa, prob_keras = gradcam(cam_base, head, tf.constant(x_norm)[0])
    calor = tf.image.resize(mapa[..., None], IMG_SIZE, method="bilinear").numpy()[..., 0]
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(x01[:, :, 0], cmap="gray")
    ax.imshow(calor, cmap="jet", alpha=0.4)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return _png_base64(fig), prob_keras


@app.get("/api/dominios")
def dominios():
    """Lo consume la UI para pintar el selector: el universal va primero."""
    salida = []
    for clave, datos in CAMPEONES.items():
        carpeta = datos["carpeta"]
        disponible = (carpeta / "modelo_quant_int8.tflite").exists() and (carpeta / "modelo.keras").exists()
        version = None
        if disponible:
            try:
                version = json.loads((carpeta / "config.json").read_text(encoding="utf-8")).get("version")
            except Exception:
                version = None
        salida.append({
            "id": clave,
            "etiqueta": datos["etiqueta"],
            "desc": datos["desc"],
            "universal": datos["universal"],
            "disponible": disponible,
            "version": version,
            "carpeta": carpeta.name,
            "por_defecto": clave == DOMINIO_POR_DEFECTO,
        })
    return salida


@app.get("/api/campeon")
def campeon():
    """Ficha del campeón universal promovido (la escribe src/promover_campeon.py)."""
    puntero = CAMPEONES["auto"]["carpeta"] / "campeon.json"
    if not puntero.exists():
        raise HTTPException(404, "aún no se ha promovido ningún campeón universal")
    ficha = json.loads(puntero.read_text(encoding="utf-8"))
    ficha.pop("historial", None)
    return ficha


@app.post("/api/predict")
def predict(imagen: UploadFile = File(...), dominio: str = Form(DOMINIO_POR_DEFECTO)):
    # def (no async): TensorFlow bloquea, y así FastAPI lo ejecuta en su pool de
    # hilos en vez de congelar el event loop para el resto de peticiones.
    with _candado:
        modelo = cargar_dominio(dominio)

    contenido = imagen.file.read()
    if len(contenido) > MAX_BYTES:
        raise HTTPException(413, "imagen demasiado grande (>15 MB)")
    if not contenido:
        raise HTTPException(400, "el archivo llegó vacío")

    temporal = RAIZ / "data" / "_demo_temp"
    temporal.mkdir(parents=True, exist_ok=True)
    ruta = temporal / f"upload_{int(time.time() * 1000)}.png"

    try:
        entrada = normalizar_subida(contenido, imagen.filename or "", ruta)

        x01 = _decodificar(tf.constant(str(ruta)), modelo["recorte"]).numpy()   # (224,224,3)
        x01 = NORMALIZADORES[modelo["normalizacion"]](x01)
        x_lote = x01[None, ...]                                                  # (1,224,224,3)

        with _candado:
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
        "modelo": {
            "version": modelo["version"],
            "arquitectura": modelo["arquitectura"],
            "universal": CAMPEONES[dominio]["universal"],
        },
        "entrada": entrada,
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
