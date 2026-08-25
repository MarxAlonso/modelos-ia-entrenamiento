"""
Fase 7 — Inferencia local con el modelo cuantizado (.tflite)

Carga un modelo_quant_int8.tflite de cualquier versión, aplica EXACTAMENTE el
preprocesamiento declarado en el config.json de esa versión (normalización +
recorte de bordes) y predice sobre una imagen suelta:

    Clase predicha: PNEUMONIA
    Probabilidad:   87.42%

También mide latencia promedio en CPU local (--benchmark N).

Uso:
    python src/predecir_tflite.py \
        --modelo models/versiones/v003_20260823-2049_mobilenetv2_fe/modelo_quant_int8.tflite \
        --imagen data/versiones/v1_kaggle_chest_xray/test/PNEUMONIA/person100_bacteria_3082.jpeg

    python src/predecir_tflite.py --modelo ... --imagen ... --benchmark 30
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from cuantizar import InterpreteTFLite
from preprocesamiento import IMG_SIZE, NORMALIZADORES, _decodificar, MAPA_CLASES


def preparar_imagen(ruta: Path, normalizacion: str, recorte_borde: int) -> np.ndarray:
    """Mismo contrato que el pipeline de entrenamiento, para UNA imagen."""
    x = _decodificar(tf.constant(str(ruta)), recorte_borde)
    x = NORMALIZADORES[normalizacion](x)
    return x[None, ...].numpy()  # (1,224,224,3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", required=True, type=Path,
                        help="ruta al .tflite (se lee config.json de la carpeta contenedora)")
    parser.add_argument("--imagen", required=True, type=Path)
    parser.add_argument("--benchmark", type=int, default=0,
                        help="repeticiones extra para latencia promedio")
    args = parser.parse_args()

    dir_version = args.modelo.parent
    config = json.loads((dir_version / "config.json").read_text(encoding="utf-8"))
    normalizacion = config.get("normalizacion", "rescale")
    recorte = int(config.get("recorte_borde_px", 0))

    if not args.imagen.exists():
        raise SystemExit(f"ERROR: no existe {args.imagen}")

    print(f"=== Inferencia TFLite · versión {config.get('version')} ===")
    print(f"modelo        : {args.modelo.name}")
    print(f"preproceso    : {normalizacion}" + (f" + recorte {recorte}px" if recorte else ""))

    interp = InterpreteTFLite(args.modelo)
    entrada = preparar_imagen(args.imagen, normalizacion, recorte)

    # warmup (primeras invocaciones cargan kernels)
    for _ in range(3):
        p = float(np.asarray(interp(entrada)).reshape(-1)[0])

    t0 = time.perf_counter()
    p = float(np.asarray(interp(entrada)).reshape(-1)[0])
    latencia_una = (time.perf_counter() - t0) * 1000

    clase = "PNEUMONIA" if p >= 0.5 else "NORMAL"
    umbral = float(config.get("umbral_decision", 0.5))

    print()
    print(f"Clase predicha: {clase}")
    print(f"Probabilidad:   {p*100:.2f}%")
    print(f"(umbral de decisión de esta versión: {umbral})")

    if args.benchmark > 0:
        tiempos = []
        for _ in range(args.benchmark):
            t = time.perf_counter()
            interp(entrada)
            tiempos.append((time.perf_counter() - t) * 1000)
        print()
        print(f"Latencia promedio CPU ({args.benchmark} repeticiones): "
              f"{np.mean(tiempos):.2f} ms ± {np.std(tiempos):.2f} ms")


if __name__ == "__main__":
    main()
