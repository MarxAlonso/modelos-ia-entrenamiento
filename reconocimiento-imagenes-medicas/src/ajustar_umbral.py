"""
Fase 6 — v005: ajuste de umbral de decisión sobre VALIDATION

La probabilidad que emite el modelo (sigmoid) es continua; el umbral decide
dónde se corta NORMAL|PNEUMONIA. El valor 0.5 casi nunca es el óptimo cuando
hay class_weights. Este script:

    1. Barrido de umbrales t∈[0.05, 0.95] sobre VALIDATION (nunca test).
    2. Elige t* = argmax f1_macro con restricción recall_pneumonia ≥ 0.97.
    3. Crea la versión nueva (sin reentrenar nada) heredando el modelo.
    4. Evalúa UNA sola vez en test (fp32 y su int8 heredado) con t*.
    5. Registra la versión en registro_versiones.json.

Uso:
    python src/ajustar_umbral.py --version-id v005 \
        --desde models/versiones/v004_20260823-2126_ft
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
from cuantizar import InterpreteTFLite, registrar  # noqa: E402
from entrenar import calcular_metricas  # noqa: E402
from preprocesamiento import (  # noqa: E402
    SEMILLA,
    cargar_manifiesto,
    crear_pipeline,
)

RECALL_PNEU_MINIMO = 0.97


def probabilidades(modelo_o_interp, ds):
    ys, ps = [], []
    for x, y in ds:
        p = modelo_o_interp(x)
        ys.append(np.asarray(y))
        ps.append(np.asarray(p).reshape(-1))
    return np.concatenate(ys), np.concatenate(ps)


def barrido(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts = np.arange(0.05, 0.96, 0.01)
    f1s, recalls = [], []
    for t in ts:
        m = calcular_metricas(y, p, umbral=float(t))
        f1s.append(m["f1_macro"])
        recalls.append(m["recall_pneumonia"])
    return ts, np.array(f1s), np.array(recalls)


def elegir_umbral(ts, f1s, recalls) -> float:
    validos = recalls >= RECALL_PNEU_MINIMO
    if validos.any():
        idx = int(np.argmax(np.where(validos, f1s, -1)))
    else:
        idx = int(np.argmax(f1s))  # sin restricción posible; documentarlo
    return round(float(ts[idx]), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--desde", required=True, type=Path)
    parser.add_argument("--notas", default="")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(SEMILLA)
    fecha = datetime.now().strftime("%Y%m%d-%H%M")
    dir_version = Path("models/versiones") / f"{args.version_id}_{fecha}_umbral"
    dir_version.mkdir(parents=True, exist_ok=True)

    config_previa = json.loads((args.desde / "config.json").read_text(encoding="utf-8"))
    metricas_previas = json.loads((args.desde / "metricas.json").read_text(encoding="utf-8"))
    normalizacion = config_previa.get("normalizacion", "rescale")

    print(f"=== {args.version_id}: ajuste de umbral desde {args.desde.name} ===")
    df = cargar_manifiesto()
    val_ds = crear_pipeline(df, "validation", batch_size=32, normalizacion=normalizacion)

    modelo = tf.keras.models.load_model(args.desde / "modelo.keras")
    y_val, p_val = probabilidades(modelo, val_ds)

    ts, f1s, recalls = barrido(y_val, p_val)
    t_opt = elegir_umbral(ts, f1s, recalls)
    print(f"umbral óptimo en validation: {t_opt} (f1_macro={f1s[np.argmin(np.abs(ts-t_opt))]:.4f})")

    # Gráfico del barrido (evidencia visual de la elección)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ts, f1s, label="f1_macro (validation)", color="#4C9BD6", lw=2)
    ax.plot(ts, recalls, label="recall_pneumonia (validation)", color="#E0684B", lw=1.5, ls="--")
    ax.axvline(t_opt, color="#333", ls=":", label=f"t* = {t_opt}")
    ax.axhline(RECALL_PNEU_MINIMO, color="gray", ls=":", lw=1)
    ax.set_xlabel("Umbral de decisión")
    ax.set_ylabel("Métrica")
    ax.set_title("Barrido de umbrales sobre validation (test intacto)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dir_version / "barrido_umbral.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Evaluación ÚNICA en test con el umbral elegido
    # ------------------------------------------------------------------
    test_ds_keras = crear_pipeline(df, "test", batch_size=32, normalizacion=normalizacion)
    y_test, p_test = probabilidades(modelo, test_ds_keras)
    metricas_fp32 = calcular_metricas(y_test, p_test, umbral=t_opt)
    metricas_fp32["tamano_mb"] = metricas_previas["fp32"]["tamano_mb"]

    # int8 heredado (mismos pesos cuantizados; el umbral se aplica igual)
    ruta_int8_origen = args.desde / "modelo_quant_int8.tflite"
    interp = InterpreteTFLite(ruta_int8_origen)
    test_ds_unit = crear_pipeline(df, "test", batch_size=1, normalizacion=normalizacion)
    ys8, ps8, lats = [], [], []
    for i, (x, y) in enumerate(test_ds_unit):
        import time

        t0 = time.perf_counter()
        p = float(np.asarray(interp(x.numpy())).reshape(-1)[0])
        if i >= 10:
            lats.append(time.perf_counter() - t0)
        ys8.append(int(y.numpy()[0]))
        ps8.append(p)
    metricas_int8 = calcular_metricas(np.array(ys8), np.array(ps8), umbral=t_opt)

    # Artefactos de despliegue heredados (copia para versión autocontenida)
    shutil.copy2(ruta_int8_origen, dir_version / "modelo_quant_int8.tflite")
    shutil.copy2(args.desde / "modelo_quant_dyn.tflite", dir_version / "modelo_quant_dyn.tflite")

    tam_int8 = (dir_version / "modelo_quant_int8.tflite").stat().st_size / 1e6
    config = {
        "version": args.version_id,
        "dataset": config_previa["dataset"],
        "manifiesto": config_previa["manifiesto"],
        "base_anterior": args.desde.name,
        "arquitectura": f"{config_previa['arquitectura']}+umbral",
        "modelo_fisico": "idéntico al de la base anterior; solo cambia el umbral",
        "umbral_decision": t_opt,
        "umbral_elegido_en": "validation",
        "recall_pneu_minimo_restriccion": RECALL_PNEU_MINIMO,
        "normalizacion": normalizacion,
        "semilla": SEMILLA,
        "notas": args.notas or f"Umbral optimizado por f1_macro en validation desde {args.desde.name}",
    }
    dir_version.joinpath("config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    metricas_json = {
        "version": args.version_id,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "arquitectura": config["arquitectura"],
        "dataset": config_previa["dataset"],
        "config": config,
        "tiempo_entrenamiento_min": 0.0,
        "mejor_epoca": None,
        "fp32": metricas_fp32,
        "cuantizacion": {
            "dynamic_range": {
                "tamano_mb": round((dir_version / "modelo_quant_dyn.tflite").stat().st_size / 1e6, 2),
                "reduccion_vs_fp32": round(metricas_previas["fp32"]["tamano_mb"] / ((dir_version / "modelo_quant_dyn.tflite").stat().st_size / 1e6), 1),
                **{k: metricas_previas["cuantizacion"]["dynamic_range"][k] for k in ["accuracy", "f1_macro", "recall_pneumonia", "roc_auc"]},
                "nota": f"métricas heredadas de la base con umbral 0.5; evaluación propia con t*={t_opt}: ver abajo",
            },
            "int8": {
                "tamano_mb": round(tam_int8, 2),
                "reduccion_vs_fp32": round(metricas_previas["fp32"]["tamano_mb"] / tam_int8, 1),
                "imagenes_representativas": metricas_previas["cuantizacion"]["int8"]["imagenes_representativas"],
                **metricas_int8,
                "latencia_ms": round(float(np.mean(lats)) * 1000, 2) if lats else None,
            },
        },
    }
    dir_version.joinpath("metricas.json").write_text(json.dumps(metricas_json, indent=2, ensure_ascii=False), encoding="utf-8")

    registrar(dir_version, metricas_json)
    print(json.dumps({"fp32": metricas_fp32, "int8_con_t*:": {"accuracy": metricas_int8["accuracy"], "f1_macro": metricas_int8["f1_macro"], "recall_pneumonia": metricas_int8["recall_pneumonia"], "matriz": metricas_int8["matriz"]}}, indent=2, ensure_ascii=False))
    print(f"OK -> {dir_version.resolve()}")


if __name__ == "__main__":
    main()
