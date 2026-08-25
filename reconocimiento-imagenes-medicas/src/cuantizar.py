"""
Fase 3 — Cuantización post-training y registro de versión

Convierte el modelo.keras maestro de una versión a dos formatos TFLite ligeros,
los evalúa en el MISMO test congelado, mide latencia en CPU local y actualiza:

    <version>/metricas.json          -> añade bloque "cuantizacion"
    models/registro_versiones.json   -> índice central del dashboard Astro

Formatos generados:
    modelo_quant_dyn.tflite   dynamic range  (pesos int8, activaciones float)
    modelo_quant_int8.tflite  full int8      (pesos+activaciones int8, IO float32,
                                              calibrado con 200 imágenes representativas)

Uso:
    python src/cuantizar.py --version-dir models/versiones/v001_20260823-1856_cnn
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

from entrenar import calcular_metricas, guardar_matriz
from preprocesamiento import cargar_manifiesto, crear_pipeline

REGISTRO = Path("models/registro_versiones.json")
N_REPRESENTATIVAS = 200


def _representative_dataset(df, n: int, normalizacion: str):
    """Generador de muestras reales para calibrar el rango de activaciones int8."""
    ds = crear_pipeline(df, "train", batch_size=1, augment=False, normalizacion=normalizacion)
    for x, _ in ds.take(n):
        yield [x.numpy()]


def convertir(dir_version: Path, modelo: tf.keras.Model, normalizacion: str, dataset: str) -> dict:
    print("Convirtiendo a TFLite ...")

    # Dynamic range: solo los pesos pasan a int8. Rápido y sin datos.
    conv_dyn = tf.lite.TFLiteConverter.from_keras_model(modelo)
    conv_dyn.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_dyn = conv_dyn.convert()

    # Full int8: requiere dataset representativo para calibrar escalas.
    df = cargar_manifiesto(dataset=dataset)
    conv_int8 = tf.lite.TFLiteConverter.from_keras_model(modelo)
    conv_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    conv_int8.representative_dataset = lambda: _representative_dataset(df, N_REPRESENTATIVAS, normalizacion)
    conv_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv_int8.inference_input_type = tf.float32   # misma interfaz que el keras -> eval idéntica
    conv_int8.inference_output_type = tf.float32
    tflite_int8 = conv_int8.convert()

    ruta_dyn = dir_version / "modelo_quant_dyn.tflite"
    ruta_int8 = dir_version / "modelo_quant_int8.tflite"
    ruta_dyn.write_bytes(tflite_dyn)
    ruta_int8.write_bytes(tflite_int8)
    print(f"  dyn  : {ruta_dyn.name} ({len(tflite_dyn)/1e6:.2f} MB)")
    print(f"  int8 : {ruta_int8.name} ({len(tflite_int8)/1e6:.2f} MB)")
    return {"dyn": ruta_dyn, "int8": ruta_int8}


class InterpreteTFLite:
    """Envoltorio mínimo para usar un .tflite como si fuera model(x)."""

    def __init__(self, ruta: Path):
        self.interp = tf.lite.Interpreter(model_path=str(ruta), num_threads=4)
        self.interp.allocate_tensors()
        self.input_detail = self.interp.get_input_details()[0]
        self.output_detail = self.interp.get_output_details()[0]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.interp.set_tensor(self.input_detail["index"], x.astype(np.float32))
        self.interp.invoke()
        return self.interp.get_tensor(self.output_detail["index"])


def evaluar_tflite(ruta: Path, df, normalizacion: str) -> tuple[dict, float]:
    """Métricas en el test congelado + latencia media por imagen (ms, CPU local)."""
    interp = InterpreteTFLite(ruta)
    ds = crear_pipeline(df, "test", batch_size=1, augment=False, normalizacion=normalizacion)  # noqa: E501

    ys, ps = [], []
    latencias = []
    i = 0
    for x, y in ds:
        arr = x.numpy()
        t0 = time.perf_counter()
        p = float(np.asarray(interp(arr)).reshape(-1)[0])
        if i >= 10:  # descarta warmup (primeras invocaciones cargan kernels)
            latencias.append(time.perf_counter() - t0)
        ys.append(int(y.numpy()[0]))
        ps.append(p)
        i += 1
    metricas = calcular_metricas(np.array(ys), np.array(ps))
    latencia_ms = round(float(np.mean(latencias)) * 1000, 2)
    return metricas, latencia_ms


def registrar(dir_version: Path, metricas_json: dict) -> None:
    """
    Inserta/actualiza la versión en el registro central.

    REGLA DE LA POLÍTICA (01-politica-versionamiento §4): solo se comparan
    versiones sobre el MISMO dataset y mismo test. Por eso hay un campeón
    POR DATASET (mejor_por_dataset) y los estados se calculan dentro de
    cada dominio, nunca entre dominios distintos.
    """
    registro = json.loads(REGISTRO.read_text(encoding="utf-8")) if REGISTRO.exists() else {
        "actualizada": None,
        "metrica_referencia": "f1_macro_test",
        "mejor_por_dataset": {},
        "versiones": [],
    }

    fp32 = metricas_json["fp32"]
    cuant = metricas_json["cuantizacion"]["int8"]
    entrada = {
        "id": metricas_json["version"],
        "carpeta": dir_version.name,
        "dataset": metricas_json["dataset"],
        "arquitectura": metricas_json["arquitectura"],
        "accuracy": fp32["accuracy"],
        "recall_normal": fp32["recall_normal"],
        "recall_pneumonia": fp32["recall_pneumonia"],
        "precision_macro": round((fp32["precision_pneumonia"] + fp32["precision_normal"]) / 2, 4),
        "recall_macro": round((fp32["recall_pneumonia"] + fp32["recall_normal"]) / 2, 4),
        "f1_macro": fp32["f1_macro"],
        "roc_auc": fp32["roc_auc"],
        "tamano_mb_fp32": fp32["tamano_mb"],
        "cuantizado": {
            "formato": "int8",
            "tamano_mb": cuant["tamano_mb"],
            "accuracy": cuant["accuracy"],
            "f1_macro": cuant["f1_macro"],
            "latencia_ms_cpu": cuant["latencia_ms"],
        },
        "tiempo_entrenamiento_min": metricas_json["tiempo_entrenamiento_min"],
        "estado": "vigente",
    }

    registro["versiones"] = [v for v in registro["versiones"] if v["id"] != entrada["id"]]
    registro["versiones"].append(entrada)

    # Estados y campeones DENTRO de cada dataset
    registro["mejor_por_dataset"] = {}
    por_dataset: dict[str, list] = {}
    for v in registro["versiones"]:
        por_dataset.setdefault(v["dataset"], []).append(v)
    for dataset_id, grupo in por_dataset.items():
        mejor = max(grupo, key=lambda v: (v["f1_macro"], v["recall_pneumonia"]))
        for v in grupo:
            v["estado"] = "vigente" if v["id"] == mejor["id"] else "superada"
        registro["mejor_por_dataset"][dataset_id] = mejor["id"]

    registro["actualizada"] = datetime.now().isoformat(timespec="seconds")
    REGISTRO.parent.mkdir(parents=True, exist_ok=True)
    REGISTRO.write_text(json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Registro actualizado: mejor_por_dataset={registro['mejor_por_dataset']} -> {REGISTRO.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version-dir", required=True, type=Path)
    parser.add_argument("--dataset", default="v1", help="dataset del manifiesto (v1|v2)")
    args = parser.parse_args()
    dir_version = args.version_dir

    print(f"=== Cuantizando {dir_version.name} ===")
    modelo = tf.keras.models.load_model(dir_version / "modelo.keras")
    config = json.loads((dir_version / "config.json").read_text(encoding="utf-8"))
    normalizacion = config.get("normalizacion", "rescale")
    print(f"normalizacion de la version: {normalizacion}")
    rutas = convertir(dir_version, modelo, normalizacion, args.dataset)

    df = cargar_manifiesto(dataset=args.dataset)
    m_dyn, lat_dyn = evaluar_tflite(rutas["dyn"], df, normalizacion)
    print(f"dyn  -> accuracy={m_dyn['accuracy']} f1_macro={m_dyn['f1_macro']} lat={lat_dyn} ms")
    m_int8, lat_int8 = evaluar_tflite(rutas["int8"], df, normalizacion)
    print(f"int8 -> accuracy={m_int8['accuracy']} f1_macro={m_int8['f1_macro']} lat={lat_int8} ms")

    metricas_json = json.loads((dir_version / "metricas.json").read_text(encoding="utf-8"))
    tamano_fp32 = metricas_json["fp32"]["tamano_mb"]
    metricas_json["cuantizacion"] = {
        "dynamic_range": {
            "tamano_mb": round(rutas["dyn"].stat().st_size / 1e6, 2),
            "reduccion_vs_fp32": round(tamano_fp32 / (rutas["dyn"].stat().st_size / 1e6), 1),
            **{k: m_dyn[k] for k in ["accuracy", "f1_macro", "recall_pneumonia", "roc_auc", "matriz"]},
            "latencia_ms": lat_dyn,
        },
        "int8": {
            "tamano_mb": round(rutas["int8"].stat().st_size / 1e6, 2),
            "reduccion_vs_fp32": round(tamano_fp32 / (rutas["int8"].stat().st_size / 1e6), 1),
            "imagenes_representativas": N_REPRESENTATIVAS,
            **{k: m_int8[k] for k in ["accuracy", "f1_macro", "recall_pneumonia", "roc_auc", "matriz"]},
            "latencia_ms": lat_int8,
        },
    }
    (dir_version / "metricas.json").write_text(json.dumps(metricas_json, indent=2, ensure_ascii=False), encoding="utf-8")

    guardar_matriz(m_int8, dir_version / "matriz_confusion_int8.png", f"{metricas_json['version']} int8 (test)")
    registrar(dir_version, metricas_json)
    print(json.dumps(metricas_json["cuantizacion"], indent=2, ensure_ascii=False))
    print("OK")


if __name__ == "__main__":
    main()
