"""
Dataset v3 — Exploración RSNA Pneumonia Detection

Analiza etiquetas_crudas.csv (26 684 pacientes) y una muestra aleatoria de
imágenes para producir:

    data/reportes/v3_exploracion/
    ├── distribucion_v3.png        # target y clase detallada
    └── resumen.json

Uso:
    python src/explorar_dataset_v3.py
"""
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

DATASET = Path("data/versiones/v3_rsna")
SALIDA = Path("data/reportes/v3_exploracion")
MUESTRA_IMAGENES = 2000
SEMILLA = 42


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET / "etiquetas_crudas.csv")
    print(f"pacientes: {len(df)}")

    conteo_target = df.target.value_counts().to_dict()
    conteo_clase = df.clase_detalle.value_counts().to_dict()
    cajas_por_positivo = df[df.target == 1].n_cajas.value_counts().sort_index().to_dict()

    # Integridad física muestreada
    rng = random.Random(SEMILLA)
    muestra = rng.sample(range(len(df)), MUESTRA_IMAGENES)
    danadas, resoluciones, modos = 0, [], Counter()
    for i in muestra:
        fila = df.iloc[i]
        try:
            with Image.open(DATASET / "images" / f"{fila.patientId}.png") as im:
                resoluciones.append(im.size)
                modos[im.mode] += 1
        except Exception:  # noqa: BLE001
            danadas += 1
    anchos = [w for w, _ in resoluciones]
    altos = [h for _, h in resoluciones]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(["NORMAL (0)", "PNEUMONIA (1)"], [conteo_target.get(0, 0), conteo_target.get(1, 0)],
                color=["#4C9BD6", "#E0684B"])
    for i, v in enumerate([conteo_target.get(0, 0), conteo_target.get(1, 0)]):
        axes[0].text(i, v + 200, str(v), ha="center", fontsize=10)
    axes[0].set_title("Target binario")
    ejes_clase = list(conteo_clase.items())
    axes[1].barh([k for k, _ in reversed(ejes_clase)], [v for _, v in reversed(ejes_clase)], color="#4C9BD6")
    axes[1].set_title("Clase detallada RSNA")
    fig.tight_layout()
    fig.savefig(SALIDA / "distribucion_v3.png", dpi=150)
    plt.close(fig)

    ratio = round(conteo_target.get(0, 0) / max(1, conteo_target.get(1, 0)), 2)
    resumen = {
        "dataset": "v3_rsna",
        "origen": "kellly/RSNA-Pneumonia-Detection (espejo del challenge oficial Kaggle/RSNA)",
        "fuente_original": "RSNA Pneumonia Detection Challenge — etiquetas por radiólogos",
        "total_pacientes": int(len(df)),
        "target": {"NORMAL": conteo_target.get(0, 0), "PNEUMONIA": conteo_target.get(1, 0)},
        "desbalance_normal_pneu": ratio,
        "clase_detalle": conteo_clase,
        "cajas_por_imagen_positiva": cajas_por_positivo,
        "integridad_muestra": {
            "muestreadas": MUESTRA_IMAGENES,
            "danadas": danadas,
            "resolucion": {"min_ancho": min(anchos), "max_ancho": max(anchos),
                            "min_alto": min(altos), "max_alto": max(altos)},
            "modos_color": dict(modos),
        },
        "implicaciones_entrenamiento": [
            f"Desbalance {ratio}:1 -> class weights + oversampling disponibles en entrenar_transfer",
            "1 imagen = 1 paciente: el manifiesto parte por paciente y la fuga es imposible por construccion",
            "Cajas conservadas en manifiesto para localizacion futura",
            "Etiqueta 'Lung Opacity' de RSNA = opacidad confirmada por radiologo (mas limpia que NIH)",
        ],
    }
    SALIDA.joinpath("resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"\nOK -> reportes en {SALIDA.resolve()}")


if __name__ == "__main__":
    main()
