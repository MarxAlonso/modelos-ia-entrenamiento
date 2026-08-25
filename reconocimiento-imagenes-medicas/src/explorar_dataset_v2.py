"""
Dataset v2 — Exploración NIH ChestX-ray14

Analiza etiquetas_crudas.csv (112 120 filas) y una MUESTRA aleatoria de imágenes
(3000, por costo de cómputo; se documenta la decisión) para producir:

    data/reportes/v2_exploracion/
    ├── distribucion_hallazgos.png     # frecuencia de cada patología NIH
    ├── mapeo_binario.png              # NORMAL / PNEUMONIA / EXCLUIDA tras el mapeo
    └── resumen.json                   # estadísticas completas

Decisiones que este script INFORMA (se aplican en generar_manifiesto_v2.py):
    NORMAL     <- labels == "No Finding"           (estricto)
    PNEUMONIA  <- "Pneumonia" presente en labels   (aunque coexista con otras)
    EXCLUIDA   <- todo lo demás (hallazgos sin neumonía)

Uso:
    python src/explorar_dataset_v2.py
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

DATASET = Path("data/versiones/v2_nih_chest_xray14")
SALIDA = Path("data/reportes/v2_exploracion")
MUESTRA_IMAGENES = 3000
SEMILLA = 42


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET / "etiquetas_crudas.csv")
    print(f"filas totales: {len(df)}")

    # ------------------------------------------------------------------
    # 1) Hallazgos individuales y combinaciones
    # ------------------------------------------------------------------
    df["lista_hallazgos"] = df["labels"].str.split("|")
    frecuencia = Counter(h for lista in df["lista_hallazgos"] for h in lista)
    combinaciones = Counter(df["labels"])

    # ------------------------------------------------------------------
    # 2) Mapeo binario propuesto
    # ------------------------------------------------------------------
    def clase_binaria(labels: str) -> str:
        if labels == "No Finding":
            return "NORMAL"
        if "Pneumonia" in labels.split("|"):
            return "PNEUMONIA"
        return "EXCLUIDA"

    df["clase_binaria"] = df["labels"].map(clase_binaria)
    conteo_binario = df.groupby(["split", "clase_binaria"]).size().unstack(fill_value=0)

    # ------------------------------------------------------------------
    # 3) Pacientes: ¿el split oficial es patient-wise?
    # ------------------------------------------------------------------
    df["paciente_id"] = df["filename"].str.slice(0, 8)
    imgs_por_paciente = df.groupby("paciente_id").size()
    cruce = {}
    for split in ["train", "validation", "test"]:
        otros = set(df[df.split != split]["paciente_id"])
        propios = set(df[df.split == split]["paciente_id"])
        cruce[split] = len(propios & otros)

    # ------------------------------------------------------------------
    # 4) Integridad física en muestra aleatoria
    # ------------------------------------------------------------------
    rng = random.Random(SEMILLA)
    muestra_idx = rng.sample(range(len(df)), MUESTRA_IMAGENES)
    danadas, resoluciones, modos = [], [], Counter()
    for i in muestra_idx:
        fila = df.iloc[i]
        ruta = DATASET / "images" / fila["split"] / fila["filename"]
        try:
            with Image.open(ruta) as im:
                im.verify()
            with Image.open(ruta) as im:
                resoluciones.append(im.size)
                modos[im.mode] += 1
        except Exception as e:  # noqa: BLE001
            danadas.append({"archivo": str(ruta), "error": str(e)})
    anchos = [w for w, _ in resoluciones]
    altos = [h for _, h in resoluciones]

    # ------------------------------------------------------------------
    # Gráficos
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    hallazgos_ordenados = frecuencia.most_common()
    ax.barh([h for h, _ in reversed(hallazgos_ordenados)], [n for _, n in reversed(hallazgos_ordenados)], color="#4C9BD6")
    ax.set_title("NIH ChestX-ray14 — frecuencia de hallazgos (una imagen puede tener varios)")
    ax.set_xlabel("Número de imágenes")
    fig.tight_layout()
    fig.savefig(SALIDA / "distribucion_hallazgos.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    conteo_binario.plot(kind="bar", ax=ax, color=["#999", "#E0684B", "#4C9BD6"])
    ax.set_title("Mapeo a tarea binaria: NORMAL vs PNEUMONIA vs EXCLUIDA")
    ax.set_ylabel("Imágenes")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(SALIDA / "mapeo_binario.png", dpi=150)
    plt.close(fig)

    resumen = {
        "dataset": "v2_nih_chest_xray14",
        "origen": f"{REPO}" if (REPO := "arudaev/chest-xray-14-320") else None,
        "fuente_original": "NIH Clinical Center ChestX-ray14 (Wang et al., CVPR 2017)",
        "total_imagenes": int(len(df)),
        "conteo_por_split": {s: int((df.split == s).sum()) for s in ["train", "validation", "test"]},
        "frecuencia_hallazgos": dict(hallazgos_ordenados),
        "top10_combinaciones": {k: int(v) for k, v in combinaciones.most_common(10)},
        "mapeo_binario": {str(k): {c: int(v) for c, v in fila.items()} for k, fila in conteo_binario.iterrows()},
        "pacientes": {
            "unicos": int(imgs_por_paciente.index.nunique()),
            "imagenes_por_paciente_media": round(float(imgs_por_paciente.mean()), 2),
            "max": int(imgs_por_paciente.max()),
            "pacientes_compartidos_entre_splits": cruce,
        },
        "integridad_muestra": {
            "muestreadas": MUESTRA_IMAGENES,
            "danadas": len(danadas),
            "resolucion": {
                "min_ancho": min(anchos), "max_ancho": max(anchos),
                "min_alto": min(altos), "max_alto": max(altos),
            },
            "modos_color": dict(modos),
        },
        "implicaciones_entrenamiento": [
            "Tarea binaria estricta: solo No Finding->NORMAL y *Pneumonia*->PNEUMONIA",
            f"Se excluyen {int(conteo_binario['EXCLUIDA'].sum())} imágenes con otros hallazgos (documentado)",
            "Split oficial ya es patient-wise: verificar cruce==0 arriba antes de confiar",
            "Dominio ADULTO: no mezclar ingenuamente con v1 pediatrico sin evaluar shift",
        ],
    }
    SALIDA.joinpath("resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: resumen[k] for k in ["total_imagenes", "mapeo_binario", "pacientes", "integridad_muestra"]}, indent=2, ensure_ascii=False))
    print(f"\nOK -> reportes en {SALIDA.resolve()}")


if __name__ == "__main__":
    main()
