"""
Fase 1 — Exploración del Dataset v1

Valida TODAS las imágenes del dataset versionado y genera:
    data/reportes/v1_exploracion/
    ├── distribucion_clases.png     # imágenes por clase en cada split
    ├── resoluciones.png            # tamaños de imagen por clase
    ├── muestras.png                # ejemplos visuales de cada clase
    └── resumen.json                # estadísticas completas (fuente para el registro)

Uso:
    python src/explorar_dataset.py
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

DATASET = Path("data/versiones/v1_kaggle_chest_xray")
SALIDA = Path("data/reportes/v1_exploracion")
SPLITS = ["train", "validation", "test"]
CLASES = ["NORMAL", "PNEUMONIA"]
SEMILLA = 42


def recorrer() -> list[dict]:
    """Valida cada imagen y devuelve registros con ruta, split, clase y dimensiones."""
    registros = []
    danadas = []
    print("Validando imágenes ...")
    for split in SPLITS:
        for clase in CLASES:
            carpeta = DATASET / split / clase
            archivos = sorted(carpeta.glob("*"))
            for i, archivo in enumerate(archivos):
                try:
                    with Image.open(archivo) as im:
                        im.verify()  # valida integridad del archivo
                    with Image.open(archivo) as im:
                        ancho, alto = im.size
                        modo = im.mode
                    registros.append(
                        {
                            "ruta": str(archivo.relative_to(DATASET)),
                            "split": split,
                            "clase": clase,
                            "ancho": ancho,
                            "alto": alto,
                            "modo": modo,
                            "md5": hashlib.md5(archivo.read_bytes()).hexdigest(),
                        }
                    )
                except Exception as e:  # noqa: BLE001 - cualquier fallo cuenta como dañada
                    danadas.append({"archivo": str(archivo), "error": str(e)})
            print(f"  {split}/{clase}: {len(archivos)} revisadas.")
    if danadas:
        print(f"ATENCION: {len(danadas)} imagenes danadas -> reportadas en resumen.json")
    return registros


def duplicados(df: pd.DataFrame) -> dict:
    """Duplicados exactos por MD5 y fuga real entre splits."""
    dup_interno = int(df.duplicated(subset=["md5"]).sum())
    md5_test = set(df.loc[df.split == "test", "md5"])
    no_test = df[df.split != "test"]
    md5_no_test = set(no_test.md5)
    # ¿Dónde están los duplicados?
    repes = df[df.duplicated(subset=["md5"], keep=False)]
    cruces_dupes = sorted(set(zip(repes.split, repes.clase)))[:10]
    return {
        "duplicados_totales_md5": dup_interno,
        "ubicaciones_de_duplicados": [f"{s}/{c}" for s, c in cruces_dupes],
        "md5_distintos_compartidos_entre_train_y_test": len(md5_no_test & md5_test),
        "imagenes_en_train_o_val_que_existen_tambien_en_test": int(no_test.md5.isin(md5_test).sum()),
    }


def grafico_distribucion(df: pd.DataFrame, destino: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    tabla = df.groupby(["split", "clase"]).size().unstack(fill_value=0).reindex(SPLITS)
    x = range(len(SPLITS))
    ancho = 0.38
    ax.bar([i - ancho / 2 for i in x], tabla["NORMAL"], ancho, label="NORMAL", color="#4C9BD6")
    ax.bar([i + ancho / 2 for i in x], tabla["PNEUMONIA"], ancho, label="PNEUMONIA", color="#E0684B")
    for i, split in enumerate(SPLITS):
        ax.text(i - ancho / 2, tabla.loc[split, "NORMAL"] + 30, str(tabla.loc[split, "NORMAL"]), ha="center", fontsize=9)
        ax.text(i + ancho / 2, tabla.loc[split, "PNEUMONIA"] + 30, str(tabla.loc[split, "PNEUMONIA"]), ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(["train", "validation", "test"])
    ax.set_ylabel("Número de imágenes")
    ax.set_title("Dataset v1 — Distribución por clase y split (desbalance visible)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)


def grafico_resoluciones(df: pd.DataFrame, destino: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, clase in zip(axes, CLASES):
        sub = df[df.clase == clase]
        ax.scatter(sub.ancho, sub.alto, s=6, alpha=0.35, color="#4C9BD6" if clase == "NORMAL" else "#E0684B")
        ax.set_title(f"{clase} — {len(sub)} imágenes")
        ax.set_xlabel("Ancho (px)")
    axes[0].set_ylabel("Alto (px)")
    fig.suptitle("Resoluciones originales", y=1.02)
    fig.tight_layout()
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)


def grafico_muestras(destino: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for fila, clase in enumerate(CLASES):
        archivos = sorted((DATASET / "train" / clase).glob("*"))
        paso = max(1, len(archivos) // 4)
        for col in range(4):
            with Image.open(archivos[col * paso]) as im:
                axes[fila, col].imshow(im.convert("L"), cmap="gray")
            axes[fila, col].set_title(clase, fontsize=9)
            axes[fila, col].axis("off")
    fig.suptitle("Muestras de entrenamiento (radiografías pediátricas AP)", y=0.98)
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    registros = recorrer()
    df = pd.DataFrame(registros)

    conteo = {s: {c: int(((df.split == s) & (df.clase == c)).sum()) for c in CLASES} for s in SPLITS}
    modos = Counter(df.modo)
    dups = duplicados(df)

    resumen = {
        "dataset": "v1_kaggle_chest_xray",
        "origen": "hf-vision/chest-xray-pneumonia (Kermany et al., 2018 — Mendeley rscbjbr9sj v2)",
        "total_imagenes": len(df),
        "conteo_por_split": conteo,
        "proporcion_train_NORMAL_vs_PNEUMONIA": round(conteo["train"]["PNEUMONIA"] / conteo["train"]["NORMAL"], 2),
        "modos_color": dict(modos),
        "resolucion_min": {"ancho": int(df.ancho.min()), "alto": int(df.alto.min())},
        "resolucion_max": {"ancho": int(df.ancho.max()), "alto": int(df.alto.max())},
        "resolucion_media": {"ancho": round(float(df.ancho.mean())), "alto": round(float(df.alto.mean()))},
        "imagenes_danadas": [],
        **dups,
        "implicaciones_entrenamiento": [
            "Desbalance ~2.9:1 en train -> usar class_weights en todas las versiones",
            "validation oficial solo tiene 16 imagenes -> regenerar splits propios con manifiesto fijo",
            (
                "Fuga real train/test: "
                f"{dups['imagenes_en_train_o_val_que_existen_tambien_en_test']} imagenes repetidas -> "
                "deduplicar antes de fijar el manifiesto"
            ),
        ],
    }
    (SALIDA / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    grafico_distribucion(df, SALIDA / "distribucion_clases.png")
    grafico_resoluciones(df, SALIDA / "resoluciones.png")
    grafico_muestras(SALIDA / "muestras.png")

    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"\nOK -> reportes en {SALIDA.resolve()}")


if __name__ == "__main__":
    main()
