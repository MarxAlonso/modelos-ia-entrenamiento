"""
Fase 1 — Manifiesto de splits fijos para Dataset v1

Decisión de diseño (ver docs/04-fase-1-dataset.md):
- TEST   = el test oficial del dataset, intacto (624 imágenes, sin fuga demostrada).
           Se congela para que TODAS las versiones del modelo sean comparables.
- TRAIN/VAL = se regeneran desde el train oficial:
           * se eliminan duplicados MD5 internos,
           * split estratificado 90/10 con semilla fija (42).

Salida:
    data/manifiestos/v1_splits.csv   -> ruta_relativa, clase, split_oficial, split_final

Uso:
    python src/generar_manifiesto.py
"""
import hashlib
from pathlib import Path

import pandas as pd

DATASET = Path("data/versiones/v1_kaggle_chest_xray")
DESTINO = Path("data/manifiestos/v1_splits.csv")
SPLITS = ["train", "validation", "test"]
CLASES = ["NORMAL", "PNEUMONIA"]
SEMILLA = 42
FRACCION_VAL = 0.10


def md5(ruta: Path) -> str:
    return hashlib.md5(ruta.read_bytes()).hexdigest()


def main() -> None:
    filas = []
    for split in SPLITS:
        for clase in CLASES:
            for archivo in sorted((DATASET / split / clase).glob("*")):
                filas.append(
                    {
                        "ruta_relativa": archivo.relative_to(DATASET).as_posix(),
                        "clase": clase,
                        "split_oficial": split,
                        "md5": md5(archivo),
                    }
                )
    df = pd.DataFrame(filas)

    # 1) Test oficial intacto
    test = df[df.split_oficial == "test"].copy()
    test["split_final"] = "test"

    # 2) Pool de entrenamiento sin duplicados internos
    pool = (
        df[df.split_oficial == "train"]
        .drop_duplicates(subset="md5", keep="first")
        .copy()
    )
    quitados = len(df[df.split_oficial == "train"]) - len(pool)

    # 3) Validación estratificada 10%
    val_indices = (
        pool.groupby("clase", group_keys=False)
        .apply(lambda g: g.sample(frac=FRACCION_VAL, random_state=SEMILLA))
        .index
    )
    pool["split_final"] = "train"
    pool.loc[val_indices, "split_final"] = "validation"

    final = pd.concat([pool.drop(columns="md5"), test.drop(columns="md5")])
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(DESTINO, index=False)

    # Resumen + class weights para el entrenamiento
    conteo = final.groupby(["split_final", "clase"]).size().unstack(fill_value=0)
    n_train = conteo.loc["train"]
    pesos = {c: round(len(final[final.split_final == "train"]) / (2 * n_train[c]), 3) for c in CLASES}

    print(f"Manifiesto guardado en {DESTINO.resolve()}")
    print(f"Duplicados eliminados del pool de train: {quitados}")
    print(conteo)
    print(f"class_weights sugeridos: {pesos}")


if __name__ == "__main__":
    main()
