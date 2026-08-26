"""
Dataset v3 — Manifiesto patient-wise (RSNA no trae split oficial)

1 imagen = 1 paciente en RSNA, así que particionar por paciente es directo.
Split ESTRATIFICADO por target con semilla fija:

    train 80% · validation 10% · test 10%

Salida:
    data/manifiestos/v3_splits.csv
      ruta_relativa, patientId, clase, target, n_cajas, cajas,
      clase_detalle, split_final

Uso:
    python src/generar_manifiesto_v3.py
"""
from pathlib import Path

import pandas as pd

DATASET = Path("data/versiones/v3_rsna")
DESTINO = Path("data/manifiestos/v3_splits.csv")
SEMILLA = 42
FRACCIONES = {"train": 0.80, "validation": 0.10, "test": 0.10}


def main() -> None:
    df = pd.read_csv(DATASET / "etiquetas_crudas.csv")
    df["clase"] = df["target"].map({0: "NORMAL", 1: "PNEUMONIA"})
    df["ruta_relativa"] = "images/" + df["patientId"] + ".png"

    # Split estratificado por clase, por paciente (1 imagen por paciente)
    partes = []
    for clase, grupo in df.groupby("clase"):
        barajado = grupo.sample(frac=1.0, random_state=SEMILLA)
        n_val = int(len(barajado) * FRACCIONES["validation"])
        n_test = int(len(barajado) * FRACCIONES["test"])
        partes.append(asignar(barajado.iloc[:n_test], "test"))
        partes.append(asignar(barajado.iloc[n_test:n_test + n_val], "validation"))
        partes.append(asignar(barajado.iloc[n_test + n_val:], "train"))

    final = pd.concat(partes)[
        ["ruta_relativa", "patientId", "clase", "target", "n_cajas", "cajas",
         "clase_detalle", "split_final"]
    ].sort_values(["split_final", "clase"]).reset_index(drop=True)

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(DESTINO, index=False)

    conteo = final.groupby(["split_final", "clase"]).size().unstack(fill_value=0)
    tr = final[final.split_final == "train"]
    pesos = {c: round(len(tr) / (2 * (tr.clase == c).sum()), 3) for c in ["NORMAL", "PNEUMONIA"]}
    print(f"Manifiesto -> {DESTINO.resolve()}")
    print(conteo)
    print(f"class_weights train: {pesos}")


def asignar(grupo: pd.DataFrame, nombre: str) -> pd.DataFrame:
    out = grupo.copy()
    out["split_final"] = nombre
    return out


if __name__ == "__main__":
    main()
