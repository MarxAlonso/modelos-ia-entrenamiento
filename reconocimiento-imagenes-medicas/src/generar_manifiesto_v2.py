"""
Dataset v2 — Manifiesto fijo con decisiones de diseño documentadas

Hallazgos de explorar_dataset_v2.py que este script resuelve:
  1. Neumonía es rara: 876 positivos en train+val oficial vs 50 500 NORMAL.
  2. La validation oficial comparte 4 571 pacientes con train (fuga).

Decisiones implementadas (todas registradas en el propio CSV y en docs):
  A. TEST   = test oficial INTACTO (limpio: 0 pacientes compartidos).
  B. VALIDATION se regenera POR PACIENTE desde el pool train+val oficial
     (8% de los pacientes), evitando la fuga detectada.
  C. SUBMUESTREO de NORMAL en train a razón 1:3 respecto a PNEUMONIA
     (laptop: época comparable a v1). Las filas excluidas quedan marcadas
     con incluir=False para auditoría.
  D. class_weights se calculan sobre el train FINAL incluido.

Salida:
    data/manifiestos/v2_splits.csv          (estándar, test y val congelados)
    data/manifiestos/v2_splits_duros.csv    (--negativos-duros N: añade N hallazgos
        visualmente similares [Infiltration|Consolidation|Lung Opacity] como
        negativos duros SOLO en train/validation; el TEST nunca cambia)

Uso:
    python src/generar_manifiesto_v2.py
    python src/generar_manifiesto_v2.py --negativos-duros 3000 --salida v2_splits_duros.csv
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd

DATASET = Path("data/versiones/v2_nih_chest_xray14")
DESTINO = Path("data/manifiestos/v2_splits.csv")
SEMILLA = 42
FRACCION_VAL = 0.08          # % de pacientes del pool para validation
RATIO_NORMAL_POR_PNEU = 3    # submuestreo NORMAL en train


def clase_binaria(labels: str) -> str:
    if labels == "No Finding":
        return "NORMAL"
    if "Pneumonia" in labels.split("|"):
        return "PNEUMONIA"
    return "EXCLUIDA"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negativos-duros", type=int, default=0,
                        help="añade N hallazgos tipo-infiltración como negativos en train/val")
    parser.add_argument("--salida", default="v2_splits.csv")
    args = parser.parse_args()

    df = pd.read_csv(DATASET / "etiquetas_crudas.csv")
    df["clase_binaria"] = df["labels"].map(clase_binaria)
    df["paciente_id"] = df["filename"].str.slice(0, 8)
    df["ruta_relativa"] = "images/" + df["split"] + "/" + df["filename"]
    df["incluir"] = True

    # ------------------------------------------------------------------
    # TEST oficial intacto (mapeado a binario; EXCLUIDA fuera)
    # ------------------------------------------------------------------
    test = df[(df.split == "test") & (df.clase_binaria != "EXCLUIDA")].copy()
    test["split_final"] = "test"

    # ------------------------------------------------------------------
    # Pool de entrenamiento: train + validation oficial (mismos pacientes
    # según exploración), solo filas mapeables a binario
    # ------------------------------------------------------------------
    pool = df[
        (df.split.isin(["train", "validation"]))
        & (df.clase_binaria != "EXCLUIDA")
    ].copy()
    n_excluidas = int((df.clase_binaria == "EXCLUIDA").sum())

    # VALIDATION por PACIENTE (no por imagen): estratificada por presencia de neumonía
    rng = pd.core.common.random_state(SEMILLA)
    pacientes_pneu = sorted(
        pool.loc[pool.clase_binaria == "PNEUMONIA", "paciente_id"].unique()
    )
    pacientes_norm = sorted(
        pool.loc[pool.clase_binaria == "NORMAL", "paciente_id"].unique()
    )
    val_pneu = set(
        pd.Series(pacientes_pneu).sample(frac=FRACCION_VAL, random_state=rng)
    )
    val_norm = set(
        pd.Series(pacientes_norm).sample(frac=FRACCION_VAL, random_state=rng)
    )
    es_val = pool.paciente_id.isin(val_pneu | val_norm)
    pool["split_final"] = "train"
    pool.loc[es_val, "split_final"] = "validation"

    # ------------------------------------------------------------------
    # Submuestreo de NORMAL SOLO en train (validation queda completa)
    # ------------------------------------------------------------------
    train_pneu = pool[(pool.split_final == "train") & (pool.clase_binaria == "PNEUMONIA")]
    train_norm_todas = pool[(pool.split_final == "train") & (pool.clase_binaria == "NORMAL")]
    objetivo_normal = min(len(train_norm_todas), RATIO_NORMAL_POR_PNEU * len(train_pneu))
    norm_muestra_idx = train_norm_todas.sample(n=objetivo_normal, random_state=rng).index
    excluir_norm = train_norm_todas.index.difference(norm_muestra_idx)
    pool.loc[excluir_norm, "incluir"] = False

    # ------------------------------------------------------------------
    # Negativos duros (opcional): hallazgos visualmente parecidos a
    # neumonía, SOLO de train/val oficial (el test se queda congelado).
    # Se etiquetan como NORMAL (no-neumonía) y quedan marcados con
    # rol="negativo_duro" para auditoría.
    # ------------------------------------------------------------------
    pool["rol"] = "estandar"
    if args.negativos_duros > 0:
        MIMICOS = ("Infiltration", "Consolidation", "Lung Opacity")
        es_mimico = (
            (df.clase_binaria == "EXCLUIDA")
            & (df.split.isin(["train", "validation"]))
            & df.labels.apply(lambda s: any(m in s.split("|") for m in MIMICOS))
        )
        duros = df[es_mimico].sample(n=min(args.negativos_duros, int(es_mimico.sum())), random_state=rng).copy()
        duros["clase_binaria"] = "NORMAL"
        duros["rol"] = "negativo_duro"
        duros["split_final"] = "train"
        duros.loc[duros.paciente_id.isin(val_pneu | val_norm), "split_final"] = "validation"
        duros["incluir"] = True
        pool = pd.concat([pool, duros])
        print(f"negativos duros añadidos: {len(duros)}")

    final = pd.concat([pool, test])[
        ["ruta_relativa", "filename", "paciente_id", "labels",
         "clase_binaria", "split", "split_final", "incluir", "rol"]
    ].rename(columns={"labels": "labels_nih", "split": "split_oficial"})
    final = final.sort_values(["split_final", "clase_binaria"]).reset_index(drop=True)
    destino = DESTINO.parent / args.salida
    final.to_csv(destino, index=False)

    # ------------------------------------------------------------------
    # Resumen + verificaciones anti-fuga + pesos
    # ------------------------------------------------------------------
    incluidos = final[final.incluir]
    conteo = incluidos.groupby(["split_final", "clase_binaria"]).size().unstack(fill_value=0)
    duros_en_uso = int((incluidos.rol == "negativo_duro").sum())
    pac_val = set(incluidos[incluidos.split_final == "validation"].paciente_id)
    pac_train = set(incluidos[incluidos.split_final == "train"].paciente_id)
    pac_test = set(incluidos[incluidos.split_final == "test"].paciente_id)

    tr = incluidos[incluidos.split_final == "train"]
    pesos = {
        c: round(len(tr) / (2 * (tr.clase_binaria == c).sum()), 3)
        for c in ["NORMAL", "PNEUMONIA"]
    }

    print(f"Manifiesto -> {destino.resolve()}")
    print(conteo)
    print(f"negativos duros incluidos: {duros_en_uso}")
    print(f"excluidas multi-etiqueta sin neumonía: {n_excluidas}")
    print(f"NORMAL submuestreadas en train: {len(excluir_norm)}")
    print(f"fuga train/validation (pacientes): {len(pac_train & pac_val)}")
    print(f"fuga train/test (pacientes):       {len(pac_train & pac_test)}")
    print(f"fuga validation/test (pacientes):  {len(pac_val & pac_test)}")
    print(f"class_weights train final: {pesos}")


if __name__ == "__main__":
    main()
