"""
Dataset v2 — NIH ChestX-ray14 (espejo arudaev/chest-xray-14-320)

Descarga los shards Parquet y extrae las imágenes a:

    data/versiones/v2_nih_chest_xray14/
    ├── images/
    │   ├── train/      <filename>.png   (77 967)
    │   ├── validation/                  (8 557)
    │   └── test/                        (25 596)
    └── etiquetas_crudas.csv             # split, filename, labels (multi-etiqueta NIH)

NOTA de diseño: al ser MULTI-ETIQUETA (una imagen puede tener varias patologías),
NO se organizan en carpetas por clase como v1. La carpeta solo ordena por split;
la etiqueta binaria se decide después en el manifiesto (generar_manifiesto_v2.py).

Uso:
    python src/preparar_datos_v2.py [--solo-extract]
"""
import argparse
import csv
import shutil
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

REPO_ID = "arudaev/chest-xray-14-320"
DESTINO = Path("data/versiones/v2_nih_chest_xray14")
TMP = Path("data/_tmp_hf_v2")


def descargar() -> None:
    print(f"[1/3] Descargando {REPO_ID} (~7.4 GB) ...")
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=TMP)
    print("      Descarga completa.")


def extraer() -> None:
    print("[2/3] Extrayendo imágenes y construyendo etiquetas_crudas.csv ...")
    DESTINO.mkdir(parents=True, exist_ok=True)
    ruta_csv = DESTINO / "etiquetas_crudas.csv"
    shards = sorted(TMP.rglob("*.parquet"))
    if not shards:
        raise SystemExit("ERROR: no hay shards en data/_tmp_hf_v2")

    with ruta_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["split", "filename", "labels"])
        for shard in shards:
            nombre_split = shard.stem.split("-")[0]  # train|validation|test
            df = pd.read_parquet(shard)
            carpeta = DESTINO / "images" / nombre_split
            carpeta.mkdir(parents=True, exist_ok=True)
            for _, fila in df.iterrows():
                destino_img = carpeta / fila["filename"]
                if not destino_img.exists():
                    destino_img.write_bytes(fila["image"]["bytes"])
                writer.writerow([nombre_split, fila["filename"], fila["labels"]])
            print(f"      {shard.name}: {len(df)} imágenes ({nombre_split})")


def limpiar() -> None:
    print("[3/3] Eliminando Parquet temporales ...")
    shutil.rmtree(TMP, ignore_errors=True)


def resumen() -> dict:
    conteo = {}
    for split in ["train", "validation", "test"]:
        n = len(list((DESTINO / "images" / split).glob("*")))
        conteo[split] = n
    return conteo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solo-extract", action="store_true")
    args = parser.parse_args()

    if not args.solo_extract:
        descargar()
    extraer()
    limpiar()
    print(resumen())
    print(f"OK -> {DESTINO.resolve()}")


if __name__ == "__main__":
    main()
