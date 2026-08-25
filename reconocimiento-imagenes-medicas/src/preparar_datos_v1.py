"""
Fase 1 — Prepara Dataset v1: Chest X-Ray Pneumonia (Kermany et al., 2018)

Descarga los shards Parquet desde HuggingFace (hf-vision/chest-xray-pneumonia),
extrae cada imagen JPEG y organiza la estructura clásica de carpetas:

    data/versiones/v1_kaggle_chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── validation/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/

Uso:
    python src/preparar_datos_v1.py            # descarga + extracción
    python src/preparar_datos_v1.py --solo-extract   # extrae desde data/_tmp_hf ya descargado
"""
import argparse
import io
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download
from PIL import Image

REPO_ID = "hf-vision/chest-xray-pneumonia"
DESTINO = Path("data/versiones/v1_kaggle_chest_xray")
TMP = Path("data/_tmp_hf")
MAPA_ETIQUETAS = {0: "NORMAL", 1: "PNEUMONIA"}


def descargar() -> None:
    print(f"[1/3] Descargando shards Parquet de {REPO_ID} ...")
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=TMP)
    print("      Descarga completa.")


def extraer() -> dict:
    print("[2/3] Extrayendo imágenes JPEG ...")
    resumen: dict = {}
    shards = sorted(TMP.rglob("*.parquet"))
    if not shards:
        sys.exit("ERROR: no hay shards en data/_tmp_hf (usa modo normal para descargar).")

    for shard in shards:
        # train-00000-of-00007.parquet -> train
        nombre_split = shard.stem.split("-")[0]
        df = pd.read_parquet(shard, columns=["image", "label"])
        for _, fila in df.iterrows():
            clase = MAPA_ETIQUETAS[int(fila["label"])]
            carpeta = DESTINO / nombre_split / clase
            carpeta.mkdir(parents=True, exist_ok=True)

            contenido = fila["image"]
            nombre_archivo = Path(contenido.get("path") or "").name or f"img_{df.index.get_loc(_)}.jpeg"
            destino_img = carpeta / nombre_archivo

            # El mismo paciente puede aparecer en shards distintos con la misma foto:
            # escribir una sola vez (deduplicación natural por nombre).
            if destino_img.exists():
                continue
            destino_img.write_bytes(contenido["bytes"])

        resumen[nombre_split] = resumen.get(nombre_split, 0) + len(df)
        print(f"      {shard.name}: {len(df)} filas procesadas.")

    return resumen


def limpiar() -> None:
    print("[3/3] Eliminando Parquet temporales ...")
    shutil.rmtree(TMP, ignore_errors=True)


def contar_final() -> dict:
    conteo = {"train": {}, "validation": {}, "test": {}}
    total = 0
    for split in conteo:
        for clase in MAPA_ETIQUETAS.values():
            n = len(list((DESTINO / split / clase).glob("*")))
            conteo[split][clase] = n
            total += n
    conteo["total"] = total
    return conteo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solo-extract", action="store_true", help="Extraer sin volver a descargar")
    args = parser.parse_args()

    if not args.solo_extract:
        descargar()
    extraer()
    limpiar()

    final = contar_final()
    print(json.dumps(final, indent=2, ensure_ascii=False))
    (DESTINO / "conteo_extraccion.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"OK -> {DESTINO.resolve()}")


if __name__ == "__main__":
    main()
