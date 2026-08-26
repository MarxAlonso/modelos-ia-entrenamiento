"""
Dataset v3 — RSNA Pneumonia Detection (espejo kellly/RSNA-Pneumonia-Detection)

Descarga el ZIP oficial del challenge (3.75 GB), extrae y convierte los DICOM
de entrenamiento a PNG, y construye etiquetas_crudas.csv con cajas incluidas:

    data/versiones/v3_rsna/
    ├── images/<patientId>.png          # 26 684 imágenes (1 por paciente)
    └── etiquetas_crudas.csv            # patientId,target,n_cajas,cajas,clase_detalle

NOTAS:
- Se omiten los 3 000 DICOM de test: sin etiquetas públicas.
- RSNA NO trae split oficial -> lo genera generar_manifiesto_v3.py (patient-wise).
- Las cajas se conservan para FUTURA localización.

Uso:
    python src/preparar_datos_v3.py [--solo-extract] [--workers 8]
"""
import argparse
import csv
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "kellly/RSNA-Pneumonia-Detection"
ZIP_NAME = "rsna-pneumonia-detection-challenge.zip"
DESTINO = Path("data/versiones/v3_rsna")
TMP = Path("data/_tmp_hf_v3")


def descargar() -> Path:
    print(f"[1/5] Descargando {ZIP_NAME} (3.75 GB, archivo único sin rate-limit) ...")
    ruta = hf_hub_download(REPO_ID, ZIP_NAME, repo_type="dataset", local_dir=TMP)
    return Path(ruta)


def extraer_zip(ruta_zip: Path) -> Path:
    print("[2/5] Extrayendo ZIP ...")
    destino_zip = TMP / "unzipped"
    if not (destino_zip / "stage_2_train_labels.csv").exists():
        with zipfile.ZipFile(ruta_zip) as z:
            z.extractall(destino_zip)
    raiz = destino_zip
    if not (raiz / "stage_2_train_labels.csv").exists():
        candidatos = list(raiz.rglob("stage_2_train_labels.csv"))
        if not candidatos:
            raise SystemExit("ERROR: no encontré stage_2_train_labels.csv dentro del zip")
        raiz = candidatos[0].parent
    print(f"      contenido en {raiz}")
    return raiz


def _convertir(lote):
    import pydicom
    from PIL import Image

    errores = []
    for origen, destino in lote:
        try:
            ds = pydicom.dcmread(origen)
            Image.fromarray(ds.pixel_array).save(destino)
        except Exception as e:  # noqa: BLE001
            errores.append((origen, str(e)))
    return errores


def convertir(raiz: Path, workers: int) -> None:
    print(f"[3/5] Convirtiendo DICOM -> PNG ({workers} hilos) ...")
    carpeta_png = DESTINO / "images"
    carpeta_png.mkdir(parents=True, exist_ok=True)

    dicoms = sorted((raiz / "stage_2_train_images").rglob("*.dcm"))
    print(f"      {len(dicoms)} DICOM encontrados.")
    lotes = [
        [(str(d), str(carpeta_png / (d.stem + ".png"))) for d in list(dicoms)[i::workers]]
        for i in range(workers)
    ]
    errores_totales = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for n, errores in enumerate(ex.map(_convertir, lotes)):
            errores_totales.extend(errores)
            hechos = sum(len(l) for l in lotes[: n + 1])
            print(f"      progreso ~{hechos}/{len(dicoms)}")
    if errores_totales:
        print(f"      ATENCION: {len(errores_totales)} errores:")
        for o, e in errores_totales[:5]:
            print(f"        {o}: {e}")


def etiquetas(raiz: Path) -> None:
    print("[4/5] Construyendo etiquetas_crudas.csv ...")
    labels = list(csv.DictReader(open(raiz / "stage_2_train_labels.csv", encoding="utf-8")))
    detalle_path = raiz / "stage_2_detailed_class_info.csv"
    detalle = (
        {r["patientId"]: r["class"] for r in csv.DictReader(open(detalle_path, encoding="utf-8"))}
        if detalle_path.exists() else {}
    )

    por_paciente: dict[str, list[dict]] = {}
    for fila in labels:
        por_paciente.setdefault(fila["patientId"], []).append(fila)

    with open(DESTINO / "etiquetas_crudas.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["patientId", "target", "n_cajas", "cajas", "clase_detalle"])
        for pid, filas in por_paciente.items():
            target = max(int(f["Target"]) for f in filas)
            cajas = ";".join(
                f"{f['x']},{f['y']},{f['width']},{f['height']}"
                for f in filas if f["Target"] == "1"
            )
            writer.writerow([pid, target, len(filas), cajas, detalle.get(pid, "?")])
    print(f"      pacientes: {len(por_paciente)}")


def limpiar(ruta_zip: Path | None) -> None:
    print("[5/5] Limpiando temporales ...")
    shutil.rmtree(TMP, ignore_errors=True)
    if ruta_zip and ruta_zip.exists():
        ruta_zip.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solo-extract", action="store_true",
                        help="usa data/_tmp_hf_v3/rsna-pneumonia-detection-challenge.zip ya descargado")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    DESTINO.mkdir(parents=True, exist_ok=True)
    if args.solo_extract:
        ruta_zip = TMP / ZIP_NAME
        if not ruta_zip.exists():
            raise SystemExit("ERROR: no existe el zip descargado previamente")
        raiz = extraer_zip(ruta_zip)
    else:
        ruta_zip = descargar()
        raiz = extraer_zip(ruta_zip)

    convertir(raiz, args.workers)
    etiquetas(raiz)
    limpiar(ruta_zip)
    n = len(list((DESTINO / "images").glob("*.png")))
    print({"imagenes_convertidas": n})
    print(f"OK -> {DESTINO.resolve()}")


if __name__ == "__main__":
    main()
