"""
Fase 2 — Preprocesamiento y pipelines tf.data

Módulo ÚNICO de preprocesamiento del proyecto: todas las versiones de modelo
deben construir sus datos a través de aquí, para que las comparaciones sean justas.

Pipeline conceptual (mapa original, sección 7):

    Imagen en disco
      ↓  decodificar JPEG + uniformar a RGB
      ↓  resize 224×224
      ↓  [solo train] data augmentation médicamente razonable
      ↓  normalización según modelo base
    batch de tensores listos para la red

Uso desde un script de entrenamiento:

    from preprocesamiento import (
        cargar_manifiesto, crear_pipeline, pesos_de_clase, IMG_SIZE, SEMILLA,
    )

    df = cargar_manifiesto()
    train_ds = crear_pipeline(df, "train", batch_size=32, augment=True, normalizacion="rescale")
    val_ds   = crear_pipeline(df, "validation", batch_size=32)
    test_ds  = crear_pipeline(df, "test", batch_size=32)
    pesos    = pesos_de_clase(df)

Verificación autónoma:
    python src/preprocesamiento.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

# ---------------------------------------------------------------------------
# Constantes del proyecto (no cambiar sin documentarlo en docs/)
# ---------------------------------------------------------------------------
SEMILLA = 42
IMG_SIZE = (224, 224)
BATCH_DEFAULT = 32
MAPA_CLASES = {"NORMAL": 0, "PNEUMONIA": 1}
MANIFIESTO = Path("data/manifiestos/v1_splits.csv")
RAIZ_DATASET = Path("data/versiones/v1_kaggle_chest_xray")

# Registro central de datasets versionados. Cada entrada normaliza su
# manifiesto a las columnas estándar del proyecto (ruta_relativa, clase,
# split_final) para que TODO el resto del código sea agnóstico al origen.
DATASETS = {
    "v1": {
        "manifiesto": Path("data/manifiestos/v1_splits.csv"),
        "raiz": Path("data/versiones/v1_kaggle_chest_xray"),
        "columna_clase": "clase",
        "solo_incluir": False,
        "id": "v1_kaggle_chest_xray",
    },
    "v2": {
        "manifiesto": Path("data/manifiestos/v2_splits.csv"),
        "raiz": Path("data/versiones/v2_nih_chest_xray14"),
        "columna_clase": "clase_binaria",
        "solo_incluir": True,   # v2 submuestrea NORMAL en train (incluir=false auditable)
        "id": "v2_nih_chest_xray14",
    },
    # Variante de tarea para NIH: añade hallazgos tipo-infiltración como
    # negativos duros en train/val. MISMO test que v2 -> comparaciones válidas.
    "v2duros": {
        "manifiesto": Path("data/manifiestos/v2_splits_duros.csv"),
        "raiz": Path("data/versiones/v2_nih_chest_xray14"),
        "columna_clase": "clase_binaria",
        "solo_incluir": True,
        "id": "v2_nih_chest_xray14",
    },
    # RSNA: etiquetas de radiólogos, 1024², cajas disponibles. Split propio.
    "v3": {
        "manifiesto": Path("data/manifiestos/v3_splits.csv"),
        "raiz": Path("data/versiones/v3_rsna"),
        "columna_clase": "clase",
        "solo_incluir": False,
        "id": "v3_rsna",
    },
    # UNIÓN multi-dominio (v1 + v2duros + v3). Las rutas del manifiesto ya
    # incluyen la carpeta del dataset de origen, por eso la raíz es el padre
    # común. Lo genera src/generar_manifiesto_union.py.
    "vu": {
        "manifiesto": Path("data/manifiestos/vu_splits.csv"),
        "raiz": Path("data/versiones"),
        "columna_clase": "clase",
        "solo_incluir": False,
        "id": "vu_union_multidominio",
    },
}

AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------------------------------------------------------
# Manifiesto
# ---------------------------------------------------------------------------
def cargar_manifiesto(ruta: Path | None = None, dataset: str = "v1") -> pd.DataFrame:
    """
    Lee el manifiesto congelado de un dataset y lo NORMALIZA:
        ruta_relativa · clase · split_final (+ atributo raiz en df.attrs)
    NUNCA se re-splittea aquí.
    """
    cfg = DATASETS[dataset]
    df = pd.read_csv(ruta or cfg["manifiesto"])
    if cfg["solo_incluir"] and "incluir" in df.columns:
        df = df[df["incluir"] == True].copy()  # noqa: E712
    if "clase" not in df.columns:
        df = df.rename(columns={cfg["columna_clase"]: "clase"})
    df.attrs["raiz"] = str(cfg["raiz"])
    df.attrs["dataset_id"] = cfg["id"]
    assert set(df["split_final"].unique()) >= {"train", "validation", "test"}
    return df


def pesos_de_clase(df: pd.DataFrame) -> dict[int, float]:
    """
    Pesos inversos a la frecuencia: peso_c = n_total / (2 * n_c).
    Compensan el desbalance 2.89:1 durante el entrenamiento.
    """
    train = df[df.split_final == "train"]
    n_total = len(train)
    return {
        MAPA_CLASES[c]: round(n_total / (2 * len(train[train.clase == c])), 3)
        for c in MAPA_CLASES
    }


# ---------------------------------------------------------------------------
# Decodificación y geometría
# ---------------------------------------------------------------------------
def _decodificar(ruta_bytes: tf.Tensor, recorte_borde_px: int = 0) -> tf.Tensor:
    """
    JPEG bytes -> tensor float32 RGB 224×224.

    tf.image.grayscale_to_rgb no basta si la imagen ya es RGB, por eso:
    convert_image_dtype a float32 y luego replicar canales solo si tiene 1.

    recorte_borde_px: elimina N píxeles de CADA borde en resolución ORIGINAL
    (antes del resize). Usado para eliminar marcadores L/R y texto quemado
    de las placas (experimento de control v006).
    """
    img = tf.io.read_file(ruta_bytes)
    img = tf.io.decode_jpeg(img, channels=0)          # respeta canales originales
    if recorte_borde_px > 0:
        forma = tf.shape(img)
        alto, ancho = forma[0], forma[1]
        img = tf.image.crop_to_bounding_box(
            img,
            offset_height=recorte_borde_px,
            offset_width=recorte_borde_px,
            target_height=alto - 2 * recorte_borde_px,
            target_width=ancho - 2 * recorte_borde_px,
        )
    img = tf.image.convert_image_dtype(img, tf.float32)  # 0..255 -> 0..1
    img = tf.cond(
        tf.equal(tf.shape(img)[-1], 1),
        lambda: tf.tile(img, [1, 1, 3]),              # L -> RGB replicando canal
        lambda: img[..., :3],                         # RGBA u otros -> recortar a RGB
    )
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    return img


def _etiqueta(clase_txt: tf.Tensor) -> tf.Tensor:
    """'NORMAL'/'PNEUMONIA' -> 0/1 (mismo orden que el dataset original)."""
    tabla = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            list(MAPA_CLASES.keys()), list(MAPA_CLASES.values())
        ),
        default_value=-1,
    )
    return tabla.lookup(clase_txt)


# ---------------------------------------------------------------------------
# Data augmentation (SOLO entrenamiento)
# ---------------------------------------------------------------------------
def capa_augmentacion() -> tf.keras.Sequential:
    """
    Augmentación médicamente razonable para radiografías AP pediátricas.

    SÍ se permite                          NO se aplica y por qué
    ─────────────────────────────         ──────────────────────────────────
    Rotación ±5°                           Flips horizontales: el corazón y la
    Zoom ≤10%                              cúpula diafragmática son laterales;
    Traslación ≤10%                        voltear crea anatomía imposible.
    Contraste ±8%                          Rotaciones grandes: simularían mal
                                           posición del paciente como variabilidad
                                           clínica falsa.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(5 / 360, seed=SEMILLA, name="rot5"),
            tf.keras.layers.RandomZoom(0.1, seed=SEMILLA, name="zoom10"),
            tf.keras.layers.RandomTranslation(0.1, 0.1, seed=SEMILLA, name="shift10"),
            tf.keras.layers.RandomContrast(0.08, seed=SEMILLA, name="contrast8"),
        ],
        name="augmentacion",
    )


# ---------------------------------------------------------------------------
# Normalizadores por arquitectura
# ---------------------------------------------------------------------------
def _normalizar_rescale(x: tf.Tensor) -> tf.Tensor:
    """CNN propia: rango [0,1]. Recorta el exceso que deja RandomContrast."""
    return tf.clip_by_value(x, 0.0, 1.0)


def _normalizar_resnet50(x: tf.Tensor) -> tf.Tensor:
    """ResNet50 ImageNet: BGR + media propia de caffe."""
    return tf.keras.applications.resnet50.preprocess_input(
        tf.clip_by_value(x, 0.0, 1.0) * 255.0
    )


def _normalizar_mobilenetv2(x: tf.Tensor) -> tf.Tensor:
    """MobileNetV2 ImageNet: [-1, 1]."""
    return tf.keras.applications.mobilenet_v2.preprocess_input(
        tf.clip_by_value(x, 0.0, 1.0) * 255.0
    )


NORMALIZADORES = {
    "rescale": _normalizar_rescale,
    "resnet50": _normalizar_resnet50,
    "mobilenetv2": _normalizar_mobilenetv2,
}


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def crear_pipeline(
    df: pd.DataFrame,
    split: str,
    batch_size: int = BATCH_DEFAULT,
    augment: bool | None = None,
    normalizacion: str = "rescale",
    recorte_borde: int = 0,
) -> tf.data.Dataset:
    """
    Construye el tf.data.Dataset de un split.

    augment: por defecto True solo para train. Las capas Random* de Keras se
    activan solas en training=True y pasan inactivas en predict/eval.
    recorte_borde: píxeles eliminados de cada borde antes del resize
    (debe coincidir entre entrenamiento y evaluación de una misma versión).
    """
    if augment is None:
        augment = split == "train"

    sub = df[df.split_final == split]
    raiz = Path(df.attrs.get("raiz", RAIZ_DATASET))
    rutas = [str(raiz / r) for r in sub["ruta_relativa"]]
    etiquetas = [MAPA_CLASES[c] for c in sub["clase"]]

    ds = tf.data.Dataset.from_tensor_slices((rutas, etiquetas))

    if split == "train":
        ds = ds.shuffle(buffer_size=len(rutas), seed=SEMILLA, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda r, y: (_decodificar(r, recorte_borde), y),
        num_parallel_calls=AUTOTUNE,
    )

    if augment:
        aug = capa_augmentacion()
        # Las capas Random* requieren saber que estamos entrenando: se aplican
        # dentro del grafo; su modo training se activa al llamar model.fit.
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)

    norm = NORMALIZADORES[normalizacion]
    ds = ds.map(lambda x, y: (norm(x), y), num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Verificación autónoma (ejecutable sin entrenar nada)
# ---------------------------------------------------------------------------
def _verificar() -> None:
    print("Cargando manifiesto ...")
    df = cargar_manifiesto()
    print(df.groupby(["split_final", "clase"]).size().unstack(fill_value=0))
    print(f"pesos_de_clase: {pesos_de_clase(df)}")

    for norm in NORMALIZADORES:
        ds = crear_pipeline(df, "train", batch_size=32, augment=True, normalizacion=norm)
        x, y = next(iter(ds))
        print(f"[{norm:>11}] batch {x.shape} {x.dtype} · rango [{x.numpy().min():.3f}, {x.numpy().max():.3f}] · labels {y.shape}")

    for split in ["train", "validation", "test"]:
        n_batches = sum(1 for _ in crear_pipeline(df, split))
        print(f"{split}: {n_batches} batches")

    print("Midiendo throughput (img/s) del pipeline de train ...")
    import time

    ds = crear_pipeline(df, "train", batch_size=32)
    it = iter(ds)
    next(it)  # warmup
    t0 = time.perf_counter()
    n = 0
    for lote in it:
        n += lote[0].shape[0]
    dt = time.perf_counter() - t0
    print(f"  {n} imágenes en {dt:.1f}s -> {n/dt:.0f} img/s")

    # Grid visual: misma imagen original vs versiones aumentadas
    print("Generando grid de augmentación ...")
    salida = Path("data/reportes/v1_preprocesamiento")
    salida.mkdir(parents=True, exist_ok=True)
    ruta_ejemplo = str(RAIZ_DATASET / df[df.split_final == "train"].iloc[0]["ruta_relativa"])
    base = _decodificar(tf.constant(ruta_ejemplo))
    aug = capa_augmentacion()
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5))
    for fila in range(2):
        for col, ax_i in enumerate(axes[fila]):
            if col == 0 and fila == 0:
                mostrar = base.numpy()
                titulo = "original"
            else:
                mostrar = aug(base[None, ...], training=True)[0].numpy()
                titulo = "augmentada"
            ax_i.imshow(mostrar)
            ax_i.set_title(titulo, fontsize=9)
            ax_i.axis("off")
    fig.suptitle("Data augmentation sobre una misma radiografía (cada llamada = variación distinta)", y=0.98)
    fig.tight_layout()
    fig.savefig(salida / "augmentacion_muestras.png", dpi=150)
    plt.close(fig)
    print(f"OK -> {salida.resolve()}")


if __name__ == "__main__":
    _verificar()
