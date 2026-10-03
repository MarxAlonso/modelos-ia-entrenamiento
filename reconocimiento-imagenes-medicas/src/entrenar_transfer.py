"""
Fase 4 — Transfer Learning: base ImageNet congelada + clasificador propio

Feature extraction: la base preentrenada actúa como extractor de características
(congelada, training=False) y SOLO se entrena la cabeza nueva.

    imagen 224×224×3 (normalizada según base)
        ↓  base congelada (MobileNetV2 | ResNet50)
        ↓  GlobalAveragePooling
        ↓  Dense(128) + Dropout(0.3)
        └→ Dense(1, sigmoid)

Uso:
    python src/entrenar_transfer.py --version-id v003 --base mobilenetv2 \
        --epochs 20 --patience 6 --reduce-lr --notas "..."
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
import preprocesamiento  # noqa: E402
from entrenar import calcular_metricas, guardar_matriz, obtener_predicciones  # noqa: E402
from preprocesamiento import (  # noqa: E402
    SEMILLA,
    cargar_manifiesto,
    crear_pipeline,
    pesos_de_clase,
)

DATASET_ID = "v1_kaggle_chest_xray"
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DROPOUT_CABEZA = 0.3

MODELS_DIR = Path("models/versiones")

BASES = {
    # nombre: (clase application, clave normalizador en preprocesamiento)
    "mobilenetv2": (tf.keras.applications.MobileNetV2, "mobilenetv2"),
    "resnet50": (tf.keras.applications.ResNet50, "resnet50"),
}


def construir_modelo(base_nombre: str) -> tf.keras.Model:
    clase_base, _ = BASES[base_nombre]
    base = clase_base(
        include_top=False,
        weights="imagenet",
        input_shape=(*preprocesamiento.IMG_SIZE, 3),
    )
    base.trainable = False  # feature extraction: nada de la base se mueve

    entradas = tf.keras.layers.Input(shape=(*preprocesamiento.IMG_SIZE, 3))
    x = base(entradas, training=False)  # BatchNorm de la base SIEMPRE en modo inferencia
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(DROPOUT_CABEZA)(x)
    salidas = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    modelo = tf.keras.Model(entradas, salidas, name=f"{base_nombre}_fe")
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return modelo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--base", choices=list(BASES), default="mobilenetv2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--monitor", default="val_auc")
    parser.add_argument("--reduce-lr", action="store_true")
    parser.add_argument("--recorte-borde", type=int, default=0,
                        help="px eliminados de cada borde (experimento de control v006)")
    parser.add_argument("--dataset", choices=list(preprocesamiento.DATASETS), default="v1",
                        help="dataset versionado a usar (manifiesto+raíz de preprocesamiento)")
    parser.add_argument("--sobremuestrear-positivos", type=int, default=1,
                        help="veces que se repite cada imagen PNEUMONIA de train (anti-desbalance)")
    parser.add_argument("--max-normal-train", type=int, default=0,
                        help="limita NORMAL de train a N imágenes (0 = sin límite; controla tiempo/época en CPU)")
    parser.add_argument("--notas", default="")
    args = parser.parse_args()

    modo_monitor = "max" if any(s in args.monitor for s in ["auc", "recall", "acc"]) else "min"
    _, normalizacion = BASES[args.base]

    tf.keras.utils.set_random_seed(SEMILLA)
    fecha = datetime.now().strftime("%Y%m%d-%H%M")
    dir_version = MODELS_DIR / f"{args.version_id}_{fecha}_{args.base}_fe"
    dir_version.mkdir(parents=True, exist_ok=True)

    print(f"=== {args.version_id}: {args.base} feature extraction ===")
    df = cargar_manifiesto(dataset=args.dataset)
    dataset_id = df.attrs["dataset_id"]

    # Oversampling de positivos SOLO en train (duplica filas, no imágenes nuevas)
    if args.sobremuestrear_positivos > 1:
        train_mask = df.split_final == "train"
        positivos = df[train_mask & (df.clase == "PNEUMONIA")]
        df = pd.concat([df, *[positivos] * (args.sobremuestrear_positivos - 1)], ignore_index=True)
        print(f"positivos sobremuestreados x{args.sobremuestrear_positivos}: "
              f"train={int(train_mask.sum())} -> {int((df.split_final == 'train').sum())} filas")

    # Límite de NORMAL en train (muestreo fijo por semilla; el manifiesto queda intacto)
    if args.max_normal_train > 0:
        en_train = df.split_final == "train"
        normales = df[en_train & (df.clase == "NORMAL")]
        if len(normales) > args.max_normal_train:
            mantener = normales.sample(n=args.max_normal_train, random_state=SEMILLA).index
            df = pd.concat([df[~(en_train & (df.clase == "NORMAL"))], df.loc[mantener]])
            print(f"NORMAL de train limitado a {args.max_normal_train}")
    train_ds = crear_pipeline(df, "train", batch_size=BATCH_SIZE, augment=True, normalizacion=normalizacion, recorte_borde=args.recorte_borde)
    val_ds = crear_pipeline(df, "validation", batch_size=BATCH_SIZE, normalizacion=normalizacion, recorte_borde=args.recorte_borde)
    test_ds = crear_pipeline(df, "test", batch_size=BATCH_SIZE, normalizacion=normalizacion, recorte_borde=args.recorte_borde)
    pesos = {int(k): v for k, v in pesos_de_clase(df).items()}

    modelo = construir_modelo(args.base)
    trainable = np.sum([np.prod(v.shape) for v in modelo.trainable_weights])
    total = modelo.count_params()
    print(f"params totales: {total:,} · entrenables: {trainable:,} ({100*trainable/total:.1f}%)")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=args.monitor, mode=modo_monitor, patience=args.patience,
            restore_best_weights=True, verbose=1,
        )
    ]
    if args.reduce_lr:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=args.monitor, mode=modo_monitor, factor=0.33, patience=2, verbose=1
            )
        )

    t0 = time.perf_counter()
    historial = modelo.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=pesos,
        callbacks=callbacks,
        verbose=2,
    )
    minutos = (time.perf_counter() - t0) / 60

    print("Evaluando en test ...")
    y_real, prob = obtener_predicciones(modelo, test_ds)
    metricas = calcular_metricas(y_real, prob)

    config = {
        "version": args.version_id,
        "dataset": dataset_id,
        "manifiesto": str(preprocesamiento.DATASETS[args.dataset]["manifiesto"]),
        "base_anterior": None,
        "arquitectura": f"{args.base}_feature_extraction",
        "base_imagenet": args.base,
        "base_congelada": True,
        "capas_descongeladas": 0,
        "img_size": list(preprocesamiento.IMG_SIZE),
        "batch_size": BATCH_SIZE,
        "epochs": args.epochs,
        "epochs_reales": len(historial.history["loss"]),
        "learning_rate": LEARNING_RATE,
        "dropout_cabeza": DROPOUT_CABEZA,
        "loss": "binary_crossentropy",
        "normalizacion": normalizacion,
        "recorte_borde_px": args.recorte_borde,
        "dataset_variante": args.dataset,
        "sobremuestreo_positivos": args.sobremuestrear_positivos,
        "max_normal_train": args.max_normal_train,
        "augmentacion": ["rotation_5deg", "zoom_10", "translation_10", "contrast_8"],
        "class_weights": {k: v for k, v in pesos.items()},
        "early_stopping": {"monitor": args.monitor, "mode": modo_monitor, "patience": args.patience},
        "reduce_lr": bool(args.reduce_lr),
        "semilla": SEMILLA,
        "notas": args.notas,
    }
    dir_version.joinpath("config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    metricas_json = {
        "version": args.version_id,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "arquitectura": config["arquitectura"],
        "dataset": dataset_id,
        "config": config,
        "tiempo_entrenamiento_min": round(minutos, 1),
        "mejor_epoca": int(np.argmin(historial.history["val_loss"])) + 1,
        "fp32": metricas,
    }

    modelo.save(dir_version / "modelo.keras")
    tamano_mb = round((dir_version / "modelo.keras").stat().st_size / 1e6, 2)
    metricas_json["fp32"]["tamano_mb"] = tamano_mb
    dir_version.joinpath("metricas.json").write_text(json.dumps(metricas_json, indent=2, ensure_ascii=False), encoding="utf-8")

    hist = historial.history
    ejes = [("recall", "val_recall", "Recall"), ("loss", "val_loss", "Loss")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (tr, va, titulo) in zip(axes, ejes):
        ax.plot(hist[tr], label=f"train {tr}")
        ax.plot(hist[va], label=f"validation {va}")
        ax.set_title(titulo)
        ax.set_xlabel("Época")
        ax.legend()
    fig.tight_layout()
    fig.savefig(dir_version / "curvas_entrenamiento.png", dpi=150)
    plt.close(fig)

    guardar_matriz(metricas, dir_version / "matriz_confusion.png", f"{args.version_id} {args.base}_fe (test)")

    print(json.dumps({"fp32": metricas, "minutos": round(minutos, 1), "tamano_mb": tamano_mb}, indent=2, ensure_ascii=False))
    print(f"OK -> {dir_version.resolve()}")


if __name__ == "__main__":
    main()
