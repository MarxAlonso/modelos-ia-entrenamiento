"""
Fase 5 — Fine Tuning: reentrenar una versión anterior con capas descongeladas

Carga el modelo.keras maestro de una versión previa, descongela las últimas N
capas de su base (BatchNorm SIEMPRE congelada), recompila con lr pequeño y
reentrena pocas épocas. Es la materialización del ciclo de mejora:
cada versión parte del aprendizaje de la anterior.

    v003 (feature extraction, base congelada)
        ↓ cargar modelo.keras
        ↓ descongelar últimas N capas de la base
        ↓ lr = 1e-5 (¡diminuto! los pesos ya son buenos, solo se pulen)
        └→ v004

Uso:
    python src/afinar.py --version-id v004 \
        --desde models/versiones/v003_20260823-2049_mobilenetv2_fe/modelo.keras \
        --capas-descongeladas 30 --epochs 10 --patience 4
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
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
from entrenar import calcular_metricas, guardar_matriz, obtener_predicciones  # noqa: E402
from preprocesamiento import (  # noqa: E402
    SEMILLA,
    cargar_manifiesto,
    crear_pipeline,
    pesos_de_clase,
)

DATASET_ID = "v1_kaggle_chest_xray"
BATCH_SIZE = 32


def localizar_base(modelo: tf.keras.Model) -> tf.keras.Model:
    """Encuentra la sub-red base (MobileNetV2/ResNet50) dentro del modelo cargado."""
    bases = [l for l in modelo.layers if isinstance(l, tf.keras.Model)]
    if not bases:
        sys.exit("ERROR: el modelo no contiene una base anidada (¿es una CNN propia?)")
    return bases[0]


def descongelar_cola(base: tf.keras.Model, n_capas: int) -> int:
    """
    Descongela las últimas n_capas de la base.
    BatchNorm queda SIEMPRE congelada: sus estadísticas de ImageNet son
    valiosas y con pocos datos reentrenarlas destruye la calibración.
    """
    base.trainable = True
    for capa in base.layers[:-n_capas]:
        capa.trainable = False
    bn_congeladas = 0
    for capa in base.layers:
        if isinstance(capa, tf.keras.layers.BatchNormalization):
            capa.trainable = False
            bn_congeladas += 1
    descongeladas = sum(1 for c in base.layers if c.trainable)
    return descongeladas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--desde", required=True, type=Path, help="modelo.keras de la versión anterior")
    parser.add_argument("--capas-descongeladas", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--monitor", default="val_auc")
    parser.add_argument("--notas", default="")
    args = parser.parse_args()

    modo_monitor = "max" if any(s in args.monitor for s in ["auc", "recall", "acc"]) else "min"
    tf.keras.utils.set_random_seed(SEMILLA)

    fecha = datetime.now().strftime("%Y%m%d-%H%M")
    dir_version = MODELS_DIR = Path("models/versiones") / f"{args.version_id}_{fecha}_ft"
    dir_version.mkdir(parents=True, exist_ok=True)

    print(f"=== {args.version_id}: fine tuning desde {args.desde.parent.name} ===")
    config_previa = json.loads((args.desde.parent / "config.json").read_text(encoding="utf-8"))
    normalizacion = config_previa.get("normalizacion", "rescale")

    df = cargar_manifiesto()
    train_ds = crear_pipeline(df, "train", batch_size=BATCH_SIZE, augment=True, normalizacion=normalizacion)
    val_ds = crear_pipeline(df, "validation", batch_size=BATCH_SIZE, normalizacion=normalizacion)
    test_ds = crear_pipeline(df, "test", batch_size=BATCH_SIZE, normalizacion=normalizacion)
    pesos = {int(k): v for k, v in pesos_de_clase(df).items()}

    modelo = tf.keras.models.load_model(args.desde)
    base = localizar_base(modelo)

    trainable_antes = int(np.sum([np.prod(v.shape) for v in modelo.trainable_weights]))
    descongeladas = descongelar_cola(base, args.capas_descongeladas)
    trainable_despues = int(np.sum([np.prod(v.shape) for v in modelo.trainable_weights]))
    print(f"base: {base.name} · capas descongeladas: {descongeladas}")
    print(f"params entrenables: {trainable_antes:,} -> {trainable_despues:,}")

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=args.monitor, mode=modo_monitor, patience=args.patience,
            restore_best_weights=True, verbose=1,
        )
    ]

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
        "dataset": DATASET_ID,
        "manifiesto": "data/manifiestos/v1_splits.csv",
        "base_anterior": args.desde.parent.name,
        "arquitectura": f"{base.name}_finetuning",
        "base_imagenet": base.name.replace("_1.00_224", ""),
        "capas_descongeladas": args.capas_descongeladas,
        "batchnorm_congeladas": True,
        "img_size": [224, 224],
        "batch_size": BATCH_SIZE,
        "epochs": args.epochs,
        "epochs_reales": len(historial.history["loss"]),
        "learning_rate_fine_tuning": args.lr,
        "loss": "binary_crossentropy",
        "normalizacion": normalizacion,
        "augmentacion": ["rotation_5deg", "zoom_10", "translation_10", "contrast_8"],
        "class_weights": {k: v for k, v in pesos.items()},
        "early_stopping": {"monitor": args.monitor, "mode": modo_monitor, "patience": args.patience},
        "semilla": SEMILLA,
        "notas": args.notas,
    }
    dir_version.joinpath("config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    metricas_json = {
        "version": args.version_id,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "arquitectura": config["arquitectura"],
        "dataset": DATASET_ID,
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

    guardar_matriz(metricas, dir_version / "matriz_confusion.png", f"{args.version_id} fine tuning (test)")

    print(json.dumps({"fp32": metricas, "minutos": round(minutos, 1), "tamano_mb": tamano_mb}, indent=2, ensure_ascii=False))
    print(f"OK -> {dir_version.resolve()}")


if __name__ == "__main__":
    main()
