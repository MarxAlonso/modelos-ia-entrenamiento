"""
Fase 3 — Versión v001: CNN propia

Entrena la arquitectura básica del mapa original (sección 8) sobre el Dataset v1
con el preprocesamiento centralizado. Guarda todo en:

    models/versiones/v001_<fecha>_cnn/
    ├── config.json               # hiperparámetros exactos
    ├── metricas.json             # evaluación en test (fp32; cuantización la añade cuantizar.py)
    ├── matriz_confusion.png
    ├── curvas_entrenamiento.png
    └── modelo.keras              # maestro reentrenable

Uso:
    python src/entrenar.py                                   # v001 con valores históricos
    python src/entrenar.py --version-id v002 --monitor val_auc \
        --patience 8 --epochs 35 --reduce-lr --notas "..."   # ciclo de mejora
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
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import preprocesamiento  # noqa: E402
from preprocesamiento import (  # noqa: E402
    SEMILLA,
    cargar_manifiesto,
    crear_pipeline,
    pesos_de_clase,
)

# ---------------------------------------------------------------------------
# Configuración por defecto (v001). Las versiones nuevas pasan sus valores
# por CLI para dejar el histórico intacto y comparable.
# ---------------------------------------------------------------------------
VERSION_ID = "v001"
ARQUITECTURA = "cnn_propia"
DATASET_ID = "v1_kaggle_chest_xray"
BATCH_SIZE = 32
EPOCHS_MAX = 25
LEARNING_RATE = 1e-3
DROPOUT = 0.5
PACIENCIA_EARLY_STOPPING = 5
MONITOR = "val_recall"
NOTAS = "Primera version: CNN basica del mapa original, baseline de referencia."

MODELS_DIR = Path("models/versiones")


def construir_modelo() -> tf.keras.Model:
    """
    Arquitectura del mapa original (sección 8):

        224×224×3 -> Conv32 -> MaxPool -> Conv64 -> MaxPool
                  -> Conv128 -> GlobalAveragePooling
                  -> Dense(128) -> Dropout(0.5) -> Dense(1, sigmoid)
    """
    capas = [
        tf.keras.layers.Input(shape=(*preprocesamiento.IMG_SIZE, 3)),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
    modelo = tf.keras.Sequential(capas, name=ARQUITECTURA)
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


def calcular_metricas(y_real: np.ndarray, prob: np.ndarray, umbral: float = 0.5) -> dict:
    """Métricas completas desde probabilidades. Compartido con cuantizar.py."""
    y_pred = (prob >= umbral).astype(int)
    tp = int(((y_pred == 1) & (y_real == 1)).sum())
    tn = int(((y_pred == 0) & (y_real == 0)).sum())
    fp = int(((y_pred == 1) & (y_real == 0)).sum())
    fn = int(((y_pred == 0) & (y_real == 1)).sum())

    precision_pneu = tp / (tp + fp) if (tp + fp) else 0.0
    recall_pneu = tp / (tp + fn) if (tp + fn) else 0.0
    precision_norm = tn / (tn + fn) if (tn + fn) else 0.0
    recall_norm = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = lambda p, r: 2 * p * r / (p + r) if (p + r) else 0.0  # noqa: E731

    f1_pneu = f1(precision_pneu, recall_pneu)
    f1_norm = f1(precision_norm, recall_norm)
    accuracy = (tp + tn) / len(y_real)

    return {
        "umbral": umbral,
        "accuracy": round(accuracy, 4),
        "precision_pneumonia": round(precision_pneu, 4),
        "recall_pneumonia": round(recall_pneu, 4),
        "f1_pneumonia": round(f1_pneu, 4),
        "precision_normal": round(precision_norm, 4),
        "recall_normal": round(recall_norm, 4),
        "f1_normal": round(f1_norm, 4),
        "f1_macro": round((f1_pneu + f1_norm) / 2, 4),
        "roc_auc": round(float(roc_auc_score(y_real, prob)), 4),
        "matriz": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "n_imagenes": int(len(y_real)),
    }


def guardar_matriz(metricas: dict, destino: Path, titulo: str) -> None:
    m = metricas["matriz"]
    tabla = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(tabla, cmap="Blues")
    for i in range(2):
        for j in range(2):
            color = "white" if tabla[i, j] > tabla.max() / 2 else "#222"
            ax.text(j, i, str(tabla[i, j]), ha="center", va="center", fontsize=16, color=color)
    ax.set_xticks([0, 1], ["pred NORMAL", "pred PNEUMONIA"], fontsize=9)
    ax.set_yticks([0, 1], ["real NORMAL", "real PNEUMONIA"], fontsize=9)
    acc, f1 = metricas["accuracy"], metricas["f1_macro"]
    ax.set_title(f"{titulo}\naccuracy={acc:.3f} · f1_macro={f1:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)


def obtener_predicciones(modelo_o_ds, ds: tf.data.Dataset):
    """Extrae (y_real, prob) tanto para modelos Keras como para intérpretes TFLite."""
    ys, ps = [], []
    for x, y in ds:
        lote_prob = modelo_o_ds(x)
        ys.append(y.numpy())
        ps.append(np.asarray(lote_prob).reshape(-1))
    return np.concatenate(ys), np.concatenate(ps)


def main() -> None:
    global VERSION_ID, EPOCHS_MAX, PACIENCIA_EARLY_STOPPING, MONITOR, NOTAS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version-id", default=VERSION_ID)
    parser.add_argument("--monitor", default=MONITOR, help="métrica a vigilar (ej: val_auc)")
    parser.add_argument("--patience", type=int, default=PACIENCIA_EARLY_STOPPING)
    parser.add_argument("--epochs", type=int, default=EPOCHS_MAX)
    parser.add_argument("--reduce-lr", action="store_true", help="ReduceLROnPlateau ÷3 patience=2")
    parser.add_argument("--notas", default=NOTAS)
    args = parser.parse_args()

    VERSION_ID = args.version_id
    EPOCHS_MAX = args.epochs
    PACIENCIA_EARLY_STOPPING = args.patience
    MONITOR = args.monitor
    NOTAS = args.notas
    modo_monitor = "max" if "auc" in MONITOR or "recall" in MONITOR or "acc" in MONITOR else "min"

    tf.keras.utils.set_random_seed(SEMILLA)
    fecha = datetime.now().strftime("%Y%m%d-%H%M")
    dir_version = MODELS_DIR / f"{VERSION_ID}_{fecha}_cnn"
    dir_version.mkdir(parents=True, exist_ok=True)

    print("=== Entrenando", VERSION_ID, ARQUITECTURA, "===")
    df = cargar_manifiesto()
    train_ds = crear_pipeline(df, "train", batch_size=BATCH_SIZE, augment=True, normalizacion="rescale")
    val_ds = crear_pipeline(df, "validation", batch_size=BATCH_SIZE, normalizacion="rescale")
    test_ds = crear_pipeline(df, "test", batch_size=BATCH_SIZE, normalizacion="rescale")
    pesos = {int(k): v for k, v in pesos_de_clase(df).items()}
    print("class_weights:", pesos)

    modelo = construir_modelo()
    modelo.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=MONITOR, mode=modo_monitor, patience=PACIENCIA_EARLY_STOPPING,
            restore_best_weights=True, verbose=1,
        )
    ]
    if args.reduce_lr:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=MONITOR, mode=modo_monitor, factor=0.33, patience=2, verbose=1
            )
        )

    t0 = time.perf_counter()
    historial = modelo.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_MAX,
        class_weight=pesos,
        callbacks=callbacks,
        verbose=2,
    )
    minutos = (time.perf_counter() - t0) / 60

    # ------------------------------------------------------------------
    # Evaluación en test (el examen se rinde UNA vez)
    # ------------------------------------------------------------------
    print("Evaluando en test ...")
    y_real, prob = obtener_predicciones(modelo, test_ds)
    metricas = calcular_metricas(y_real, prob)

    config = {
        "version": VERSION_ID,
        "dataset": DATASET_ID,
        "manifiesto": "data/manifiestos/v1_splits.csv",
        "base_anterior": None,
        "arquitectura": ARQUITECTURA,
        "capas_descongeladas": None,
        "img_size": list(preprocesamiento.IMG_SIZE),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS_MAX,
        "epochs_reales": len(historial.history["loss"]),
        "learning_rate": LEARNING_RATE,
        "dropout": DROPOUT,
        "loss": "binary_crossentropy",
        "augmentacion": ["rotation_5deg", "zoom_10", "translation_10", "contrast_8"],
        "class_weights": {k: v for k, v in pesos.items()},
        "early_stopping": {"monitor": MONITOR, "mode": modo_monitor, "patience": PACIENCIA_EARLY_STOPPING},
        "reduce_lr": bool(args.reduce_lr),
        "semilla": SEMILLA,
        "notas": NOTAS,
    }

    metricas_json = {
        "version": VERSION_ID,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "arquitectura": ARQUITECTURA,
        "dataset": DATASET_ID,
        "config": config,
        "tiempo_entrenamiento_min": round(minutos, 1),
        "mejor_epoca": int(np.argmin(historial.history["val_loss"])) + 1,
        "fp32": metricas,
    }
    dir_version.joinpath("config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    dir_version.joinpath("metricas.json").write_text(json.dumps(metricas_json, indent=2, ensure_ascii=False), encoding="utf-8")

    modelo.save(dir_version / "modelo.keras")

    # Curvas de entrenamiento
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

    guardar_matriz(metricas, dir_version / "matriz_confusion.png", f"{VERSION_ID} {ARQUITECTURA} (test)")

    tamano_mb = round((dir_version / "modelo.keras").stat().st_size / 1e6, 2)
    metricas_json["fp32"]["tamano_mb"] = tamano_mb
    dir_version.joinpath("metricas.json").write_text(json.dumps(metricas_json, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"fp32": metricas, "minutos": round(minutos, 1), "tamano_mb": tamano_mb}, indent=2, ensure_ascii=False))
    print(f"OK -> {dir_version.resolve()}")


if __name__ == "__main__":
    main()
