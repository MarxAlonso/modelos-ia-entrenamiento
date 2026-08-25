"""
Fase 7 — Grad-CAM: ¿dónde mira el modelo?

Genera mapas de activación del mejor modelo (v003) sobre casos representativos
del test (TP, TN, FP, FN) y un chequeo objetivo de plausibilidad médica:

    energía del heatmap en el CENTRO (campos pulmonares)
    VS energía en los BORDES (marcadores/artefactos de la placa)

Salidas:
    data/reportes/v1_gradcam/
    ├── gradcam_TP.png / TN / FP / FN      # overlays por categoría
    ├── gradcam_galeria.png                # grid resumen 4×3
    └── resumen_gradcam.json               # estadística centro vs borde

Uso:
    python src/gradcam.py --version-dir models/versiones/v003_20260823-2049_mobilenetv2_fe
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
from entrenar import obtener_predicciones  # noqa: E402
from preprocesamiento import SEMILLA, cargar_manifiesto, crear_pipeline

SALIDA_RAIZ = Path("data/reportes")
CASOS_POR_CATEGORIA = 2


def localizar_base(modelo: tf.keras.Model) -> tf.keras.Model:
    return next(l for l in modelo.layers if isinstance(l, tf.keras.Model))


def construir_modelo_cam(modelo: tf.keras.Model):
    """
    Divide el modelo cargado en dos piezas reconectables:
        cam_base : imagen -> [última conv (7×7×1280), vector de la base]
        head     : vector de la base -> probabilidad
    (Tras reload, el grafo interno de la base no es trazable desde los inputs
    externos; partir por la frontera de la base siempre funciona.)
    """
    base = localizar_base(modelo)
    ultima_conv = base.get_layer("out_relu")
    cam_base = tf.keras.Model(base.inputs, [ultima_conv.output, base.output])

    cabeza_entrada = tf.keras.Input(shape=tuple(base.output_shape[1:]))
    x = cabeza_entrada
    despues_de_base = False
    for capa in modelo.layers:
        if capa is base:
            despues_de_base = True
            continue
        if despues_de_base:
            x = capa(x)
    head = tf.keras.Model(cabeza_entrada, x)
    return cam_base, head


def gradcam(cam_base, head, img: tf.Tensor) -> tuple[np.ndarray, float]:
    """
    Grad-CAM clásico:
        1. forward hasta la última conv + predicción
        2. gradiente de la salida respecto a las activaciones conv
        3. pesos = promedio global del gradiente por canal
        4. suma ponderada de canales -> ReLU -> normalización
    """
    x = img[None, ...]
    with tf.GradientTape() as tape:
        feats, vec = cam_base(x, training=False)
        preds = head(feats, training=False)
        score = preds[:, 0]
    grads = tape.gradient(score, feats)              # (1,7,7,1280)
    pesos = tf.reduce_mean(grads, axis=(0, 1))       # (1280,)
    mapa = tf.reduce_sum(feats[0] * pesos, axis=-1)  # (7,7)
    mapa = tf.nn.relu(mapa).numpy()
    if mapa.max() > 0:
        mapa = mapa / mapa.max()
    return mapa, float(preds[0, 0])


def superponer(img: np.ndarray, mapa7x7: np.ndarray) -> np.ndarray:
    """Upsample del mapa 7×7 a 224×224 y overlay tipo jet."""
    calor = tf.image.resize(mapa7x7[..., None], (224, 224), method="bilinear").numpy()[..., 0]
    return calor


def energia_centro_vs_borde(calor: np.ndarray) -> dict:
    """Fracción de la energía del heatmap dentro de una elipse central."""
    h, w = calor.shape
    yy, xx = np.mgrid[0:h, 0:w]
    elipse = (((yy - h / 2) / (h * 0.38)) ** 2 + ((xx - w / 2) / (w * 0.45)) ** 2) <= 1.0
    total = calor.sum() + 1e-9
    return {
        "energia_centro": round(float(calor[elipse].sum() / total), 4),
        "energia_borde": round(float(calor[~elipse].sum() / total), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version-dir", required=True, type=Path)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(SEMILLA)
    SALIDA = SALIDA_RAIZ / f"gradcam_{args.version_dir.name}"
    SALIDA.mkdir(parents=True, exist_ok=True)

    print(f"=== Grad-CAM sobre {args.version_dir.name} ===")
    config = json.loads((args.version_dir / "config.json").read_text(encoding="utf-8"))
    modelo = tf.keras.models.load_model(args.version_dir / "modelo.keras")
    cam_base, head = construir_modelo_cam(modelo)

    df = cargar_manifiesto()
    test_ds = crear_pipeline(df, "test", batch_size=32, normalizacion=config["normalizacion"])
    y_real, probs = obtener_predicciones(modelo, test_ds)
    y_pred = (probs >= 0.5).astype(int)

    # Recuperar imágenes individuales del test por ruta (mismo orden que el pipeline)
    sub = df[df.split_final == "test"].reset_index(drop=True)

    categorias = {
        "TP": np.where((y_pred == 1) & (y_real == 1))[0],
        "TN": np.where((y_pred == 0) & (y_real == 0))[0],
        "FP": np.where((y_pred == 1) & (y_real == 0))[0],
        "FN": np.where((y_pred == 0) & (y_real == 1))[0],
    }
    # casos MÁS representativos: más confiantes en su categoría
    ordenar = {
        "TP": lambda idx: idx[np.argsort(-probs[idx])],
        "TN": lambda idx: idx[np.argsort(probs[idx])],
        "FP": lambda idx: idx[np.argsort(-probs[idx])],
        "FN": lambda idx: idx[np.argsort(probs[idx])],
    }

    stats_centro, stats_borde, n_mapas = [], [], 0
    imagenes_categoria: dict[str, list] = {}
    for nombre_cat, indices in categorias.items():
        seleccion = ordenar[nombre_cat](indices)[: CASOS_POR_CATEGORIA * 3]  # extra para estadística
        guardados = []
        # reconstrucción directa por ruta (determinista):
        for i, idx_global in enumerate(seleccion):
            fila = sub.iloc[idx_global]
            from preprocesamiento import RAIZ_DATASET, _decodificar

            ruta = str(RAIZ_DATASET / fila["ruta_relativa"])
            img = _decodificar(tf.constant(ruta))
            mapa, prob = gradcam(cam_base, head, img)
            calor = superponer(img.numpy(), mapa)
            e = energia_centro_vs_borde(calor)
            stats_centro.append(e["energia_centro"])
            stats_borde.append(e["energia_borde"])
            n_mapas += 1
            if i < CASOS_POR_CATEGORIA:
                guardados.append((img.numpy(), calor, float(prob), int(fila["clase"] == "PNEUMONIA")))

            fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
            axes[0].imshow(img.numpy(), cmap="gray")
            axes[0].set_title(f"original · real={fila['clase']}", fontsize=9)
            axes[1].imshow(calor, cmap="jet")
            axes[1].set_title("Grad-CAM 224²", fontsize=9)
            axes[2].imshow(img.numpy(), cmap="gray")
            axes[2].imshow(calor, cmap="jet", alpha=0.4)
            axes[2].set_title(f"overlay · p(PNEU)={prob:.3f} · pred={'PNEUMONIA' if prob>=0.5 else 'NORMAL'}", fontsize=9)
            for ax in axes:
                ax.axis("off")
            fig.suptitle(f"{nombre_cat} · caso {idx_global}", y=1.02, fontsize=10)
            fig.tight_layout()
            fig.savefig(SALIDA / f"gradcam_{nombre_cat}_{i}.png", dpi=140, bbox_inches="tight")
            plt.close(fig)
        imagenes_categoria[nombre_cat] = guardados

    # Grid resumen: filas = categorías, cols = original|calor|overlay del caso 1
    fig, axes = plt.subplots(4, 3, figsize=(11, 14))
    for r, (nombre_cat, casos) in enumerate(imagenes_categoria.items()):
        img, calor, prob, _ = casos[0]
        ejemplos = [(img, None, None), (None, calor, None), (img, calor, prob)]
        titulos = ["original", "Grad-CAM", f"overlay p={prob:.2f}"]
        for c, ax in enumerate(axes[r]):
            if c == 1:
                ax.imshow(ejemplos[c][1], cmap="jet")
            elif c == 2:
                ax.imshow(ejemplos[c][0], cmap="gray")
                ax.imshow(ejemplos[c][1], cmap="jet", alpha=0.4)
            else:
                ax.imshow(ejemplos[c][0], cmap="gray")
            ax.set_title(titulos[c] if r < 4 else "", fontsize=8)
            ax.axis("off")
        axes[r, 0].set_ylabel(nombre_cat, fontsize=12)
    fig.suptitle(f"Grad-CAM {args.version_dir.name}: TP/TN/FP/FN (caso más representativo)", y=0.995)
    fig.tight_layout()
    fig.savefig(SALIDA / "gradcam_galeria.png", dpi=130)
    plt.close(fig)

    resumen = {
        "version": args.version_dir.name,
        "mapas_generados": n_mapas,
        "energia_media_en_centro": round(float(np.mean(stats_centro)), 4),
        "energia_media_en_borde": round(float(np.mean(stats_borde)), 4),
        "interpretacion": (
            "El heatmap concentra su energía en el centro (campos pulmonares)"
            if np.mean(stats_centro) > 0.75
            else "Revisar: parte relevante del heatmap cae fuera de la zona pulmonar"
        ),
        "archivos": sorted(p.name for p in SALIDA.glob("*.png")),
    }
    SALIDA.joinpath("resumen_gradcam.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"OK -> {SALIDA.resolve()}")


if __name__ == "__main__":
    main()
