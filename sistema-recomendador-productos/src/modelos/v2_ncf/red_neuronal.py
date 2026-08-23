
import sys
from pathlib import Path
_RUTA_MODULO = Path(__file__).resolve()
while _RUTA_MODULO.name != "src" and _RUTA_MODULO != _RUTA_MODULO.parent:
    _RUTA_MODULO = _RUTA_MODULO.parent
sys.path.insert(0, str(_RUTA_MODULO.parent))

from src.rutas import RUTA_RAIZ
import os
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
import tensorflow as tf
from sklearn.decomposition import TruncatedSVD, PCA

from src.modelos.v1_colaborativo_svd_knn.colaborativo import construir_indices, evaluar_ranking

SEED = 42
DIM_EMBEDDING = 32
DIM_CONTENIDO = 64
N_NEGATIVOS = 4
EPOCHS = 30
BATCH = 512
TOP_K = 10
UMBRAL_RELEVANTE = 4.0

RUTA_DATOS = RUTA_RAIZ / "data"
RUTA_MODELOS = RUTA_RAIZ / "models"


def fijar_semillas() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    interacciones = pd.read_csv(RUTA_DATOS / "interacciones.csv")
    productos_info = pd.read_csv(
        RUTA_DATOS / "productos.csv",
        usecols=["product_id", "name", "main_category"],
    )
    return interacciones, productos_info


def construir_modelo(n_usuarios: int, n_productos: int) -> tf.keras.Model:
    entrada_usuario = tf.keras.Input(shape=(1,), name="usuario", dtype=tf.int32)
    entrada_producto = tf.keras.Input(shape=(1,), name="producto", dtype=tf.int32)
    entrada_contenido = tf.keras.Input(
        shape=(DIM_CONTENIDO,), name="contenido_espanol", dtype=tf.float32
    )

    emb_usuario = tf.keras.layers.Flatten()(
        tf.keras.layers.Embedding(
            n_usuarios + 1, DIM_EMBEDDING, name="embedding_usuario"
        )(entrada_usuario)
    )
    emb_producto = tf.keras.layers.Flatten()(
        tf.keras.layers.Embedding(
            n_productos + 1, DIM_EMBEDDING, name="embedding_producto"
        )(entrada_producto)
    )

    sesgo_usuario = tf.keras.layers.Flatten()(
        tf.keras.layers.Embedding(
            n_usuarios + 1, 1,
            embeddings_initializer="zeros", name="sesgo_usuario",
        )(entrada_usuario)
    )
    sesgo_producto = tf.keras.layers.Flatten()(
        tf.keras.layers.Embedding(
            n_productos + 1, 1,
            embeddings_initializer="zeros", name="sesgo_producto",
        )(entrada_producto)
    )

    gmf = tf.keras.layers.Multiply(name="interaccion_gmf")([emb_usuario, emb_producto])
    rama_mlp = tf.keras.layers.Concatenate(name="concatenar")([emb_usuario, emb_producto])
    rama_mlp = tf.keras.layers.Dense(64, activation="relu", name="oculta_1")(rama_mlp)
    rama_mlp = tf.keras.layers.Dropout(0.2, name="dropout")(rama_mlp)
    rama_mlp = tf.keras.layers.Dense(32, activation="relu", name="oculta_2")(rama_mlp)

    contenido = tf.keras.layers.Dense(32, activation="relu", name="texto_espanol")(
        entrada_contenido
    )

    fusion = tf.keras.layers.Concatenate(name="fusion")([
        gmf, rama_mlp, contenido,
    ])
    fusion = tf.keras.layers.Dense(16, activation="relu", name="fusion_oculta")(fusion)
    logit = tf.keras.layers.Dense(
        1,
        name="logit",
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer="zeros",
    )(fusion)

    salida = tf.keras.layers.Add(name="sumar_logit")([logit, sesgo_usuario, sesgo_producto])
    salida = tf.keras.layers.Activation("sigmoid", name="probabilidad_compra")(salida)

    modelo = tf.keras.Model(
        inputs=[entrada_usuario, entrada_producto, entrada_contenido], outputs=salida,
        name="recomendador_ncf",
    )
    modelo.compile(optimizer=tf.keras.optimizers.Adam(0.002), loss="binary_crossentropy")
    return modelo


def inicializar_con_word2vec(
    modelo: tf.keras.Model,
    productos_unicos: list[str],
    compradas_train: list[set[int]],
) -> float:
    """Inicializa embeddings de la red con vectores Word2Vec proyectados.

    Cada producto arranca con su vector semantico (PCA 100D->32D de Word2Vec)
    y cada usuario con el promedio de los vectores de sus compras en train.
    La escala se ajusta al orden de magnitud de la inicializacion aleatoria
    de Keras para no saturar las activaciones al comienzo del entrenamiento.
    """
    datos = np.load(RUTA_MODELOS / "embeddings_word2vec.npz", allow_pickle=True)
    w2v = datos["embeddings"].astype(np.float32)
    mapa_pid = {pid: i for i, pid in enumerate(datos["product_ids"].tolist())}

    pca = PCA(n_components=DIM_EMBEDDING, random_state=SEED)
    proyectado = pca.fit_transform(w2v).astype(np.float32)
    varianza = float(pca.explained_variance_ratio_.sum())
    proyectado *= 0.03 / (proyectado.std() + 1e-9)

    filas = np.array([mapa_pid.get(pid, -1) for pid in productos_unicos])
    matriz_producto = np.zeros((len(productos_unicos) + 1, DIM_EMBEDDING), dtype=np.float32)
    validos = filas >= 0
    matriz_producto[:len(productos_unicos)][validos] = proyectado[filas[validos]]

    proy_catalogo = matriz_producto[:len(productos_unicos)]
    matriz_usuario = np.zeros((len(compradas_train) + 1, DIM_EMBEDDING), dtype=np.float32)
    for u, items in enumerate(compradas_train):
        if items:
            matriz_usuario[u] = proy_catalogo[sorted(items)].mean(axis=0)

    modelo.get_layer("embedding_producto").set_weights([matriz_producto])
    modelo.get_layer("embedding_usuario").set_weights([matriz_usuario])
    return varianza


def generar_pares_entrenamiento(
    x_usuario: np.ndarray,
    x_producto: np.ndarray,
    compradas_train: list[set[int]],
    n_productos: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    usuarios_pos = []
    productos_pos = []
    for u, p in zip(x_usuario, x_producto):
        negativos = []
        while len(negativos) < N_NEGATIVOS:
            candidato = int(rng.integers(n_productos))
            if candidato not in compradas_train[u] and candidato not in negativos:
                negativos.append(candidato)
        usuarios_pos.extend([u] * (1 + len(negativos)))
        productos_pos.append(p)
        productos_pos.extend(negativos)

    usuarios_totales = np.array(usuarios_pos, dtype=np.int32)
    productos_totales = np.array(productos_pos, dtype=np.int32)
    etiquetas = np.zeros(len(usuarios_totales), dtype=np.float32)
    posiciones_positivas = np.arange(0, len(usuarios_totales), 1 + N_NEGATIVOS)
    etiquetas[posiciones_positivas] = 1.0

    orden = rng.permutation(len(usuarios_totales))
    return (
        usuarios_totales[orden],
        productos_totales[orden],
        etiquetas[orden],
    )


def main() -> None:
    RUTA_MODELOS.mkdir(exist_ok=True)
    fijar_semillas()

    print("[1/10] Cargando datos del dominio en espanol...")
    interacciones, productos_info = cargar_datos()
    usuarios, productos_unicos, idx_usuario, idx_producto = construir_indices(interacciones)
    forma = (len(usuarios), len(productos_unicos))
    print(f"      {len(interacciones)} compras | {forma[0]} usuarios | {forma[1]} productos")

    print("[2/10] Division train/test 80/20 (misma semilla que la Fase 3)...")
    rng = np.random.default_rng(SEED)
    mascara = rng.random(len(interacciones)) < 0.8
    train = interacciones[mascara].reset_index(drop=True)
    test = interacciones[~mascara].reset_index(drop=True)
    print(f"      train={len(train)} | test={len(test)}")

    x_usuario = train["user_id"].map(idx_usuario).to_numpy(dtype=np.int32)
    x_producto = train["product_id"].map(idx_producto).to_numpy(dtype=np.int32)

    compradas_train = [set() for _ in range(forma[0])]
    for u, p in zip(x_usuario, x_producto):
        compradas_train[u].add(p)

    print(f"[3/10] Reduciendo TF-IDF en espanol a {DIM_CONTENIDO} dimensiones (LSA)...")
    tfidf = sparse.load_npz(RUTA_MODELOS / "tfidf_matrix.npz")
    lsa = TruncatedSVD(n_components=DIM_CONTENIDO, random_state=SEED)
    contenido_productos = lsa.fit_transform(tfidf).astype(np.float32)
    filas_catalogo = np.array([int(pid[1:]) for pid in productos_unicos])
    contenido_interacciones = contenido_productos[filas_catalogo]
    varianza_texto = lsa.explained_variance_ratio_.sum()
    print(f"      Matriz de contenido: {contenido_productos.shape} | Varianza explicada: {varianza_texto:.1%}")

    print(f"[4/10] Generando pares de entrenamiento (+{N_NEGATIVOS} negativos por compra)...")
    ent_u, ent_p, ent_y = generar_pares_entrenamiento(
        x_usuario, x_producto, compradas_train, forma[1], rng
    )
    positivos = int(ent_y.sum())
    print(f"      Total pares: {len(ent_y)} ({positivos} compras reales + "
          f"{len(ent_y) - positivos} no-compras muestreadas)")

    print("[5/10] Construyendo red NCF hibrida (embeddings + texto espanol)...")
    modelo = construir_modelo(forma[0], forma[1])

    ruta_w2v = RUTA_MODELOS / "embeddings_word2vec.npz"
    if not ruta_w2v.exists():
        raise FileNotFoundError(
            "Falta models/embeddings_word2vec.npz. Ejecute primero: "
            "python src/entrenar_word2vec.py"
        )
    varianza_w2v = inicializar_con_word2vec(modelo, productos_unicos, compradas_train)
    print(f"[6/10] Embeddings inicializados con Word2Vec "
          f"(PCA 100D->{DIM_EMBEDDING}D, varianza explicada {varianza_w2v:.1%})")
    print("       Productos: vector semantico propio | Usuarios: perfil de sus compras")

    print("[7/10] Entrenando con perdida binaria y negativos frescos por epoca...")
    historial_final = None
    for epoca in range(EPOCHS):
        e_u, e_p, e_y = generar_pares_entrenamiento(
            x_usuario, x_producto, compradas_train, forma[1], rng
        )
        h = modelo.fit(
            [e_u, e_p, contenido_interacciones[e_p]],
            e_y,
            validation_split=0.1,
            epochs=1,
            batch_size=BATCH,
            verbose=0,
        )
        historial_final = h
        print(
            f"      Epoca {epoca + 1:>2}/{EPOCHS} | "
            f"perdida={h.history['loss'][0]:.4f} | val={h.history['val_loss'][0]:.4f}"
        )

    f_te = test["user_id"].map(idx_usuario).to_numpy(dtype=np.int32)
    c_te = test["product_id"].map(idx_producto).to_numpy(dtype=np.int32)
    reales = test["rating"].to_numpy()

    print(f"[8/10] Generando puntajes para todo el catalogo ({forma[0]}x{forma[1]})...")
    puntajes = np.empty(forma, dtype=np.float32)
    todos_p = np.arange(forma[1], dtype=np.int32)
    for u in range(forma[0]):
        usuarios_lote = np.full(forma[1], u, dtype=np.int32)
        puntajes[u] = modelo.predict(
            [usuarios_lote, todos_p, contenido_interacciones], batch_size=65536, verbose=0
        ).ravel()

    relevantes_test: dict[int, set[int]] = {}
    for u, p, r in zip(f_te, c_te, reales):
        if r >= UMBRAL_RELEVANTE:
            relevantes_test.setdefault(u, set()).add(p)

    print(f"[9/10] Metricas de ranking (relevante = rating>={UMBRAL_RELEVANTE})...")
    precision_at_k, recall_at_k = evaluar_ranking(puntajes, compradas_train, relevantes_test)
    pop = np.tile(
        np.bincount(x_producto, minlength=forma[1]).astype(np.float32), (forma[0], 1)
    )
    precision_pop, _ = evaluar_ranking(pop, compradas_train, relevantes_test)

    ruta_metricas_previas = RUTA_MODELOS / "red_neuronal_metricas.pkl"
    precision_anterior = None
    if ruta_metricas_previas.exists():
        precision_anterior = joblib.load(ruta_metricas_previas).get("precision_at_k")

    fase3 = joblib.load(RUTA_MODELOS / "colaborativo_svd.pkl")
    comparacion = pd.DataFrame(
        {
            "Modelo": ["Popularidad (base)", "KNN item-item (Fase 3)", "Red neuronal NCF (Fase 4)"],
            f"Precision@{TOP_K}": [
                round(precision_pop, 4),
                round(float(fase3["precision_at_k"]), 4),
                round(precision_at_k, 4),
            ],
            f"Recall@{TOP_K}": [
                "-",
                round(float(fase3["recall_at_k"]), 4),
                round(recall_at_k, 4),
            ],
        }
    )
    print(comparacion.to_string(index=False))

    if precision_anterior is not None:
        mejora = (precision_at_k - precision_anterior) / max(precision_anterior, 1e-9) * 100
        print(f"\n      Comparacion de inicializaciones (misma semilla y datos):")
        print(f"      NCF init aleatoria : P@{TOP_K}={precision_anterior:.4f}")
        print(f"      NCF init Word2Vec  : P@{TOP_K}={precision_at_k:.4f} ({mejora:+.1f}%)")

    print("[10/10] DEMO en espanol - Top-5 para u0007:")
    u_demo = idx_usuario["u0007"]
    fila_demo = puntajes[u_demo].copy()
    fila_demo[list(compradas_train[u_demo])] = -np.inf
    top = np.argsort(fila_demo)[::-1][:5]
    inv_prod = {v: k for k, v in idx_producto.items()}
    info = productos_info.set_index("product_id")
    favoritas = pd.read_csv(RUTA_DATOS / "usuarios.csv").set_index("user_id").loc[
        "u0007", "categorias_favoritas"
    ]
    print(f"\nPreferencias declaradas de u0007: {favoritas}\n")
    for rank, col in enumerate(top, start=1):
        pid = inv_prod[col]
        nombre = info.loc[pid, "name"]
        categoria = info.loc[pid, "main_category"]
        print(f"  {rank}. [{fila_demo[col]:.2f}] ({categoria}) {nombre}")

    print("\nGuardando artefactos...")
    modelo.save(RUTA_MODELOS / "red_neuronal.keras")
    np.savez_compressed(
        RUTA_MODELOS / "predicciones_red_neuronal.npz",
        matriz=puntajes,
        usuarios=np.array(usuarios),
        productos=np.array(productos_unicos),
    )
    joblib.dump(
        {
            "idx_usuario": idx_usuario,
            "idx_producto": idx_producto,
            "dim_embedding": DIM_EMBEDDING,
            "dim_contenido": DIM_CONTENIDO,
            "n_negativos": N_NEGATIVOS,
            "epocas_entrenadas": EPOCHS,
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "precision_popularidad": precision_pop,
            "varianza_texto": varianza_texto,
            "inicializacion_word2vec": True,
            "varianza_word2vec_pca": varianza_w2v,
        },
        RUTA_MODELOS / "red_neuronal_metricas.pkl",
    )
    print("Artefactos: models/red_neuronal.keras, models/red_neuronal_metricas.pkl,")
    print("            models/predicciones_red_neuronal.npz")
    print("Fase 4 completada (NCF inicializada con Word2Vec).")


if __name__ == "__main__":
    main()
