import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_USERS = 500

RUTA_RAW = Path("data/raw/train/metadata.csv")
RUTA_SALIDA = Path("data")


def limpiar_productos() -> pd.DataFrame:
    df = pd.read_csv(RUTA_RAW)
    df["product_id"] = [f"p{i:04d}" for i in range(len(df))]
    df["ai_generated_description"] = df["ai_generated_description"].fillna("")
    df["texto"] = df["name"] + ". " + np.where(
        df["ai_generated_description"] == "",
        "Producto de " + df["subcategory"].str.replace("-", " ", regex=False),
        df["ai_generated_description"],
    )
    columnas = [
        "product_id", "file_name", "name", "supermarket_category",
        "main_category", "subcategory", "texto",
    ]
    return df[columnas]


def generar_usuarios(categorias: list[str], rng: np.random.Generator) -> pd.DataFrame:
    filas = []
    for i in range(N_USERS):
        n_favoritas = rng.integers(1, 4)
        favoritas = rng.choice(categorias, size=n_favoritas, replace=False)
        filas.append({
            "user_id": f"u{i:04d}",
            "categorias_favoritas": "|".join(favoritas),
        })
    return pd.DataFrame(filas)


def generar_interacciones(
    productos: pd.DataFrame,
    usuarios: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    pesos_categoria = (
        productos.groupby("main_category")
        .size()
        .reindex(productos["main_category"].cat.categories)
    ) if False else productos["main_category"].value_counts()

    indices_por_categoria = {
        cat: grupo.index.to_numpy()
        for cat, grupo in productos.groupby("main_category")
    }
    categorias = productos["main_category"].unique()
    fecha_base = pd.Timestamp("2026-01-01")

    filas = []
    for _, usuario in usuarios.iterrows():
        favoritas = usuario["categorias_favoritas"].split("|")
        n_compras = int(rng.integers(20, 81))
        for _ in range(n_compras):
            if rng.random() < 0.75:
                categoria = rng.choice(favoritas)
            else:
                categoria = rng.choice(categorias)
            candidatos = indices_por_categoria[categoria]
            idx = rng.choice(candidatos)
            producto = productos.loc[idx]
            es_favorita = categoria in favoritas
            rating = float(np.clip(
                rng.normal(4.3 if es_favorita else 3.0, 0.6), 1.0, 5.0
            ))
            dias = int(rng.integers(0, 180))
            filas.append({
                "user_id": usuario["user_id"],
                "product_id": producto["product_id"],
                "categoria_producto": categoria,
                "rating": round(rating, 1),
                "cantidad": int(rng.integers(1, 4)),
                "fecha": fecha_base + pd.Timedelta(days=dias),
            })
    return pd.DataFrame(filas)


def main() -> None:
    rng = np.random.default_rng(SEED)

    print("[1/4] Limpiando catalogo de productos...")
    productos = limpiar_productos()
    productos.to_csv(RUTA_SALIDA / "productos.csv", index=False)
    print(f"      {len(productos)} productos guardados")

    print("[2/4] Generando usuarios con preferencias...")
    usuarios = generar_usuarios(sorted(productos["main_category"].unique()), rng)
    usuarios.to_csv(RUTA_SALIDA / "usuarios.csv", index=False)
    print(f"      {len(usuarios)} usuarios guardados")

    print("[3/4] Generando historial de compras...")
    interacciones = generar_interacciones(productos, usuarios, rng)
    interacciones.to_csv(RUTA_SALIDA / "interacciones.csv", index=False)
    print(f"      {len(interacciones)} compras guardadas")

    print("[4/4] Resumen:")
    densidad = len(interacciones) / (N_USERS * len(productos)) * 100
    print(f"      Densidad matriz usuario-producto: {densidad:.2f}%")
    print(f"      Rating promedio: {interacciones['rating'].mean():.2f}")
    print(f"      Compras en categoria favorita: "
          f"{interacciones.apply(lambda r: r['categoria_producto'] in usuarios.set_index('user_id').loc[r['user_id'], 'categorias_favoritas'], axis=1).mean():.0%}")
    print("      Listo.")


if __name__ == "__main__":
    main()
