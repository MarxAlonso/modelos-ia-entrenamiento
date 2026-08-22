import ast
import numpy as np
import pandas as pd
from pathlib import Path

RUTA_RAW = Path("data/raw_amazon/data")
RUTA_SALIDA = Path("data/amazon")
MIN_RESENAS_USUARIO = 2
MIN_RESENAS_PRODUCTO = 3


def cargar_parquets() -> pd.DataFrame:
    archivos = sorted(RUTA_RAW.glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in archivos], ignore_index=True)


def _lista_a_texto(valor) -> str:
    if isinstance(valor, str):
        try:
            valor = ast.literal_eval(valor)
        except (ValueError, SyntaxError):
            return str(valor)
    if isinstance(valor, list):
        return " ".join(str(v) for v in valor)
    if valor is None:
        return ""
    return str(valor)


def construir_productos(df: pd.DataFrame) -> pd.DataFrame:
    productos = df[df["parent_asin"].notna()].copy()
    productos["precio"] = pd.to_numeric(productos["price"], errors="coerce")
    productos["texto_producto"] = (
        productos["title_y"].fillna("").astype(str)
        + ". "
        + productos["features"].map(_lista_a_texto)
        + " "
        + productos["description"].map(_lista_a_texto)
    )
    agregados = productos.groupby("parent_asin").agg(
        titulo=("title_y", "first"),
        tienda=("store", "first"),
        precio=("precio", "mean"),
        rating_promedio=("average_rating", "first"),
        numero_resenas=("rating_number", "first"),
        texto_producto=("texto_producto", "first"),
    )
    agregados = agregados.reset_index().rename(columns={"parent_asin": "product_id"})
    agregados["product_id"] = "az_" + agregados["product_id"].astype(str)
    agregados["texto_producto"] = agregados["texto_producto"].str.slice(0, 1500)
    columnas = [
        "product_id", "titulo", "tienda", "precio",
        "rating_promedio", "numero_resenas", "texto_producto",
    ]
    return agregados[columnas]


def construir_resenas(df: pd.DataFrame) -> pd.DataFrame:
    resenas = df[
        ["user_id", "parent_asin", "rating", "title_x", "text", "timestamp", "helpful_vote", "verified_purchase"]
    ].copy()
    resenas.columns = [
        "user_id", "product_id", "rating", "titulo_resena", "texto_resena",
        "timestamp_ms", "votos_utiles", "compra_verificada",
    ]
    resenas["product_id"] = "az_" + resenas["product_id"].astype(str)
    resenas["rating"] = pd.to_numeric(resenas["rating"], errors="coerce")
    resenas["fecha"] = pd.to_datetime(
        pd.to_numeric(resenas["timestamp_ms"], errors="coerce"), unit="ms", errors="coerce"
    ).dt.date
    resenas["texto_resena"] = resenas["texto_resena"].fillna("").astype(str)
    resenas["titulo_resena"] = resenas["titulo_resena"].fillna("").astype(str)
    resenas = resenas.dropna(subset=["user_id", "product_id", "rating"])
    resenas = resenas[resenas["rating"].between(1, 5)]
    resenas["texto"] = (
        resenas["titulo_resena"].str.strip() + ". " + resenas["texto_resena"].str.strip()
    ).str.strip(". ")
    return resenas


def filtrar_nucleo(resenas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    activos_u = resenas["user_id"].value_counts()
    usuarios_nucleo = set(activos_u[activos_u >= MIN_RESENAS_USUARIO].index)
    activos_p = resenas["product_id"].value_counts()
    productos_nucleo = set(activos_p[activos_p >= MIN_RESENAS_PRODUCTO].index)
    nucleo = resenas[
        resenas["user_id"].isin(usuarios_nucleo) & resenas["product_id"].isin(productos_nucleo)
    ].copy()
    nucleo = nucleo.groupby(["user_id", "product_id"], as_index=False).agg(
        rating=("rating", "mean"),
        fecha=("fecha", "max"),
        texto=("texto", "first"),
    )
    nucleo["rating"] = nucleo["rating"].round(1)
    return nucleo, resenas


def main() -> None:
    RUTA_SALIDA.mkdir(exist_ok=True)

    print("[1/5] Cargando 3 archivos parquet...")
    raw = cargar_parquets()
    print(f"      {len(raw)} resenas brutas")

    print("[2/5] Construyendo catalogo de productos...")
    productos = construir_productos(raw)
    productos.to_csv(RUTA_SALIDA / "productos.csv", index=False)
    print(f"      {len(productos)} productos")

    print("[3/5] Construyendo resenas limpias...")
    resenas = construir_resenas(raw)
    columnas = [
        "user_id", "product_id", "rating", "fecha",
        "titulo_resena", "texto_resena", "texto", "compra_verificada",
    ]
    resenas[columnas].to_csv(RUTA_SALIDA / "resenas.csv", index=False)
    print(f"      {len(resenas)} resenas validas")

    print("[4/5] Filtrando nucleo colaborativo "
          f"(usuarios>={MIN_RESENAS_USUARIO} resenas, productos>={MIN_RESENAS_PRODUCTO})...")
    nucleo, _ = filtrar_nucleo(resenas)
    nucleo.to_csv(RUTA_SALIDA / "interacciones.csv", index=False)
    print(f"      {len(nucleo)} interacciones del nucleo")

    print("[5/5] Resumen:")
    print(f"      Usuarios unicos en nucleo: {nucleo['user_id'].nunique()}")
    print(f"      Productos unicos en nucleo: {nucleo['product_id'].nunique()}")
    densidad = len(nucleo) / (
        nucleo['user_id'].nunique() * nucleo['product_id'].nunique()
    ) * 100
    print(f"      Densidad matriz nucleo: {densidad:.2f}%")
    print(f"      Rating promedio: {resenas['rating'].mean():.2f}")
    print(f"      Longitud promedio de resena: {int(resenas['texto'].str.len().mean())} caracteres")


if __name__ == "__main__":
    main()
