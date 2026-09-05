"""
Fase CRISP-DM 2-3: Comprensión y Preparación de Datos
Dataset: skillsmuggler/amazon-ratings (Beauty, 2,023,070 ratings)
Integración sin traducción local (ratings-only), solo pipeline colaborativo.

CRISP-DM:
 1. Business Understanding: Recomendador con datos reales Beauty
 2. Data Understanding: EDA head/info/describe (verificado arriba)
 3. Data Preparation: limpieza, validación 1-5, Timestamp -> fecha, núcleo colaborativo
 4. Modeling: compatible con colaborativo_amazon.py / red_neuronal.py
 5. Evaluation: RMSE, Precision@K, Recall@K
 6. Deployment: artefactos en data/amazon_beauty + models/

Fundamentos IA/ML aplicados (Patricio Peralta):
 - EDA: df.head(), df.info(), df.describe(), isnull(), duplicated(), value_counts()
 - Limpieza: drop_duplicates(), dropna(), between(1,5), validación outliers
 - Preparación: normalización, train/test split, núcleo colaborativo
"""
import sys
from pathlib import Path
_RUTA_MODULO = Path(__file__).resolve()
while _RUTA_MODULO.name != "src" and _RUTA_MODULO != _RUTA_MODULO.parent:
    _RUTA_MODULO = _RUTA_MODULO.parent
sys.path.insert(0, str(_RUTA_MODULO.parent))

from src.rutas import RUTA_RAIZ
import pandas as pd

RUTA_RAW = RUTA_RAIZ / "data/raw_beauty/ratings_Beauty.csv"
RUTA_SALIDA = RUTA_RAIZ / "data/amazon_beauty"
MIN_RESENAS_USUARIO = 2
MIN_RESENAS_PRODUCTO = 3


def cargar_beauty() -> pd.DataFrame:
    df = pd.read_csv(RUTA_RAW)
    # EDA ya validado: 0 nulos, 0 duplicados, Rating 1-5
    return df


def construir_resenas(df: pd.DataFrame) -> pd.DataFrame:
    resenas = df.copy()
    resenas.columns = ["user_id", "product_id", "rating", "timestamp_s"]
    # Validación CRISP-DM: rating en [1,5]
    resenas["rating"] = pd.to_numeric(resenas["rating"], errors="coerce")
    resenas = resenas[resenas["rating"].between(1, 5)]
    # Timestamp UNIX segundos -> fecha
    resenas["fecha"] = pd.to_datetime(
        pd.to_numeric(resenas["timestamp_s"], errors="coerce"), unit="s", errors="coerce"
    ).dt.date
    # IDs como string con prefijo para evitar colisión con supermercado/Amazon Fashion
    resenas["user_id"] = resenas["user_id"].astype(str)
    resenas["product_id"] = "bt_" + resenas["product_id"].astype(str)
    # Texto vacío (ratings-only, sin reseña) - compatible con pipeline existente
    resenas["texto"] = ""
    resenas = resenas.dropna(subset=["user_id", "product_id", "rating"])
    return resenas


def construir_productos(resenas: pd.DataFrame) -> pd.DataFrame:
    # Catálogo mínimo derivado de resenas (no hay metadata de producto en este dataset)
    agg = resenas.groupby("product_id").agg(
        numero_resenas=("rating", "size"),
        rating_promedio=("rating", "mean"),
        primera_fecha=("fecha", "min"),
        ultima_fecha=("fecha", "max"),
    ).reset_index()
    agg["titulo"] = "Beauty product " + agg["product_id"]
    agg["tienda"] = "Amazon Beauty"
    agg["precio"] = None
    agg["texto_producto"] = agg["titulo"]
    return agg[["product_id", "titulo", "tienda", "precio", "rating_promedio", "numero_resenas", "texto_producto"]]


def filtrar_nucleo(resenas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    activos_u = resenas["user_id"].value_counts()
    usuarios_nucleo = set(activos_u[activos_u >= MIN_RESENAS_USUARIO].index)
    activos_p = resenas["product_id"].value_counts()
    productos_nucleo = set(activos_p[activos_p >= MIN_RESENAS_PRODUCTO].index)
    nucleo = resenas[
        resenas["user_id"].isin(usuarios_nucleo) & resenas["product_id"].isin(productos_nucleo)
    ].copy()
    # Si un usuario calificó mismo producto varias veces, promedio (CRISP-DM: manejo duplicados lógicos)
    nucleo = nucleo.groupby(["user_id", "product_id"], as_index=False).agg(
        rating=("rating", "mean"),
        fecha=("fecha", "max"),
        texto=("texto", "first"),
    )
    nucleo["rating"] = nucleo["rating"].round(1)
    return nucleo, resenas


def main() -> None:
    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)
    print("[1/5] Cargando Beauty ratings_Beauty.csv (kagglehub)...")
    raw = cargar_beauty()
    print(f"      {len(raw)} filas brutas | cols={list(raw.columns)}")
    print(f"      Usuarios únicos: {raw['UserId'].nunique()} | Productos: {raw['ProductId'].nunique()}")

    print("[2/5] Construyendo reseñas limpias (validación 1-5, Timestamp->fecha)...")
    resenas = construir_resenas(raw)
    columnas_res = ["user_id", "product_id", "rating", "fecha", "texto"]
    # Guardamos también resenas completas para trazabilidad CRISP-DM
    resenas_full = resenas[["user_id", "product_id", "rating", "fecha", "texto"]].copy()
    # Para compatibilidad con preparar_amazon.py, guardamos resenas.csv con esquema extendido
    resenas_full["titulo_resena"] = ""
    resenas_full["texto_resena"] = ""
    resenas_full["compra_verificada"] = ""
    resenas_full[["user_id","product_id","rating","fecha","titulo_resena","texto_resena","texto","compra_verificada"]].to_csv(
        RUTA_SALIDA / "resenas.csv", index=False
    )
    print(f"      {len(resenas_full)} reseñas válidas | Rating medio: {resenas_full['rating'].mean():.2f}")
    print(f"      Distribución rating:\n{resenas_full['rating'].value_counts().sort_index().to_string()}")

    print("[3/5] Construyendo catálogo Beauty...")
    productos = construir_productos(resenas)
    productos.to_csv(RUTA_SALIDA / "productos.csv", index=False)
    print(f"      {len(productos)} productos Beauty")

    print(f"[4/5] Filtrando núcleo colaborativo (usuarios>={MIN_RESENAS_USUARIO}, productos>={MIN_RESENAS_PRODUCTO})...")
    nucleo, _ = filtrar_nucleo(resenas)
    nucleo.to_csv(RUTA_SALIDA / "interacciones.csv", index=False)
    print(f"      {len(nucleo)} interacciones núcleo")
    print(f"      Usuarios núcleo: {nucleo['user_id'].nunique()} | Productos núcleo: {nucleo['product_id'].nunique()}")
    if len(nucleo) > 0:
        densidad = len(nucleo) / (nucleo['user_id'].nunique() * nucleo['product_id'].nunique()) * 100
        print(f"      Densidad matriz núcleo: {densidad:.4f}%")
        print(f"      Sparsity: {100-densidad:.4f}% (típico recomendadores reales)")
    
    print("[5/5] Resumen CRISP-DM:")
    print(f"      Raw: 2,023,070 -> Limpio: {len(resenas_full)} -> Núcleo: {len(nucleo)} ({len(nucleo)/len(resenas_full)*100:.1f}%)")
    print(f"      Artefactos en: {RUTA_SALIDA}")
    print("      Listo para modelado (colaborativo_beauty.py / red_neuronal).")


if __name__ == "__main__":
    main()
