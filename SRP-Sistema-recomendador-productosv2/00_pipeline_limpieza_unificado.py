"""
PIPELINE UNIFICADO DE LIMPIEZA - Sistema Recomendador de Productos
=================================================================
Rol: Experto IA | Enfoque: simple, ejecutable y fácil de exponer en clase.

Qué hace (5 pasos, legati a tus clases aprendizajes6.md / aprendizajes7.md):
  1. EXTRAE solo 4 columnas canónicas de cada raw (datos estructurados).
  2. ESTANDARIZA a esquema único: user_id, product_id, rating, fecha.
  3. LIMPIA: nulos, duplicados user-producto, rating en [1,5], fecha válida.
  4. CONSOLIDA los 6 interim + filtro k-core (user>=2, item>=3).
  5. CATALOGA productos.csv solo con metadata REAL (no inventa títulos).

Uso:
  python 00_pipeline_limpieza_unificado.py
  # o corre cada función por celda en Jupyter para exponer.

Entradas: data/raw/*.csv, *.json
Salidas : data/interim/interacciones_*.csv, data/processed/interacciones.csv, productos.csv
"""
import os, json, html
import pandas as pd

RUTA_RAW = os.path.join("data", "raw")
RUTA_INTERIM = os.path.join("data", "interim")
RUTA_PROCESSED = os.path.join("data", "processed")

os.makedirs(RUTA_INTERIM, exist_ok=True)
os.makedirs(RUTA_PROCESSED, exist_ok=True)

# Prefijo por dominio: evita colisión de ASINs entre categorías.
# Ej: mismo "B00XYZ" en Beauty y Electronics -> bt_B00XYZ vs el_B00XYZ.
PREFIJOS = {"kindle": "kn_", "videogames": "vg_", "beauty": "bt_",
            "books": "bk_", "fashion": "af_", "electronics": "el_"}

CATEGORIA_DEFAULT = {"kn_": "Kindle Store", "vg_": "Video Games",
                     "bt_": "Beauty & Personal Care", "bk_": "Books",
                     "af_": "Clothing, Shoes & Jewelry", "el_": "Electronics"}


def limpiar_base(df, col_fecha="fecha"):
    """Limpieza común: nulos, duplicados, rating [1,5], fecha YYYY-MM-DD."""
    df = df.dropna(subset=["user_id", "product_id", "rating"])
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].clip(1, 5)  # corrige 0 o 6 accidentales
    if col_fecha in df.columns:
        df[col_fecha] = pd.to_datetime(df[col_fecha], errors="coerce").dt.strftime("%Y-%m-%d")
        df = df.dropna(subset=[col_fecha])
        df = df[df[col_fecha] >= "1990-01-01"]  # descarta unix corruptos (1969)
    df = df.drop_duplicates(subset=["user_id", "product_id"], keep="last")
    return df[["user_id", "product_id", "rating", "fecha"]]


def limpiar_texto(s: str) -> str:
    """Para rama NLP: decodifica &amp;, mojibake leve, minúsculas, espacios."""
    if pd.isna(s):
        return ""
    s = html.unescape(str(s))
    s = s.replace("Ã¢Â", "'").replace("â€™", "'").replace("â€œ", '"')
    return " ".join(str(s).lower().split())

# ---------- 1-6: un extractor por dominio (solo columnas necesarias) ----------

def proceso_kindle():
    df = pd.read_csv(os.path.join(RUTA_RAW, "all_kindle_review .csv"),
                     usecols=["reviewerID", "asin", "rating", "reviewTime"])
    df = df.rename(columns={"reviewerID": "user_id", "asin": "product_id", "reviewTime": "fecha"})
    df["product_id"] = PREFIJOS["kindle"] + df["product_id"].astype(str)
    df = limpiar_base(df)
    df.to_csv(os.path.join(RUTA_INTERIM, "interacciones_kindle.csv"), index=False)
    print(f"Kindle: {len(df):,} filas"); return df


def proceso_beauty():
    df = pd.read_csv(os.path.join(RUTA_RAW, "ratings_Beauty.csv"),
                     usecols=["UserId", "ProductId", "Rating", "Timestamp"])
    df = df.rename(columns={"UserId": "user_id", "ProductId": "product_id",
                             "Rating": "rating", "Timestamp": "fecha"})
    df["product_id"] = PREFIJOS["beauty"] + df["product_id"].astype(str)
    df["fecha"] = pd.to_datetime(df["fecha"], unit="s", errors="coerce")
    df = limpiar_base(df)
    df.to_csv(os.path.join(RUTA_INTERIM, "interacciones_beauty.csv"), index=False)
    print(f"Beauty: {len(df):,} filas"); return df


def proceso_books():
    # Books_rating.csv ~2.8GB: leer por chunks para no saturar RAM
    partes = []
    for ch in pd.read_csv(os.path.join(RUTA_RAW, "Books_rating.csv"),
                          usecols=["User_id", "Id", "review/score", "review/time"],
                          chunksize=200000):
        ch = ch.rename(columns={"User_id": "user_id", "Id": "product_id",
                                "review/score": "rating", "review/time": "fecha"})
        ch["product_id"] = PREFIJOS["books"] + ch["product_id"].astype(str)
        ch["fecha"] = pd.to_datetime(ch["fecha"], unit="s", errors="coerce")
        ch = limpiar_base(ch)
        partes.append(ch)
    df = pd.concat(partes, ignore_index=True).drop_duplicates(subset=["user_id", "product_id"])
    df.to_csv(os.path.join(RUTA_INTERIM, "interacciones_books.csv"), index=False)
    print(f"Books: {len(df):,} filas"); return df


def proceso_fashion():
    df = pd.read_csv(os.path.join(RUTA_RAW, "amazon-fashion-800k+-user-reviews-dataset.csv"),
                     usecols=["user_id", "parent_asin", "rating", "timestamp"])
    df = df.rename(columns={"parent_asin": "product_id", "timestamp": "fecha"})
    df["product_id"] = PREFIJOS["fashion"] + df["product_id"].astype(str)
    df["fecha"] = pd.to_datetime(df["fecha"], unit="ms", errors="coerce")  # viene en ms
    df = limpiar_base(df)
    df.to_csv(os.path.join(RUTA_INTERIM, "interacciones_fashion.csv"), index=False)
    print(f"Fashion: {len(df):,} filas"); return df


def _json_a_interim(nombre_json, prefijo, salida):
    regs = []
    with open(os.path.join(RUTA_RAW, nombre_json), encoding="utf-8") as f:
        for linea in f:
            r = json.loads(linea)
            regs.append({"user_id": r.get("reviewerID"),
                         "product_id": prefijo + str(r.get("asin")),
                         "rating": r.get("overall"),
                         "fecha": pd.to_datetime(r.get("unixReviewTime"), unit="s", errors="coerce")})
    df = limpiar_base(pd.DataFrame(regs))
    df.to_csv(os.path.join(RUTA_INTERIM, salida), index=False)
    print(f"{salida}: {len(df):,} filas"); return df


def proceso_videogames():
    return _json_a_interim("Video_Games_5.json", PREFIJOS["videogames"], "interacciones_videogames.csv")


def proceso_electronics():
    return _json_a_interim("Electronics_5.json", PREFIJOS["electronics"], "interacciones_electronics.csv")

# ---------- 7: consolidar + k-core ----------

def consolidar_kcore(min_user=2, min_item=3):
    """Une 6 interim y aplica k-core iterativo (quita cold-start extremo).
    Por qué (aprendizajes7.md): un usuario con 1 rating o un item con 1-2 ratings
    no generaliza -> overfitting / matriz ultra-dispersa. k-core lo poda."""
    archivos = ["interacciones_kindle.csv", "interacciones_videogames.csv",
                "interacciones_beauty.csv", "interacciones_books.csv",
                "interacciones_fashion.csv", "interacciones_electronics.csv"]
    dfs = [pd.read_csv(os.path.join(RUTA_INTERIM, a)) for a in archivos
           if os.path.exists(os.path.join(RUTA_INTERIM, a))]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Combinado pre k-core: {len(df):,}")
    while True:
        antes = len(df)
        u_ok = df["user_id"].value_counts(); u_ok = u_ok[u_ok >= min_user].index
        i_ok = df["product_id"].value_counts(); i_ok = i_ok[i_ok >= min_item].index
        df = df[df["user_id"].isin(u_ok) & df["product_id"].isin(i_ok)]
        print(f"  k-core iter: {len(df):,}")
        if len(df) == antes:
            break
    df.to_csv(os.path.join(RUTA_PROCESSED, "interacciones.csv"), index=False)
    print(f"OK interacciones.csv | users={df.user_id.nunique():,} items={df.product_id.nunique():,}")
    return df

# ---------- 8: catálogo honesto (NO inventar títulos) ----------

def catalogar(df_inter):
    """Solo Books tiene título/descripción real (books_data.csv).
    Beauty: categoría real + título desde slug URL (parcial).
    Kindle/VG/Electronics/Fashion: placeholder honesto 'Producto <id>'.
    ERROR que corrige de intento3: Fashion usaba el TÍTULO DE LA RESEÑA
    ('DON\\'T BUY THIS PRODUCT!') como nombre del producto. Eso contamina."""
    stats = df_inter.groupby("product_id").agg(
        rating_promedio=("rating", "mean"), numero_resenas=("rating", "count")).reset_index()
    stats["rating_promedio"] = stats["rating_promedio"].round(2)
    validos = set(stats["product_id"])
    meta = {}

    # Books: puente Id->Title + metadata (mejor dominio)
    id_a_titulo = {}
    for ch in pd.read_csv(os.path.join(RUTA_RAW, "Books_rating.csv"),
                          usecols=["Id", "Title"], chunksize=200000):
        for _, f in ch.dropna(subset=["Id", "Title"]).iterrows():
            pid = "bk_" + str(f["Id"])
            if pid in validos and pid not in id_a_titulo:
                id_a_titulo[pid] = str(f["Title"])
    m = pd.read_csv(os.path.join(RUTA_RAW, "books_data.csv"),
                    usecols=["Title", "description", "categories"]).dropna(subset=["Title"])
    m = m.drop_duplicates("Title")
    d_desc, d_cat = dict(zip(m.Title, m.description)), dict(zip(m.Title, m.categories))
    for pid, tit in id_a_titulo.items():
        meta[pid] = {"titulo": tit, "categoria": str(d_cat.get(tit) or "Books"),
                     "descripcion": limpiar_texto(d_desc.get(tit) or "")}
    del m, id_a_titulo
    print(f"Books catalogados: {len(meta):,}")

    # Beauty: ProductType + slug URL (solo los que existan, resto fallback)
    try:
        b = pd.read_csv(os.path.join(RUTA_RAW, "Amazon_Beauty_Recommendation.csv"),
                        usecols=["ProductId", "ProductType", "URL"]).drop_duplicates("ProductId")
        for _, f in b.iterrows():
            pid = "bt_" + str(f["ProductId"])
            if pid in validos and pid not in meta:
                slug = str(f["URL"]).split("/")[3].replace("-", " ").strip().title() \
                    if "/" in str(f["URL"]) else str(f["ProductType"]).title()
                meta[pid] = {"titulo": slug,
                             "categoria": f"Beauty & Personal Care - {str(f['ProductType']).title()}",
                             "descripcion": ""}
        del b
    except Exception as e:
        print("Beauty meta omitida:", e)
    print(f"Total con metadata real: {len(meta):,} / {len(validos):,}")

    # Ensamble final con fallback honesto
    out = []
    for _, r in stats.iterrows():
        pid = r["product_id"]
        mm = meta.get(pid, {})
        out.append({"product_id": pid,
                    "titulo": mm.get("titulo", f"Producto {pid}"),
                    "categoria": mm.get("categoria", CATEGORIA_DEFAULT.get(pid[:3], "General")),
                    "rating_promedio": r["rating_promedio"],
                    "numero_resenas": r["numero_resenas"],
                    "descripcion": mm.get("descripcion", "")})
    dfp = pd.DataFrame(out)
    dfp.to_csv(os.path.join(RUTA_PROCESSED, "productos.csv"), index=False)
    print(f"OK productos.csv: {len(dfp):,} | sin_descripcion={(dfp.descripcion=='').sum():,}")
    return dfp


def auditar():
    di = pd.read_csv(os.path.join(RUTA_PROCESSED, "interacciones.csv"))
    dp = pd.read_csv(os.path.join(RUTA_PROCESSED, "productos.csv"))
    print(f"duplicados_prod={dp.product_id.duplicated().sum()} "
          f"| huerfanos={len(set(di.product_id)-set(dp.product_id))} "
          f"| rating=[{di.rating.min()},{di.rating.max()}] "
          f"| min_user={di.groupby('user_id').size().min()} "
          f"| min_item={di.groupby('product_id').size().min()}")
    print(dp["categoria"].value_counts().head(10).to_string())
    print(di["product_id"].str[:3].value_counts().to_string())


if __name__ == "__main__":
    proceso_kindle(); proceso_beauty(); proceso_books()
    proceso_fashion(); proceso_videogames(); proceso_electronics()
    df = consolidar_kcore()
    catalogar(df)
    auditar()
