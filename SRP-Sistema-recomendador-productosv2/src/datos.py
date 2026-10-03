"""
Paso 1: catalogo + particion de interacciones. Lo usan el two-tower, el SASRec y Laya,
asi los tres se evaluan sobre los mismos usuarios y los mismos candidatos.

Uso:  python src/datos.py

Particion (leave-last-out temporal, estandar en recomendacion secuencial):
  usuarios con >=2 resenas:  ultima -> test, penultima -> val (si tiene >=3), resto -> train
  usuarios con 1 resena:     5 % -> test_frio (usuario nuevo: solo se puede recomendar por
                             texto), resto -> train
Salidas en data/splits/:
  catalogo.parquet            product_id, item_idx, titulo, tienda, precio, categoria, publico, texto
  train/val/test/test_frio.parquet
  candidatos_{val,test}.npy   [n, 100] col 0 = producto real, 99 negativos fuera del historial
"""
import os

import numpy as np
import pandas as pd

from comun import (CANDIDATOS_EVAL, DATA_AZ, DATA_RAW, DATA_SPLITS, SEMILLA, categoria_de,
                   publico_de)


def construir_catalogo():
    p = pd.read_csv(os.path.join(DATA_AZ, "productos_clean.csv")).drop_duplicates("product_id")
    crudo = pd.read_csv(os.path.join(DATA_RAW, "productos.csv"), usecols=["product_id", "texto_producto"])
    p = p.merge(crudo.drop_duplicates("product_id"), on="product_id", how="left").reset_index(drop=True)
    p["item_idx"] = np.arange(len(p))
    p["categoria"] = p["titulo"].map(categoria_de)
    p["publico"] = p["titulo"].map(publico_de)
    desc = p["texto_producto"].fillna("").str.slice(0, 400)
    # Texto natural (no el *_clean en minusculas): el tokenizer del Transformer aprovecha
    # mayusculas y puntuacion.
    p["texto"] = (p["titulo"].fillna("") + ". Tienda: " + p["tienda"].fillna("") + ". " + desc).str.strip()
    return p[["product_id", "item_idx", "titulo", "tienda", "precio_final", "precio_faltante",
              "rating_promedio", "numero_resenas", "categoria", "publico", "texto"]]


def particionar(r):
    r = r.sort_values(["user_id", "fecha", "product_id"]).reset_index(drop=True)
    n = r.groupby("user_id")["item_idx"].transform("size")
    desde_final = r.groupby("user_id").cumcount(ascending=False)  # 0 = ultima
    rng = np.random.default_rng(SEMILLA)
    multi = n >= 2
    test = multi & (desde_final == 0)
    val = (n >= 3) & (desde_final == 1)
    frio = (~multi) & (rng.random(len(r)) < 0.05)
    train = ~(test | val | frio)
    return r[train], r[val], r[test], r[frio]


def candidatos(filas, train, n_items, rng):
    """Por fila: el producto real + 99 negativos que el usuario no reseno en train."""
    vistos = train.groupby("user_id")["item_idx"].agg(set).to_dict()
    salida = np.empty((len(filas), CANDIDATOS_EVAL), dtype=np.int64)
    for k, (uid, pos) in enumerate(zip(filas["user_id"], filas["item_idx"])):
        prohibidos = vistos.get(uid, set()) | {pos}
        negs = []
        while len(negs) < CANDIDATOS_EVAL - 1:
            c = int(rng.integers(n_items))
            if c not in prohibidos:
                negs.append(c); prohibidos.add(c)
        salida[k, 0], salida[k, 1:] = pos, negs
    return salida


def main():
    os.makedirs(DATA_SPLITS, exist_ok=True)
    cat = construir_catalogo()
    r = pd.read_csv(os.path.join(DATA_AZ, "interacciones_clean.csv"),
                    usecols=["user_id", "product_id", "rating", "fecha", "titulo_resena", "texto_resena"])
    r = r.merge(cat[["product_id", "item_idx"]], on="product_id", how="inner")
    r["resena"] = (r["titulo_resena"].fillna("") + ". " + r["texto_resena"].fillna("")).str.strip(". ")
    r = r.drop(columns=["titulo_resena", "texto_resena"])
    train, val, test, frio = particionar(r)

    rng = np.random.default_rng(SEMILLA)
    np.save(os.path.join(DATA_SPLITS, "candidatos_val.npy"), candidatos(val, train, len(cat), rng))
    np.save(os.path.join(DATA_SPLITS, "candidatos_test.npy"), candidatos(test, train, len(cat), rng))
    cat.to_parquet(os.path.join(DATA_SPLITS, "catalogo.parquet"), index=False)
    for nombre, df in [("train", train), ("val", val), ("test", test), ("test_frio", frio)]:
        df.to_parquet(os.path.join(DATA_SPLITS, f"{nombre}.parquet"), index=False)
        print(f"  {nombre:<10} {len(df):>8,} filas  {df['user_id'].nunique():>8,} usuarios")
    print(f"  catalogo   {len(cat):>8,} productos")
    print("  categorias:", cat["categoria"].value_counts().to_dict())
    print("  publico   :", cat["publico"].value_counts().to_dict())


if __name__ == "__main__":
    main()
