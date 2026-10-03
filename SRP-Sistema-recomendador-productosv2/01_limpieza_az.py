"""
LIMPIEZA AZ - Recomendador híbrido ML + DL
==========================================
Universo: data/raw/resenas.csv (204k) + productos.csv (71k)
Salida  : data/processed_az/interacciones_clean.csv, productos_clean.csv, usuarios_clean.csv

Mapeo al proyecto:
- historial compras -> compra_verificada + fecha
- preferencias      -> rating + texto_clean (para TF-IDF / embeddings DL)
- comportamiento    -> frecuencia, recencia, tasa_verificada (usuarios_clean.csv)

Uso: python 01_limpieza_az.py
"""
import os, html, re
import pandas as pd

RAW = os.path.join("data", "raw")
OUT = os.path.join("data", "processed_az")
os.makedirs(OUT, exist_ok=True)


def limpiar_texto(s: str) -> str:
    """Texto para ML (TF-IDF) y DL (embeddings): minúsculas, sin HTML, sin ruido."""
    if pd.isna(s):
        return ""
    s = html.unescape(str(s))          # &amp; -> &
    s = s.lower()
    s = re.sub(r"http\S+|www\S+", " ", s)
    s = re.sub(r"[^a-záéíóúñü0-9\s]", " ", s)
    return " ".join(s.split())


print("1/4 Leyendo raw...")
r = pd.read_csv(os.path.join(RAW, "resenas.csv"))
p = pd.read_csv(os.path.join(RAW, "productos.csv"))
print(f"  resenas={len(r):,} users={r.user_id.nunique():,} items={r.product_id.nunique():,}")
print(f"  productos={len(p):,}")

print("2/4 Limpieza interacciones (historial + preferencias)...")
r["titulo_resena"] = r["titulo_resena"].fillna("")
r["texto_resena"] = r["texto_resena"].fillna("")
r["texto"] = (r["titulo_resena"] + ". " + r["texto_resena"]).str.strip()
r["texto_clean"] = r["texto"].apply(limpiar_texto)
r = r[r["texto_clean"].str.len() >= 3]                       # sin texto no hay contenido
r["rating"] = pd.to_numeric(r["rating"], errors="coerce").clip(1, 5)
r["fecha"] = pd.to_datetime(r["fecha"], errors="coerce")
r = r.dropna(subset=["user_id", "product_id", "rating", "fecha"])
r = r.drop_duplicates(subset=["user_id", "product_id"], keep="last")  # 1,529 dups
r["fecha"] = r["fecha"].dt.strftime("%Y-%m-%d")
r["compra_verificada"] = r["compra_verificada"].astype(str).str.lower().isin(["true", "1"])
print(f"  tras limpieza: {len(r):,} (rating 5*={ (r.rating==5).mean():.0%} -> sesgo a tratar en ML)")

print("3/4 Limpieza productos (catálogo)...")
p["titulo"] = p["titulo"].fillna("Producto sin titulo")
p["tienda"] = p["tienda"].fillna("Sin tienda").str.strip()
p["texto_producto"] = p["texto_producto"].fillna("")
p["titulo_clean"] = p["titulo"].apply(limpiar_texto)
p["texto_producto_clean"] = p["texto_producto"].apply(limpiar_texto)
mediana_global = p["precio"].median()                        # 90% nulo: no es feature core
p["precio_mediana_tienda"] = p.groupby("tienda")["precio"].transform("median")
p["precio_final"] = p["precio"].fillna(p["precio_mediana_tienda"]).fillna(mediana_global)
p["precio_faltante"] = p["precio"].isna().astype(int)        # flag para el modelo
print(f"  precio nulo={p['precio'].isna().mean():.0%} -> imputado + flag")

print("4/4 Usuarios (comportamiento) + guardado...")
u = r.groupby("user_id").agg(num_resenas=("rating", "count"),
                             rating_promedio=("rating", "mean"),
                             tasa_verificada=("compra_verificada", "mean"),
                             ultima_fecha=("fecha", "max")).reset_index()
r.to_csv(os.path.join(OUT, "interacciones_clean.csv"), index=False)
p[["product_id", "titulo", "titulo_clean", "tienda", "precio_final",
   "precio_faltante", "rating_promedio", "numero_resenas",
   "texto_producto_clean"]].to_csv(os.path.join(OUT, "productos_clean.csv"), index=False)
u.to_csv(os.path.join(OUT, "usuarios_clean.csv"), index=False)

# Subset denso SOLO para colaborativo (el full se usa para contenido/DL)
denso = r.groupby("user_id").filter(lambda x: len(x) >= 3)
denso = denso[denso["product_id"].isin(denso["product_id"].value_counts()[lambda s: s >= 5].index)]
denso.to_csv(os.path.join(OUT, "interacciones_denso.csv"), index=False)
print(f"OK -> {OUT}/ | full={len(r):,} denso_colab={len(denso):,} "
      f"users_denso={denso.user_id.nunique():,} (el 91% tiene 1 resena: por eso el hibrido)")
