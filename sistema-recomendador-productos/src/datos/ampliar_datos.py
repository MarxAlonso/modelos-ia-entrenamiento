"""Ampliacion de datasets: mas compras + dataset de comportamiento.

1) Enriquece interacciones.csv generando compras adicionales para los 500
   usuarios respetando sus preferencias declaradas en usuarios.csv
   (75% categorias favoritas / 25% exploracion, como la Fase 1).

2) Crea data/comportamiento.csv: embudo implicito realista
   (vista -> carrito -> compra) con sesiones, dispositivo, franja horaria y
   duracion. Representa la senal de COMPORTAMIENTO del enunciado y queda
   disponible para modelos de secuencia / feedback implicito.
"""

import sys
from pathlib import Path
_RUTA_MODULO = Path(__file__).resolve()
while _RUTA_MODULO.name != "src" and _RUTA_MODULO != _RUTA_MODULO.parent:
    _RUTA_MODULO = _RUTA_MODULO.parent
sys.path.insert(0, str(_RUTA_MODULO.parent))

from src.rutas import RUTA_RAIZ

import numpy as np
import pandas as pd
from pathlib import Path

RUTA_BASE = RUTA_RAIZ
RUTA_DATOS = RUTA_BASE / "data"
SEED = 42
FECHA_INICIO = pd.Timestamp("2026-01-01")
DIAS_TOTALES = 300


def cargar() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(RUTA_DATOS / "productos.csv"),
        pd.read_csv(RUTA_DATOS / "usuarios.csv"),
        pd.read_csv(RUTA_DATOS / "interacciones.csv"),
    )


def generar_compras_extra(
    productos: pd.DataFrame,
    usuarios: pd.DataFrame,
    existentes: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Compras adicionales sesgadas por las preferencias de cada usuario."""
    prods_por_cat = {
        cat: df["product_id"].to_numpy()
        for cat, df in productos.groupby("main_category")
    }
    favoritas_por_usuario = {
        r["user_id"]: [c for c in str(r["categorias_favoritas"]).split("|") if c]
        for _, r in usuarios.iterrows()
    }
    filas = []
    for uid, favoritas in favoritas_por_usuario.items():
        validas = [c for c in favoritas if c in prods_por_cat]
        if not validas:
            continue
        n_extra = int(rng.integers(28, 46))
        pesos = np.array([rng.uniform(0.8, 1.5) for _ in validas])
        pesos /= pesos.sum()
        for _ in range(n_extra):
            if rng.random() < 0.75:
                cat = str(rng.choice(validas, p=pesos))
                rating = float(np.clip(rng.normal(4.25, 0.6), 1.0, 5.0))
            else:
                cat = str(rng.choice(list(prods_por_cat)))
                rating = float(np.clip(rng.normal(3.05, 0.6), 1.0, 5.0))
            pid = str(rng.choice(prods_por_cat[cat]))
            filas.append({
                "user_id": uid,
                "product_id": pid,
                "categoria_producto": cat,
                "rating": round(rating, 1),
                "cantidad": int(rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1])),
                "fecha": (FECHA_INICIO + pd.Timedelta(
                    days=int(rng.integers(0, DIAS_TOTALES)),
                    hours=int(rng.integers(6, 23)),
                )).strftime("%Y-%m-%d"),
            })
    nuevas = pd.DataFrame(filas)
    vistas_previas = set(
        zip(existentes["user_id"], existentes["product_id"]))
    mascara = [
        (u, p) not in vistas_previas
        for u, p in zip(nuevas["user_id"], nuevas["product_id"])
    ]
    return nuevas[mascara].reset_index(drop=True)


def generar_comportamiento(
    interacciones: pd.DataFrame,
    usuarios: pd.DataFrame,
    productos: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Embudo vista->carrito->compra coherente con las compras reales."""
    cats_usuario = {
        r["user_id"]: [c for c in str(r["categorias_favoritas"]).split("|") if c]
        for _, r in usuarios.iterrows()
    }
    pool_cats = productos["main_category"].dropna().unique().tolist()
    eventos = []
    contador_sesion = 0

    def nueva_sesion(dispositivo: str) -> str:
        nonlocal contador_sesion
        contador_sesion += 1
        return f"s{contador_sesion:06d}"

    for fila in interacciones.itertuples(index=False):
        uid = fila.user_id
        favoritas = cats_usuario.get(uid, [])
        dia = FECHA_INICIO + pd.Timedelta(days=rng.integers(0, DIAS_TOTALES))
        dispositivo = str(rng.choice(
            ["movil", "escritorio", "app"], p=[0.55, 0.25, 0.20]))
        sesion = nueva_sesion(dispositivo)
        hora_base = int(rng.integers(7, 23))

        # vistas exploratorias previas que no convierten (ruido realista)
        if rng.random() < 0.65:
            for _ in range(int(rng.integers(1, 5))):
                cat_ruido = (str(rng.choice(favoritas)) if favoritas and rng.random() < 0.7
                             else str(rng.choice(pool_cats)))
                candidatos = productos[productos["main_category"] == cat_ruido]
                if candidatos.empty:
                    continue
                pid_ruido = str(candidatos.iloc[rng.integers(len(candidatos))]["product_id"])
                eventos.append({
                    "user_id": uid, "product_id": pid_ruido,
                    "categoria_producto": cat_ruido, "tipo_evento": "view",
                    "sesion_id": sesion, "dispositivo": dispositivo,
                    "fecha": dia.strftime("%Y-%m-%d"),
                    "hora": int(np.clip(hora_base - int(rng.integers(1, 40)), 0, 23)),
                    "duracion_seg": int(rng.integers(4, 90)),
                })

        # vistas del producto comprado
        for _ in range(int(rng.integers(1, 4))):
            eventos.append({
                "user_id": uid, "product_id": fila.product_id,
                "categoria_producto": fila.categoria_producto,
                "tipo_evento": "view",
                "sesion_id": sesion, "dispositivo": dispositivo,
                "fecha": fila.fecha,
                "hora": int(np.clip(hora_base - int(rng.integers(1, 15)), 0, 23)),
                "duracion_seg": int(rng.integers(15, 300)),
            })

        # carrito antes de comprar (85% de las compras)
        if rng.random() < 0.85:
            eventos.append({
                "user_id": uid, "product_id": fila.product_id,
                "categoria_producto": fila.categoria_producto,
                "tipo_evento": "cart",
                "sesion_id": sesion, "dispositivo": dispositivo,
                "fecha": fila.fecha,
                "hora": int(np.clip(hora_base - int(rng.integers(0, 6)), 0, 23)),
                "duracion_seg": int(rng.integers(10, 120)),
            })

        eventos.append({
            "user_id": uid, "product_id": fila.product_id,
            "categoria_producto": fila.categoria_producto,
            "tipo_evento": "purchase",
            "sesion_id": sesion, "dispositivo": dispositivo,
            "fecha": fila.fecha, "hora": hora_base,
            "duracion_seg": int(rng.integers(20, 400)),
        })
    return pd.DataFrame(eventos)


def main() -> None:
    print("=" * 60)
    print("AMPLIACION DE DATASETS (compras + comportamiento)")
    print("=" * 60)
    rng = np.random.default_rng(SEED)

    productos, usuarios, interacciones = cargar()
    print(f"Estado inicial: {len(interacciones):,} compras | "
          f"{interacciones['user_id'].nunique()} usuarios")

    print("[1/3] Generando compras adicionales...")
    extras = generar_compras_extra(productos, usuarios, interacciones, rng)
    totales = pd.concat([interacciones, extras], ignore_index=True)
    totales.to_csv(RUTA_DATOS / "interacciones.csv", index=False)
    print(f"      +{len(extras):,} compras -> total {len(totales):,} "
          f"(+{len(extras)/len(interacciones)*100:.0f}%)")
    en_favoritas = totales.merge(usuarios, on="user_id").apply(
        lambda r: r["categoria_producto"] in str(r["categorias_favoritas"]).split("|"),
        axis=1)
    print(f"      Compras en categoria favorita: {en_favoritas.mean():.0%}")

    print("[2/3] Generando dataset de comportamiento (embudo view->cart->purchase)...")
    comportamiento = generar_comportamiento(totales, usuarios, productos, rng)
    comportamiento.to_csv(RUTA_DATOS / "comportamiento.csv", index=False)
    dist = comportamiento["tipo_evento"].value_counts()
    for tipo, n in dist.items():
        print(f"      {tipo}: {n:,}")
    print(f"      Sesiones: {comportamiento['sesion_id'].nunique():,} | "
          f"Dispositivos: {dict(comportamiento['dispositivo'].value_counts())}")

    print("[3/3] Guardado completado.")
    print("Archivos: data/interacciones.csv (ampliado) | data/comportamiento.csv (nuevo)")


if __name__ == "__main__":
    main()
