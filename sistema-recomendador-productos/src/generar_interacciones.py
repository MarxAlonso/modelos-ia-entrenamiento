"""Genera interacciones realistas para usuarios con diferentes preferencias.

Crea perfiles de usuario con gustos variados para entrenar el modelo.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RUTA_BASE = Path(__file__).resolve().parent.parent
RUTA_DATOS = RUTA_BASE / "data"

# Perfiles de usuario con preferencias claras
PERFILES_USUARIOS = {
    "u0007": {
        "nombre": "María - Amante de los snacks",
        "categorias_preferidas": ["pasabocas", "dulces-y-postres", "bebidas"],
        "pesos": [0.5, 0.3, 0.2],
        "rating_base": 4.0,
    },
    "u0100": {
        "nombre": "Carlos - Carnívoro",
        "categorias_preferidas": ["carne-y-pollo", "charcuteria", "vinos-y-licores"],
        "pesos": [0.5, 0.3, 0.2],
        "rating_base": 4.2,
    },
    "u0250": {
        "nombre": "Ana - Vinos y fiestas",
        "categorias_preferidas": ["vinos-y-licores", "pasabocas", "dulces-y-postres"],
        "pesos": [0.4, 0.35, 0.25],
        "rating_base": 3.8,
    },
    "u0300": {
        "nombre": "Pedro - Saludable",
        "categorias_preferidas": ["frutas-y-verduras", "lacteos-huevos-y-refrigerados", "pescados-y-mariscos"],
        "pesos": [0.5, 0.3, 0.2],
        "rating_base": 4.5,
    },
    "u0400": {
        "nombre": "Laura - Panadera",
        "categorias_preferidas": ["panaderia-y-pasteleria", "dulces-y-postres", "lacteos-huevos-y-refrigerados"],
        "pesos": [0.5, 0.3, 0.2],
        "rating_base": 4.3,
    },
}


def cargar_datos():
    """Carga datos existentes."""
    df_prod = pd.read_csv(RUTA_DATOS / "productos.csv")
    df_int = pd.read_csv(RUTA_DATOS / "interacciones.csv")
    return df_prod, df_int


def generar_interacciones_usuario(
    usuario_id: str,
    perfil: dict,
    df_prod: pd.DataFrame,
    n_compras: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Genera interacciones realistas para un usuario."""
    rng = np.random.default_rng(seed)
    
    # Filtrar productos por categorías preferidas
    productos_por_cat = {}
    for cat in perfil["categorias_preferidas"]:
        prods = df_prod[df_prod["main_category"] == cat]
        if len(prods) > 0:
            productos_por_cat[cat] = prods
    
    interacciones = []
    
    for _ in range(n_compras):
        # Seleccionar categoría según pesos
        cat_idx = rng.choice(len(perfil["categorias_preferidas"]), p=perfil["pesos"])
        cat = perfil["categorias_preferidas"][cat_idx]
        
        if cat not in productos_por_cat or len(productos_por_cat[cat]) == 0:
            continue
        
        # Seleccionar producto aleatorio de la categoría
        prod = productos_por_cat[cat].sample(n=1, random_state=rng.integers(10000)).iloc[0]
        
        # Generar rating con variación
        rating = np.clip(
            perfil["rating_base"] + rng.normal(0, 0.5),
            1.0, 5.0
        )
        
        # Generar cantidad
        cantidad = rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        
        # Generar fecha aleatoria en 2026
        mes = rng.integers(1, 13)
        dia = rng.integers(1, 29)
        fecha = f"2026-{mes:02d}-{dia:02d}"
        
        interacciones.append({
            "user_id": usuario_id,
            "product_id": prod["product_id"],
            "categoria_producto": cat,
            "rating": round(rating, 1),
            "cantidad": cantidad,
            "fecha": fecha,
        })
    
    return pd.DataFrame(interacciones)


def main():
    print("=" * 60)
    print("GENERACIÓN DE INTERACCIONES REALISTAS")
    print("=" * 60)
    
    # Cargar datos
    df_prod, df_int = cargar_datos()
    print(f"\nDatos existentes:")
    print(f"  - Productos: {len(df_prod)}")
    print(f"  - Interacciones: {len(df_int)}")
    print(f"  - Usuarios únicos: {df_int['user_id'].nunique()}")
    
    # Generar interacciones para cada perfil
    nuevas_ints = []
    for uid, perfil in PERFILES_USUARIOS.items():
        print(f"\nGenerando para {uid} ({perfil['nombre']})...")
        ints = generar_interacciones_usuario(uid, perfil, df_prod, n_compras=60, seed=hash(uid) % 10000)
        nuevas_ints.append(ints)
        print(f"  - {len(ints)} interacciones generadas")
        
        # Mostrar distribución
        dist = ints["categoria_producto"].value_counts()
        for cat, count in dist.head(3).items():
            print(f"    - {cat}: {count}")
    
    # Combinar con existentes
    df_nuevas = pd.concat(nuevas_ints, ignore_index=True)
    df_todas = pd.concat([df_int, df_nuevas], ignore_index=True)
    
    # Guardar
    df_todas.to_csv(RUTA_DATOS / "interacciones.csv", index=False)
    print(f"\n{'='*60}")
    print(f"Total interacciones: {len(df_todas)} (+{len(df_nuevas)} nuevas)")
    print(f"Guardado en: {RUTA_DATOS / 'interacciones.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
