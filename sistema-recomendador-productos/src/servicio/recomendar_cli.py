"""Demo interactivo del sistema recomendador hibrido v2.

Uso:
    python src/recomendar_cli.py                 # modo interactivo
    python src/recomendar_cli.py --usuario u0007 # consulta directa
"""

import sys
from pathlib import Path
_RUTA_MODULO = Path(__file__).resolve()
while _RUTA_MODULO.name != "src" and _RUTA_MODULO != _RUTA_MODULO.parent:
    _RUTA_MODULO = _RUTA_MODULO.parent
sys.path.insert(0, str(_RUTA_MODULO.parent))

from src.rutas import RUTA_RAIZ

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

RUTA_DATOS = RUTA_RAIZ / "data"
RUTA_MODELOS = RUTA_RAIZ / "models"


def estrellas(rating: float) -> str:
    llenas = int(float(rating))
    return "★" * llenas + "☆" * (5 - llenas)


def etiqueta_resena(rating: float) -> str:
    if rating >= 4.0:
        return "positiva"
    if rating >= 3.0:
        return "neutra"
    return "negativa"


def cargar_sistema() -> dict:
    artefacto = joblib.load(RUTA_MODELOS / "hibrido_v2.pkl")
    puntajes = np.load(RUTA_MODELOS / "puntajes_hibrido_v2.npz")
    productos_info = pd.read_csv(
        RUTA_DATOS / "productos.csv",
        usecols=["product_id", "name", "main_category"],
    ).set_index("product_id")
    interacciones = pd.read_csv(RUTA_DATOS / "interacciones.csv")
    interacciones["fecha"] = pd.to_datetime(interacciones["fecha"])
    historial = {
        u: grupo.sort_values("fecha", ascending=False)
        for u, grupo in interacciones.groupby("user_id")
    }
    return {
        "artefacto": artefacto,
        "matriz": puntajes["matriz"],
        "productos_info": productos_info,
        "historial": historial,
    }


def mostrar_usuario(sistema: dict, usuario_id: str, n_recomendaciones: int = 10) -> None:
    art = sistema["artefacto"]
    idx_usuario = art["idx_usuario"]
    inv_prod = {v: k for k, v in art["idx_producto"].items()}
    lista_cats = art["lista_cats"]
    p_cat_u = art["p_categoria_usuario"]
    info = sistema["productos_info"]
    matriz = sistema["matriz"]
    u = idx_usuario[usuario_id]
    hist = sistema["historial"].get(usuario_id)

    favoritas = art.get("favoritas_declaradas", {}).get(usuario_id, [])

    print("=" * 78)
    print(f"  USUARIO {usuario_id}")
    print("=" * 78)

    if hist is None or len(hist) == 0:
        print("  Sin historial de compras registrado.\n")
        return
    else:
        rating_medio = hist["rating"].mean()
        n_pos = int((hist["rating"] >= 4).sum())
        n_neu = int(((hist["rating"] >= 3) & (hist["rating"] < 4)).sum())
        n_neg = int((hist["rating"] < 3).sum())
        print("\nPERFIL")
        print(f"  Compras totales : {len(hist)}")
        print(f"  Rating medio    : {rating_medio:.2f} / 5")
        print(f"  Preferencias    : {', '.join(favoritas or [])}")

        print("\nRESEÑAS (calificaciones)")
        print(f"  Positivas ≥4★   : {n_pos:>3} ({n_pos/len(hist):.0%})")
        print(f"  Neutras 3-3.9★  : {n_neu:>3} ({n_neu/len(hist):.0%})")
        print(f"  Negativas <3★   : {n_neg:>3} ({n_neg/len(hist):.0%})")

        print("\nHISTORIAL RECIENTE")
        for _, fila in hist.head(8).iterrows():
            nombre = info.loc[fila["product_id"], "name"]
            cat = info.loc[fila["product_id"], "main_category"]
            print(f"  {fila['fecha']:%Y-%m-%d} | {estrellas(fila['rating'])} "
                  f"{fila['rating']:.1f} [{etiqueta_resena(fila['rating']):<8}] "
                  f"({cat}) {nombre}")

        print("\nCATEGORIAS PREDICHAS PARA TI")
        orden = np.argsort(p_cat_u[u])[::-1][:3]
        for j in orden:
            print(f"  {lista_cats[j]:<28} {p_cat_u[u, j]:>5.1%} de afinidad")

    fila_scores = matriz[u].copy()
    fila_scores[~np.isfinite(fila_scores)] = -np.inf
    ya_comprados = set(hist["product_id"]) if hist is not None else set()
    top = np.argsort(fila_scores)[::-1][:n_recomendaciones]

    print(f"\nTOP-{n_recomendaciones} RECOMENDACIONES PERSONALIZADAS")
    print(f"{'#':>2} {'Producto':<52} {'Categoria':<24} {'Afin.':>6} Puntaje")
    mostrados = 0
    for col in top:
        if not np.isfinite(fila_scores[col]):
            continue
        pid = inv_prod[col]
        nombre = str(info.loc[pid, "name"])
        cat = str(info.loc[pid, "main_category"])
        afin = p_cat_u[u, _indice_cat(art, pid, info)] * 100
        marca = " (ya la compraste)" if pid in ya_comprados else ""
        print(f"{mostrados + 1:>2}. {nombre[:50]:<52} {cat[:22]:<24} {afin:>5.0f}% "
              f"{fila_scores[col]:.2f}{marca}")
        mostrados += 1
        if mostrados >= n_recomendaciones:
            break
    print()


def _indice_cat(art: dict, pid: str, info: pd.DataFrame) -> int:
    cat = info.loc[pid, "main_category"]
    return art["lista_cats"].index(cat)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recomendador personalizado")
    parser.add_argument("--usuario", type=str, default=None,
                        help="ID de usuario (ej. u0007); si se omite entra en modo interactivo")
    parser.add_argument("--top", type=int, default=10, help="Numero de recomendaciones")
    args = parser.parse_args()

    sistema = cargar_sistema()
    usuarios_validos = list(sistema["artefacto"]["idx_usuario"].keys())

    if args.usuario is not None:
        mostrar_usuario(sistema, args.usuario.strip(), args.top)
        return

    print("=" * 78)
    print("  SISTEMA RECOMENDADOR - DEMO INTERACTIVO")
    print("=" * 78)
    print(f"  Precision categoria @10 : "
          f"{sistema['artefacto']['precision_categoria_test']:.1%}")
    print(f"  HitRate categoria @10   : "
          f"{sistema['artefacto']['hitrate_categoria_test']:.1%}")
    print("  Comandos: <user_id> | lista | aleatorio | salir\n")

    while True:
        try:
            entrada = input("Tu ID de usuario > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if entrada in ("salir", "exit", "q"):
            break
        if entrada == "lista":
            muestra = usuarios_validos[:10]
            print(f"  Ejemplos validos: {', '.join(muestra)} ... ({len(usuarios_validos)} en total)\n")
            continue
        if entrada == "aleatorio":
            entrada = str(np.random.choice(usuarios_validos))
        if entrada not in usuarios_validos:
            print(f"  Usuario '{entrada}' no existe. Escribe 'lista' para ver ejemplos.\n")
            continue
        mostrar_usuario(sistema, entrada, args.top)


if __name__ == "__main__":
    main()
