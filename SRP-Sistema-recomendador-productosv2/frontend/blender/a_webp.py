"""Convierte los PNG renderizados por Blender a WebP (con transparencia) dentro de public/3d.

    python a_webp.py <carpeta_png> <carpeta_destino>   (requiere: pip install pillow)
"""
import os
import sys

from PIL import Image

origen, destino = sys.argv[1], sys.argv[2]
total = 0
for raiz, _, archivos in os.walk(origen):
    for nombre in archivos:
        if not nombre.lower().endswith(".png"):
            continue
        rel = os.path.relpath(os.path.join(raiz, nombre), origen)
        salida = os.path.join(destino, os.path.splitext(rel)[0] + ".webp")
        os.makedirs(os.path.dirname(salida), exist_ok=True)
        im = Image.open(os.path.join(raiz, nombre)).convert("RGBA")
        im.save(salida, "WEBP", quality=88, alpha_quality=100, method=6)
        total += os.path.getsize(salida)
        print(f"{rel} -> {os.path.getsize(salida) // 1024} KB")
print(f"total {total // 1024} KB")
