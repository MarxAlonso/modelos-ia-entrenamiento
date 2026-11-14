"""Renderiza productos del catálogo.

  blender -b -P render_productos.py -- <salida_dir> <nombre|todos> [variante|todas] [muestras] [tam]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comun  # noqa: E402
import productos  # noqa: E402

args = comun.args_script()
salida = args[0]
quienes = args[1] if len(args) > 1 else "todos"
vars_ = args[2] if len(args) > 2 else "todas"
muestras = int(args[3]) if len(args) > 3 else 64
tam = int(args[4]) if len(args) > 4 else 640

NOMBRES = ["relojes", "joyeria", "lentes", "bolsos", "calzado", "ropa_h", "ropa_m", "accesorios", "ropa_interior", "otros"]
lista = NOMBRES if quienes == "todos" else quienes.split(",")
variantes = [1, 2, 3] if vars_ == "todas" else [int(v) for v in vars_.split(",")]

for nombre in lista:
    for v in variantes:
        comun.reiniciar()
        comun.configurar_render(tam, tam, muestras)
        productos.construir(nombre, v)
        comun.estudio(**productos.CAMARA.get(nombre, {}))
        comun.renderizar(os.path.join(salida, f"{nombre}-{v}.png"))
