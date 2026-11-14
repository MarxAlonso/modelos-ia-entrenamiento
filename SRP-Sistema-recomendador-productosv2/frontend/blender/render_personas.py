"""Renderiza los personajes 3D y sus escenas de perfil.

  blender -b -P render_personas.py -- <salida_dir> <solo|perfiles|todo> [genero|ambos] [muestras] [tam]
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402

import comun  # noqa: E402
import m_personas  # noqa: E402
import productos  # noqa: E402
from m_complementos import orientar  # noqa: E402

args = comun.args_script()
salida = args[0]
modo = args[1] if len(args) > 1 else "todo"
generos = ["hombre", "mujer"] if (len(args) < 3 or args[2] == "ambos") else args[2].split(",")
muestras = int(args[3]) if len(args) > 3 else 64
tam = int(args[4]) if len(args) > 4 else 640

# perfil -> (producto, variante, escala, posición del objeto, giro extra)
PERFILES = {
    "relojes": ("relojes", None, 0.55, (2.5, -2.9, 1.1)),
    "joyeria": ("joyeria", 1, 0.82, (2.5, -2.9, 1.0)),
    "calzado": ("calzado", None, 0.66, (1.9, -3.0, 0.45)),
    "bolsos": ("bolsos", None, 0.66, (2.45, -2.9, 0.95)),
    "lentes": None,
    "variado": ("otros", None, 0.82, (2.35, -2.9, 0.85)),
}


def paleta_de(genero):
    return comun.PALETAS[1 if genero == "mujer" else 2], (1 if genero == "mujer" else 2)


def escena(genero, perfil):
    comun.reiniciar()
    comun.configurar_render(tam, tam, muestras)
    pal, var = paleta_de(genero)
    av = m_personas.avatar(genero, pal)
    if perfil == "lentes":
        m_personas.con_gafas(av, pal)
    orientar(av, giro=-14, inclina=0, z=2.7)
    if perfil in PERFILES and PERFILES[perfil]:
        nombre, v, esc, pos = PERFILES[perfil]
        prop = productos.construir(nombre, v or var)
        prop.scale = (esc, esc, esc)
        prop.location = pos
    if perfil == "lentes":
        # cubo fantasma (no se renderiza) para que el encuadre sea el mismo que en los demás perfiles
        bpy.ops.mesh.primitive_cube_add(size=1.6, location=(2.5, -2.9, 1.0))
        bpy.context.active_object.hide_render = True
    comun.estudio(azimut=-8, elevacion=5, lente=85, relleno=0.88 if perfil != "ninguno" else 0.8)


for genero in generos:
    if modo in ("solo", "todo"):
        escena(genero, "ninguno")
        comun.renderizar(os.path.join(salida, "avatares", f"{genero}.png"))
    if modo in ("perfiles", "todo"):
        for perfil in PERFILES:
            escena(genero, perfil)
            comun.renderizar(os.path.join(salida, "perfiles", f"{genero}-{perfil}.png"))
