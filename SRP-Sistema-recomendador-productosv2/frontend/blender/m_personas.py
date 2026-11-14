"""Personajes 3D estilizados (busto). Frente en +Z, cabeza centrada en el origen, radio ~1."""
import math

import bmesh
import bpy
from mathutils import Vector

from comun import *  # noqa: F401,F403
from m_complementos import tubo


def zsup(x, y):
    """z de la superficie frontal de la cabeza (elipsoide 1 x 1.1 x 1) en (x, y)."""
    v = 1 - x * x - (y / 1.1) ** 2
    return math.sqrt(max(v, 0.0))


def _casquete(nombre, radio, escala, centro, mat, quitar, grosor=0.07):
    """Esfera a la que se le quitan los vértices donde quitar(x, y, z) es verdadero (cabello, gorro)."""
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=32, radius=radio)
    borrar = []
    for v in bm.verts:
        p = Vector((v.co.x * escala[0], v.co.y * escala[1], v.co.z * escala[2]))
        if quitar(p.x, p.y, p.z):
            borrar.append(v)
    bmesh.ops.delete(bm, geom=borrar, context="VERTS")
    for v in bm.verts:
        v.co = Vector((v.co.x * escala[0], v.co.y * escala[1], v.co.z * escala[2])) + Vector(centro)
    o = desde_bmesh(nombre, bm, mat, subdiv=1)
    md = o.modifiers.new("grosor", "SOLIDIFY")
    md.thickness = grosor
    md.offset = 1
    return o


def _linea(nombre, pts, radio, mat):
    return tubo(nombre, [Vector(p) for p in pts], radio, mat, res=8)


def avatar(genero, pal, piel_hex="#D9A27C", pelo_hex=None):
    piel = material("piel", piel_hex, rugosidad=0.5, subsurface=0.3)
    piel_o = material("piel_o", "#C98A66", rugosidad=0.5)
    pelo_hex = pelo_hex or ("#3B241B" if genero == "mujer" else "#2A1A14")
    pelo = material("pelo", pelo_hex, rugosidad=0.35, brillo=0.5)
    ojo = material("ojo", "#21140F", rugosidad=0.1, brillo=1.0)
    blanco = material("brillo", "#FFFFFF", rugosidad=0.2)
    mejilla = material("mejilla", "#EE8F84", rugosidad=0.6)
    boca_m = material("boca", "#8E2F2F", rugosidad=0.4)
    ropa = tela("ropa", pal["a"], rugosidad=0.85, sheen=0.7)
    ropa2 = tela("ropa2", pal["b"], rugosidad=0.85, sheen=0.6)
    oro = material("oro", "#F2C75C", rugosidad=0.16, metal=1.0)
    g = []
    # ---- cabeza y rasgos
    g.append(esfera("cabeza", 1.0, (0, 0, 0), piel, escala=(1.0, 1.1, 1.0), seg=64))
    for s in (-1, 1):
        g.append(esfera(f"oreja{s}", 0.2, (s * 0.98, -0.08, -0.05), piel, escala=(0.55, 1.0, 0.85)))
        # ojos
        x, y = s * 0.36, 0.06
        z = zsup(x, y)
        g.append(esfera(f"ojo{s}", 0.115, (x, y, z - 0.04), ojo, escala=(0.9, 1.15, 0.7)))
        g.append(esfera(f"brillo{s}", 0.036, (x + 0.035, y + 0.045, z + 0.045), blanco))
        # cejas
        cj = [(s * (0.24 + 0.2 * t), 0.33 + 0.03 * (t if s > 0 else t) * 0 + 0.03 * math.sin(t * math.pi) + (0.0 if genero == "mujer" else -0.02),
               zsup(s * (0.24 + 0.2 * t), 0.33) + 0.012) for t in [i / 6 for i in range(7)]]
        g.append(_linea(f"ceja{s}", cj, 0.032 if genero == "hombre" else 0.026, pelo))
        # mejillas
        mx, my = s * 0.6, -0.2
        g.append(esfera(f"mejilla{s}", 0.15, (mx, my, zsup(mx, my) - 0.03), mejilla, escala=(1.0, 0.7, 0.3)))
    g.append(esfera("nariz", 0.115, (0, -0.14, zsup(0, -0.14) + 0.02), piel_o, escala=(0.9, 0.9, 0.8)))
    sonrisa = [(0.3 * t, -0.43 - 0.1 * (1 - t * t), zsup(0.3 * t, -0.43 - 0.1 * (1 - t * t)) + 0.012) for t in [i / 8 * 2 - 1 for i in range(9)]]
    g.append(_linea("boca", sonrisa, 0.03, boca_m))
    # ---- cuello y torso
    g.append(cilindro("cuello", 0.4, 0.7, (0, -1.15, -0.02), piel, bisel=0.0, seg=40))
    g[-1].rotation_euler = (math.radians(90), 0, 0)
    g.append(esfera("torso", 1.0, (0, -2.5, -0.05), ropa, escala=(2.0, 1.1, 0.92), seg=64))
    g.append(toro("cuello_ropa", 0.5, 0.11, (0, -1.5, 0.0), ropa2, rot=(math.radians(90), 0, 0), seg_m=48, seg_n=16))
    g[-1].rotation_euler = (math.radians(90), 0, 0)
    g[-1].scale = (1, 1, 1)
    if genero == "hombre":
        # solapas de camisa
        for s in (-1, 1):
            sol = caja(f"solapa{s}", (0.45, 0.35, 0.08), (s * 0.3, -1.5, 0.36), ropa2, bisel=0.03, seg=3,
                       rot=(math.radians(-25), 0, s * math.radians(-28)))
            g.append(sol)
    # ---- pelo
    if genero == "mujer":
        g.append(_casquete("pelo_alto", 1.07, (1.0, 1.12, 1.05), (0, 0.04, -0.05), pelo,
                           lambda x, y, z: z > 0.2 and y < 0.55 - 0.45 * x * x))
        g.append(esfera("pelo_atras", 1.0, (0, -0.85, -0.5), pelo, escala=(1.12, 1.6, 0.72), seg=48))
        for k, (px, py, ang) in enumerate(((-0.38, 0.72, 25), (0.05, 0.8, 0), (0.46, 0.7, -28))):
            f = esfera(f"flequillo{k}", 0.42, (px, py, 0.72), pelo, escala=(0.78, 0.34, 0.4), seg=32)
            f.rotation_euler = (math.radians(-20), 0, math.radians(ang))
            g.append(f)
        for s in (-1, 1):
            g.append(esfera(f"mechon{s}", 0.3, (s * 0.96, -0.45, 0.12), pelo, escala=(0.42, 1.7, 0.45), seg=32))
            g.append(esfera(f"arete{s}", 0.075, (s * 1.0, -0.42, 0.18), oro))
        # pasador de color
        g.append(esfera("pasador", 0.17, (0.72, 0.62, 0.62), material("pasador", pal["d"], rugosidad=0.3, brillo=0.5), escala=(1.0, 0.6, 0.5)))
    else:
        g.append(_casquete("pelo_alto", 1.06, (1.0, 1.1, 1.05), (0, 0.05, -0.04), pelo,
                           lambda x, y, z: (z > 0.15 and y < 0.64 - 0.35 * x * x) or y < -0.12, grosor=0.08))
        copete = esfera("copete", 0.5, (0.14, 0.98, 0.55), pelo, escala=(1.2, 0.42, 0.9), seg=40)
        copete.rotation_euler = (math.radians(-18), 0, math.radians(-10))
        g.append(copete)
        for s in (-1, 1):
            g.append(esfera(f"patilla{s}", 0.2, (s * 0.94, -0.02, 0.2), pelo, escala=(0.3, 1.1, 0.35), seg=24))
        # botones de la camisa
        for k in range(3):
            g.append(esfera(f"boton{k}", 0.07, (0, -1.85 - 0.38 * k, 0.9 - 0.05 * k * 0 + 0.0), ropa2, escala=(1, 1, 0.55)))
    return agrupar("avatar_" + genero, g)


def con_gafas(av, pal, escala=0.47, y=0.1, z=1.02):
    """Pone unas gafas (construidas en m_complementos) sobre la cara del avatar."""
    from m_complementos import gafas
    gg = gafas(pal, 1)
    gg.scale = (escala, escala, escala)
    gg.location = (0, y, z)
    gg.parent = av
    return gg
