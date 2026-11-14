"""Zapatilla deportiva procedural. Eje X = largo (punta en +X), Y = ancho, Z = alto."""
import math

import bmesh
import bpy
from mathutils import Vector

from comun import *  # noqa: F401,F403
from m_complementos import tubo

L = 3.4          # largo total
ZS = 0.34        # altura de la media suela
S_ABRE = 0.42    # fracción del largo (desde el talón) donde termina la boca del zapato


def _ancho(x):
    """Semianchura de la planta en x."""
    s = (x + L / 2) / L
    s = min(max(s, 0.0005), 0.9995)
    base = 0.60 * (math.sin(math.pi * s) ** 0.40)
    cintura = 1 - 0.13 * math.exp(-((x + 0.3) / 0.5) ** 2)
    talon = 0.86 + 0.14 * min(1.0, s * 3.5)
    return base * cintura * talon


def _alto(x):
    """Altura de la capellada sobre la suela."""
    s = (x + L / 2) / L
    empeine = math.sin(math.pi * min(1.0, max(0.0, s * 1.12))) ** 1.1
    talon = math.exp(-(s / 0.2) ** 2)
    return 0.30 + 0.80 * empeine + 0.62 * talon


def _gap(s):
    """Semiángulo (rad) de la apertura superior en la fracción s del largo (0 = cerrada)."""
    if s >= S_ABRE:
        return 0.0
    return 1.0 * ((S_ABRE - s) / S_ABRE) ** 0.55


def _planta(n=44, escala=1.0):
    xs = [-L / 2 + L * i / n for i in range(n + 1)]
    arriba = [(x * escala, _ancho(x) * escala) for x in xs[1:-1]]
    abajo = [(x * escala, -_ancho(x) * escala) for x in xs[1:-1][::-1]]
    return [(-L / 2 * escala + 0.0001, 0.0)] + arriba + [(L / 2 * escala - 0.0001, 0.0)] + abajo


def _losa(nombre, contorno, espesor, z0, mat, bisel):
    bm = bmesh.new()
    vs = [bm.verts.new((x, y, 0)) for x, y in contorno]
    f = bm.faces.new(vs)
    res = bmesh.ops.extrude_face_region(bm, geom=[f])
    for v in [e for e in res["geom"] if isinstance(e, bmesh.types.BMVert)]:
        v.co.z += espesor
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    o = desde_bmesh(nombre, bm, mat, subdiv=0, bisel=bisel, bisel_seg=5)
    o.location = (0, 0, z0)
    return o


def _punto(s, th, k_w=0.94, k_h=1.0):
    x = -L / 2 + 0.05 + (L - 0.1) * s
    w = _ancho(x) * k_w
    h = _alto(x) * k_h
    return Vector((x, w * math.cos(th), ZS - 0.02 + h * (math.sin(th) ** 0.72)))


def _capa(nombre, s0, s1, mat, grosor, offset, k_w=0.94, k_h=1.0, n=20, pasos=30, con_boca=True, tapas=False):
    """Capa de la capellada entre las fracciones s0..s1 del largo, con apertura superior si `con_boca`."""
    bm = bmesh.new()
    filas = []
    huecos = []
    mitad = n // 2
    for i in range(pasos + 1):
        s = s0 + (s1 - s0) * i / pasos
        d = _gap(s) if con_boca else 0.0
        fila = []
        for j in range(n + 1):
            if j <= mitad:
                th = (math.pi / 2 - d) * j / mitad
            else:
                th = (math.pi / 2 + d) + (math.pi / 2 - d) * (j - mitad - 1) / (n - mitad - 1) if d > 0 else math.pi * j / n
            if d == 0:
                th = math.pi * j / n
            fila.append(bm.verts.new(_punto(s, th, k_w, k_h)))
        filas.append(fila)
        huecos.append(d > 0)
    for i in range(pasos):
        for j in range(n):
            if j == mitad and (huecos[i] or huecos[i + 1]):
                continue                      # no hay superficie en la boca
            bm.faces.new((filas[i][j], filas[i][j + 1], filas[i + 1][j + 1], filas[i + 1][j]))
    if tapas:
        for fila in (filas[0], filas[-1]):
            try:
                bm.faces.new(fila)
            except ValueError:
                pass
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    o = desde_bmesh(nombre, bm, mat, subdiv=1)
    md = o.modifiers.new("grosor", "SOLIDIFY")
    md.thickness = grosor
    md.offset = offset
    md2 = o.modifiers.new("subdiv2", "SUBSURF")
    md2.levels = 1
    md2.render_levels = 1
    return o


def zapatilla(pal):
    suela_m = material("suela", "#FBF7F2", rugosidad=0.45, brillo=0.2)
    pisada_m = material("pisada", pal["c"], rugosidad=0.7)
    malla = tela("malla", pal["b"], rugosidad=0.75, sheen=0.5)
    cuero = material("cuero", pal["a"], rugosidad=0.38, brillo=0.35)
    oscuro = material("cuello", pal["c"], rugosidad=0.55, brillo=0.1)
    acento = material("acento", pal["d"], rugosidad=0.4, brillo=0.3)
    cordon = material("cordon", "#FFFFFF", rugosidad=0.75)
    g = []
    g.append(_losa("media", _planta(), 0.34, 0.0, suela_m, 0.09))
    g.append(_losa("pisada", _planta(escala=0.985), 0.1, -0.08, pisada_m, 0.03))
    # capellada completa (malla) + refuerzos de cuero
    g.append(_capa("capellada", 0.0, 1.0, malla, 0.07, -1, tapas=True))
    g.append(_capa("puntera", 0.80, 1.0, cuero, 0.06, 1, k_w=0.97, k_h=1.04, pasos=14, con_boca=False, tapas=False))
    g.append(_capa("talon", 0.0, 0.26, cuero, 0.06, 1, k_w=0.97, k_h=1.03, pasos=14, tapas=False))

    # ---- cuello acolchado: tubo siguiendo el borde de la boca
    def borde(signo):
        pts = []
        for i in range(0, 26):
            s = 0.012 + (S_ABRE - 0.012) * i / 25
            d = _gap(s)
            th = math.pi / 2 - d if signo > 0 else math.pi / 2 + d
            pts.append(_punto(s, th, 0.945, 1.0) + Vector((0, 0, 0.0)))
        return pts

    izq, der = borde(1), borde(-1)
    g.append(tubo("cuello", izq + der[::-1] + [izq[0]], 0.075, oscuro, res=10))
    # ---- lengüeta: sale de la garganta, sube y se apoya hacia atrás
    pts = []
    for i in range(10):
        u = i / 9
        s = 0.47 - 0.14 * u
        p = _punto(s, math.pi / 2, 0.0, 1.0)
        pts.append(Vector((p.x, 0, p.z + 0.02 + 0.50 * u ** 1.4)))
    g.append(barrido("lengueta", pts, perfil_rect(0.62, 0.11, 0.45), cuero, arriba=(0, 0, 1), subdiv=1))
    # ---- cordones: barras cruzadas sobre el empeine
    for k in range(5):
        s = 0.58 + 0.075 * k
        pts = [_punto(s, math.radians(34 + 112 * j / 8), 0.95, 1.012) + Vector((0, 0, 0.05)) for j in range(9)]
        g.append(tubo(f"cord{k}", pts, 0.036, cordon, res=8))
        g.append(esfera(f"ojalA{k}", 0.05, pts[0], acento))
        g.append(esfera(f"ojalB{k}", 0.05, pts[-1], acento))
    # lazo
    p0 = _punto(0.58, math.pi / 2, 0.0, 1.012) + Vector((0, 0, 0.11))
    for sgn in (-1, 1):
        a = []
        for j in range(0, 11):
            u = math.pi * j / 10
            a.append(Vector((p0.x - 0.06 * math.sin(u), sgn * 0.34 * math.sin(u), p0.z + 0.20 * math.sin(u / 2) * math.sin(u) ** 0.5)))
        g.append(tubo(f"lazo{sgn}", a, 0.04, cordon, res=8))
    # ---- franja lateral de acento
    for signo in (-1, 1):
        pts = []
        for i in range(32):
            u = i / 31
            s = 0.20 + 0.50 * u
            th = math.radians(26 + 16 * math.sin(math.pi * u))
            p = _punto(s, th if signo > 0 else math.pi - th, 0.99, 1.0)
            pts.append(Vector((p.x, p.y, p.z)))
        g.append(tubo(f"franja{signo}", pts, 0.055, acento, res=8))
    # ---- pestaña del talón
    p = _punto(0.0, math.pi / 2, 0.0, 1.0)
    g.append(caja("pestana", (0.12, 0.26, 0.28), (p.x + 0.04, 0, p.z - 0.02), acento, bisel=0.035, seg=3))
    return agrupar("zapatilla", g)
