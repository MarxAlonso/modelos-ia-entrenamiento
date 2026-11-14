"""Camiseta, vestido, gorro, calcetín y caja de regalo. Frente en +Z (prendas en el plano XY)."""
import math
import random

import bmesh
import bpy
from mathutils import Vector

from comun import *  # noqa: F401,F403
from m_complementos import tubo


def _bezier(p0, p1, p2, p3, n=10):
    out = []
    for i in range(n + 1):
        t = i / n
        a = (1 - t) ** 3
        b = 3 * (1 - t) ** 2 * t
        c = 3 * (1 - t) * t * t
        d = t ** 3
        out.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0], a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
    return out


# ================================================================== CAMISETA
def camiseta(pal):
    tel = tela("tela", pal["a"], rugosidad=0.85, sheen=0.6)
    cuello_m = tela("cuello", pal["c"], rugosidad=0.8, sheen=0.4)
    puño = tela("puño", pal["d"], rugosidad=0.8, sheen=0.4)
    A, B, C, D, E, F = (0.82, 1.5), (1.58, 1.2), (1.76, 1.64), (1.0, 2.22), (0.36, 2.4), (0.0, 2.02)
    der = [(0.0, 0.0), (0.86, 0.0)]
    der += _bezier((0.86, 0.0), (0.88, 0.5), (0.84, 1.1), A, 8)[1:]
    der += _bezier(A, (1.05, 1.42), (1.3, 1.3), B, 6)[1:]
    der += [C]
    der += _bezier(C, (1.5, 1.9), (1.25, 2.1), D, 8)[1:]
    der += _bezier(D, (0.8, 2.3), (0.55, 2.38), E, 6)[1:]
    der += _bezier(E, (0.34, 2.12), (0.2, 2.02), F, 8)[1:]
    izq = [(-x, y) for x, y in der[::-1][1:]]
    cont = der + izq
    cont = [(x, y) for i, (x, y) in enumerate(cont) if i == 0 or abs(x - cont[i - 1][0]) + abs(y - cont[i - 1][1]) > 1e-4]
    g = [relleno_2d("camiseta", cont, grosor=0.14, mat=tel, subdiv=2, hinchar=0.30, resolucion=0.09)]
    cu = _bezier(E, (0.34, 2.12), (0.2, 2.02), F, 12)
    c = [Vector((x, y, 0.02)) for x, y in cu] + [Vector((-x, y, 0.02)) for x, y in cu[::-1][1:]]
    g.append(tubo("canale", c, 0.07, cuello_m, res=8))
    for s in (-1, 1):
        g.append(tubo(f"puño{s}", [Vector((s * B[0], B[1], 0.03)), Vector((s * C[0], C[1], 0.03))], 0.055, puño, res=8))
    return agrupar("camiseta", g)


# ================================================================== VESTIDO
def vestido(pal):
    tel = tela("tela", pal["a"], rugosidad=0.8, sheen=0.7)
    faja_m = material("faja", pal["d"], rugosidad=0.4, brillo=0.3)
    borde_m = tela("borde", pal["b"], rugosidad=0.8, sheen=0.5)
    der = []
    der += [(0.0, 0.0)]
    # falda con vuelo (dobladillo ondulado)
    for i in range(0, 25):
        t = i / 24
        x = 1.75 * t
        y = 0.04 * math.sin(t * math.pi * 5) * 1.0
        der.append((x, y))
    der += _bezier((1.75, 0.0), (1.55, 0.9), (0.85, 1.35), (0.62, 1.78), 12)[1:]   # falda hacia cintura
    der += _bezier((0.62, 1.78), (0.58, 2.1), (0.7, 2.35), (0.75, 2.62), 8)[1:]      # busto
    der += [(0.6, 2.62), (0.52, 3.4)]                                              # tirante
    der += [(0.43, 3.4), (0.34, 2.75)]
    der += _bezier((0.34, 2.75), (0.3, 2.55), (0.12, 2.4), (0.0, 2.4), 8)[1:]        # escote
    izq = [(-x, y) for x, y in der[::-1][1:]]
    cont = der + izq
    cont = [(x, y) for i, (x, y) in enumerate(cont) if i == 0 or abs(x - cont[i - 1][0]) + abs(y - cont[i - 1][1]) > 1e-4]
    cuerpo = relleno_2d("vestido", cont, grosor=0.1, mat=tel, subdiv=2, hinchar=0.3, resolucion=0.1)
    g = [cuerpo]
    # faja en la cintura (barrido plano que sigue el contorno de la cintura)
    pts = [Vector((x, 1.78 + 0.02 * math.cos(x * 2.2) - 0.06 * abs(x) * 0.0, 0.34)) for x in [-0.64 + 1.28 * i / 20 for i in range(21)]]
    g.append(tubo("faja", pts, 0.1, faja_m, res=10))
    # lazo de la faja
    for s in (-1, 1):
        p = [Vector((0.0 + s * 0.02, 1.78, 0.4))]
        for j in range(1, 10):
            u = math.pi * j / 9
            p.append(Vector((s * 0.32 * math.sin(u), 1.78 - 0.18 * math.sin(u * 2), 0.42 + 0.12 * math.sin(u))))
        g.append(tubo(f"lazo{s}", p, 0.065, faja_m, res=8))
    g.append(esfera("nudo", 0.1, (0, 1.78, 0.44), faja_m))
    return agrupar("vestido", g)


# ================================================================== GORRO
def gorro(pal):
    lana = tela("lana", pal["a"], rugosidad=0.95, sheen=1.0)
    lana2 = tela("lana2", pal["b"], rugosidad=0.95, sheen=1.0)
    pompon_m = tela("pompon", pal["d"], rugosidad=1.0, sheen=1.0)
    rng = random.Random(7)
    g = []

    def tejido(nombre, radio, alto, z0, mat, costillas=36, amp=0.035, redondeo=1.0, cierre=False, seg_v=30):
        bm = bmesh.new()
        filas = []
        for i in range(seg_v + 1):
            u = i / seg_v
            if cierre:
                th = u * math.pi / 2
                r = radio * math.cos(th)
                z = z0 + alto * math.sin(th) ** 0.9
            else:
                r = radio
                z = z0 + alto * u
            fila = []
            for j in range(costillas * 4):
                a = 2 * math.pi * j / (costillas * 4)
                rr = r * (1 + amp * math.cos(a * costillas) * (1 - 0.6 * u if cierre else 1.0))
                fila.append(bm.verts.new((rr * math.cos(a), rr * math.sin(a), z)))
            filas.append(fila)
        m = costillas * 4
        for i in range(seg_v):
            for j in range(m):
                k = (j + 1) % m
                bm.faces.new((filas[i][j], filas[i][k], filas[i + 1][k], filas[i + 1][j]))
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        o = desde_bmesh(nombre, bm, mat, subdiv=1)
        md = o.modifiers.new("grosor", "SOLIDIFY")
        md.thickness = 0.05
        return o

    g.append(tejido("cupula", 1.0, 1.2, 0.3, lana, costillas=24, amp=0.035, cierre=True))
    g.append(tejido("puño", 1.1, 0.7, 0.0, lana2, costillas=26, amp=0.03))
    g.append(esfera("pompon", 0.42, (0, 0, 1.78), pompon_m, escala=(1, 1, 0.95), seg=48))
    # pelusa del pompón
    for _ in range(0):
        pass
    return agrupar("gorro", g)


# ================================================================== CALCETÍN
def calcetin(pal):
    base = tela("base", pal["b"], rugosidad=0.9, sheen=0.9)
    color = tela("color", pal["a"], rugosidad=0.9, sheen=0.9)
    oscuro = tela("oscuro", pal["c"], rugosidad=0.9, sheen=0.9)
    g = []
    # trayectoria: pierna recta -> talón curvo -> pie
    pts = [Vector((0, 0, 2.7 - 2.1 * i / 12)) for i in range(13)]
    R = 0.55
    for i in range(1, 13):
        a = math.pi / 2 * i / 12
        pts.append(Vector((R - R * math.cos(a) + 0.0, 0, 0.6 + R * (1 - 1) - R * math.sin(a) + 0.55 * 0 + 0.0)))
    # reconstruir limpio
    pts = []
    for i in range(14):
        pts.append(Vector((0, 0, 2.6 - 2.0 * i / 13)))
    R = 0.62
    for i in range(1, 11):
        a = math.pi / 2 * i / 10
        pts.append(Vector((R * (1 - math.cos(a)), 0, 0.6 - R * math.sin(a) + 0.0)))
    for i in range(1, 17):
        pts.append(Vector((R + 1.55 * i / 16, 0, -0.02 + 0.0)))
    # alinear: el pie debe quedar a z≈0 -> ajustar
    dz = 0.0 - min(p.z for p in pts) + 0.0
    pts = [Vector((p.x, p.y, p.z + dz + 0.3)) for p in pts]
    n = len(pts)
    esc = []
    for i, p in enumerate(pts):
        t = i / (n - 1)
        a = 0.42 + 0.0 * t                    # grosor (adelante-atrás / vertical)
        b = 0.50 - 0.1 * max(0, t - 0.7) * 2   # ancho (lado a lado)
        if t > 0.8:                           # punta redondeada
            u = (t - 0.8) / 0.2
            a *= (1 - u ** 3) ** 0.5 if u < 1 else 0.05
            b *= (1 - u ** 3) ** 0.5 if u < 1 else 0.05
        esc.append((max(a, 0.03), max(b, 0.03)))
    cuerpo = barrido("calcetin", pts, perfil_elipse(1.0, 1.0, 28), base, arriba=(0, 1, 0), escalas=esc, subdiv=1,
                     cerrar_extremos=True)
    g.append(cuerpo)
    # puño con franjas
    for k, (zz, mat) in enumerate(((2.55, color), (2.38, oscuro), (2.21, color))):
        r = cilindro(f"franja{k}", 0.5, 0.15, (0, 0, zz + 0.0), mat, seg=48, bisel=0.02)
        r.scale = (0.9, 1.0, 1.0)
        g.append(r)
    # talón y puntera de color
    heel = esfera("talon", 0.5, (R * 0.0 + 0.62 - 0.55, 0, pts[14 + 7].z), color, escala=(0.8, 1.0, 0.8))
    g.append(heel)
    toe = esfera("punta", 0.5, (pts[-1].x - 0.28, 0, pts[-1].z), color, escala=(0.75, 1.02, 0.86))
    g.append(toe)
    return agrupar("calcetin", g)


# ================================================================== REGALO
def regalo(pal):
    caja_m = material("caja", pal["a"], rugosidad=0.4, brillo=0.4)
    tapa_m = material("tapa", pal["a"], rugosidad=0.4, brillo=0.4)
    cinta = material("cinta", pal["d"], rugosidad=0.25, brillo=0.6)
    g = []
    g.append(caja("caja", (2.2, 1.6, 2.2), (0, 0.8, 0), caja_m, bisel=0.06, seg=4))
    g.append(caja("tapa", (2.4, 0.42, 2.4), (0, 1.8, 0), tapa_m, bisel=0.06, seg=4))
    # cintas en cruz
    g.append(caja("cinta1", (0.4, 1.66, 2.24), (0, 0.8, 0), cinta, bisel=0.03, seg=3))
    g.append(caja("cinta2", (2.24, 1.66, 0.4), (0, 0.8, 0), cinta, bisel=0.03, seg=3))
    g.append(caja("cinta3", (0.42, 0.46, 2.44), (0, 1.8, 0), cinta, bisel=0.03, seg=3))
    g.append(caja("cinta4", (2.44, 0.46, 0.42), (0, 1.8, 0), cinta, bisel=0.03, seg=3))
    # lazo: dos lazadas
    for s in (-1, 1):
        pts = []
        for j in range(0, 25):
            u = math.pi * j / 24
            pts.append(Vector((s * 0.55 * math.sin(u) * (0.35 + 0.65 * (j / 24)), 2.1 + 0.55 * math.sin(u) * 0.7, 0.6 * math.sin(u) * 0.0)))
        g.append(tubo(f"lazo{s}", pts, 0.12, cinta, res=10))
    for s in (-1, 1):
        pts = []
        for j in range(0, 25):
            u = math.pi * j / 24
            pts.append(Vector((s * 0.8 * math.sin(u), 2.08 + 0.78 * math.sin(u) ** 0.8 * math.cos(u * 0.4), 0.0)))
        g.append(tubo(f"lazada{s}", pts, 0.13, cinta, res=10))
    g.append(esfera("nudo", 0.2, (0, 2.1, 0), cinta))
    return agrupar("regalo", g)
