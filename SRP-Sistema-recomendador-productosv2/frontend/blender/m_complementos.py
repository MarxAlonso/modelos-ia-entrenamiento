"""Joyería, gafas de sol y bolso. Convención: el frente del modelo mira a +Z (se orienta luego)."""
import math

import bmesh
import bpy
from mathutils import Vector

from comun import *  # noqa: F401,F403


# ------------------------------------------------------------------ helpers
def tubo(nombre, puntos, radio, mat, cerrado=False, res=10):
    """Curva poligonal con sección circular (bevel de curva)."""
    cv = bpy.data.curves.new(nombre, "CURVE")
    cv.dimensions = "3D"
    cv.bevel_depth = radio
    cv.bevel_resolution = res
    cv.use_fill_caps = True
    sp = cv.splines.new("POLY")
    sp.points.add(len(puntos) - 1)
    for i, p in enumerate(puntos):
        sp.points[i].co = (p[0], p[1], p[2], 1)
    sp.use_cyclic_u = cerrado
    o = bpy.data.objects.new(nombre, cv)
    bpy.context.collection.objects.link(o)
    o.data.materials.append(mat)
    return o


def superelipse(a, b, n=3.0, pts=64, taper=0.0, yoff=0.0):
    out = []
    for i in range(pts):
        th = 2 * math.pi * i / pts
        c, s = math.cos(th), math.sin(th)
        x = a * math.copysign(abs(c) ** (2 / n), c)
        y = b * math.copysign(abs(s) ** (2 / n), s)
        x *= 1 + taper * (y / b)
        out.append((x, y + yoff))
    return out


def degradado(nombre, color_arriba, color_abajo, rug=0.08, y0=-0.7, y1=0.7):
    """Material con degradado vertical (coordenada Y del objeto)."""
    m = material(nombre, color_arriba, rugosidad=rug, brillo=0.8)
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    co = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = y0
    mr.inputs["From Max"].default_value = y1
    rp = nt.nodes.new("ShaderNodeValToRGB")
    rp.color_ramp.elements[0].color = hexa(color_abajo)
    rp.color_ramp.elements[1].color = hexa(color_arriba)
    nt.links.new(co.outputs["Object"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Y"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], rp.inputs["Fac"])
    nt.links.new(rp.outputs["Color"], p.inputs["Base Color"])
    return m


def orientar(g, giro=-25, inclina=0.0, z=0.0, x=0.0):
    """Pone de pie un modelo con el frente en +Z, mirando hacia la cámara (−Y) y girado `giro` grados."""
    g.rotation_euler = (math.radians(90 - inclina), 0, math.radians(giro))
    g.location = (x, 0, z)
    return g


def oro_m():
    return material("oro", "#F2C75C", rugosidad=0.16, metal=1.0)


# ================================================================== JOYERÍA
def gema(nombre, r, color, alto=0.55, lados=8, mat=None):
    """Talla brillante simplificada (mesa + corona + pabellón); cara plana hacia +Z."""
    bm = bmesh.new()
    mesa = [bm.verts.new((r * 0.55 * math.cos(2 * math.pi * i / lados), r * 0.55 * math.sin(2 * math.pi * i / lados), alto * 0.35))
            for i in range(lados)]
    cint = [bm.verts.new((r * math.cos(2 * math.pi * (i + 0.5) / lados), r * math.sin(2 * math.pi * (i + 0.5) / lados), 0.0))
            for i in range(lados)]
    punta = bm.verts.new((0, 0, -alto))
    bm.faces.new(mesa)
    for i in range(lados):
        j = (i + 1) % lados
        bm.faces.new((mesa[i], cint[i], mesa[j]) if False else (mesa[j], mesa[i], cint[i]))
        bm.faces.new((mesa[j], cint[i], cint[j]))
        bm.faces.new((cint[j], cint[i], punta))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(nombre)
    bm.to_mesh(me)
    bm.free()
    o = bpy.data.objects.new(nombre, me)
    bpy.context.collection.objects.link(o)
    o.data.materials.append(mat or material(nombre, color, rugosidad=0.03, transmision=0.85, ior=2.1, brillo=1.0))
    return o


def anillo(pal):
    oro = oro_m()
    g = []
    aro = toro("aro", 1.0, 0.17, (0, 0, 0), oro, seg_m=96, seg_n=32)
    g.append(aro)                              # círculo en el plano XY, gema arriba (+Y)
    g.append(cilindro("base", 0.42, 0.26, (0, 1.12, 0), oro, rot=(math.radians(90), 0, 0), bisel=0.05, seg=32))
    for i in range(4):
        a = math.pi / 4 + i * math.pi / 2
        g.append(esfera(f"garra{i}", 0.07, (0.4 * math.cos(a), 1.38, 0.4 * math.sin(a)), oro))
    gm = gema("gema", 0.46, pal["a"], alto=0.55, lados=8)
    gm.rotation_euler = (math.radians(-90), 0, 0)
    gm.location = (0, 1.4, 0)
    g.append(gm)
    for k in (-1, 1):
        s = gema(f"p{k}", 0.1, "#FFFFFF", alto=0.11)
        ang = math.radians(90 + k * 24)
        s.location = (math.cos(ang) * 1.12, math.sin(ang) * 1.12, 0.0)
        s.rotation_euler = (math.radians(-90), 0, ang - math.radians(90))
        g.append(s)
    return agrupar("anillo", g)


def aros(pal):
    oro = oro_m()
    perla = material("perla", "#FFF7EC", rugosidad=0.1, brillo=1.0, sheen=0.5)
    g = []
    for k, x in enumerate((-1.0, 1.0)):
        gm = gema(f"stud{k}", 0.3, pal["a"], alto=0.34, lados=8)
        gm.rotation_euler = (math.radians(-90), 0, 0)
        gm.location = (x, 2.15, 0)
        g.append(gm)
        g.append(toro(f"anilla{k}", 0.3, 0.055, (x, 1.62, 0), oro))
        v = cilindro(f"varilla{k}", 0.045, 0.34, (x, 1.17, 0), oro, seg=16)
        v.rotation_euler = (math.radians(90), 0, 0)
        g.append(v)
        g.append(esfera(f"perla{k}", 0.44, (x, 0.7, 0), perla))
    return agrupar("aros", g)


def colgante(pal):
    oro = oro_m()
    g = []
    pts = [Vector((t * 2.0, 2.45 - 1.2 * (1 - t * t) ** 1.0 + 0.0 * t, 0)) for t in [-1 + 2 * i / 44 for i in range(45)]]
    g.append(tubo("cadena", pts, 0.04, oro))
    for i in range(1, 44, 3):
        g.append(esfera(f"e{i}", 0.06, pts[i], oro))
    gm = gema("colgante", 0.6, pal["a"], alto=0.9, lados=8)
    gm.rotation_euler = (math.radians(-90), 0, 0)
    gm.location = (0, 0.82, 0)
    g.append(gm)
    g.append(toro("anilla", 0.13, 0.035, (0, 1.3, 0), oro, rot=(0, 0, 0)))
    return agrupar("colgante", g)


def pulsera(pal):
    """Brazalete esmaltado con incrustaciones de oro y piedras."""
    oro = oro_m()
    esmalte = material("esmalte", pal["a"], rugosidad=0.2, brillo=0.9)
    g = []
    g.append(toro("banda", 1.3, 0.24, (0, 0, 0), esmalte, seg_m=96, seg_n=32, escala=(1, 1, 1.1)))
    g.append(toro("filete1", 1.3, 0.05, (0, 0, 0.27), oro, seg_m=96, seg_n=16))
    g.append(toro("filete2", 1.3, 0.05, (0, 0, -0.27), oro, seg_m=96, seg_n=16))
    for i in range(9):
        ang = math.radians(-70 + i * 17.5)
        r = 1.3
        gm = gema(f"p{i}", 0.12 if i != 4 else 0.2, "#FFFFFF" if i != 4 else pal["d"], alto=0.1 if i != 4 else 0.14, lados=8)
        gm.location = (math.cos(ang + math.pi / 2) * r, math.sin(ang + math.pi / 2) * r, 0.0)
        gm.rotation_euler = (math.radians(-90), 0, ang)
        gm.location = (math.cos(ang + math.pi / 2) * (r + 0.24), math.sin(ang + math.pi / 2) * (r + 0.24), 0.0)
        g.append(gm)
    return agrupar("pulsera", g)


# ================================================================== GAFAS DE SOL
def gafas(pal, estilo=1):
    marco = material("marco", pal["c"] if estilo != 2 else pal["a"], rugosidad=0.25, brillo=0.8)
    metal = oro_m()
    lente = degradado("lente", "#171B28", pal["a"] if estilo == 1 else "#4F5B9A", rug=0.04)
    g = []
    sep = 1.3
    for k, x in enumerate((-sep, sep)):
        c2 = superelipse(0.9, 0.66, n=3.0, pts=72, taper=0.22)
        lz = relleno_2d(f"lente{k}", c2, grosor=0.05, mat=lente, subdiv=1, hinchar=0.05, resolucion=0.14)
        lz.location = (x, 0, 0.0)
        g.append(lz)
        c3 = [Vector((cx + x, cy, 0.03)) for cx, cy in superelipse(0.94, 0.70, n=3.0, pts=72, taper=0.22)]
        g.append(tubo(f"aro{k}", c3, 0.075, marco, cerrado=True, res=8))
    puente = [Vector((-0.45 + 0.9 * i / 8, 0.3 + 0.12 * math.sin(math.pi * i / 8), 0.03)) for i in range(9)]
    g.append(tubo("puente", puente, 0.08, marco))
    for s in (-1, 1):
        base = [Vector((s * (sep + 0.9), 0.36, 0.03)), Vector((s * (sep + 0.97), 0.34, -0.45)),
                Vector((s * (sep + 0.98), 0.3, -1.45)), Vector((s * (sep + 0.92), 0.18, -2.05)),
                Vector((s * (sep + 0.86), -0.05, -2.4))]
        sm = []
        for i in range(len(base) - 1):
            for j in range(8):
                u = j / 8
                sm.append(base[i] * (1 - u) + base[i + 1] * u)
        sm.append(base[-1])
        g.append(barrido(f"patilla{s}", sm, perfil_rect(0.17, 0.3, 0.45), marco, arriba=(0, 1, 0), subdiv=1))
        g.append(esfera(f"bisagra{s}", 0.075, (s * (sep + 0.9), 0.36, 0.03), metal))
    return agrupar("gafas", g)


# ================================================================== BOLSO
def _tono(hexa_, f):
    h = hexa_.lstrip("#")
    r, g_, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02X%02X%02X" % (min(255, int(r * f)), min(255, int(g_ * f)), min(255, int(b * f)))


def _bez(p0, p1, p2, p3, n=8):
    out = []
    for i in range(n + 1):
        u = i / n
        a, b, c, d = (1 - u) ** 3, 3 * (1 - u) ** 2 * u, 3 * (1 - u) * u * u, u ** 3
        out.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0], a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
    return out


def bolso(pal):
    cuero = material("cuero", pal["a"], rugosidad=0.36, brillo=0.45)
    solapa_m = material("solapa", _tono(pal["a"], 0.78), rugosidad=0.36, brillo=0.45)
    oscuro = material("oscuro", pal["c"], rugosidad=0.35, brillo=0.35)
    oro = oro_m()
    costura = material("costura", pal["d"], rugosidad=0.6)
    der = [(0.0, 0.0), (1.2, 0.0)] + _bez((1.2, 0.0), (1.6, 0.0), (1.66, 0.2), (1.62, 0.5))[1:]
    der += [(1.28, 2.0)] + _bez((1.28, 2.0), (1.28, 2.1), (1.2, 2.14), (1.1, 2.14))[1:] + [(0.0, 2.14)]
    izq = [(-x, y) for x, y in der[::-1][1:]]
    cont = der + izq
    g = [relleno_2d("cuerpo", cont, grosor=0.9, mat=cuero, subdiv=2, hinchar=0.28, resolucion=0.1)]
    sd = [(1.33, 2.08), (1.33, 1.55)] + _bez((1.33, 1.55), (1.3, 1.1), (0.7, 0.98), (0.0, 0.92))[1:]
    sol = [(-x, y) for x, y in sd[::-1][1:]]
    sol = sd + sol
    fl = relleno_2d("solapa", sol, grosor=0.12, mat=solapa_m, subdiv=2, hinchar=0.16, resolucion=0.1)
    fl.location = (0, 0, 0.88)
    g.append(fl)
    c = [Vector((x * 0.93, y * 0.97 + 0.05, 1.18)) for x, y in sd[1:]] + [Vector((-x * 0.93, y * 0.97 + 0.05, 1.18)) for x, y in sd[::-1][1:-1]]
    g.append(tubo("costura", c, 0.014, costura))
    cz = cilindro("cierre", 0.2, 0.1, (0, 1.12, 1.25), oro, bisel=0.02, seg=40)
    cz.rotation_euler = (math.radians(90), 0, 0)
    g.append(cz)
    g.append(esfera("cierre_p", 0.08, (0, 1.12, 1.32), oscuro))
    pts = [Vector((-0.8 * math.cos(math.pi * i / 30), 2.1 + 1.25 * math.sin(math.pi * i / 30), 0.0)) for i in range(31)]
    g.append(tubo("asa", pts, 0.12, oscuro, res=12))
    g.append(esfera("ra", 0.17, (-0.8, 2.1, 0.0), oro))
    g.append(esfera("rb", 0.17, (0.8, 2.1, 0.0), oro))
    g.append(esfera("pompon", 0.2, (1.45, 0.2, 1.05), solapa_m))
    return agrupar("bolso", g)
