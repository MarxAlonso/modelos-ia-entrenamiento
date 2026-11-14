"""Modelos 3D procedurales de cada categoría del catálogo. Cada función construye el objeto
apoyado sobre z=0 y devuelve el Empty padre. `pal` es una paleta de comun.PALETAS."""
import math

import bmesh
import bpy
from mathutils import Vector

from comun import *  # noqa: F401,F403


def _giro_z(o, grados):
    o.rotation_euler = (0, 0, math.radians(grados))
    return o


# ================================================================== RELOJ
def _aguja(nombre, largo, ancho, ang, z, mat, cola=0.12):
    """Aguja con pivote en el centro, apuntando a `ang` grados desde las 12 (sentido horario)."""
    b = caja(nombre, (ancho, largo + cola, 0.03), (0, (largo - cola) / 2, 0), mat, bisel=ancho * 0.3, seg=3)
    piv = bpy.data.objects.new(nombre + "_p", None)
    bpy.context.collection.objects.link(piv)
    piv.location = (0, 0, z)
    piv.rotation_euler = (0, 0, math.radians(-ang))
    b.parent = piv
    return piv


def reloj(pal, dorado=False):
    metal = material("caja", "#E9C46A" if dorado else "#DADFE6", rugosidad=0.2, metal=1.0)
    esfera_m = material("esfera", pal["a"], rugosidad=0.32, brillo=0.5)
    anillo_m = material("anillo", pal["b"], rugosidad=0.4, brillo=0.3)
    oscuro = material("oscuro", pal["c"], rugosidad=0.4)
    claro = material("marca", "#FFFFFF", rugosidad=0.35)
    correa_m = material("correa", pal["c"], rugosidad=0.5, brillo=0.25)
    costura = material("costura", pal["d"], rugosidad=0.6)
    seg_m = material("segundero", pal["d"], rugosidad=0.3, brillo=0.2)
    objs = []
    objs.append(cilindro("caja", 1.0, 0.36, (0, 0, 0.18), metal, bisel=0.09, seg=96))
    objs.append(toro("bisel", 0.9, 0.07, (0, 0, 0.37), metal, seg_m=96, seg_n=24))
    objs.append(cilindro("esfera", 0.85, 0.05, (0, 0, 0.36), esfera_m, seg=96))
    objs.append(cilindro("anillo", 0.8, 0.012, (0, 0, 0.392), anillo_m, seg=96))
    objs.append(cilindro("fondo", 0.7, 0.014, (0, 0, 0.4), esfera_m, seg=96))
    for i in range(12):
        a = math.radians(i * 30)
        grande = i % 3 == 0
        l, w = (0.16, 0.06) if grande else (0.09, 0.035)
        r = 0.66 if grande else 0.69
        objs.append(caja(f"h{i}", (w, l, 0.03), (math.sin(a) * r, math.cos(a) * r, 0.42), claro, bisel=0.01, seg=2,
                         rot=(0, 0, -a)))
    objs.append(_aguja("hora", 0.36, 0.1, 305, 0.44, claro))
    objs.append(_aguja("minuto", 0.58, 0.07, 60, 0.46, claro))
    objs.append(_aguja("segundero", 0.62, 0.025, 160, 0.48, seg_m, cola=0.2))
    objs.append(cilindro("centro", 0.075, 0.06, (0, 0, 0.5), seg_m, seg=32, bisel=0.01))
    objs.append(cilindro("corona", 0.1, 0.2, (1.08, 0, 0.18), metal, rot=(0, math.radians(90), 0), bisel=0.02, seg=24))
    for signo in (1, -1):
        n = 30
        pts = []
        for i in range(n):
            t = i / (n - 1)
            pts.append(Vector((0, signo * (0.92 + 1.5 * t), 0.18 - 0.55 * (t ** 2.2))))
        anc = [0.74 - 0.08 * (i / (n - 1)) for i in range(n)]
        objs.append(barrido("correa", pts, perfil_rect(1.0, 0.22, 0.4), correa_m, arriba=(0, 0, 1),
                            escalas=[(a, 1) for a in anc], subdiv=1))
        # costura y agujeros
        objs.append(barrido("costura", [Vector((0, p.y, p.z + 0.115)) for p in pts[3:-3]], perfil_rect(0.06, 0.02, 0.5),
                            costura, arriba=(0, 0, 1), subdiv=0))
    # hebilla en el tramo inferior
    objs.append(toro("hebilla", 0.3, 0.04, (0, -2.1, -0.15), metal, rot=(math.radians(80), 0, 0), seg_m=48, seg_n=16))
    return agrupar("reloj", objs)


from m_complementos import anillo, aros, colgante, gafas, bolso, orientar, pulsera  # noqa: E402
from m_calzado import zapatilla  # noqa: E402
from m_ropa import camiseta, vestido, gorro, calcetin, regalo  # noqa: E402


def construir(nombre, variante):
    # joyería tiene 6 variantes: 1 anillo, 2 aros, 3 colgante, 4 pulsera, 5 colgante coral, 6 anillo índigo
    pal = PALETAS[{1: 1, 2: 2, 3: 3, 4: 2, 5: 1, 6: 3}[variante]] if nombre == "joyeria" else PALETAS[variante]
    if nombre == "relojes":
        g = reloj(pal, dorado=(variante == 1))
        g.rotation_euler = (math.radians(90), 0, math.radians(-26))
        return g
    if nombre == "joyeria":
        g = {1: anillo, 2: aros, 3: colgante, 4: pulsera, 5: colgante, 6: anillo}[variante](pal)
        return orientar(g, giro={1: -52, 2: -22, 3: -18, 4: -50, 5: -15, 6: -40}[variante], inclina=10)
    if nombre == "lentes":
        return orientar(gafas(pal, variante), giro=-52, inclina=8)
    if nombre == "bolsos":
        return orientar(bolso(pal), giro=-28, inclina=8)
    if nombre == "calzado":
        g = zapatilla(pal)
        g.rotation_euler = (0, 0, math.radians(-32))
        return g
    if nombre == "ropa_h":
        return orientar(camiseta(pal), giro=-8)
    if nombre == "ropa_m":
        return orientar(vestido(pal), giro=-14)
    if nombre == "accesorios":
        g = gorro(pal)
        g.rotation_euler = (math.radians(-12), 0, math.radians(20))
        return g
    if nombre == "ropa_interior":
        g = calcetin(pal)
        g.rotation_euler = (0, 0, math.radians(-24))
        return g
    if nombre == "otros":
        return orientar(regalo(pal), giro=-30, inclina=10)
    raise ValueError(nombre)


# azimut / elevación de cámara por producto
CAMARA = {
    "calzado": dict(azimut=-18, elevacion=24),
    "gorro": dict(azimut=-10, elevacion=16),
    "accesorios": dict(azimut=-10, elevacion=14),
}
