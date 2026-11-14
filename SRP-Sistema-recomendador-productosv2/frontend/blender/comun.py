"""Utilidades comunes para generar los renders 3D del frontend con Blender (bpy).

Estilo: "arcilla suave" (clay) — materiales mate con un poco de brillo, bordes biselados,
luz de estudio suave y sombra de contacto. Fondo transparente: la tarjeta pone el color.

Uso:  blender -b -P <script>.py -- <argumentos>
"""
import math
import os
import sys

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

# ------------------------------------------------------------------ color
def lin(c):
    """sRGB 0-1 -> lineal."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def hexa(h, alfa=1.0):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return (lin(r), lin(g), lin(b), alfa)


# Paletas por variante: (principal, secundario, acento, detalle oscuro)
PALETAS = {
    1: dict(a="#E4572E", b="#F6D8C6", c="#2A2421", d="#FFC857"),   # terracota
    2: dict(a="#2A9D8F", b="#D9F0EC", c="#1D3557", d="#F4A261"),   # verde azulado
    3: dict(a="#5B5BD6", b="#E3E3FA", c="#1B1B3A", d="#FF8FA3"),   # índigo
}

# ------------------------------------------------------------------ escena
def reiniciar():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for col in list(bpy.data.collections):
        bpy.data.collections.remove(col)


def configurar_render(ancho=640, alto=640, muestras=64):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = muestras
    sc.cycles.use_denoising = True
    try:
        sc.cycles.denoiser = "OPENIMAGEDENOISE"
    except Exception:
        pass
    sc.cycles.max_bounces = 6
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.render.film_transparent = True
    sc.render.resolution_x = ancho
    sc.render.resolution_y = alto
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.image_settings.compression = 15
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.view_settings.gamma = 1.0
    # mundo: luz ambiente suave y cálida
    mundo = bpy.data.worlds.new("mundo")
    mundo.use_nodes = True
    bg = mundo.node_tree.nodes["Background"]
    bg.inputs[0].default_value = hexa("#FFF6EE")
    bg.inputs[1].default_value = 0.45
    sc.world = mundo
    return sc


def luz_area(nombre, loc, energia, tam, color="#FFFFFF", apuntar=(0, 0, 0.6)):
    datos = bpy.data.lights.new(nombre, "AREA")
    datos.energy = energia
    datos.size = tam
    datos.color = hexa(color)[:3]
    obj = bpy.data.objects.new(nombre, datos)
    bpy.context.collection.objects.link(obj)
    obj.location = loc
    objetivo = Vector(apuntar)
    direccion = objetivo - Vector(loc)
    obj.rotation_euler = direccion.to_track_quat("-Z", "Y").to_euler()
    return obj


def caja_global():
    """Caja envolvente (mín, máx) de todos los objetos con geometría, en coordenadas de mundo."""
    bpy.context.view_layer.update()
    mn = Vector((1e9, 1e9, 1e9))
    mx = Vector((-1e9, -1e9, -1e9))
    dg = bpy.context.evaluated_depsgraph_get()
    for o in bpy.context.scene.objects:
        if o.type not in ("MESH", "CURVE"):
            continue
        oe = o.evaluated_get(dg)
        for corner in oe.bound_box:
            w = oe.matrix_world @ Vector(corner)
            mn = Vector((min(mn.x, w.x), min(mn.y, w.y), min(mn.z, w.z)))
            mx = Vector((max(mx.x, w.x), max(mx.y, w.y), max(mx.z, w.z)))
    return mn, mx


def _esquinas_mundo():
    """Vértices evaluados de todos los objetos (mallas y curvas con bisel) en coordenadas de mundo."""
    import numpy as np
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    salida = []
    for o in bpy.context.scene.objects:
        if o.type not in ("MESH", "CURVE"):
            continue
        oe = o.evaluated_get(dg)
        me = oe.to_mesh()
        n = len(me.vertices)
        if n:
            co = np.empty(n * 3, dtype=np.float32)
            me.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3)
            m = np.array(oe.matrix_world)
            w = co @ m[:3, :3].T + m[:3, 3]
            salida.append(w[:: max(1, n // 4000)])
        oe.to_mesh_clear()
    todo = np.concatenate(salida)
    return [Vector(p) for p in todo]


def estudio(azimut=-28, elevacion=18, lente=70, escala_luz=1.0, relleno=0.80, centro=None, desplazar=(0.0, 0.0), **_):
    """Cámara + luces de estudio. Se encuadra proyectando la caja del modelo: ocupa `relleno` del cuadro."""
    from bpy_extras.object_utils import world_to_camera_view
    sc = bpy.context.scene
    pts = _esquinas_mundo()
    mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    c = Vector(centro) if centro else (mn + mx) / 2
    radio = max((mx - mn).length / 2, 0.5)
    cam = bpy.data.cameras.new("cam")
    cam.lens = lente
    cam.sensor_width = 36
    co = bpy.data.objects.new("cam", cam)
    bpy.context.collection.objects.link(co)
    sc.camera = co
    az, el = math.radians(azimut), math.radians(elevacion)
    dir_ = Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
    dist = radio * 4.0
    objetivo = c.copy()
    for _it in range(6):
        co.location = objetivo + dir_ * dist
        co.rotation_euler = (objetivo - co.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        proy = [world_to_camera_view(sc, co, p) for p in pts]
        x0, x1 = min(p.x for p in proy), max(p.x for p in proy)
        y0, y1 = min(p.y for p in proy), max(p.y for p in proy)
        ext = max(x1 - x0, y1 - y0)
        # reencuadrar: mover el objetivo hacia el centro de la proyección y acercar/alejar
        cx, cy = (x0 + x1) / 2 - 0.5 - desplazar[0], (y0 + y1) / 2 - 0.5 - desplazar[1]
        ancho_vis = 2 * dist * math.tan(math.atan(18 / lente))
        derecha = co.matrix_world.to_quaternion() @ Vector((1, 0, 0))
        arriba = co.matrix_world.to_quaternion() @ Vector((0, 1, 0))
        objetivo = objetivo + derecha * cx * ancho_vis + arriba * cy * ancho_vis
        dist *= max(ext / relleno, 0.2)
    co.location = objetivo + dir_ * dist
    co.rotation_euler = (objetivo - co.location).to_track_quat("-Z", "Y").to_euler()
    c = objetivo
    k = (radio / 2.6) ** 2 * escala_luz
    d = radio
    luz_area("key", c + Vector((-1.1 * d, -1.3 * d, 1.9 * d)), 700 * k, d * 1.8, "#FFF2E4", c)
    luz_area("fill", c + Vector((1.9 * d, -1.0 * d, 0.7 * d)), 220 * k, d * 2.2, "#E6EEFF", c)
    luz_area("rim", c + Vector((0.6 * d, 1.8 * d, 1.4 * d)), 380 * k, d * 1.2, "#FFFFFF", c)
    luz_area("top", c + Vector((0, 0, 2.6 * d)), 120 * k, d * 2.4, "#FFFFFF", c)
    return co


def renderizar(ruta):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    sc = bpy.context.scene
    sc.render.filepath = ruta
    bpy.ops.render.render(write_still=True)
    print("RENDER ->", ruta)


# ------------------------------------------------------------------ materiales
def material(nombre, color, rugosidad=0.5, metal=0.0, brillo=0.0, sheen=0.0, subsurface=0.0, alfa=1.0,
             transmision=0.0, ior=1.45, emision=None, ss_color=None, bump=0.0):
    m = bpy.data.materials.new(nombre)
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    col = hexa(color) if isinstance(color, str) else color
    p.inputs["Base Color"].default_value = col
    p.inputs["Roughness"].default_value = rugosidad
    p.inputs["Metallic"].default_value = metal
    p.inputs["IOR"].default_value = ior
    if brillo:
        for k in ("Coat Weight", "Clearcoat"):
            if k in p.inputs:
                p.inputs[k].default_value = brillo
                break
        if "Coat Roughness" in p.inputs:
            p.inputs["Coat Roughness"].default_value = 0.08
    if sheen:
        for k in ("Sheen Weight", "Sheen"):
            if k in p.inputs:
                p.inputs[k].default_value = sheen
                break
        if "Sheen Roughness" in p.inputs:
            p.inputs["Sheen Roughness"].default_value = 0.5
    if subsurface:
        for k in ("Subsurface Weight", "Subsurface"):
            if k in p.inputs:
                p.inputs[k].default_value = subsurface
                break
        if "Subsurface Radius" in p.inputs:
            p.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.2)
        if "Subsurface Scale" in p.inputs:
            p.inputs["Subsurface Scale"].default_value = 0.08
    if transmision:
        for k in ("Transmission Weight", "Transmission"):
            if k in p.inputs:
                p.inputs[k].default_value = transmision
                break
    if emision:
        for k in ("Emission Color", "Emission"):
            if k in p.inputs:
                p.inputs[k].default_value = hexa(emision)
                break
        if "Emission Strength" in p.inputs:
            p.inputs["Emission Strength"].default_value = 1.0
    p.inputs["Alpha"].default_value = alfa
    return m


def tela(nombre, color, rugosidad=0.85, sheen=0.6):
    """Tela con textura de tejido muy sutil (bump procedural)."""
    m = material(nombre, color, rugosidad=rugosidad, sheen=sheen)
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = 220.0
    tex.inputs["Detail"].default_value = 6.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    bump.inputs["Distance"].default_value = 0.02
    nt.links.new(tex.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], p.inputs["Normal"])
    return m


def asignar(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return obj


# ------------------------------------------------------------------ geometría
def _enlazar(obj):
    bpy.context.collection.objects.link(obj)
    return obj


def desde_bmesh(nombre, bm, mat=None, suave=True, subdiv=0, bisel=0.0, bisel_seg=4, unir_normales=False):
    me = bpy.data.meshes.new(nombre)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(nombre, me)
    _enlazar(obj)
    if mat:
        obj.data.materials.append(mat)
    acabado(obj, suave=suave, subdiv=subdiv, bisel=bisel, bisel_seg=bisel_seg)
    return obj


def acabado(obj, suave=True, subdiv=0, bisel=0.0, bisel_seg=4):
    if bisel:
        md = obj.modifiers.new("bisel", "BEVEL")
        md.width = bisel
        md.segments = bisel_seg
        md.limit_method = "ANGLE"
        md.angle_limit = math.radians(35)
    if subdiv:
        md = obj.modifiers.new("subdiv", "SUBSURF")
        md.levels = subdiv
        md.render_levels = subdiv
    if suave:
        for pl in obj.data.polygons:
            pl.use_smooth = True
    return obj


def caja(nombre, tam, loc=(0, 0, 0), mat=None, bisel=0.05, rot=(0, 0, 0), seg=5):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = nombre
    o.scale = tam
    bpy.ops.object.transform_apply(scale=True)
    if mat:
        asignar(o, mat)
    acabado(o, suave=True, bisel=bisel, bisel_seg=seg)
    return o


def cilindro(nombre, radio, prof, loc=(0, 0, 0), mat=None, rot=(0, 0, 0), bisel=0.0, seg=48, bisel_seg=4):
    bpy.ops.mesh.primitive_cylinder_add(radius=radio, depth=prof, vertices=seg, location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = nombre
    if mat:
        asignar(o, mat)
    acabado(o, suave=True, bisel=bisel, bisel_seg=bisel_seg)
    return o


def esfera(nombre, radio, loc=(0, 0, 0), mat=None, escala=(1, 1, 1), seg=48, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radio, location=loc, segments=seg, ring_count=seg // 2, rotation=rot)
    o = bpy.context.active_object
    o.name = nombre
    o.scale = escala
    bpy.ops.object.transform_apply(scale=True)
    if mat:
        asignar(o, mat)
    acabado(o, suave=True)
    return o


def toro(nombre, mayor, menor, loc=(0, 0, 0), mat=None, rot=(0, 0, 0), seg_m=64, seg_n=24, escala=(1, 1, 1)):
    bpy.ops.mesh.primitive_torus_add(major_radius=mayor, minor_radius=menor, location=loc, rotation=rot,
                                     major_segments=seg_m, minor_segments=seg_n)
    o = bpy.context.active_object
    o.name = nombre
    o.scale = escala
    bpy.ops.object.transform_apply(scale=True)
    if mat:
        asignar(o, mat)
    acabado(o, suave=True)
    return o


def perfil_rect(ancho, grosor, redondeo=0.35, n=4):
    """Rectángulo redondeado como lista de (u, v) para barridos."""
    hw, ht = ancho / 2, grosor / 2
    r = min(hw, ht) * redondeo * 2
    pts = []
    for cx, cy, a0 in ((hw - r, ht - r, 0), (-hw + r, ht - r, 90), (-hw + r, -ht + r, 180), (hw - r, -ht + r, 270)):
        for i in range(n + 1):
            a = math.radians(a0 + 90 * i / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def perfil_elipse(a, b, n=20):
    return [(a * math.cos(2 * math.pi * i / n), b * math.sin(2 * math.pi * i / n)) for i in range(n)]


def barrido(nombre, puntos, perfil, mat=None, arriba=(0, 0, 1), escalas=None, cerrar_extremos=True, subdiv=1, bisel=0.0,
            giro=None):
    """Barre `perfil` (lista de (u,v)) a lo largo de `puntos` (lista de Vector).

    El marco usa `arriba` como referencia: u = binormal (lado), v = normal. `escalas` (lista) escala el perfil.
    `giro` (lista de ángulos) rota el perfil alrededor de la tangente.
    """
    pts = [Vector(p) for p in puntos]
    n = len(pts)
    bm = bmesh.new()
    anillos = []
    ref = Vector(arriba)
    for i, p in enumerate(pts):
        if i == 0:
            t = pts[1] - pts[0]
        elif i == n - 1:
            t = pts[-1] - pts[-2]
        else:
            t = pts[i + 1] - pts[i - 1]
        t.normalize()
        lado = t.cross(ref)
        if lado.length < 1e-4:
            lado = t.cross(Vector((1, 0, 0)))
        lado.normalize()
        norm = lado.cross(t).normalized()
        if giro:
            q = Matrix.Rotation(giro[i], 3, t)
            lado, norm = q @ lado, q @ norm
        s = escalas[i] if escalas else 1.0
        if isinstance(s, (int, float)):
            su, sv = s, s
        else:
            su, sv = s
        anillos.append([bm.verts.new(p + lado * u * su + norm * v * sv) for (u, v) in perfil])
    m = len(perfil)
    for i in range(n - 1):
        for j in range(m):
            k = (j + 1) % m
            bm.faces.new((anillos[i][j], anillos[i][k], anillos[i + 1][k], anillos[i + 1][j]))
    if cerrar_extremos:
        bm.faces.new(anillos[0][::-1])
        bm.faces.new(anillos[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return desde_bmesh(nombre, bm, mat, subdiv=subdiv, bisel=bisel)


def relleno_2d(nombre, contorno, grosor=0.0, mat=None, subdiv=2, hinchar=0.0, resolucion=0.08):
    """Superficie rellena a partir de un contorno 2D (x, y) -> 'almohadón' (para prendas).

    Triangula el contorno, la subdivide y desplaza cada vértice en Z según la distancia al borde (hinchado).
    Devuelve un objeto con grosor en Z (cara delantera y trasera).
    """
    import math as _m
    bm = bmesh.new()
    vs = [bm.verts.new((x, y, 0)) for x, y in contorno]
    bm.faces.new(vs)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    # subdividir hasta la resolución pedida
    for _ in range(6):
        aristas = [e for e in bm.edges if e.calc_length() > resolucion * 1.5]
        if not aristas:
            break
        bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=1, use_grid_fill=True)
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
    # distancia al contorno
    seg = [(Vector((contorno[i][0], contorno[i][1], 0)), Vector((contorno[(i + 1) % len(contorno)][0],
                                                                  contorno[(i + 1) % len(contorno)][1], 0)))
           for i in range(len(contorno))]

    def dist(p):
        d = 1e9
        for a, b in seg:
            ab = b - a
            t = max(0.0, min(1.0, (p - a).dot(ab) / max(ab.length_squared, 1e-9)))
            d = min(d, (p - (a + ab * t)).length)
        return d

    for v in bm.verts:
        d = dist(v.co)
        v.co.z = hinchar * (1 - _m.exp(-d * 6.0)) ** 0.7 if hinchar else 0.0
    me_front = bm
    obj = desde_bmesh(nombre, me_front, mat, subdiv=0)
    if grosor:
        md = obj.modifiers.new("solid", "SOLIDIFY")
        md.thickness = grosor
        md.offset = 0
    if subdiv:
        md = obj.modifiers.new("subdiv", "SUBSURF")
        md.levels = subdiv
        md.render_levels = subdiv
    return obj


def mover(obj, loc=None, rot=None, esc=None):
    if loc is not None:
        obj.location = loc
    if rot is not None:
        obj.rotation_euler = Euler(rot)
    if esc is not None:
        obj.scale = esc
    return obj


def agrupar(nombre, objetos, loc=(0, 0, 0)):
    """Empty padre para mover/rotar un conjunto."""
    e = bpy.data.objects.new(nombre, None)
    _enlazar(e)
    e.location = loc
    for o in objetos:
        o.parent = e
    return e


def args_script():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
