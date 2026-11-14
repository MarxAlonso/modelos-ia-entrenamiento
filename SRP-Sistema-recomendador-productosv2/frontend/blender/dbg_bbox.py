import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy, comun, productos
from mathutils import Vector
for nombre in ("ropa_h", "joyeria"):
    v = 3 if nombre == "joyeria" else 1
    comun.reiniciar(); comun.configurar_render(64, 64, 1)
    productos.construir(nombre, v)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    for o in bpy.context.scene.objects:
        if o.type in ("MESH", "CURVE"):
            oe = o.evaluated_get(dg)
            ws = [oe.matrix_world @ Vector(c) for c in oe.bound_box]
            mn = [min(w[i] for w in ws) for i in range(3)]; mx = [max(w[i] for w in ws) for i in range(3)]
            print("DBG", nombre, o.name, o.type, [round(a, 2) for a in mn], [round(a, 2) for a in mx])
