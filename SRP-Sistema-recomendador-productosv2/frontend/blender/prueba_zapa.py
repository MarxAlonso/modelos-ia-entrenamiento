import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comun, productos
args = comun.args_script()
out = args[0]
for nombre, az, el in (("a", -35, 28), ("b", -110, 22), ("c", 160, 30)):
    comun.reiniciar(); comun.configurar_render(420, 420, 16)
    g = productos.zapatilla(productos.PALETAS[1])
    comun.estudio(azimut=az, elevacion=el)
    comun.renderizar(os.path.join(out, f"zapa_{nombre}.png"))
