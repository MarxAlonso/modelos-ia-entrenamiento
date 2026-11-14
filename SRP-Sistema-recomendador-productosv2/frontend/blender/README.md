# Ilustraciones 3D del frontend (Blender)

Todas las imágenes de `frontend/public/3d/` son **renders propios** hechos con Blender (Python/`bpy`, sin modelos ni
texturas descargadas). Los scripts de esta carpeta las regeneran de cero.

| Archivo | Qué hace |
|---|---|
| `comun.py` | Escena, materiales "arcilla", luces de estudio, encuadre automático y barrido de perfiles |
| `productos.py` | Despacho por categoría del catálogo (`relojes`, `joyeria`, `lentes`, `bolsos`, `calzado`, `ropa_h`, `ropa_m`, `accesorios`, `ropa_interior`, `otros`) y su paleta |
| `m_complementos.py` | Reloj, joyería (anillo, aros, colgante), gafas de sol, bolso |
| `m_calzado.py` | Zapatilla (suela, capellada por *loft*, cuello, lengüeta, cordones) |
| `m_ropa.py` | Camiseta, vestido, gorro, calcetín, caja de regalo |
| `m_personas.py` | Personajes hombre y mujer |
| `render_productos.py` | Renderiza los productos (3 variantes de color cada uno) |
| `render_personas.py` | Renderiza los personajes y las 12 escenas de perfil (6 perfiles × 2 géneros) |
| `a_webp.py` | Convierte los PNG a WebP con transparencia |

## Regenerar

Requiere Blender 5.x (se usó 5.3) y Pillow para convertir.

```powershell
$B = "C:\Program Files\Blender Foundation\Blender 5.3\blender.exe"
$OUT = "$env:TEMP\srp3d"

# 1. Personajes y perfiles (~8 min en CPU)
& $B -b -P render_personas.py -- $OUT todo ambos 64 640
# 2. Productos: 10 categorías x 3 variantes (~15 min en CPU)
& $B -b -P render_productos.py -- "$OUT\productos" todos todas 64 640
# 3. A WebP, dentro del frontend
python a_webp.py $OUT ..\public\3d
```

Argumentos: `render_productos.py -- <salida> <categoria[,categoria]|todos> <variante[,v]|todas> <muestras> <tamaño>`.
Para probar rápido: `... -- $OUT relojes 1 16 320`.

## Notas de diseño

- Estilo "arcilla": materiales mate con un poco de brillo, bordes biselados, luz suave de estudio, fondo transparente.
  La sombra de contacto la dibuja el CSS (`.arte::before`), no Blender.
- Paletas: variante 1 = coral (mujer), 2 = verde azulado (hombre), 3 = índigo (unisex/niños).
- Cada categoría tiene un solo modelo: las imágenes representan la **categoría** del producto, no su foto real.
- El encuadre se calcula proyectando los vértices evaluados en cámara (la caja de las curvas de Blender no es fiable).
