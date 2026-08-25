# Plan — Visualización con Astro

**ESTADO: ✅ Implementada (`viz/`).** Build estático verificado: 11 páginas. Ejecutar con:

```powershell
cd reconocimiento-imagenes-medicas/viz
pnpm dev        # sincroniza datos + http://localhost:4321
pnpm build      # genera dist/
```

Diseño: estética de sala de lectura radiológica (lightbox casi negro, métricas en IBM Plex Mono como anotaciones DICOM de esquina, Space Grotesk para display; teal = dominio v1, ámbar = dominio v2; esquinas de panel estilo visor). Datos: `scripts/sincronizar-datos.mjs` copia registro/configs/métricas/PNGs a `src/data` y `public` en cada dev/build.

Páginas implementadas: `/` (campeones por dominio + gráfico evolución f1_macro), `/versiones/` (tabla completa), `/versiones/[id]` (7 detalles con config, matriz, curvas, cuantización), `/datasets/` (v1+v2+descartados), `/gradcam/` (2 galerías con estadística centro/borde).

---

Documento original de planificación (referencia):

Dashboard estático dentro de `viz/` que muestra la evolución del modelo versión tras versión. Sin backend: Astro lee los JSON de registro en build time.

---

## 1. Fuente de datos

```text
models/registro_versiones.json   → métricas y estado de cada entrenamiento
models/versiones/*/config.json   → configuración detallada por versión
data/registro_datasets.json      → datasets usados
models/versiones/*/*.png         → matrices, curvas, galería Grad-CAM
```

Astro importa los JSON en el frontmatter (build time) y genera páginas 100% estáticas.

---

## 2. Páginas

| Ruta | Contenido |
|---|---|
| `/` | Resumen: mejor versión actual, tarjetas de métricas clave, gráfico de evolución Recall/F1/Accuracy por versión |
| `/versiones/` | Tabla comparativa completa (fp32 vs cuantizado, tamaños, tiempos) |
| `/versiones/[id]` | Detalle: config.json renderizado, matriz de confusión, curvas, notas |
| `/datasets/` | Datasets versionados, balance de clases, origen |
| `/gradcam/` | Galería de mapas de calor comparados entre versiones |

---

## 3. Stack

```text
astro + chart.js (gráficos ligeros)
Sin Tailwind inicialmente si no hace falta; CSS simple
pnpm como gestor (consistente con ver_grafos)
```

---

## 4. Comandos previstos

```powershell
cd reconocimiento-imagenes-medicas/viz
pnpm create astro@latest .
pnpm dev        # http://localhost:4321
pnpm build
```

---

## 5. Cuándo construirlo

Al terminar **Fase 3** (habrá al menos una versión v001 con métricas reales). Construirlo antes sería un dashboard vacío.

---

## 6. Nota sobre imágenes

Los PNG de cada versión se referencian por ruta relativa; para el build, Astro los sirve desde `public/` vía symlink/copy script o `vite` assets — decidir en implementación (Fase 8 del plan maestro).
