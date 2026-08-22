# Fase 1 — Datos: catálogo real + historial de interacciones

## Objetivo

Obtener y preparar los datos que alimentarán todos los modelos: el catálogo de productos (con texto para NLP) y las interacciones usuario-producto (para filtrado colaborativo).

## 1. Origen del dataset

**Fuente real:** [valentinafevu/productos-supermercado](https://huggingface.co/datasets/valentinafevu/productos-supermercado) (HuggingFace Hub), descargada con `huggingface_hub`.

### Contenido del repositorio original

| Tipo de archivo | Cantidad | ¿Se usó? |
|-----------------|----------|----------|
| Imágenes `.webp` de productos | 7450 | ❌ No (fuera del alcance NLP/tabular) |
| `train/metadata.csv` | 1 | ✅ **Sí — fuente principal** |
| README.md, .gitattributes | 2 | ❌ No |

> La descarga completa falló con error **429 Too Many Requests** (7453 archivos sin autenticación). Solución: descargar **solo el CSV** con `hf_hub_download`, dejando las imágenes fuera del proyecto.

- Tamaño del CSV: **3.14 MB**
- Ubicación local: `data/raw/train/metadata.csv`

## 2. Exploración del catálogo

### Esquema original (`metadata.csv`)

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `file_name` | str | Identificador de imagen (`0.webp`, …) |
| `name` | str | Nombre comercial del producto |
| `supermarket_category` | str | Categoría raíz (todos: "supermercado") |
| `main_category` | str | Categoría principal (12 valores) |
| `subcategory` | str | Subcategoría fina (67 valores) |
| `full_image_path` | str | Ruta de la imagen |
| `ai_generated_description` | str | Descripción en español generada por IA |

### Estadísticas encontradas

| Métrica | Valor |
|---------|-------|
| Productos totales | **7449** |
| Nombres únicos | 6054 (diferencias por presentación/peso: x10kg vs x5kg) |
| Categorías principales | 12 |
| Subcategorías | 67 |
| Longitud de descripción | mín 91 · promedio 379 · máx 1141 caracteres |
| Descripciones nulas | **1239** |

### Distribución por categoría principal

| main_category | Productos |
|---------------|-----------|
| despensa | 1770 |
| lacteos-huevos-y-refrigerados | 1058 |
| vinos-y-licores | 992 |
| pasabocas | 840 |
| panaderia-y-pasteleria | 596 |
| dulces-y-postres | 548 |
| charcuteria | 424 |
| bebidas | 414 |
| frutas-y-verduras | 304 |
| pescados-y-mariscos | 220 |
| carne-y-pollo | 152 |
| cuidado-personal | 131 |

## 3. Hallazgo crítico y decisión

El dataset es solo un **catálogo**: tiene productos pero **ni usuarios ni compras ni ratings**, que son indispensables para el filtrado colaborativo.

| Alternativa | Decisión |
|-------------|----------|
| Buscar otro dataset transaccional | Se perderían las descripciones reales en español |
| **Conservar el catálogo real + simular comportamiento de compra** | ✅ Elegida: práctica estándar en proyectos académicos |

La simulación no es aleatoria: cada usuario recibe **preferencias por categoría** y compra sesgado hacia ellas, generando patrones aprendibles (esto representa el "historial de compras" y el "comportamiento" del enunciado).

## 4. Diseño de la generación sintética

Parámetros implementados en `src/preparar_datos.py`:

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| Semilla | 42 | Reproducibilidad total |
| Usuarios | 500 | Volumen suficiente para SVD y embeddings |
| Categorías favoritas por usuario | 1 a 3 | Representa "preferencias" |
| Compras por usuario | 20 a 80 | Historial variado |
| Elección de producto | 75% categoría favorita / 25% exploración | Simula comportamiento real |
| Rating en favoritas | normal(4.3, 0.6) recortado a [1, 5] | Satisfacción alta |
| Rating fuera de favoritas | normal(3.0, 0.6) recortado a [1, 5] | Satisfacción media |
| Fechas | 180 días desde 2026-01-01 | Permite análisis temporal/recencia |
| ID de producto asignado | `p0000` … `p7448` | Clave primaria estable para uniones |
| Limpieza de nulos | descripción vacía → "Producto de \<subcategoría\>" | El texto nunca queda vacío para TF-IDF |

## 5. Tablas resultantes

### `data/productos.csv` — 7449 filas

| Columna | Origen |
|---------|--------|
| `product_id` | **Generado** (clave primaria) |
| `file_name`, `name`, `main_category`, `subcategory` | Del dataset real |
| `texto` | **Unión** de `name + ". " + descripcion` (rellenada si era nula) |

### `data/usuarios.csv` — 500 filas

| Columna | Descripción |
|---------|-------------|
| `user_id` | `u0000` … `u0499` (clave primaria) |
| `categorias_favoritas` | 1–3 categorías separadas por `\|` |

### `data/interacciones.csv` — 25 854 filas

| Columna | Descripción |
|---------|-------------|
| `user_id` | Clave foránea → usuarios |
| `product_id` | Clave foránea → productos |
| `categoria_producto` | Denormalizada para auditoría rápida |
| `rating` | 1.0 – 5.0 (feedback explícito) |
| `cantidad` | 1 – 3 unidades (comportamiento) |
| `fecha` | Fecha de la compra |

## 6. Modelo relacional y uniones (joins)

```
┌──────────────┐         ┌──────────────────┐         ┌──────────────┐
│  usuarios    │         │  interacciones   │         │  productos   │
│──────────────│         │──────────────────│         │──────────────│
│ user_id  PK  │◄───1:N──│ user_id       FK │         │ product_id PK│
│ categorias_  │         │ product_id    FK │───N:1──►│ name         │
│ favoritas    │         │ rating           │         │ main_category│
└──────────────┘         │ cantidad, fecha  │         │ texto        │
                         └──────────────────┘         └──────────────┘
```

Ejemplo típico de unión usado en las siguientes fases:

```python
import pandas as pd
df = (interacciones
      .merge(productos, on="product_id", how="left")
      .merge(usuarios, on="user_id", how="left"))
```

Con esta unión cada fila de compra queda enriquecida con: qué se compró, en qué categoría, si coincide con las preferencias del usuario y el texto descriptivo del producto.

## 7. Verificaciones de calidad ejecutadas

| Chequeo | Resultado | Interpretación |
|---------|-----------|----------------|
| Densidad matriz usuario×producto | 25 854 / (500 × 7449) = **0.69 %** | Dispersión típica de recomendadores reales |
| Rating promedio global | 4.00 | Coherente con el diseño |
| Compras en categoría favorita | **79 %** | Patrón fuerte → los modelos podrán aprender |
| Nulos en `texto` | 0 | NLP sin pérdida de datos |

## Resultados de la fase

- [x] Dataset real descargado y auditado (solo lo necesario)
- [x] Catálogo limpio con IDs estables y texto unificado
- [x] 500 usuarios con preferencias declaradas
- [x] 25 854 interacciones con ratings, cantidades y fechas
- [x] Esquema relacional documentado para las uniones
