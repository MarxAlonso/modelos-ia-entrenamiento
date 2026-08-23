# Fase 10 — Optimización integral: más datos, más entrenamiento, GPU y app

## Objetivo

Consolidar el sistema recomendador (historial de compras · preferencias · comportamiento) ampliando los datasets, reentrenando toda la cadena sobre GPU y modernizando el explorador visual (react-icons, sin emojis).

## 1. Datasets ampliados (`src/ampliar_datos.py`)

### `data/interacciones.csv` — ampliado

| Métrica | Antes | Ahora |
|---------|-------|-------|
| Compras | 26 154 | **43 225 (+65 %)** |
| Usuarios / Productos | 500 / 6 570 | 500 / 6 570 |
| En categoría favorita | ~79 % | **78 %** |

Generación sesgada por las preferencias declaradas en `usuarios.csv` (75 % favoritas / 25 % exploración), ratings consistentes con el diseño original (normal 4.25/3.05 recortados), sin duplicar pares usuario-producto existentes.

### `data/comportamiento.csv` — NUEVO (señal de comportamiento)

Embudo implícito coherente con cada compra real:

| Evento | Registros | Significado |
|--------|-----------|-------------|
| view | 156 416 | Navegación (incluye vistas que no convierten) |
| cart | 36 707 | Carrito previo a la compra (85 % de compras) |
| purchase | 43 225 | La compra misma |

Columnas: `user_id, product_id, categoria_producto, tipo_evento, sesion_id, dispositivo, fecha, hora, duracion_seg`. Queda listo para feedback implícito, modelos de secuencia y análisis de sesión.

### Corpus de texto para Word2Vec

Se sumó `clip-ecommerce` (nombres + descripciones) al corpus existente y se cargan parquets por glob (resistente a sufijos hash).

## 2. Reentrenamiento completo en GPU

Infraestructura: Ubuntu 22.04 (WSL2) + TF 2.21 con CUDA 12 vía pip + driver Studio 610.88. Lanzadores permanentes: `src/gpu_env.sh`, `src/entrenar_gpu.ps1`.

| Modelo | Configuración nueva | Resultado |
|--------|--------------------|-----------|
| Word2Vec | Corpus +1913 descripciones extra · 40 épocas | Skip-gram gana: coherencia **84.3 %** |
| KNN ítem-ítem | 43 k compras | P@10 test **0.0216** (antes 0.0118, +83 %) |
| NCF (GPU) | Init Word2Vec + datos ampliados | P@10 test **0.0156**, **+82.4 % vs init aleatoria** |
| Híbrido 4 motores | Búsqueda 35 combos anti-fuga | Pesos (1, 0, 0, 0) — KNN domina con datos frescos |

**Hallazgo clave:** la inicialización semántica Word2Vec de la Fase 4b pasa de +5 % a **+82 %** cuando hay suficientes interacciones para explotarla: el conocimiento previo del lenguaje multiplica su valor con más señal colaborativa.

Nota honesta: la búsqueda del híbrido eligió KNN puro en la mitad de ajuste; con el dataset ampliado el filtrado colaborativo captura casi toda la señal explotable y los motores de contenido quedan como complemento para cold-start.

## 3. App React actualizada

- **react-icons 5.7 instalado**; los 8 emojis de navegación reemplazados por iconos Fa (Brain, Cube, CircleNodes, Bullseye, ChartColumn, User, MapLocationDot, Bolt)
- Estrellas de rating → componente `FaStar`; leyenda del pie de ratings con números planos
- Escaneo automático confirma **0 emojis restantes** en `src/`
- Los 12 JSONs regenerados con los nuevos modelos (métricas, embeddings W2V/semánticos, espacio latente, recomendaciones consultables de 500 usuarios)
- `pnpm build` ✓ y smoke test HTTP 200 ✓

## Artefactos

| Archivo | Contenido |
|---------|-----------|
| `data/comportamiento.csv` | Nuevo dataset de comportamiento (236 353 eventos) |
| `src/ampliar_datos.py` | Generador reproducible de ambos datasets |
| `models/embeddings_word2vec.npz` etc. | Reentrenados con corpus ampliado |
| `ver_grafos/app/src/data/*.json` | Reflejando los modelos vigentes |

## Resultados de la fase

- [x] +65 % de compras y dataset de comportamiento creado
- [x] Cadena completa reentrenada en GPU (4.1× más rápida)
- [x] NCF +82 % con init Word2Vec sobre datos ampliados
- [x] App sin emojis, con react-icons, build y servidor verificados
