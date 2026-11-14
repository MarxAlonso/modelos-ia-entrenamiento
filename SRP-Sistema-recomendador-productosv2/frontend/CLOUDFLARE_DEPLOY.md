# Despliegue de SRP v2 Frontend a Cloudflare Pages

Este frontend de Astro consume la API del backend FastAPI (modelo entrenado) a través de un proxy (endpoint de Astro en `src/pages/api/[...ruta].ts`).

## Estructura

```
frontend/
  public/3d/         Ilustraciones 3D (WebP con transparencia) hechas con Blender
  blender/           Scripts de Blender que generan esas imágenes (ver blender/README.md)
  src/
    pages/           Rutas: / (elegir género y perfil), /perfil (recomendaciones), /metricas, /resenas
      api/[...ruta].ts   Proxy genérico: /api/<ruta> -> <API_URL>/<ruta> (backend FastAPI)
    components/      Navbar y Footer
    layouts/         Layout base y sistema de diseño (CSS global)
    lib/             api.ts (cliente), arte.ts (producto -> ilustración 3D, perfiles), ui.ts (formato)
  astro.config.mjs   Configuración con adapter @astrojs/cloudflare
  wrangler.jsonc     Configuración de Wrangler
  .dev.vars          Variables de entorno para desarrollo local
```

El navegador solo habla con este origen (`/api/...`); el proxy reenvía al backend con la misma ruta, así que no hay CORS
ni se expone la URL del backend.

## Variables de entorno

El frontend necesita saber la URL del backend FastAPI. Configura `API_URL` en:

### Desarrollo local
Archivo `frontend/.dev.vars`:
```
API_URL=http://localhost:8000
```

### Producción (Cloudflare Pages)
En el dashboard de Cloudflare Pages → Configuración → Variables de entorno, agregar:
- `API_URL` = `https://tu-backend-api.com` (donde esté desplegado el FastAPI)

## Despliegue

### 1. Autenticarse con Cloudflare
```bash
cd frontend
npx wrangler login
```

### 2. Desplegar a Cloudflare Pages
```bash
cd frontend
npm run build
npx wrangler pages deploy dist/ --project-name=srp-v2-frontend
```

O usar GitHub Actions para despliegue automático en cada push.

### 3. Desplegar el backend FastAPI

El backend FastAPI debe estar desplegado en algún servidor accesible públicamente (por ejemplo, en una VM, Railway, Render, o Cloudflare Workers con ONNX Runtime).

El backend sirve en `MODELOS_DIR=models/servir/` y expone:
- `GET /salud` - Estado del modelo
- `GET /recomendar/usuario/{user_id}` - Recomendaciones por usuario
- `POST /recomendar/consulta` - Recomendaciones por consulta de texto
- `POST /resenas/analizar` - Análisis de reseñas (estrellas + recomendaría)
- `GET /metricas` - Métricas de evaluación

## Archivos del modelo publicados

Los modelos entrenados están en `models/servir/`:
- `encoder.onnx` - Two-tower encoder (470 MB)
- `items.npy` - Embeddings de productos
- `sasrec.onnx` - SASRec secuencial
- `modelo_gbdt.txt` - Re-ranker GBDT (ganador, HR@10 0.9501)
- `calibrador.json` - Calibrador de confianza (isotonia)
- `rasgos.py` - Rasgos compartidos (36 rasgos)
- `calidad.onnx` - Modelo de calidad de reseñas (470 MB)
- `catalogo.csv` - Catálogo de productos (71,214 items)
- `historial.npz` + `usuarios.json` - Historiales de usuarios
- `manifiesto.json` - Versiones y métricas

## Métricas del modelo publicado

| Métrica | Valor |
|---|---|
| HR@10 (GBDT) | 0.9501 |
| NDCG@10 | 0.8488 |
| Selectivo 90% top-10 | 0.9501 precisión |
| Selectivo 90% top-1 | 0.9043 precisión |
| RMSE reseñas | 0.6343 |
| Accuracy "recomendaría" | 96.77% |

## Notas

- Las páginas son estáticas (HTML/CSS/JS); solo el proxy `/api/*` corre en el servidor (`prerender = false`).
- Los modelos ONNX son demasiado grandes para incluirlos en el frontend estático; se consumen a través de la API backend.
- Para desarrollo local, ejecuta el backend FastAPI en `localhost:8000` (`iniciar_demo.cmd` lo levanta) y el frontend con `npm run dev`.
