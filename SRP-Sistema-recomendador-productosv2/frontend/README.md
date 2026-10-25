# SRP v2 Frontend

Sistema recomendador de productos con frontend en Astro desplegado en Cloudflare Pages.

## Desarrollo

```bash
cd frontend
npm run dev
```

Abre http://localhost:4321

## Build

```bash
npm run build
```

## Despliegue a Cloudflare Pages

```bash
npx wrangler login
npm run build
npx wrangler pages deploy dist/ --project-name=srp-v2-frontend
```

Ver `CLOUDFLARE_DEPLOY.md` para instrucciones completas.

## Variables de entorno

Crea `.dev.vars` con:
```
API_URL=http://localhost:8011
```

## Páginas

| Ruta | Descripción |
|---|---|
| `/` | Inicio con buscador de recomendaciones por usuario |
| `/recomendar/[userId]` | Recomendaciones para un usuario específico |
| `/resenas` | Analizador de reseñas (estrellas + recomendaría) |
| `/metricas` | Métricas de evaluación del modelo |
| `/404` | Página no encontrada |

## API Functions (proxy a backend)

| Ruta | Método | Descripción |
|---|---|---|
| `/api/salud` | GET | Estado del modelo |
| `/api/recomendar/usuario/:userId` | GET | Recomendaciones por usuario |
| `/api/recomendar` | POST | Recomendaciones por consulta de texto |
| `/api/resenas` | POST | Análisis de reseñas |
| `/api/metricas` | GET | Métricas de evaluación |
