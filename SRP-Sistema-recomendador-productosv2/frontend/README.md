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
API_URL=http://localhost:8000
```

## Páginas

| Ruta | Descripción |
|---|---|
| `/` | Elegir si eres hombre o mujer (personajes 3D) y tu perfil de compra (6 escenas 3D) |
| `/perfil/?g=mujer&p=relojes` | Tu perfil, "Pregúntale a la IA", lo más probable que compres y recomendaciones con arte 3D |
| `/metricas/` | Ranking, confiabilidad y calidad de reseñas, con sus límites |
| `/resenas/` | Estrellas estimadas y probabilidad de "recomendaría" para una reseña |

Necesita el backend en marcha (`iniciar_demo.cmd`, puerto 8000). Las imágenes de `public/3d/` se regeneran con
`blender/` (ver `blender/README.md`).
