// @ts-check
import { defineConfig } from 'astro/config';
import cloudflare from '@astrojs/cloudflare';

// https://astro.build/config
export default defineConfig({
  adapter: cloudflare(),
  site: 'https://srp-demo.pages.dev',
  // 'ignore': las rutas /api/... (proxy al backend) funcionan con o sin barra final
  trailingSlash: 'ignore',
  build: { format: 'directory' },
});
