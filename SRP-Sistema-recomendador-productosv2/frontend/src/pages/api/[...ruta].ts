import type { APIContext } from 'astro';

// Proxy genérico: /api/<ruta> -> <API_URL>/<ruta> (el backend FastAPI del modelo).
// El navegador solo habla con este origen, así no hay CORS ni se expone la URL del backend.
export const prerender = false;

async function urlBackend(): Promise<string> {
  let base: string | undefined;
  try {
    // Cloudflare (producción y `astro dev` con el adaptador): variables del Worker / .dev.vars
    const mod = await import('cloudflare:workers');
    base = (mod as any).env?.API_URL;
  } catch {
    /* fuera de Cloudflare */
  }
  base ||= (import.meta as any).env?.API_URL || (globalThis as any).process?.env?.API_URL;
  return (base || 'http://localhost:8000').replace(/\/+$/, '');
}

async function reenviar({ request, params, url }: APIContext): Promise<Response> {
  const base = await urlBackend();
  // En producción, sin API_URL público el Worker no puede llegar a un localhost: se avisa en vez de fallar raro.
  if ((import.meta as any).env?.PROD && /^https?:\/\/(localhost|127\.|0\.0\.0\.0)/.test(base)) {
    return Response.json({ error: 'backend_no_configurado', detalle: 'Falta la variable API_URL con la URL pública del backend.' }, { status: 502 });
  }
  const destino = `${base}/${params.ruta ?? ''}${url.search}`;
  const init: RequestInit = { method: request.method, headers: { Accept: 'application/json' } };
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    init.body = await request.text();
    (init.headers as Record<string, string>)['Content-Type'] = request.headers.get('Content-Type') || 'application/json';
  }
  try {
    const res = await fetch(destino, init);
    return new Response(res.body, {
      status: res.status,
      headers: { 'Content-Type': res.headers.get('Content-Type') || 'application/json', 'Cache-Control': 'no-store' },
    });
  } catch (err: any) {
    return Response.json(
      { error: 'backend_no_disponible', detalle: String(err?.message || err), destino },
      { status: 502 },
    );
  }
}

export const GET = reenviar;
export const POST = reenviar;
