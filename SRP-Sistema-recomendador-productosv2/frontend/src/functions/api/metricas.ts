import type { APIContext } from 'astro';

const API_URL = import.meta.env.API_URL || 'http://localhost:8011';

export async function GET(): Promise<Response> {
  const url = `${API_URL}/metricas`;

  try {
    const res = await fetch(url);
    const data = await res.json();
    return Response.json(data, { status: res.status });
  } catch (err) {
    return Response.json({ error: err.message }, { status: 502 });
  }
}
