import type { APIContext } from 'astro';

const API_URL = import.meta.env.API_URL || 'http://localhost:8011';

export async function POST({ request }: APIContext): Promise<Response> {
  const url = `${API_URL}/recomendar/consulta`;

  try {
    const body = await request.json();
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    return Response.json(data, { status: res.status });
  } catch (err) {
    return Response.json({ error: err.message }, { status: 502 });
  }
}
