// Cliente de API para consumir el backend del SRP v2 a través de Cloudflare Pages Functions
// Las rutas /api/* son proxy funciones que redirigen al backend FastAPI

export const API_BASE = '/api';

export async function salud() {
  const res = await fetch(`${API_BASE}/salud`);
  if (!res.ok) throw new Error(`Error ${res.status}`);
  return res.json();
}

export async function recomendarUsuario(userId: string, k = 10) {
  const res = await fetch(`${API_BASE}/recomendar/usuario/${encodeURIComponent(userId)}?k=${k}`);
  if (!res.ok) throw new Error(`Error ${res.status}`);
  return res.json();
}

export async function recomendarConsulta(texto: string, k = 10, userId?: string, historialIds?: string[], usarLaya = false) {
  const body: any = { texto, k, usar_laya: usarLaya };
  if (userId) body.user_id = userId;
  if (historialIds) body.historial_ids = historialIds;

  const res = await fetch(`${API_BASE}/recomendar`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`Error ${res.status}`);
  return res.json();
}

export async function analizarResena(texto: string, producto?: string) {
  const body: any = { texto };
  if (producto) body.producto = producto;

  const res = await fetch(`${API_BASE}/resenas`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`Error ${res.status}`);
  return res.json();
}

export async function metricas() {
  const res = await fetch(`${API_BASE}/metricas`);
  if (!res.ok) throw new Error(`Error ${res.status}`);
  return res.json();
}

export async function usuariosEjemplo(n = 5, minimo = 3) {
  const res = await fetch(`${API_BASE}/usuarios/ejemplo?n=${n}&minimo=${minimo}`);
  if (!res.ok) throw new Error(`Error ${res.status}`);
  return res.json();
}

export async function productosBuscar(q: string, k = 20) {
  const res = await fetch(`${API_BASE}/productos/buscar?q=${encodeURIComponent(q)}&k=${k}`);
  if (!res.ok) throw new Error(`Error ${res.status}`);
  return res.json();
}

export async function productosSimilares(productId: string, k = 10) {
  const res = await fetch(`${API_BASE}/productos/${encodeURIComponent(productId)}/similares?k=${k}`);
  if (!res.ok) throw new Error(`Error ${res.status}`);
  return res.json();
}

export async function escenario() {
  const res = await fetch(`${API_BASE}/escenarios`);
  if (!res.ok) throw new Error(`Error ${res.status}`);
  return res.json();
}
