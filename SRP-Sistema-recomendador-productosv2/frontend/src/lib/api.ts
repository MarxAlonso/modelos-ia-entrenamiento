// Cliente de la API del SRP v2. Todas las rutas pasan por el proxy /api/* (ver src/pages/api/[...ruta].ts),
// que las reenvía al backend FastAPI con la misma ruta.
export const API_BASE = '/api';

export class ErrorApi extends Error {
  constructor(message: string, public estado = 0, public sinBackend = false) {
    super(message);
  }
}

async function pedir<T = any>(ruta: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${ruta}`, init);
  } catch {
    throw new ErrorApi('No se pudo conectar', 0, true);
  }
  if (res.status === 502) throw new ErrorApi('El backend del modelo no está disponible', 502, true);
  if (!res.ok) throw new ErrorApi(`Error ${res.status}`, res.status);
  return res.json();
}

const post = (cuerpo: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(cuerpo),
});

export const salud = () => pedir('/salud');
export const metricas = () => pedir('/metricas');
export const escenarios = () => pedir('/escenarios');
export const perfilUsuario = (id: string) => pedir(`/usuarios/${encodeURIComponent(id)}/perfil`);

export interface OpcionesUsuario {
  k?: number;
  publico?: string;
  categoria?: string;
}
export function recomendarUsuario(id: string, { k = 12, publico, categoria }: OpcionesUsuario = {}) {
  const q = new URLSearchParams({ k: String(k) });
  if (publico) q.set('publico', publico);
  if (categoria) q.set('categoria', categoria);
  return pedir(`/recomendar/usuario/${encodeURIComponent(id)}?${q}`);
}

export const recomendarConsulta = (texto: string, k = 6, userId?: string) =>
  pedir('/recomendar/consulta', post({ texto, k, user_id: userId, usar_laya: false }));

export const analizarResena = (texto: string) => pedir('/resenas/analizar', post({ texto }));
