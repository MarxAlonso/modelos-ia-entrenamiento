// Utilidades de presentación compartidas por las páginas.
import { arteProducto, TINTE, NOMBRE_CATEGORIA } from './arte';

// Algunos títulos del dataset vienen con UTF-8 leído como cp1252 ("Womenâ€™s"): se repara al mostrarlos.
const CP1252: Record<string, number> = {
  '€': 0x80, '‚': 0x82, 'ƒ': 0x83, '„': 0x84, '…': 0x85, '†': 0x86, '‡': 0x87, 'ˆ': 0x88, '‰': 0x89, 'Š': 0x8a, '‹': 0x8b,
  'Œ': 0x8c, 'Ž': 0x8e, '‘': 0x91, '’': 0x92, '“': 0x93, '”': 0x94, '•': 0x95, '–': 0x96, '—': 0x97, '˜': 0x98, '™': 0x99,
  'š': 0x9a, '›': 0x9b, 'œ': 0x9c, 'ž': 0x9e, 'Ÿ': 0x9f,
};
export function repararTexto(s: string): string {
  if (!/[âÃÂ]/.test(s)) return s;
  const bytes: number[] = [];
  for (const ch of s) {
    const c = ch.codePointAt(0)!;
    if (c < 256) bytes.push(c);
    else if (CP1252[ch] !== undefined) bytes.push(CP1252[ch]);
    else return s; // no es el patrón conocido: se deja igual
  }
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(new Uint8Array(bytes));
  } catch {
    return s;
  }
}

export const esc = (s: unknown) =>
  repararTexto(String(s ?? '')).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);

// Coma decimal; miles con espacio duro (71 214); sin separador en 4 cifras (4726).
const nf = (d: number) => new Intl.NumberFormat('es-ES', { minimumFractionDigits: d, maximumFractionDigits: d, useGrouping: 'min2' as any });
export const num = (n: number, d = 1) => nf(d).format(n).replace(/\.(?=\d{3}(\D|$))/g, '\u00A0');
export const pct = (frac: number, d = 1) => `${num(frac * 100, d)} %`;

export const corto = (t: string, n = 60) => {
  const s = (t || '').replace(/\s+/g, ' ').trim();
  return s.length > n ? s.slice(0, n - 1).trimEnd() + '…' : s;
};

export function estrellas(valor = 0) {
  const r = Math.max(0, Math.min(5, Math.round(valor)));
  return '★'.repeat(r) + '☆'.repeat(5 - r);
}

export const etiquetaPublico = (p = '') =>
  ({ mujer: 'Mujer', hombre: 'Hombre', unisex: 'Unisex', ninos: 'Niños' } as Record<string, string>)[p] || p;

export const nombreCategoria = (c = '') => NOMBRE_CATEGORIA[c] || c;

export function fecha(iso?: string) {
  if (!iso) return '';
  const d = new Date(iso + 'T00:00:00');
  return isNaN(+d) ? iso : d.toLocaleDateString('es-PE', { day: 'numeric', month: 'short', year: 'numeric' });
}

const PARASITAS = new Set(
  ('the and for with from this that your you our not are was but all any can new set pack size sizes inch men mens women womens ' +
    'man woman boys girls kids teen unisex adult adults style styles color colors wear comfort comfortable lightweight design ' +
    'designed perfect gift gifts quality high best soft fit fits made use compatible replacement band bands ' +
    'para con las los una uno del por').split(' '),
);

/** Palabras que más se repiten en los títulos de una lista de productos. */
export function palabrasFrecuentes(titulos: string[], n = 4): string[] {
  const cuenta = new Map<string, number>();
  for (const t of titulos) {
    const vistas = new Set<string>();
    for (const w of (t || '').toLowerCase().replace(/[^a-záéíóúñ ]+/g, ' ').split(/\s+/)) {
      if (w.length < 4 || PARASITAS.has(w) || vistas.has(w)) continue;
      vistas.add(w);
      cuenta.set(w, (cuenta.get(w) || 0) + 1);
    }
  }
  return [...cuenta.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, n)
    .map(([w]) => w);
}

/** Explicación en una frase de por qué se recomienda un producto (campo `porque` de la API). */
export function textoPorque(p: any, conComillas = true): string {
  const q = p?.porque;
  if (!q) return '';
  if (q.parecido_a) {
    const t = corto(q.parecido_a.titulo, 46);
    const sim = q.parecido_a.similitud != null ? ` (${Math.round(q.parecido_a.similitud * 100)} % de parecido)` : '';
    return `Se parece a tu compra #${q.parecido_a.posicion}: ${conComillas ? '“' + t + '”' : t}${sim}`;
  }
  if (q.popular) return `Es el #${q.popular.puesto} de los más comprados (${q.popular.compras} compras) y suele comprarse junto a lo que llevas`;
  return '';
}

/** Tarjeta de producto con su ilustración 3D. `i` es la posición (1-based). */
export function tarjetaProducto(p: any, i: number, opciones: { grande?: boolean; maxPuntaje?: number } = {}) {
  const tinte = TINTE[p.categoria] || '#fff3d6';
  const rating = Number(p.rating_promedio || 0);
  const afin = Math.max(6, Math.min(100, Math.round(((p.puntaje ?? 0) / (opciones.maxPuntaje || 1)) * 100)));
  const precio = p.precio != null ? `<span class="chip sol">$${num(Number(p.precio), 2)}</span>` : '';
  const porque = textoPorque(p);
  return `
  <article class="prod${opciones.grande ? ' prod-grande' : ''}" style="--tinte:${tinte}">
    <div class="arte"><span class="rank">${i}</span><img src="${arteProducto(p)}" alt="Ilustración 3D de ${esc(nombreCategoria(p.categoria))}" width="420" height="420" /></div>
    <div class="prod-cuerpo">
      <div class="chips"><span class="chip">${esc(nombreCategoria(p.categoria))}</span><span class="chip gris">${esc(etiquetaPublico(p.publico))}</span>${precio}</div>
      <h3 title="${esc(p.titulo)}">${esc(corto(p.titulo, opciones.grande ? 110 : 70))}</h3>
      <div class="estrellas" aria-label="${rating.toFixed(1)} de 5"><span>${estrellas(rating)}</span> <b>${num(rating, 1)}</b> <small>(${num(Number(p.numero_resenas || 0), 0)} ${Number(p.numero_resenas) === 1 ? 'reseña' : 'reseñas'})</small></div>
      <div class="afinidad" title="Qué tan alto puntúa el re-ranker este producto frente al primero"><span>Afinidad</span><div class="barra"><i style="width:${afin}%"></i></div></div>
      ${porque ? `<p class="porque">${esc(porque)}</p>` : ''}
    </div>
  </article>`;
}
