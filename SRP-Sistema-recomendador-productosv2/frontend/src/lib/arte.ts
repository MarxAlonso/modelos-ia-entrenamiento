// Ilustraciones 3D (renders de Blender, ver /blender) y su asignación a productos y perfiles.
export const BASE_3D = '/3d';

export type Genero = 'hombre' | 'mujer';

export interface ProductoLike {
  product_id?: string;
  titulo?: string;
  categoria?: string;
  publico?: string;
}

const hash = (s: string) => {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return (h >>> 0) % 1000003;
};

/** Variante de color (1 coral, 2 verde azulado, 3 índigo) según para quién es el producto. */
function variantePorPublico(publico = ''): number {
  if (publico === 'mujer') return 1;
  if (publico === 'hombre') return 2;
  return 3; // unisex / niños
}

/** Ruta de la ilustración 3D de un producto del catálogo (modelo por categoría y por palabras del título). */
export function arteProducto(p: ProductoLike): string {
  const t = (p.titulo || '').toLowerCase();
  const id = p.product_id || t;
  const cat = p.categoria || 'otros';
  const v = variantePorPublico(p.publico);
  switch (cat) {
    case 'relojes':
      return `${BASE_3D}/productos/relojes-${v}.webp`;
    case 'joyeria': {
      // familias de ilustraciones: anillo [1, 6] · aros [2] · colgante/charm [3, 5] · pulsera [4]
      const h = hash(id);
      let n: number;
      if (/\b(ring|rings|anillo)\b/.test(t)) n = [1, 6][h % 2];
      else if (/(earring|stud|hoop|dangle|pearl)/.test(t)) n = 2;
      else if (/(bracelet|bangle|cuff|wristband|anklet|pulsera)/.test(t)) n = 4;
      else if (/(necklace|pendant|chain|choker|charm|locket)/.test(t)) n = [3, 5][h % 2];
      else n = [1, 2, 3, 4, 5, 6][h % 6];
      return `${BASE_3D}/productos/joyeria-${n}.webp`;
    }
    case 'lentes':
      return `${BASE_3D}/productos/lentes-${v}.webp`;
    case 'bolsos':
      return `${BASE_3D}/productos/bolsos-${v}.webp`;
    case 'calzado':
      return `${BASE_3D}/productos/calzado-${v}.webp`;
    case 'ropa': {
      const vestido = /(dress|gown|skirt|vestido|romper|jumpsuit)/.test(t) ||
        (p.publico === 'mujer' && !/(shirt|tee|t-shirt|hoodie|sweat|jacket|coat|pants|shorts|top|polo|jersey)/.test(t));
      return `${BASE_3D}/productos/${vestido ? 'ropa_m' : 'ropa_h'}-${v}.webp`;
    }
    case 'accesorios':
      return `${BASE_3D}/productos/accesorios-${v}.webp`;
    case 'ropa_interior':
      return `${BASE_3D}/productos/ropa_interior-${v}.webp`;
    default:
      return `${BASE_3D}/productos/otros-${v}.webp`;
  }
}

/** Color de fondo pastel de la tarjeta según la categoría. */
export const TINTE: Record<string, string> = {
  relojes: '#E3F1EE',
  joyeria: '#FBEFD3',
  lentes: '#FBE3DA',
  bolsos: '#FADFD5',
  calzado: '#E5E5FA',
  ropa: '#FBE1E8',
  accesorios: '#E3F1EE',
  ropa_interior: '#EDE3F6',
  otros: '#FBEFD3',
};

export const NOMBRE_CATEGORIA: Record<string, string> = {
  relojes: 'Relojes',
  joyeria: 'Joyería',
  lentes: 'Lentes',
  bolsos: 'Bolsos',
  calzado: 'Calzado',
  ropa: 'Ropa',
  accesorios: 'Accesorios',
  ropa_interior: 'Calcetines y básicos',
  otros: 'Otros',
};

// ---- perfiles (los 6 clientes de ejemplo del backend) ----------------------------------------
export interface Perfil {
  id: string; // para la URL y los renders: /3d/perfiles/<genero>-<id>.webp
  categoria: string; // categoría del backend (/escenarios)
  titulo: string;
  descripcion: string;
  tinte: string;
}

export const PERFILES: Perfil[] = [
  { id: 'relojes', categoria: 'relojes', titulo: 'Fan de los relojes', descripcion: 'Cronógrafos, deportivos y clásicos', tinte: '#E3F1EE' },
  { id: 'joyeria', categoria: 'joyeria', titulo: 'Amante de la joyería', descripcion: 'Anillos, aros y colgantes', tinte: '#FBEFD3' },
  { id: 'calzado', categoria: 'calzado', titulo: 'Siempre busca calzado', descripcion: 'Zapatillas para todos los días', tinte: '#E5E5FA' },
  { id: 'bolsos', categoria: 'bolsos', titulo: 'Fiel a los bolsos', descripcion: 'Del diario al de ocasión', tinte: '#FADFD5' },
  { id: 'lentes', categoria: 'lentes', titulo: 'Lentes de sol', descripcion: 'Para la playa y la ciudad', tinte: '#FBE3DA' },
  { id: 'variado', categoria: 'otros', titulo: 'Gustos variados', descripcion: 'De todo un poco', tinte: '#EDE3F6' },
];

export const perfilPorId = (id: string) => PERFILES.find((p) => p.id === id);
export const arteAvatar = (g: Genero) => `${BASE_3D}/avatares/${g}.webp`;
export const artePerfil = (g: Genero, id: string) => `${BASE_3D}/perfiles/${g}-${id}.webp`;
