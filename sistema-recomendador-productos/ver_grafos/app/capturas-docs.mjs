import { chromium } from '@playwright/test';
import { mkdirSync } from 'fs';

const DEST = '../../docs/capturas';
mkdirSync(DEST, { recursive: true });

const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1400, height: 900 } });
await p.goto('http://localhost:5173/', { waitUntil: 'networkidle' });
await p.waitForTimeout(3000);

async function capturar(nombre, opciones = {}) {
  await p.screenshot({ path: `${DEST}/${nombre}.png`, ...opciones });
  console.log('OK', nombre);
}

// 1. Red neuronal 3D (torre doble)
await p.locator('nav button', { hasText: 'Red Neuronal' }).click();
await p.waitForTimeout(5000);
await capturar('01-red-neuronal-torre-doble');

// 2-5. Espacio latente: un metodo por captura
await p.locator('nav button', { hasText: 'Espacio Latente' }).click();
await p.waitForTimeout(2500);
const metodos = [
  ['NCF (red neuronal)', '02-latente-ncf'],
  ['Word2Vec', '03-latente-word2vec'],
  ['Semantico Transformer', '04-latente-semantico'],
  ['LightGCN', '05-latente-lightgcn'],
];
for (const [texto, archivo] of metodos) {
  await p.locator('main button', { hasText: texto }).first().click();
  await p.waitForTimeout(4500);
  await capturar(archivo);
}

// 6. Grafo usuario-producto
await p.locator('nav button', { hasText: 'Grafo Usuario' }).click();
await p.waitForTimeout(6000);
await capturar('06-grafo-usuario-producto');

// 7. Consultar recomendaciones
await p.locator('nav button', { hasText: 'Consultar' }).click();
await p.waitForTimeout(3500);
await capturar('07-consultar-recomendaciones');

// 8. Metricas (pagina completa para la tabla)
await p.locator('nav button', { hasText: 'Métricas' }).click();
await p.waitForTimeout(2500);
await capturar('08-metricas-modelos', { fullPage: true });

// 9. Perfiles
await p.locator('nav button', { hasText: 'Perfiles' }).click();
await p.waitForTimeout(2500);
await capturar('09-perfiles-usuarios', { fullPage: true });

// 10. Afinidad
await p.locator('nav button', { hasText: 'Afinidad' }).click();
await p.waitForTimeout(2500);
await capturar('10-afinidad-categorias', { fullPage: true });

// 11. Flujo
await p.locator('nav button', { hasText: 'Flujo' }).click();
await p.waitForTimeout(2500);
await capturar('11-flujo-hibrido-v5');

await b.close();
console.log('CAPTURAS LISTAS');
