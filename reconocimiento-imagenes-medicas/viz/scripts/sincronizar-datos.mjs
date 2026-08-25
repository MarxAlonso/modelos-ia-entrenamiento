/**
 * Sincroniza los artefactos del experimento (fuera de viz/) hacia src/data y
 * public/ para que Astro los consuma estáticamente en build.
 *
 *  - models/registro_versiones.json      -> src/data/versiones.json (enriquecido)
 *  - data/registro_datasets.json         -> src/data/datasets.json
 *  - config/metricas de cada versión     -> embebidos en versiones.json
 *  - PNGs por versión                    -> public/artefactos/<carpeta>/
 *  - galerías Grad-CAM                   -> public/reportes/<carpeta>/ + src/data/galerias.json
 *
 * Uso: node scripts/sincronizar-datos.mjs
 */
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const RAIZ = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const SRC_DATA = join(RAIZ, "viz", "src", "data");
const PUBLIC = join(RAIZ, "viz", "public");
mkdirSync(SRC_DATA, { recursive: true });

const leerJSON = (p) => JSON.parse(readFileSync(p, "utf-8"));

// ---------------------------------------------------------------------------
// Versiones: registro + config + metricas + lista de artefactos copiados
// ---------------------------------------------------------------------------
const registro = leerJSON(join(RAIZ, "models", "registro_versiones.json"));
const versiones = [];
for (const entrada of registro.versiones) {
  const carpeta = join(RAIZ, "models", "versiones", entrada.carpeta);
  const config = existsSync(join(carpeta, "config.json"))
    ? leerJSON(join(carpeta, "config.json"))
    : {};
  const metricas = existsSync(join(carpeta, "metricas.json"))
    ? leerJSON(join(carpeta, "metricas.json"))
    : {};

  // Copiar PNGs disponibles de la versión
  const imagenes = [];
  for (const nombre of [
    "matriz_confusion.png",
    "matriz_confusion_int8.png",
    "curvas_entrenamiento.png",
    "barrido_umbral.png",
  ]) {
    const origen = join(carpeta, nombre);
    if (existsSync(origen)) {
      const destinoDir = join(PUBLIC, "artefactos", entrada.carpeta);
      mkdirSync(destinoDir, { recursive: true });
      cpSync(origen, join(destinoDir, nombre));
      imagenes.push(`/artefactos/${entrada.carpeta}/${nombre}`);
    }
  }

  versiones.push({ ...entrada, config, metricas, imagenes });
}

writeFileSync(
  join(SRC_DATA, "versiones.json"),
  JSON.stringify(
    { actualizada: registro.actualizada, mejorPorDataset: registro.mejor_por_dataset ?? {}, versiones },
    null,
    2
  )
);

// ---------------------------------------------------------------------------
// Datasets
// ---------------------------------------------------------------------------
cpSync(join(RAIZ, "data", "registro_datasets.json"), join(SRC_DATA, "datasets.json"));

// ---------------------------------------------------------------------------
// Galerías Grad-CAM
// ---------------------------------------------------------------------------
const reportesRaiz = join(RAIZ, "data", "reportes");
const galerias = [];
for (const dir of readdirSync(reportesRaiz)) {
  if (!dir.startsWith("gradcam_")) continue;
  const origen = join(reportesRaiz, dir);
  const destino = join(PUBLIC, "reportes", dir);
  mkdirSync(destino, { recursive: true });
  cpSync(origen, destino, { recursive: true });
  let resumen = null;
  try {
    resumen = leerJSON(join(origen, "resumen_gradcam.json"));
  } catch {}
  galerias.push({
    carpeta: dir,
    versionId: dir.replace("gradcam_", ""),
    base: `/reportes/${dir}`,
    overlays: readdirSync(origen).filter((f) => f.startsWith("gradcam_") && f.endsWith(".png")).sort(),
    resumen,
  });
}
writeFileSync(join(SRC_DATA, "galerias.json"), JSON.stringify(galerias, null, 2));

console.log(`sync OK: ${versiones.length} versiones · ${galerias.length} galerías Grad-CAM`);
