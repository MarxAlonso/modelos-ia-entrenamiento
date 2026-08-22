"""Genera grafos 3D interactivos con 3d-force-graph (Three.js).

Uso:
    python ver_grafos/generar_grafos_3d.py

Genera en ver_grafos/salidas/:
    - 01_red_neuronal_3d.html      : arquitectura NCF como grafo 3D arrastrable
    - 02_embeddings_3d.html        : embeddings de productos en un espacio 3D real
    - 03_usuarios_categorias_3d.html : red usuario-categoria por afinidad
    - index.html                   : panel unificado (3D + vistas analiticas)

Los nodos se pueden ARRASTRAR con el mouse, la camara orbita/zoom con el
mouse y todo funciona sin servidor ni internet (libreria local).
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import sys
import time
import webbrowser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RUTA_BASE = Path(__file__).resolve().parents[1]
RUTA_MODELOS = RUTA_BASE / "models"
RUTA_DATOS = RUTA_BASE / "data"
SALIDAS = Path(__file__).resolve().parent / "salidas"
RUTA_LIB = SALIDAS / "lib" / "3d-force-graph.min.js"

URL_LIB = [
    "https://unpkg.com/3d-force-graph@1/dist/3d-force-graph.min.js",
    "https://cdn.jsdelivr.net/npm/3d-force-graph/dist/3d-force-graph.min.js",
]

COLORES_TIPO = {
    "InputLayer": "#60a5fa", "Embedding": "#a78bfa", "Flatten": "#94a3b8",
    "Multiply": "#fbbf24", "Concatenate": "#fbbf24", "Add": "#fbbf24",
    "Dense": "#34d399", "Dropout": "#cbd5e1", "Activation": "#f87171",
}

PALETA = ["#6366f1", "#f97316", "#10b981", "#ef4444", "#8b5cf6", "#0891b2",
          "#d946ef", "#84cc16", "#f59e0b", "#0ea5e9", "#ec4899", "#14b8a6"]


def asegurar_libreria() -> bool:
    if RUTA_LIB.exists():
        return True
    print("      descargando 3d-force-graph...")
    try:
        import requests
        RUTA_LIB.parent.mkdir(parents=True, exist_ok=True)
        for url in URL_LIB:
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                RUTA_LIB.write_bytes(r.content)
                return True
            except Exception:
                continue
    except ImportError:
        pass
    return False


def cargar_modelo_y_datos():
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    modelo = tf.keras.models.load_model(RUTA_MODELOS / "red_neuronal.keras")
    art = joblib.load(RUTA_MODELOS / "hibrido_v2.pkl")
    productos_info = pd.read_csv(
        RUTA_DATOS / "productos.csv", usecols=["product_id", "name", "main_category"]
    ).set_index("product_id")
    return modelo, art, productos_info


def info_capa(capa) -> str:
    tipo = type(capa).__name__
    unidades = getattr(capa, "units", None)
    activacion = getattr(capa, "activation", None)
    if callable(activacion):
        activacion = getattr(activacion, "__name__", "")
    try:
        params = capa.count_params()
    except Exception:
        params = 0
    partes = [f"<b>{capa.name}</b>", tipo]
    if unidades is not None:
        partes.append(f"unidades={unidades}")
    if activacion:
        partes.append(f"activacion={activacion}")
    partes.append(f"params={params:,}")
    return "<br>".join(partes)


def _nombres_keras_tensor(obj, salida):
    if isinstance(obj, dict):
        if obj.get("class_name") == "__keras_tensor__":
            historial = obj.get("config", {}).get("keras_history")
            if isinstance(historial, (list, tuple)) and historial:
                salida.append(str(historial[0]))
            return
        for valor in obj.values():
            _nombres_keras_tensor(valor, salida)
    elif isinstance(obj, (list, tuple)):
        for valor in obj:
            _nombres_keras_tensor(valor, salida)


def extraer_aristas(modelo):
    try:
        cfg = modelo.get_config()
        validos = {str(c.get("name")) for c in cfg["layers"]}
        aristas = []
        for capa_cfg in cfg["layers"]:
            nombre = str(capa_cfg.get("name"))
            nodos = capa_cfg.get("inbound_nodes") or []
            origenes = []
            _nombres_keras_tensor(nodos, origenes)
            for nodo in nodos:
                if not isinstance(nodo, (list, tuple)):
                    continue
                for ref in nodo:
                    if isinstance(ref, (list, tuple)) and ref and isinstance(ref[0], str):
                        origenes.append(str(ref[0]))
                    elif isinstance(ref, str):
                        origenes.append(ref)
            for origen in origenes:
                if origen != nombre and origen in validos and (origen, nombre) not in aristas:
                    aristas.append((origen, nombre))
        return aristas
    except Exception:
        return []


def datos_red_neuronal(modelo):
    capas = list(modelo.layers)
    aristas = extraer_aristas(modelo)
    nodos = []
    params_por_nombre = {}
    for c in capas:
        try:
            p = int(c.count_params())
        except Exception:
            p = 0
        params_por_nombre[c.name] = p
        nodos.append({
            "id": c.name,
            "tipo": type(c).__name__,
            "color": COLORES_TIPO.get(type(c).__name__, "#e2e8f0"),
            "val": float(np.log1p(p) * 2.2 + 2),
            "label": info_capa(c),
        })
    enlaces = [{"source": a, "target": b} for a, b in aristas]
    return {"nodes": nodos, "links": enlaces}


def datos_embeddings(art, productos_info, max_puntos=1500):
    from sklearn.decomposition import PCA
    modelo_path = RUTA_MODELOS / "red_neuronal.keras"
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    modelo = tf.keras.models.load_model(modelo_path)
    pesos = modelo.get_layer("embedding_producto").get_weights()[0].astype(np.float32)

    inv_prod = {col: pid for pid, col in art["idx_producto"].items()}
    n_validos = len(inv_prod)
    coords = PCA(n_components=3, random_state=42).fit_transform(pesos[:n_validos])
    coords *= 70.0 / max(np.abs(coords).max(), 1e-9)

    rng = np.random.default_rng(42)
    muestra = rng.choice(n_validos, size=min(max_puntos, n_validos), replace=False)

    cats = sorted(productos_info["main_category"].dropna().unique().tolist())
    color_cat = {c: PALETA[i % len(PALETA)] for i, c in enumerate(cats)}

    nodos = []
    for i in muestra:
        pid = inv_prod[int(i)]
        fila = productos_info.loc[pid]
        x, y, z = coords[i]
        nodos.append({
            "id": f"p{int(i)}",
            "name": str(fila["name"])[:60],
            "cat": str(fila["main_category"]),
            "color": color_cat[str(fila["main_category"])],
            "val": 0.9,
            "fx": float(x), "fy": float(y), "fz": float(z),
            "label": f"<b>{str(fila['name'])[:60]}</b><br>{fila['main_category']}",
        })
    return {"nodes": nodos, "links": [], "cats": cats,
            "colors": [color_cat[c] for c in cats]}


def datos_usuarios_categorias(art):
    p_cat = art["p_categoria_usuario"]
    lista_cats = art["lista_cats"]
    usuarios = list(art["idx_usuario"].keys())

    nodos, enlaces = [], []
    color_cat = {c: PALETA[i % len(PALETA)] for i, c in enumerate(lista_cats)}
    for j, cat in enumerate(lista_cats):
        nodos.append({"id": f"cat::{cat}", "name": cat, "cat": cat,
                      "color": color_cat[cat], "val": 14.0,
                      "label": f"<b>CATEGORIA</b><br>{cat}"})
    for u_idx, uid in enumerate(usuarios):
        orden = np.argsort(p_cat[u_idx])[::-1][:3]
        resumen = "<br>".join(f"{lista_cats[j]}: {p_cat[u_idx, j]:.0%}"
                              for j in orden[:2])
        nodos.append({"id": f"u::{uid}", "name": uid, "cat": "__usuario",
                      "color": "#38bdf8", "val": 0.7,
                      "label": f"<b>{uid}</b><br>{resumen}"})
        for j in orden:
            enlaces.append({
                "source": f"u::{uid}", "target": f"cat::{lista_cats[j]}",
                "w": float(p_cat[u_idx, j]),
            })
    return {"nodes": nodos, "links": enlaces}


HTML_PLANTILLA = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>{titulo}</title>
<style>
  body {{ margin:0; background:#0b1020; color:#e2e8f0;
         font-family:'Segoe UI', system-ui, sans-serif; overflow:hidden; }}
  #cont {{ width:100vw; height:100vh; }}
  .panel {{ position:fixed; top:16px; left:16px; z-index:10; max-width:330px;
           background:rgba(17,24,39,.92); padding:14px 18px; border-radius:12px;
           border:1px solid #334155; }}
  .panel h1 {{ margin:0 0 4px; font-size:17px; }}
  .panel .desc {{ font-size:12px; color:#94a3b8; line-height:1.45; margin-bottom:10px; }}
  select, button {{ background:#1e293b; color:#e2e8f0; border:1px solid #334155;
                   border-radius:6px; padding:4px 8px; font-size:12px; cursor:pointer; }}
  button:hover, select:hover {{ border-color:#60a5fa; }}
  .leyenda {{ position:fixed; bottom:16px; left:16px; z-index:10; display:flex;
             gap:12px; flex-wrap:wrap; background:rgba(17,24,39,.92); padding:10px 14px;
             border-radius:10px; border:1px solid #334155; font-size:11px; }}
  .leyenda span {{ display:inline-flex; align-items:center; gap:5px; }}
  .punto {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
</style>
</head>
<body>
<div class="panel">
  <h1>{titulo}</h1>
  <div class="desc">{descripcion}</div>
  <div id="controles">{controles}</div>
  <div class="hint" style="font-size:11px;color:#64748b;margin-top:8px">
    Arrastra los nodos &middot; click enfoca la camara &middot;
    arrastrar fondo = rotar &middot; rueda = zoom
  </div>
</div>
<div class="leyenda">{leyenda}</div>
<div id="cont"></div>

<script src="{ruta_js}"></script>
<script>
const DATOS = {datos_json};

const grafico = ForceGraph3D()(document.getElementById("cont"))
  .graphData(DATOS.grafo)
  .backgroundColor("#0b1020")
  .nodeColor(n => n.color)
  .nodeLabel(n => n.label)
  .nodeVal(n => n.val || 1)
  .nodeOpacity(0.95)
  .linkColor(l => "{color_enlace}")
  .linkOpacity({opacidad_enlace})
  .linkWidth(l => l.w ? l.w * {ancho_enlace} : 0.6)
  .linkDirectionalParticles({particulas})
  .linkDirectionalParticleWidth(1.6)
  .linkDirectionalParticleSpeed(0.006)
  .onNodeClick(nodo => {{
    const dist = 220 / Math.cbrt(Math.pow(nodo.x,2)+Math.pow(nodo.y,2)+Math.pow(nodo.z,2) || 1);
    grafico.cameraPosition(
      {{ x: nodo.x*dist, y: nodo.y*dist, z: nodo.z*dist }}, nodo, 900);
  }})
  .onEngineStop(() => grafico.zoomToFit(700, 60));

{script_extra}
window.addEventListener("resize", () => grafico.width(window.innerWidth).height(window.innerHeight));
</script>
</body>
</html>
"""

LEYENDA_RED = "".join(
    f'<span><span class="punto" style="background:{c}"></span>{t}</span>'
    for t, c in [("Entrada", COLORES_TIPO["InputLayer"]),
                 ("Embedding", COLORES_TIPO["Embedding"]),
                 ("Operacion", COLORES_TIPO["Multiply"]),
                 ("Dense", COLORES_TIPO["Dense"]),
                 ("Dropout", COLORES_TIPO["Dropout"]),
                 ("Salida", COLORES_TIPO["Activation"])]
)


def _etiqueta_libreria(ruta_js: str) -> str:
    """Incrusta la libreria dentro del HTML para evitar restricciones file://."""
    if RUTA_LIB.exists():
        codigo = RUTA_LIB.read_text(encoding="utf-8", errors="replace")
        return "<script>\n" + codigo + "\n</script>"
    return f'<script src="https://unpkg.com/3d-force-graph@1/dist/3d-force-graph.min.js"></script>'


def escribir_grafo(nombre: str, titulo: str, descripcion: str, datos: dict,
                   leyenda: str, controles: str = "", script_extra: str = "",
                   particulas: float = 0, color_enlace: str = "#475569",
                   opacidad_enlace: float = 0.35, ancho_enlace: float = 6.0,
                   ruta_js: str = "lib/3d-force-graph.min.js") -> Path:
    html = HTML_PLANTILLA.format(
        titulo=titulo, descripcion=descripcion,
        controles=controles, leyenda=leyenda,
        ruta_js=ruta_js,
        datos_json=json.dumps(datos, ensure_ascii=False),
        script_extra=script_extra, particulas=particulas,
        color_enlace=color_enlace, opacidad_enlace=opacidad_enlace,
        ancho_enlace=ancho_enlace,
    )
    html = html.replace(
        f'<script src="{ruta_js}"></script>', _etiqueta_libreria(ruta_js), 1
    )
    ruta = SALIDAS / nombre
    ruta.write_text(html, encoding="utf-8")
    return ruta


SCRIPT_FILTRO_EMBEDDINGS = """
const sel = document.getElementById("filtro");
sel.addEventListener("change", () => {
  const cat = sel.value;
  const nodos = cat === "__todas"
    ? DATOS.grafo.nodes
    : DATOS.grafo.nodes.filter(n => n.cat === cat);
  grafico.graphData({ nodes: nodos, links: [] });
  grafico.zoomToFit(600, 50);
});
"""

SCRIPT_PARTICULAS_TOGGLE = """
const btn = document.getElementById("toggle-flujo");
btn.addEventListener("click", () => {
  const activo = btn.dataset.on === "1";
  grafico.linkDirectionalParticles(activo ? 0 : 4);
  btn.dataset.on = activo ? "0" : "1";
  btn.textContent = activo ? "Mostrar flujo de datos" : "Pausar flujo de datos";
});
"""


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Dashboard 3D del recomendador")
    parser.add_argument("--servir", action="store_true",
                        help="sirve las salidas por http://127.0.0.1:8765 en vez de file://")
    parser.add_argument("--json", action="store_true",
                        help="genera JSONs en ver_grafos/app/src/data/ para la app React")
    args = parser.parse_args()

    inicio = time.time()
    SALIDAS.mkdir(parents=True, exist_ok=True)

    print("[1/4] Cargando modelos entrenados...")
    modelo, art, productos_info = cargar_modelo_y_datos()

    if args.json:
        generar_jsons_react(modelo, art, productos_info)
        print(f"JSONs listos en {time.time() - inicio:.1f}s")
        print("Para ver la app: cd ver_grafos/app && pnpm dev")
        return

    if not asegurar_libreria():
        print("AVISO: no se pudo descargar 3d-force-graph localmente;")
        print("       los HTML intentaran cargarlo desde CDN (requiere internet).")

    ruta_js = "lib/3d-force-graph.min.js" if RUTA_LIB.exists() \
        else "https://unpkg.com/3d-force-graph@1/dist/3d-force-graph.min.js"

    print("[2/4] Grafo 3D de la arquitectura de la red neuronal...")
    escribir_grafo(
        "01_red_neuronal_3d.html",
        "Red neuronal NCF en 3D",
        "Cada esfera es una capa entrenada; el tamano refleja sus parametros y las "
        "particulas muestran el flujo de datos desde las entradas hasta la salida.",
        datos_red_neuronal(modelo), LEYENDA_RED,
        controles='<button id="toggle-flujo" data-on="1">Pausar flujo de datos</button>',
        script_extra=SCRIPT_PARTICULAS_TOGGLE,
        particulas=4, opacidad_enlace=0.28,
    )

    print("[3/4] Espacio 3D de embeddings de productos...")
    datos_emb = datos_embeddings(art, productos_info)
    opciones = ['<select id="filtro"><option value="__todas">Todas las categorias</option>']
    opciones += [f'<option value="{c}">{c}</option>' for c in datos_emb["cats"]]
    opciones.append("</select>")
    ley_emb = "".join(
        f'<span><span class="punto" style="background:{datos_emb["colors"][i]}"></span>'
        f'{datos_emb["cats"][i]}</span>'
        for i in range(len(datos_emb["cats"]))
    )
    escribir_grafo(
        "02_embeddings_3d.html",
        "Embeddings de productos en 3D",
        "Posicion 3D aprendida por la red (PCA de los vectores latentes de 32 "
        "dimensiones). Productos cercanos son percibidos como similares.",
        datos_emb, ley_emb, controles="".join(opciones),
        script_extra=SCRIPT_FILTRO_EMBEDDINGS,
        particulas=0, opacidad_enlace=0.0,
    )

    print("[4/4] Red usuario-categoria y panel index...")
    datos_red_uc = datos_usuarios_categorias(art)
    ley_uc = ('<span><span class="punto" style="background:#38bdf8"></span>Usuario'
              '</span>' + "".join(
                  f'<span><span class="punto" style="background:{PALETA[i % len(PALETA)]}">'
                  f'</span>{c}</span>'
                  for i, c in enumerate(art["lista_cats"])))
    escribir_grafo(
        "03_usuarios_categorias_3d.html",
        "Red usuario-categoria (500 usuarios)",
        "Grafo bipartito: cada usuario se conecta a sus 3 categorias mas afines; "
        "el grosor del enlace es la probabilidad predicha. Se forman clusters "
        "naturales por gustos.",
        datos_red_uc, ley_uc,
        particulas=0, color_enlace="#38bdf855", ancho_enlace=9.0,
        ruta_js=ruta_js,
    )

    regenerar_index(art, len(modelo.layers))
    index_ruta = SALIDAS / "index.html"
    print(f"Listo en {time.time() - inicio:.1f}s -> {index_ruta}")

    if args.servir:
        import functools
        import http.server
        import socketserver
        import threading

        puerto = 8765
        manejador = functools.partial(
            http.server.SimpleHTTPRequestHandler, directory=str(SALIDAS)
        )
        with socketserver.TCPServer(("127.0.0.1", puerto), manejador) as httpd:
            url = f"http://127.0.0.1:{puerto}/index.html"
            print(f"Sirviendo en {url}  (Ctrl+C para detener)")
            threading.Timer(0.8, lambda: webbrowser.open(url)).start()
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nServidor detenido.")
    else:
        webbrowser.open(index_ruta.as_uri())


def generar_jsons_react(modelo, art, productos_info):
    """Genera JSONs en ver_grafos/app/src/data/ para la app React."""
    DATA_DIR = RUTA_BASE / "ver_grafos" / "app" / "src" / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Red neuronal / Torre doble ---
    print("  [json] Arquitectura torre doble...")
    ruta_arq_tt = RUTA_BASE / "models" / "torre_doble_arquitectura.json"
    if ruta_arq_tt.exists():
        red = json.loads(ruta_arq_tt.read_text(encoding="utf-8"))
    else:
        red = datos_red_neuronal(modelo)
    (DATA_DIR / "red_neuronal.json").write_text(
        json.dumps(red, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- 2. Embeddings 3D ---
    print("  [json] Embeddings 3D...")
    emb = datos_embeddings(art, productos_info)
    emb_nodes = []
    for i, nodo in enumerate(emb["nodes"]):
        emb_nodes.append({
            "id": nodo["id"],
            "nombre": nodo["name"],
            "category": nodo["cat"],
            "x": float(nodo["fx"]),
            "y": float(nodo["fy"]),
            "z": float(nodo["fz"]),
        })
    (DATA_DIR / "embeddings.json").write_text(
        json.dumps({"nodes": emb_nodes, "cats": emb["cats"], "colors": emb["colors"]},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # --- 3. Usuarios-Categorias 3D ---
    print("  [json] Usuarios-Categorias...")
    uc = datos_usuarios_categorias(art)
    uc_nodes = []
    for nodo in uc["nodes"]:
        uc_nodes.append({
            "id": nodo["id"],
            "type": "categoria" if nodo.get("fx") is not None else "usuario",
            "color": nodo.get("color", "#64748b"),
        })
    uc_links = []
    for link in uc["links"]:
        uc_links.append({
            "source": link["source"],
            "target": link["target"],
            "weight": float(link["w"]),
        })
    (DATA_DIR / "usuarios_categorias.json").write_text(
        json.dumps({"nodes": uc_nodes, "links": uc_links},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # --- 4. Metricas 2D ---
    print("  [json] Metricas...")
    ind = art["precision_individuales"]
    metricas = {
        "productos": {
            "labels": ["Popularidad", "Contenido TF-IDF", "NCF red neuronal",
                       "Afinidad categorias", "KNN item-item", "HIBRIDO V2"],
            "valores": [ind["popularidad"], ind["contenido"], ind["ncf"],
                       ind["afinidad"], ind["knn"], art["precision_at_k"]],
            "colores": ["#94a3b8"] * 5 + ["#ef4444"],
        },
        "categorias": {
            "labels": ["Precision categoria @10", "HitRate categoria @10"],
            "valores": [art["precision_categoria_test"], art["hitrate_categoria_test"]],
            "colores": ["#10b981", "#059669"],
        },
    }
    (DATA_DIR / "metricas.json").write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- 5. Arquitectura 2D ---
    print("  [json] Arquitectura 2D...")
    cfg = modelo.get_config()
    capas_data = []
    for capa_cfg in cfg["layers"]:
        nombre = capa_cfg.get("name")
        tipo = capa_cfg.get("class_name", "")
        capas_data.append({"name": nombre, "type": tipo})
    aristas_2d = extraer_aristas(modelo) or []
    (DATA_DIR / "arquitectura_2d.json").write_text(
        json.dumps({"nodes": capas_data, "links": [{"source": a, "target": b} for a, b in aristas_2d]},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # --- 6. Embeddings 2D (PCA) ---
    print("  [json] Embeddings 2D...")
    from sklearn.decomposition import PCA
    pesos = modelo.get_layer("embedding_producto").get_weights()[0].astype(np.float32)
    inv_prod = {col: pid for pid, col in art["idx_producto"].items()}
    n_validos = len(inv_prod)
    coords_2d = PCA(n_components=2, random_state=42).fit_transform(pesos[:n_validos])
    rng = np.random.default_rng(42)
    muestra = rng.choice(n_validos, size=min(3000, n_validos), replace=False)
    emb2d_nodes = []
    for i in muestra:
        pid = inv_prod[int(i)]
        fila = productos_info.loc[pid]
        emb2d_nodes.append({
            "x": float(coords_2d[i, 0]),
            "y": float(coords_2d[i, 1]),
            "name": str(fila["name"])[:40],
            "category": str(fila["main_category"]),
        })
    cats_2d = sorted(productos_info["main_category"].dropna().unique().tolist())
    (DATA_DIR / "embeddings_2d.json").write_text(
        json.dumps({"nodes": emb2d_nodes, "cats": cats_2d},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # --- 7. Afinidad Categorias ---
    print("  [json] Afinidad...")
    p_cat = art["p_categoria_usuario"]
    lista_cats = art["lista_cats"]
    usuarios = list(art["idx_usuario"].keys())
    afinidad = {
        "matrix": p_cat.tolist(),
        "categories": lista_cats,
        "usuarios_demo": ["u0007", "u0100", "u0250"],
    }
    (DATA_DIR / "afinidad.json").write_text(
        json.dumps(afinidad, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- 8. Flujo Hibrido (arquitectura final v4) ---
    print("  [json] Flujo Hibrido...")
    ruta_res34_f = RUTA_BASE / "models" / "nivel34_resultados.json"
    res_v4 = {}
    if ruta_res34_f.exists():
        res_v4 = json.loads(ruta_res34_f.read_text(encoding="utf-8")).get("HIBRIDO V4", {})
    cfg_art = art["config"]
    pesos_v4 = {"two_tower": 0.30, "lightgcn": 0.10, "ncf": 0.25,
                "knn": 0.15, "nlp_resenas": 0.20}
    flujo = {
        "pasos": [
            {"pos": 1.1, "titulo": "DATOS\nusuarios - productos - reseñas",
             "color_fondo": "#e2e8f0", "color_texto": "#334155"},
            {"pos": 3.2, "titulo": "EMBEDDINGS NLP\nSentence-BERT 384D + Word2Vec\nText Embedding reseñas ≥4★",
             "color_fondo": "#ddd6fe", "color_texto": "#4c1d95"},
            {"pos": 5.3, "titulo": "MODELOS DL/GNN\nTwo-Tower · LightGCN · NCF",
             "color_fondo": "#dbeafe", "color_texto": "#1e3a8a"},
            {"pos": 7.4, "titulo": f"HÍBRIDO V4\nTT {pesos_v4['two_tower']} · GNN {pesos_v4['lightgcn']} · NCF {pesos_v4['ncf']}\nKNN {pesos_v4['knn']} · NLP {pesos_v4['nlp_resenas']}",
             "color_fondo": "#fef3c7", "color_texto": "#78350f"},
            {"pos": 9.5, "titulo": "RANKING FINAL\n(excluye comprados)",
             "color_fondo": "#d1fae5", "color_texto": "#064e3b"},
            {"pos": 11.4, "titulo": f"TOP-10\nP@10={res_v4.get('precision_producto', 0):.4f}\nCatPrec={res_v4.get('precision_categoria', 0):.1%}",
             "color_fondo": "#fee2e2", "color_texto": "#7f1d1d"},
        ],
        "resultado": {
            "precision_categoria": res_v4.get("precision_categoria", art["precision_categoria_test"]),
            "hitrate": art["hitrate_categoria_test"],
            "precision_producto": res_v4.get("precision_producto", art["precision_at_k"]),
        },
    }
    (DATA_DIR / "flujo.json").write_text(
        json.dumps(flujo, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- 9. Word2Vec Embeddings ---
    print("  [json] Word2Vec Embeddings...")
    try:
        from gensim.models import Word2Vec
        w2v_model = Word2Vec.load(str(RUTA_BASE / "models" / "word2vec_productos.model"))
        w2v_data = np.load(RUTA_BASE / "models" / "embeddings_word2vec.npz", allow_pickle=True)
        embeddings_w2v = w2v_data["embeddings"]
        product_ids_w2v = w2v_data["product_ids"]
        
        # PCA 3D de embeddings Word2Vec
        from sklearn.decomposition import PCA
        pca_w2v = PCA(n_components=3, random_state=42)
        coords_w2v = pca_w2v.fit_transform(embeddings_w2v)
        
        # Muestra para visualización
        rng = np.random.default_rng(42)
        n_muestra = min(2000, len(embeddings_w2v))
        indices_muestra = rng.choice(len(embeddings_w2v), size=n_muestra, replace=False)
        
        w2v_nodes = []
        for idx in indices_muestra:
            pid = product_ids_w2v[idx]
            if pid in productos_info.index:
                row = productos_info.loc[pid]
                w2v_nodes.append({
                    "id": f"w2v_{idx}",
                    "nombre": str(row["name"])[:40],
                    "category": str(row["main_category"]),
                    "x": float(coords_w2v[idx, 0]),
                    "y": float(coords_w2v[idx, 1]),
                    "z": float(coords_w2v[idx, 2]),
                })
        
        w2v_cats = sorted(productos_info["main_category"].dropna().unique().tolist())
        
        (DATA_DIR / "word2vec.json").write_text(
            json.dumps({"nodes": w2v_nodes, "cats": w2v_cats,
                       "vocab_size": len(w2v_model.wv),
                       "vector_size": w2v_model.vector_size},
                      ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"    AVISO: No se pudo generar JSON Word2Vec: {e}")

    # --- 9b. Embeddings semanticos Transformer ---
    print("  [json] Semanticos Transformer...")
    try:
        ruta_sem = RUTA_BASE / "models" / "embeddings_semanticos.npz"
        if ruta_sem.exists():
            sem = np.load(ruta_sem, allow_pickle=True)
            emb_sem = sem["embeddings"]
            pids_sem = sem["product_ids"]
            modelo_st = str(sem["model"])
            from sklearn.decomposition import PCA
            coords_sem = PCA(n_components=3, random_state=42).fit_transform(emb_sem)
            rng2 = np.random.default_rng(42)
            idx_m = rng2.choice(len(emb_sem), size=min(2000, len(emb_sem)), replace=False)
            nodos_sem = []
            for i in idx_m:
                pid = pids_sem[i]
                if pid in productos_info.index:
                    fila = productos_info.loc[pid]
                    nodos_sem.append({
                        "id": f"s{int(i)}",
                        "nombre": str(fila["name"])[:40],
                        "category": str(fila["main_category"]),
                        "x": float(coords_sem[i, 0]),
                        "y": float(coords_sem[i, 1]),
                        "z": float(coords_sem[i, 2]),
                    })
            (DATA_DIR / "semantico.json").write_text(
                json.dumps({
                    "nodes": nodos_sem,
                    "cats": sorted(productos_info["main_category"].dropna().unique().tolist()),
                    "modelo": modelo_st,
                    "dim": int(emb_sem.shape[1]),
                    "benchmark": {"transformer": 81.6, "word2vec": 85.0, "tfidf": 88.7},
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            print("    AVISO: ejecuta primero src/embeddings_semanticos.py")
    except Exception as e:
        print(f"    AVISO: No se pudo generar JSON semantico: {e}")

    # --- 9c. Espacio latente unificado (NCF / W2V / Semantico / LightGCN) ---
    print("  [json] Espacio latente unificado...")
    try:
        from sklearn.decomposition import PCA
        metodos = {}

        def _pca3d(emb, ids, nombres, cats_prod, n_muestra=1200, seed=42):
            coords = PCA(n_components=3, random_state=seed).fit_transform(emb)
            coords *= 70.0 / max(float(np.abs(coords).max()), 1e-9)   # misma escala que NCF
            rng_m = np.random.default_rng(seed)
            idx_m = rng_m.choice(len(emb), size=min(n_muestra, len(emb)), replace=False)
            nodos = []
            for i in idx_m:
                pid = str(ids[i])
                if pid in productos_info.index:
                    fila = productos_info.loc[pid]
                    nodos.append({
                        "id": pid,
                        "nombre": str(fila["name"])[:40],
                        "category": str(fila["main_category"]),
                        "x": float(coords[i, 0]),
                        "y": float(coords[i, 1]),
                        "z": float(coords[i, 2]),
                    })
            return nodos

        pids_w2v = w2v_data["product_ids"]
        # NCF: reutilizar nodos ya proyectados por datos_embeddings
        metodos["ncf"] = [
            {"id": n["id"], "nombre": str(n["name"])[:40],
             "category": str(n["cat"]),
             "x": float(n["fx"]), "y": float(n["fy"]), "z": float(n["fz"])}
            for n in emb["nodes"]
        ]
        metodos["w2v"] = _pca3d(embeddings_w2v, pids_w2v, None, None)
        if ruta_sem.exists():
            sem2 = np.load(ruta_sem, allow_pickle=True)
            metodos["semantico"] = _pca3d(sem2["embeddings"], sem2["product_ids"], None, None)
        ruta_gnn = RUTA_BASE / "models" / "lightgcn_embeddings.npz"
        if ruta_gnn.exists():
            gnn = np.load(ruta_gnn)
            Ei = gnn["productos"]
            inv_idx = {col: pid for pid, col in art["idx_producto"].items()}
            ids_g = [inv_idx.get(i, f"p{i}") for i in range(len(Ei))]
            metodos["lightgcn"] = _pca3d(Ei, ids_g, None, None)

        (DATA_DIR / "espacio_latente.json").write_text(
            json.dumps({"metodos": metodos}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"    AVISO: espacio latente: {e}")

    # --- metricas extendidas con niveles 3-4 ---
    ruta_res34 = RUTA_BASE / "models" / "nivel34_resultados.json"
    if ruta_res34.exists():
        res34 = json.loads(ruta_res34.read_text(encoding="utf-8"))
        metricas["niveles34"] = res34
        metricas["nota_svd"] = ("SVD/NCF/Hibrido provienen de fases previas; SVD muestra "
                                "fuga de datos al evaluar sobre este split unificado.")
        (DATA_DIR / "metricas.json").write_text(
            json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # --- 10. Usuarios Demo con historial ---
    print("  [json] Usuarios Demo...")
    import pandas as pd
    df_int = pd.read_csv(RUTA_DATOS / "interacciones.csv")
    usuarios_demo_data = {}
    for uid in ["u0007", "u0100", "u0250", "u0300", "u0400"]:
        hist = df_int[df_int["user_id"] == uid]
        if len(hist) == 0:
            continue
        
        # Productos comprados
        prods_comprados = []
        for _, row in hist.head(10).iterrows():
            pid = row["product_id"]
            if pid in productos_info.index:
                prods_comprados.append({
                    "id": pid,
                    "nombre": str(productos_info.loc[pid, "name"])[:50],
                    "categoria": str(productos_info.loc[pid, "main_category"]),
                    "rating": float(row["rating"]),
                })
        
        # Categorías preferidas
        cats_pref = hist["categoria_producto"].value_counts().head(3)
        cats_data = [{"cat": c, "count": int(n)} for c, n in cats_pref.items()]
        
        usuarios_demo_data[uid] = {
            "historial": prods_comprados,
            "categorias_preferidas": cats_data,
            "total_compras": len(hist),
        }
    
    (DATA_DIR / "usuarios_demo.json").write_text(
        json.dumps(usuarios_demo_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- 11. Index/Dashboard ---
    print("  [json] Dashboard...")
    import time
    n_capas = len(modelo.layers)
    dashboard = {
        "fecha": time.strftime("%Y-%m-%d %H:%M"),
        "metricas": {
            "precision_categoria": f"{art['precision_categoria_test']:.1%}",
            "hitrate": f"{art['hitrate_categoria_test']:.0%}",
            "precision_producto": f"{art['precision_at_k']:.4f}",
            "recall": f"{art['recall_at_k']:.4f}",
            "n_capas": n_capas,
        },
    }
    (DATA_DIR / "dashboard.json").write_text(
        json.dumps(dashboard, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"  [json] 11 archivos generados en {DATA_DIR}")


def regenerar_index(art, n_capas: int) -> None:
    tarjetas_3d = [
        ("01_red_neuronal_3d.html", "Red neuronal en 3D",
         "Arquitectura completa de la red NCF como grafo arrastrable con flujo "
         "de datos animado entre capas."),
        ("02_embeddings_3d.html", "Embeddings en 3D",
         "Nube de puntos del espacio latente de productos; filtra por categoria "
         "y arrastra libremente."),
        ("03_usuarios_categorias_3d.html", "Usuarios y categorias",
         "Red bipartita de afinidad: clusters de usuarios alrededor de sus "
         "categorias preferidas."),
    ]
    tarjetas_2d = [
        ("01_arquitectura_red.html", "Arquitectura (2D)",
         "Version plana del grafo de capas con detalle de configuracion."),
        ("02_metricas_modelos.html", "Metricas comparadas",
         "Precision producto y categoria de todos los modelos."),
        ("03_embeddings_productos.html", "Embeddings (PCA 2D)",
         "Proyeccion 2D interactiva del espacio latente."),
        ("04_afinidad_categorias.html", "Afinidad (heatmap)",
         "Mapa de calor usuario-categoria."),
        ("05_flujo_hibrido.html", "Flujo del hibrido v2",
         "Pipeline explicado paso a paso."),
    ]

    def tarjeta(ruta, titulo, texto):
        return (f'<a class="tarjeta" href="{ruta}"><h3>{titulo}</h3>'
                f'<p>{texto}</p></a>')

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Sistema Recomendador - Dashboard 3D</title>
<style>
  body {{ font-family:'Segoe UI', system-ui, sans-serif; background:#0b1020; color:#e2e8f0;
         margin:0; padding:40px; }}
  h1 {{ font-size:30px; margin:0 0 6px; }}
  h2 {{ font-size:18px; margin:34px 0 14px; color:#93c5fd;
       border-bottom:1px solid #334155; padding-bottom:8px; }}
  .sub {{ color:#94a3b8; margin-bottom:26px; }}
  .metricas {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:10px; }}
  .chip {{ background:#111827; border:1px solid #334155; border-radius:12px;
          padding:14px 22px; }}
  .chip .num {{ font-size:26px; font-weight:700; }}
  .verde {{ color:#34d399; }} .azul {{ color:#60a5fa; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(270px,1fr));
          gap:16px; }}
  .tarjeta {{ background:#111827; border-radius:14px; padding:20px;
             border:1px solid #334155; transition:transform .15s,border-color .15s;
             display:block; color:inherit; text-decoration:none; }}
  .tarjeta:hover {{ transform:translateY(-3px); border-color:#60a5fa; }}
  .tarjeta h3 {{ margin:0 0 8px; font-size:16px; color:#93c5fd; }}
  .tarjeta p {{ margin:0; font-size:13px; color:#94a3b8; line-height:1.5; }}
  footer {{ margin-top:36px; color:#64748b; font-size:12px; }}
</style>
</head>
<body>
<h1>Panel 3D del Sistema Recomendador</h1>
<div class="sub">Grafos interactivos con Three.js / 3d-force-graph &middot;
arrastra nodos, orbita la camara &middot; generado el {time.strftime('%Y-%m-%d %H:%M')}</div>

<div class="metricas">
  <div class="chip"><div class="num verde">{art['precision_categoria_test']:.1%}</div>Precision categoria @10</div>
  <div class="chip"><div class="num verde">{art['hitrate_categoria_test']:.0%}</div>HitRate categoria @10</div>
  <div class="chip"><div class="num azul">{art['precision_at_k']:.4f}</div>P@10 producto</div>
  <div class="chip"><div class="num azul">{art['recall_at_k']:.4f}</div>Recall@10</div>
</div>

<h2>Grafos 3D interactivos</h2>
<div class="grid">{"".join(tarjeta(*t) for t in tarjetas_3d)}</div>

<h2>Vistas analiticas 2D</h2>
<div class="grid">{"".join(tarjeta(*t) for t in tarjetas_2d)}</div>

<footer>Regenerar: python ver_grafos/generar_grafos.py (2D plotly) |
python ver_grafos/generar_grafos_3d.py (3D Three.js)</footer>
</body>
</html>
"""
    (SALIDAS / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
