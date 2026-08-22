"""Genera el dashboard interactivo de los modelos entrenados.

Uso:
    python ver_grafos/generar_grafos.py

Genera archivos HTML interactivos en ver_grafos/salidas/ y abre el index
en el navegador. Todo funciona sin servidor ni internet (plotly.js local).
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import time
import webbrowser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

RUTA_BASE = Path(__file__).resolve().parents[1]
RUTA_MODELOS = RUTA_BASE / "models"
RUTA_DATOS = RUTA_BASE / "data"
SALIDAS = Path(__file__).resolve().parent / "salidas"

COLORES_TIPO = {
    "InputLayer": "#3b82f6",
    "Embedding": "#8b5cf6",
    "Flatten": "#64748b",
    "Multiply": "#f59e0b",
    "Concatenate": "#f59e0b",
    "Add": "#f59e0b",
    "Dense": "#10b981",
    "Dropout": "#cbd5e1",
    "Activation": "#ef4444",
}

ARISTAS_FIJAAS_FALLBACK = [
    ("usuario", "embedding_usuario"), ("usuario", "sesgo_usuario"),
    ("producto", "embedding_producto"), ("producto", "sesgo_producto"),
    ("embedding_usuario", "flatten"), ("embedding_producto", "flatten_1"),
    ("sesgo_usuario", "flatten_2"), ("sesgo_producto", "flatten_3"),
    ("flatten", "interaccion_gmf"), ("flatten_1", "interaccion_gmf"),
    ("flatten", "concatenar"), ("flatten_1", "concatenar"),
    ("concatenar", "oculta_1"), ("oculta_1", "dropout"), ("dropout", "oculta_2"),
    ("contenido_espanol", "texto_espanol"),
    ("interaccion_gmf", "fusion"), ("oculta_2", "fusion"),
    ("texto_espanol", "fusion"),
    ("fusion", "fusion_oculta"), ("fusion_oculta", "logit"),
    ("logit", "sumar_logit"), ("sesgo_usuario", "sumar_logit"),
    ("sesgo_producto", "sumar_logit"),
    ("sumar_logit", "probabilidad_compra"),
]


def extraer_aristas(modelo):
    try:
        cfg = modelo.get_config()
        aristas = []
        for capa_cfg in cfg["layers"]:
            nombre = str(capa_cfg.get("name"))
            for nodo in capa_cfg.get("inbound_nodes") or []:
                if isinstance(nodo, dict):
                    refs = nodo.get("inbound_layers") or []
                    if isinstance(refs, str):
                        refs = [refs]
                    for r in refs:
                        if isinstance(r, (list, tuple)):
                            r = r[0]
                        aristas.append((str(r), nombre))
                else:
                    for ref in nodo:
                        if isinstance(ref, (list, tuple)):
                            aristas.append((str(ref[0]), nombre))
                        elif isinstance(ref, str):
                            aristas.append((ref, nombre))
        nombres_validos = {str(c.get("name")) for c in cfg["layers"]}
        aristas = [(a, b) for a, b in aristas if a in nombres_validos]
        return aristas or None
    except Exception:
        return None


def info_capa(capa) -> str:
    tipo = type(capa).__name__
    unidades = getattr(capa, "units", None)
    activacion = getattr(capa, "activation", None)
    if callable(activacion):
        activacion = getattr(activacion, "__name__", str(activacion))
    try:
        params = f"{capa.count_params():,}"
    except Exception:
        params = "-"
    detalle = [f"<b>{capa.name}</b>", tipo]
    if unidades is not None:
        detalle.append(f"unidades={unidades}")
    if activacion:
        detalle.append(f"activacion={activacion}")
    detalle.append(f"params={params}")
    return "<br>".join(detalle)


def grafo_arquitectura(modelo) -> go.Figure:
    capas = {c.name: c for c in modelo.layers}
    aristas = extraer_aristas(modelo) or ARISTAS_FIJAAS_FALLBACK
    aristas = [(a, b) for a, b in aristas if a in capas and b in capas]

    profundidad = {}
    def calcular(nombre):
        if nombre in profundidad:
            return profundidad[nombre]
        padres = [a for a, b in aristas if b == nombre]
        d = 0 if not padres else max(calcular(a) for a in padres) + 1
        profundidad[nombre] = d
        return d

    for nombre in capas:
        try:
            calcular(nombre)
        except RecursionError:
            profundidad[nombre] = 0

    columnas: dict[int, list[str]] = {}
    for nombre, d in profundidad.items():
        columnas.setdefault(d, []).append(nombre)

    pos = {}
    for d, grupo in sorted(columnas.items()):
        n = len(grupo)
        for i, nombre in enumerate(sorted(grupo)):
            pos[nombre] = (d * 240.0, (len(grupo) - i - 1) * 110.0)

    fig = go.Figure()
    trazos_x, trazos_y, textos_arista = [], [], []
    for a, b in aristas:
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        trazos_x += [x0, (x0 + x1) / 2, x1, None]
        trazos_y += [y0, (y0 + y1) / 2 + 18, y1, None]
    fig.add_trace(go.Scatter(
        x=trazos_x, y=trazos_y, mode="lines", line={"width": 1.4, "color": "#94a3b8"},
        hoverinfo="skip", showlegend=False,
    ))

    for nombre, capa in capas.items():
        x, y = pos[nombre]
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker={"size": 34, "color": COLORES_TIPO.get(type(capa).__name__, "#334155"),
                    "line": {"width": 2, "color": "white"}},
            text=nombre, textposition="top center",
            textfont={"size": 11, "color": "#0f172a"},
            customdata=[info_capa(capa)],
            hovertemplate="%{customdata}<extra></extra>",
            name=nombre, showlegend=False,
        ))
    fig.update_layout(
        title=f"Arquitectura de la red NCF entrenada ({len(capas)} capas)",
        height=820, template="plotly_white",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        annotations=[{
            "text": "Pasa el cursor sobre cada capa para ver su configuracion. "
                    "Arrastra para mover, rueda para zoom.",
            "xref": "paper", "yref": "paper", "x": 0, "y": -0.03, "showarrow": False,
            "font": {"size": 12, "color": "#64748b"},
        }],
    )
    return fig


def grafico_metricas(art) -> go.Figure:
    ind = art["precision_individuales"]
    modelos_prod = ["Popularidad", "Contenido TF-IDF", "NCF red neuronal",
                    "Afinidad categorias", "KNN item-item", "HIBRIDO V2"]
    valores_prod = [ind["popularidad"], ind["contenido"], ind["ncf"],
                    ind["afinidad"], ind["knn"], art["precision_at_k"]]
    colores_prod = ["#94a3b8"] * 5 + ["#ef4444"]

    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Precision@10 sobre PRODUCTO exacto", "Objetivo CATEGORIA (test completo)"))
    fig.add_trace(go.Bar(
        x=modelos_prod, y=valores_prod, marker_color=colores_prod,
        text=[f"{v:.4f}" for v in valores_prod], textposition="outside",
        hovertemplate="%{x}: %{y:.4f}<extra></extra>", showlegend=False,
    ), row=1, col=1)
    etiquetas_cat = ["Precision categoria @10", "HitRate categoria @10"]
    valores_cat = [art["precision_categoria_test"], art["hitrate_categoria_test"]]
    fig.add_trace(go.Bar(
        x=etiquetas_cat, y=valores_cat, marker_color=["#10b981", "#059669"],
        text=[f"{v:.1%}" for v in valores_cat], textposition="outside",
        hovertemplate="%{x}: %{y:.1%}<extra></extra>", showlegend=False,
    ), row=1, col=2)
    fig.update_yaxes(range=[0, 0.008], title_text="P@10", row=1, col=1)
    fig.update_yaxes(range=[0, 1.15], tickformat=".0%", row=1, col=2)
    fig.update_layout(title="Comparativa de metricas de los modelos entrenados",
                      template="plotly_white", height=520)
    return fig


def grafico_embeddings(modelo, art, productos_info) -> go.Figure:
    from sklearn.decomposition import PCA

    pesos = modelo.get_layer("embedding_producto").get_weights()[0].astype(np.float32)
    inv_prod = {col: pid for pid, col in art["idx_producto"].items()}
    n_validos = len(inv_prod)
    vectores = pesos[:n_validos]
    coords = PCA(n_components=2, random_state=42).fit_transform(vectores)

    rng = np.random.default_rng(42)
    muestra = rng.choice(n_validos, size=min(3000, n_validos), replace=False)

    fig = go.Figure()
    cats = productos_info["main_category"].dropna().unique().tolist()
    paleta = px_colors(len(cats))
    visibilidad_todas, botones = [], []
    for j, cat in enumerate(cats):
        idxs = [i for i in muestra
                if productos_info.loc[inv_prod[i], "main_category"] == cat]
        if not idxs:
            continue
        xs = coords[idxs, 0]
        ys = coords[idxs, 1]
        nombres = [productos_info.loc[inv_prod[i], "name"][:40] for i in idxs]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name=cat,
            marker={"size": 6, "color": paleta[j % len(paleta)], "opacity": 0.75},
            text=nombres,
            hovertemplate="%{text}<br>(" + cat + ")<extra></extra>",
        ))
        visibilidad_todas.append(True)
    botones.append({"label": "Todas", "method": "update",
                    "args": [{"visible": visibilidad_todas}, {}]})
    for j, cat in enumerate(cats):
        estado = [False] * len(visibilidad_todas)
        if j < len(estado):
            estado[j] = True
        botones.append({"label": cat, "method": "update",
                        "args": [{"visible": estado}, {}]})
    fig.update_layout(
        title="Embeddings de productos (32D -> 2D con PCA), aprendidos por la red NCF",
        template="plotly_white", height=780,
        legend={"title": {"text": "Categoria"}, "font": {"size": 10}},
        updatemenus=[{"buttons": botones, "direction": "down", "x": 1.02, "y": 1.15}],
        annotations=[{"text": "Cada punto es un producto; los cercanos son similares "
                              "para la red. Usa el menu o la leyenda para filtrar.",
                      "xref": "paper", "yref": "paper", "x": 0, "y": -0.05,
                      "showarrow": False, "font": {"size": 12, "color": "#64748b"}}],
    )
    return fig


def px_colors(n: int) -> list[str]:
    base = ["#6366f1", "#f97316", "#10b981", "#ef4444", "#8b5cf6", "#0891b2",
            "#d946ef", "#84cc16", "#f59e0b", "#0ea5e9", "#ec4899", "#14b8a6"]
    return (base * ((n // len(base)) + 1))[:n]


def grafico_afinidad(art, usuarios_demo) -> go.Figure:
    p_cat = art["p_categoria_usuario"]
    lista_cats = art["lista_cats"]
    usuarios = list(art["idx_usuario"].keys())

    fig = make_subplots(rows=2, cols=1, row_heights=[0.68, 0.32],
                        vertical_spacing=0.12,
                        subplot_titles=("Afinidad usuario-categoria (500 usuarios x 12 categorias)",
                                        "Top-3 categorias predichas para usuarios demo"))
    fig.add_trace(go.Heatmap(
        z=p_cat, x=lista_cats, y=list(range(len(usuarios))),
        colorscale="Viridis", colorbar={"title": "prob."}),
        row=1, col=1)
    fig.update_yaxes(title_text="# usuario", dtick=25, row=1, col=1)

    paleta = px_colors(len(lista_cats))
    for k, uid in enumerate(usuarios_demo):
        u = art["idx_usuario"][uid]
        orden = np.argsort(p_cat[u])[::-1][:3]
        fig.add_trace(go.Bar(
            x=[lista_cats[j] for j in orden],
            y=[p_cat[u, j] for j in orden],
            name=uid, marker_color=paleta[k % len(paleta)],
            text=[f"{p_cat[u, j]:.0%}" for j in orden], textposition="outside",
            hovertemplate="%{x}: %{y:.1%} (" + uid + ")<extra></extra>",
        ), row=2, col=1)
    fig.update_yaxes(tickformat=".0%", range=[0, 0.55], row=2, col=1)
    fig.update_layout(template="plotly_white", height=900,
                      title="Modelo de afinidad por categoria (etapa 1 del hibrido)")
    return fig


def grafico_flujo(art) -> go.Figure:
    cfg = art["config"]
    pesos = cfg["pesos_producto"]
    cajas = [
        (1.2, "Historial de compras\n+ preferencias declaradas", "#e2e8f0", "#334155"),
        (3.4, f"Predictor de categorias\nesquema={cfg['esquema_rating']}, alfa={cfg['alfa_declaradas']}, beta={cfg['beta_suavizado']}", "#ddd6fe", "#4c1d95"),
        (5.6, f"Cuotas entre top-{cfg['top_categorias']} categorias\nexponente={cfg['exponente_cuotas']}", "#fef3c7", "#78350f"),
        (7.8, f"Ranking de productos dentro\nde cada cuota\nknn={pesos['knn']:.2f} ncf={pesos['ncf']:.2f} texto={pesos['contenido']:.2f}", "#d1fae5", "#064e3b"),
        (10.0, "Top-10 recomendado\npersonalizado", "#fee2e2", "#7f1d1d"),
    ]
    fig = go.Figure()
    for x, texto, relleno, borde in cajas:
        fig.add_shape(type="rect", x0=x - 0.9, x1=x + 0.9, y0=-0.55, y1=0.55,
                      fillcolor=relleno, line={"color": borde, "width": 2})
        fig.add_annotation(x=x, y=0, text=texto.replace("\n", "<br>"),
                           showarrow=False, font={"size": 12, "color": borde})
        fig.add_annotation(x=x + 1.1, y=0, ax=x + 0.92, ay=0, text="",
                           xanchor="left", arrowhead=3, arrowsize=1.6, arrowwidth=2,
                           arrowcolor="#94a3b8")
    pc = art["precision_categoria_test"]
    hr = art["hitrate_categoria_test"]
    fig.add_annotation(
        x=5.6, y=1.35,
        text=f"<b>Resultado en test:</b> precision categoria @10 = <b>{pc:.1%}</b> | "
             f"hitrate @10 = <b>{hr:.0%}</b> | P@10 producto = {art['precision_at_k']:.4f}",
        showarrow=False, font={"size": 14, "color": "#0f172a"},
    )
    fig.update_xaxes(visible=False, range=[-0.2, 11.4])
    fig.update_yaxes(visible=False, range=[-1.2, 1.9])
    fig.update_layout(title="Flujo del sistema hibrido v2 (reranking en 2 etapas)",
                      template="plotly_white", height=430)
    return fig


def guardar(fig: go.Figure, nombre: str) -> Path:
    ruta = SALIDAS / nombre
    fig.write_html(ruta, include_plotlyjs=False, full_html=True)
    html = ruta.read_text(encoding="utf-8")
    html = html.replace("<head>", '<head>\n<script src="plotly.min.js"></script>', 1)
    ruta.write_text(html, encoding="utf-8")
    return ruta


INDEX_PLANTILLA = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Sistema Recomendador - Dashboard</title>
<style>
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background:#0f172a; color:#e2e8f0;
         margin:0; padding:40px; }}
  h1 {{ font-size:28px; margin:0 0 6px; }}
  .sub {{ color:#94a3b8; margin-bottom:30px; }}
  .metricas {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:34px; }}
  .chip {{ background:#1e293b; border:1px solid #334155; border-radius:12px;
          padding:14px 22px; }}
  .chip .num {{ font-size:26px; font-weight:700; }}
  .verde {{ color:#34d399; }} .rojo {{ color:#f87171; }} .azul {{ color:#60a5fa; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
          gap:18px; }}
  .tarjeta {{ background:#1e293b; border-radius:14px; padding:20px; border:1px solid #334155;
             transition:transform .15s, border-color .15s; display:block; color:inherit;
             text-decoration:none; }}
  .tarjeta:hover {{ transform:translateY(-3px); border-color:#60a5fa; }}
  .tarjeta h3 {{ margin:0 0 8px; font-size:17px; color:#93c5fd; }}
  .tarjeta p {{ margin:0; font-size:13px; color:#94a3b8; line-height:1.5; }}
  footer {{ margin-top:36px; color:#64748b; font-size:12px; }}
</style>
</head>
<body>
<h1>Panel del Sistema Recomendador</h1>
<div class="sub">Visualizacion interactiva de todo lo entrenado &middot; generado el {fecha}</div>

<div class="metricas">
  <div class="chip"><div class="num verde">{pcat}</div>Precision categoria @10</div>
  <div class="chip"><div class="num verde">{hrcat}</div>HitRate categoria @10</div>
  <div class="chip"><div class="num azul">{pprod}</div>P@10 producto exacto</div>
  <div class="chip"><div class="num azul">{recall}</div>Recall@10 producto</div>
</div>

<div class="grid">
  <a class="tarjeta" href="01_arquitectura_red.html"><h3>Arquitectura de la red neuronal</h3>
    <p>Grafo interactivo de las {n_capas} capas de la red NCF: entradas, embeddings,
       ramas GMF/MLP/texto, fusion y salida. Hover para ver configuracion.</p></a>
  <a class="tarjeta" href="02_metricas_modelos.html"><h3>Metricas comparadas</h3>
    <p>Barras comparando popularidad, contenido, NCF, KNN, afinidad e hibrido v2.</p></a>
  <a class="tarjeta" href="03_embeddings_productos.html"><h3>Embeddings de productos</h3>
    <p>Mapa 2D (PCA) del espacio latente que aprendio la red. Filtra por categoria.</p></a>
  <a class="tarjeta" href="04_afinidad_categorias.html"><h3>Afinidad usuario-categoria</h3>
    <p>Mapa de calor del predictor de categorias y ejemplos de usuarios demo.</p></a>
  <a class="tarjeta" href="05_flujo_hibrido.html"><h3>Flujo del hibrido v2</h3>
    <p>Como se generan las recomendaciones: predictor, cuotas, ranking interno.</p></a>
</div>

<footer>Regenera con: python ver_grafos/generar_grafos.py</footer>
</body>
</html>
"""


def main() -> None:
    inicio = time.time()
    SALIDAS.mkdir(parents=True, exist_ok=True)
    js = SALIDAS / "plotly.min.js"
    if not js.exists():
        print("[1/7] Copiando plotly.js local...")
        origen = Path(__import__("plotly").__file__).parent / "package_data" / "plotly.min.js"
        js.write_bytes(origen.read_bytes())

    print("[2/7] Cargando artefactos entrenados...")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    modelo = tf.keras.models.load_model(RUTA_MODELOS / "red_neuronal.keras")
    art = joblib.load(RUTA_MODELOS / "hibrido_v2.pkl")
    productos_info = pd.read_csv(
        RUTA_DATOS / "productos.csv", usecols=["product_id", "name", "main_category"]
    ).set_index("product_id")
    usuarios_demo = ["u0007", "u0100", "u0250"]

    print("[3/7] Grafo de arquitectura de la red...")
    guardar(grafo_arquitectura(modelo), "01_arquitectura_red.html")

    print("[4/7] Metricas comparadas...")
    guardar(grafico_metricas(art), "02_metricas_modelos.html")

    print("[5/7] Embeddings de productos (PCA)...")
    guardar(grafico_embeddings(modelo, art, productos_info),
            "03_embeddings_productos.html")

    print("[6/7] Afinidad usuario-categoria y flujo del hibrido...")
    guardar(grafico_afinidad(art, usuarios_demo), "04_afinidad_categorias.html")
    guardar(grafico_flujo(art), "05_flujo_hibrido.html")

    index = INDEX_PLANTILLA.format(
        fecha=time.strftime("%Y-%m-%d %H:%M"),
        pcat=f"{art['precision_categoria_test']:.1%}",
        hrcat=f"{art['hitrate_categoria_test']:.0%}",
        pprod=f"{art['precision_at_k']:.4f}",
        recall=f"{art['recall_at_k']:.4f}",
        n_capas=len(modelo.layers),
    )
    index_ruta = SALIDAS / "index.html"
    index_ruta.write_text(index, encoding="utf-8")

    print(f"[7/7] Listo en {time.time() - inicio:.1f}s -> {index_ruta}")
    webbrowser.open(index_ruta.as_uri())


if __name__ == "__main__":
    main()
