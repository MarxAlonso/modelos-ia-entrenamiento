import { useState, useEffect } from "react";
import Plot from "react-plotly.js";

const PALETA = [
  "#6366f1", "#f97316", "#10b981", "#ef4444", "#8b5cf6", "#0891b2",
  "#d946ef", "#84cc16", "#f59e0b", "#0ea5e9", "#ec4899", "#14b8a6",
];

export default function Embeddings2D() {
  const [datos, setDatos] = useState(null);
  const [filtro, setFiltro] = useState("Todas");

  useEffect(() => {
    fetch("/src/data/embeddings_2d.json")
      .then(r => r.json())
      .then(setDatos)
      .catch(() => fetch("../src/data/embeddings_2d.json").then(r => r.json()).then(setDatos));
  }, []);

  if (!datos) return <div style={{ padding: 40, color: "#94a3b8" }}>Cargando...</div>;

  const cats = datos.cats;
  const visibilidad = cats.map(c => filtro === "Todas" || c === filtro);

  const traces = cats.map((cat, j) => {
    const pts = datos.nodes.filter(n => n.category === cat);
    return {
      x: pts.map(p => p.x),
      y: pts.map(p => p.y),
      mode: "markers",
      name: cat,
      marker: { size: 6, color: PALETA[j % PALETA.length], opacity: 0.75 },
      text: pts.map(p => p.name),
      hovertemplate: "%{text}<br>(" + cat + ")<extra></extra>",
      visible: visibilidad[j],
    };
  });

  return (
    <div style={{ width: "100%", height: "100%", overflow: "auto", padding: 20 }}>
      <h2 style={{ color: "#e2e8f0", marginBottom: 20 }}>Embeddings de productos (PCA 2D)</h2>
      <div style={{ marginBottom: 16 }}>
        <select value={filtro} onChange={e => setFiltro(e.target.value)} style={{
          padding: "8px 12px", borderRadius: 6, border: "1px solid #334155",
          background: "#1e293b", color: "#e2e8f0", fontSize: 13,
        }}>
          <option value="Todas">Todas las categorias ({datos.nodes.length})</option>
          {cats.map(c => (
            <option key={c} value={c}>{c} ({datos.nodes.filter(n => n.category === c).length})</option>
          ))}
        </select>
      </div>
      <div style={{ background: "#fff", borderRadius: 8 }}>
        <Plot
          data={traces}
          layout={{
            title: "Embeddings de productos (32D -> 2D con PCA)",
            height: 700,
            legend: { title: { text: "Categoria" }, font: { size: 10 } },
            margin: { l: 50, r: 20, t: 60, b: 50 },
            annotations: [{
              text: "Cada punto es un producto; los cercanos son similares para la red.",
              xref: "paper", yref: "paper", x: 0, y: -0.08, showarrow: false,
              font: { size: 12, color: "#64748b" },
            }],
          }}
          config={{ responsive: true }}
        />
      </div>
    </div>
  );
}
