import { useState, useEffect } from "react";
import Plot from "react-plotly.js";

const PALETA = [
  "#6366f1", "#f97316", "#10b981", "#ef4444", "#8b5cf6", "#0891b2",
  "#d946ef", "#84cc16", "#f59e0b", "#0ea5e9", "#ec4899", "#14b8a6",
];

export default function AfinidadCategorias() {
  const [datos, setDatos] = useState(null);

  useEffect(() => {
    fetch("/src/data/afinidad.json")
      .then(r => r.json())
      .then(setDatos)
      .catch(() => fetch("../src/data/afinidad.json").then(r => r.json()).then(setDatos));
  }, []);

  if (!datos) return <div style={{ padding: 40, color: "#94a3b8" }}>Cargando...</div>;

  const { matrix, categories, usuarios_demo } = datos;

  const barTraces = usuarios_demo.map((uid, k) => {
    const uIdx = uid.replace("u", "");
    const row = matrix[parseInt(uIdx)] || matrix[0];
    const sorted = row.map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v).slice(0, 3);
    return {
      x: sorted.map(s => categories[s.i]),
      y: sorted.map(s => s.v),
      name: uid,
      type: "bar",
      marker: { color: PALETA[k % PALETA.length] },
      text: sorted.map(s => (s.v * 100).toFixed(0) + "%"),
      textposition: "outside",
      hovertemplate: "%{x}: %{y:.1%} (" + uid + ")<extra></extra>",
    };
  });

  return (
    <div style={{ width: "100%", height: "100%", overflow: "auto", padding: 20 }}>
      <h2 style={{ color: "#e2e8f0", marginBottom: 20 }}>Afinidad usuario-categoria</h2>
      <div style={{ background: "#fff", borderRadius: 8, marginBottom: 20 }}>
        <Plot
          data={[{
            z: matrix, x: categories,
            y: Array.from({ length: matrix.length }, (_, i) => i),
            type: "heatmap",
            colorscale: "Viridis",
            colorbar: { title: "prob." },
          }]}
          layout={{
            title: "Afinidad usuario-categoria (500 usuarios x 12 categorias)",
            height: 550,
            yaxis: { title: "# usuario", dtick: 25 },
            margin: { l: 50, r: 20, t: 60, b: 100 },
          }}
          config={{ responsive: true }}
        />
      </div>
      <div style={{ background: "#fff", borderRadius: 8 }}>
        <Plot
          data={barTraces}
          layout={{
            title: "Top-3 categorias predichas para usuarios demo",
            barmode: "group",
            yaxis: { tickformat: ".0%", range: [0, 0.55] },
            height: 350,
            margin: { l: 50, r: 20, t: 60, b: 100 },
          }}
          config={{ responsive: true }}
        />
      </div>
    </div>
  );
}
