import { useState, useEffect } from "react";
import Plot from "react-plotly.js";

const COLORES = {
  InputLayer: "#3b82f6", Embedding: "#8b5cf6", Flatten: "#64748b",
  Multiply: "#f59e0b", Concatenate: "#f59e0b", Add: "#f59e0b",
  Dense: "#10b981", Dropout: "#cbd5e1", Activation: "#ef4444",
};

export default function Arquitectura2D() {
  const [datos, setDatos] = useState(null);

  useEffect(() => {
    fetch("/src/data/arquitectura_2d.json")
      .then(r => r.json())
      .then(setDatos)
      .catch(() => fetch("../src/data/arquitectura_2d.json").then(r => r.json()).then(setDatos));
  }, []);

  if (!datos) return <div style={{ padding: 40, color: "#94a3b8" }}>Cargando...</div>;

  const capas = {};
  datos.nodes.forEach(n => { capas[n.name] = n; });

  const profundidad = {};
  const calcular = (nombre) => {
    if (profundidad[nombre] !== undefined) return profundidad[nombre];
    const padres = datos.links.filter(l => l.target === nombre).map(l => l.source);
    const d = padres.length === 0 ? 0 : Math.max(...padres.map(calcular)) + 1;
    profundidad[nombre] = d;
    return d;
  };
  datos.nodes.forEach(n => { try { calcular(n.name); } catch { profundidad[n.name] = 0; } });

  const columnas = {};
  Object.entries(profundidad).forEach(([nombre, d]) => {
    if (!columnas[d]) columnas[d] = [];
    columnas[d].push(nombre);
  });

  const pos = {};
  Object.entries(columnas).forEach(([d, grupo]) => {
    const dNum = parseInt(d);
    grupo.forEach((nombre, i) => {
      pos[nombre] = [dNum * 240, (grupo.length - i - 1) * 110];
    });
  });

  const edges_x = [], edges_y = [];
  datos.links.forEach(l => {
    if (pos[l.source] && pos[l.target]) {
      const [x0, y0] = pos[l.source];
      const [x1, y1] = pos[l.target];
      edges_x.push(x0, (x0 + x1) / 2, x1, null);
      edges_y.push(y0, (y0 + y1) / 2 + 18, y1, null);
    }
  });

  const node_x = [], node_y = [], node_text = [], node_color = [], node_hover = [];
  datos.nodes.forEach(n => {
    if (pos[n.name]) {
      const [x, y] = pos[n.name];
      node_x.push(x);
      node_y.push(y);
      node_text.push(n.name);
      node_color.push(COLORES[n.type] || "#334155");
      node_hover.push(`${n.name}<br>${n.type}`);
    }
  });

  return (
    <div style={{ width: "100%", height: "100%", overflow: "auto", padding: 20 }}>
      <h2 style={{ color: "#e2e8f0", marginBottom: 20 }}>Arquitectura de la red neuronal</h2>
      <div style={{ background: "#fff", borderRadius: 8 }}>
        <Plot
          data={[
            { x: edges_x, y: edges_y, mode: "lines", line: { width: 1.4, color: "#94a3b8" }, hoverinfo: "skip", showlegend: false },
            {
              x: node_x, y: node_y, mode: "markers+text",
              marker: { size: 34, color: node_color, line: { width: 2, color: "white" } },
              text: node_text, textposition: "top center",
              textfont: { size: 10, color: "#0f172a" },
              hovertext: node_hover, hovertemplate: "%{hovertext}<extra></extra>",
              showlegend: false,
            },
          ]}
          layout={{
            title: `Arquitectura de la red NCF (${datos.nodes.length} capas)`,
            height: 750,
            margin: { l: 20, r: 20, t: 60, b: 20 },
            xaxis: { visible: false },
            yaxis: { visible: false },
            annotations: [{
              text: "Pasa el cursor sobre cada capa para ver su configuracion.",
              xref: "paper", yref: "paper", x: 0, y: -0.02, showarrow: false,
              font: { size: 12, color: "#64748b" },
            }],
          }}
          config={{ responsive: true }}
        />
      </div>
    </div>
  );
}
