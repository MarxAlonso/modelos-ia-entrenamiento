import { useState, useEffect } from "react";
import Plot from "react-plotly.js";

export default function FlujoHibrido() {
  const [datos, setDatos] = useState(null);

  useEffect(() => {
    fetch("/src/data/flujo.json")
      .then(r => r.json())
      .then(setDatos)
      .catch(() => fetch("../src/data/flujo.json").then(r => r.json()).then(setDatos));
  }, []);

  if (!datos) return <div style={{ padding: 40, color: "#94a3b8" }}>Cargando...</div>;

  const { pasos, resultado } = datos;

  const shapes = pasos.map(p => ({
    type: "rect", x0: p.pos - 0.9, x1: p.pos + 0.9, y0: -0.55, y1: 0.55,
    fillcolor: p.color_fondo, line: { color: p.color_texto, width: 2 },
  }));

  const annotations = [];
  pasos.forEach((p, i) => {
    annotations.push({
      x: p.pos, y: 0, text: p.titulo.replace(/\n/g, "<br>"),
      showarrow: false, font: { size: 12, color: p.color_texto },
    });
    if (i < pasos.length - 1) {
      annotations.push({
        x: p.pos + 1.1, y: 0, ax: p.pos + 0.92, ay: 0, text: "",
        xanchor: "left", arrowhead: 3, arrowsize: 1.6, arrowwidth: 2,
        arrowcolor: "#94a3b8",
      });
    }
  });

  annotations.push({
    x: 5.6, y: 1.35,
    text: `<b>Resultado en test:</b> precision categoria @10 = <b>${(resultado.precision_categoria * 100).toFixed(1)}%</b> | hitrate @10 = <b>${(resultado.hitrate * 100).toFixed(0)}%</b> | P@10 producto = ${resultado.precision_producto.toFixed(4)}`,
    showarrow: false, font: { size: 14, color: "#0f172a" },
  });

  return (
    <div style={{ width: "100%", height: "100%", overflow: "auto", padding: 20 }}>
      <h2 style={{ color: "#e2e8f0", marginBottom: 20 }}>Flujo del sistema hibrido v2</h2>
      <div style={{ background: "#fff", borderRadius: 8 }}>
        <Plot
          data={[{ x: [null], y: [null], mode: "markers", showlegend: false }]}
          layout={{
            title: "Flujo del sistema hibrido v2 (reranking en 2 etapas)",
            shapes: shapes,
            annotations: annotations,
            xaxis: { visible: false, range: [-0.2, 11.4] },
            yaxis: { visible: false, range: [-1.2, 1.9] },
            height: 430,
            margin: { l: 20, r: 20, t: 60, b: 20 },
          }}
          config={{ responsive: true }}
        />
      </div>
    </div>
  );
}
