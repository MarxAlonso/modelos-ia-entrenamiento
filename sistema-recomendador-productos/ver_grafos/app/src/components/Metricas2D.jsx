import { useState, useEffect } from "react";
import Plot from "react-plotly.js";

export default function Metricas2D() {
  const [datos, setDatos] = useState(null);

  useEffect(() => {
    fetch("/src/data/metricas.json")
      .then(r => r.json())
      .then(setDatos)
      .catch(() => fetch("../src/data/metricas.json").then(r => r.json()).then(setDatos));
  }, []);

  if (!datos) return <div style={{ padding: 40, color: "#94a3b8" }}>Cargando...</div>;

  const n34 = datos.niveles34 || {};

  return (
    <div style={{ width: "100%", height: "100%", overflow: "auto", padding: 20 }}>
      <h2 style={{ color: "#e2e8f0", marginBottom: 20 }}>Metricas comparadas</h2>
      {Object.keys(n34).length > 0 && (
        <div style={{ background: "#111827", borderRadius: 12, padding: 20, marginBottom: 20 }}>
          <h3 style={{ color: "#e2e8f0", marginBottom: 4 }}>Evaluacion unificada (split SEED=42)</h3>
          <p style={{ fontSize: 11, color: "#64748b", marginBottom: 12 }}>
            {datos.nota_svd || ""} · CatPrec@10 = % del top-10 cuya categoria esta en el test del usuario
          </p>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead>
              <tr style={{ color: "#94a3b8", borderBottom: "1px solid #334155" }}>
                <th style={{ textAlign: "left", padding: "6px 10px" }}>Modelo</th>
                <th style={{ textAlign: "right", padding: "6px 10px" }}>P@10 producto</th>
                <th style={{ textAlign: "right", padding: "6px 10px" }}>CatPrec@10</th>
                <th style={{ textAlign: "left", padding: "6px 10px", width: "35%" }}>CatPrec (barra)</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(n34).map(([nombre, r]) => (
                <tr key={nombre} style={{
                  borderBottom: "1px solid #1e293b",
                  background: ["Two-Tower", "LightGCN"].includes(nombre) ? "rgba(56,189,248,0.06)" : "transparent",
                }}>
                  <td style={{ padding: "7px 10px", color: ["Two-Tower", "LightGCN"].includes(nombre) ? "#7dd3fc" : "#e2e8f0" }}>
                    {nombre}
                  </td>
                  <td style={{ padding: "7px 10px", textAlign: "right" }}>{r.precision_producto.toFixed(4)}</td>
                  <td style={{ padding: "7px 10px", textAlign: "right" }}>{(r.precision_categoria * 100).toFixed(1)}%</td>
                  <td style={{ padding: "7px 10px" }}>
                    <div style={{ height: 8, background: "#1e293b", borderRadius: 4 }}>
                      <div style={{
                        width: `${r.precision_categoria * 100}%`, height: "100%",
                        borderRadius: 4,
                        background: nombre.includes("Hibrido") ? "#34d399"
                          : ["Two-Tower"].includes(nombre) ? "#38bdf8"
                          : nombre.includes("SVD") ? "#a78bfa" : "#64748b",
                      }} />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <h3 style={{ color: "#e2e8f0", marginBottom: 12 }}>Comparativa historica (fases previas)</h3>
      <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
        <div style={{ flex: 1, minWidth: 400, background: "#fff", borderRadius: 8 }}>
          <Plot
            data={[{
              type: "bar",
              x: datos.productos.labels,
              y: datos.productos.valores,
              marker: { color: datos.productos.colores },
              text: datos.productos.valores.map(v => v.toFixed(4)),
              textposition: "outside",
              hovertemplate: "%{x}: %{y:.4f}<extra></extra>",
            }]}
            layout={{
              title: "Precision@10 sobre PRODUCTO exacto",
              yaxis: { range: [0, 0.008], title: "P@10" },
              height: 450,
              margin: { l: 50, r: 20, t: 50, b: 100 },
            }}
            config={{ responsive: true }}
          />
        </div>
        <div style={{ flex: 1, minWidth: 400, background: "#fff", borderRadius: 8 }}>
          <Plot
            data={[{
              type: "bar",
              x: datos.categorias.labels,
              y: datos.categorias.valores,
              marker: { color: datos.categorias.colores },
              text: datos.categorias.valores.map(v => (v * 100).toFixed(1) + "%"),
              textposition: "outside",
              hovertemplate: "%{x}: %{y:.1%}<extra></extra>",
            }]}
            layout={{
              title: "Objetivo CATEGORIA (test completo)",
              yaxis: { range: [0, 1.15], tickformat: ".0%" },
              height: 450,
              margin: { l: 50, r: 20, t: 50, b: 100 },
            }}
            config={{ responsive: true }}
          />
        </div>
      </div>
    </div>
  );
}
