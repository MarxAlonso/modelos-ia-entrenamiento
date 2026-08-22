import { useState, useEffect } from "react";
import ForceGraph3D from "react-force-graph-3d";

const PALETA = {
  "pasabocas": "#f97316", "dulces-y-postres": "#ec4899", "bebidas": "#3b82f6",
  "carne-y-pollo": "#ef4444", "charcuteria": "#f43f5e", "vinos-y-licores": "#8b5cf6",
  "frutas-y-verduras": "#22c55e", "lacteos-huevos-y-refrigerados": "#06b6d4",
  "pescados-y-mariscos": "#14b8a6", "panaderia-y-pasteleria": "#f59e0b",
  "despensa": "#64748b", "cuidado-personal": "#a855f7",
};

export default function Semantico3D() {
  const [datos, setDatos] = useState(null);
  const [filtro, setFiltro] = useState("todos");

  useEffect(() => {
    fetch("/src/data/semantico.json")
      .then(r => r.json())
      .then(setDatos)
      .catch(() => setCargando(false));
  }, []);

  if (!datos) return <div style={{ padding: 40, color: "#94a3b8" }}>Cargando embeddings semanticos...</div>;

  const categorias = [...new Set(datos.nodes.map(n => n.category))].sort();
  const nodosFiltrados = filtro === "todos"
    ? datos.nodes
    : datos.nodes.filter(n => n.category === filtro);

  const bench = datos.benchmark || {};

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <div style={{ position: "absolute", top: 16, left: 16, zIndex: 10, background: "rgba(15,23,42,0.92)", padding: "16px 20px", borderRadius: 8, color: "#e2e8f0", maxWidth: 310 }}>
        <h3 style={{ fontSize: 16, marginBottom: 4 }}>Embeddings Semanticos</h3>
        <p style={{ fontSize: 11, color: "#64748b", marginBottom: 10 }}>
          Transformer: {datos.modelo}<br />Dimension: {datos.dim}D | {datos.nodes.length} productos
        </p>
        <div style={{ fontSize: 10, marginBottom: 10 }}>
          <div style={{ color: "#94a3b8", marginBottom: 4 }}>Coherencia k-NN (misma categoria):</div>
          {[["TF-IDF", bench.tfidf, "#64748b"], ["Word2Vec", bench.word2vec, "#38bdf8"], ["Transformer", bench.transformer, "#a78bfa"]].map(([n, v, c]) => (
            <div key={n} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
              <span style={{ width: 70, color: "#94a3b8" }}>{n}</span>
              <div style={{ flex: 1, height: 6, background: "#1e293b", borderRadius: 3 }}>
                <div style={{ width: `${v}%`, height: "100%", background: c, borderRadius: 3 }} />
              </div>
              <span style={{ width: 36, textAlign: "right" }}>{v}%</span>
            </div>
          ))}
        </div>
        <select value={filtro} onChange={e => setFiltro(e.target.value)} style={{
          width: "100%", padding: "6px 10px", borderRadius: 4,
          background: "#1e293b", color: "#e2e8f0", border: "1px solid #334155", fontSize: 12,
        }}>
          <option value="todos">Todas ({datos.nodes.length})</option>
          {categorias.map(c => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
      </div>
      <ForceGraph3D
        graphData={{ nodes: nodosFiltrados.map(n => ({ ...n, val: 6 })), links: [] }}
        nodeLabel={n => `<b>${n.nombre}</b><br>${n.category}`}
        nodeColor={n => PALETA[n.category] || "#94a3b8"}
        nodeVal={n => n.val}
        backgroundColor="#020617"
      />
    </div>
  );
}
