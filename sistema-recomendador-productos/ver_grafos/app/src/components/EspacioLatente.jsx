import { useState, useEffect } from "react";
import ForceGraph3D from "react-force-graph-3d";

const PALETA = {
  "pasabocas": "#f97316", "dulces-y-postres": "#ec4899", "bebidas": "#3b82f6",
  "carne-y-pollo": "#ef4444", "charcuteria": "#f43f5e", "vinos-y-licores": "#8b5cf6",
  "frutas-y-verduras": "#22c55e", "lacteos-huevos-y-refrigerados": "#06b6d4",
  "pescados-y-mariscos": "#14b8a6", "panaderia-y-pasteleria": "#f59e0b",
  "despensa": "#64748b", "cuidado-personal": "#a855f7",
};

const METODOS = [
  { id: "ncf", label: "NCF (red neuronal)", desc: "Embedding del filtrado colaborativo" },
  { id: "w2v", label: "Word2Vec", desc: "Titulos/descripciones, 100D -> PCA" },
  { id: "semantico", label: "Semantico Transformer", desc: "Sentence-BERT, 384D -> PCA" },
  { id: "lightgcn", label: "LightGCN", desc: "Grafo usuario-producto (Nivel 4)" },
];

export default function EspacioLatente() {
  const [datos, setDatos] = useState(null);
  const [metodo, setMetodo] = useState("ncf");
  const [cat, setCat] = useState("todos");

  useEffect(() => {
    fetch("/src/data/espacio_latente.json")
      .then(r => r.json())
      .then(setDatos)
      .catch(() => {});
  }, []);

  if (!datos) return <div style={{ padding: 40, color: "#94a3b8" }}>Cargando espacio latente...</div>;

  const disponibles = METODOS.filter(m => datos.metodos[m.id]?.length);
  const todos = datos.metodos[metodo] || [];
  const categorias = [...new Set(todos.map(n => n.category))].sort();
  const nodos = cat === "todos" ? todos : todos.filter(n => n.category === cat);

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <div style={{ position: "absolute", top: 16, left: 16, zIndex: 10, background: "rgba(15,23,42,0.92)", padding: "16px 20px", borderRadius: 8, color: "#e2e8f0", maxWidth: 320 }}>
        <h3 style={{ fontSize: 15, marginBottom: 10 }}>Espacio Latente de Productos</h3>
        {disponibles.map(m => (
          <button key={m.id} onClick={() => setMetodo(m.id)} style={{
            display: "block", width: "100%", textAlign: "left", marginBottom: 6,
            padding: "7px 10px", borderRadius: 6, cursor: "pointer", fontSize: 12,
            border: metodo === m.id ? "1px solid #38bdf8" : "1px solid #334155",
            background: metodo === m.id ? "#164e63" : "#1e293b",
            color: metodo === m.id ? "#7dd3fc" : "#94a3b8",
          }}>
            <b>{m.label}</b><br />
            <span style={{ fontSize: 10 }}>{m.desc} · {(datos.metodos[m.id] || []).length} pts</span>
          </button>
        ))}
        <select value={cat} onChange={e => setCat(e.target.value)} style={{
          width: "100%", marginTop: 8, padding: "6px 10px", borderRadius: 4,
          background: "#1e293b", color: "#e2e8f0", border: "1px solid #334155", fontSize: 12,
        }}>
          <option value="todos">Todas las categorias ({nodos.length})</option>
          {categorias.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
      </div>
      <ForceGraph3D
        key={metodo + cat}
        graphData={{ nodes: nodos.map(n => ({ ...n, val: 5 })), links: [] }}
        nodeLabel={n => `<b>${n.nombre}</b><br>${n.category}`}
        nodeColor={n => PALETA[n.category] || "#94a3b8"}
        nodeVal={n => n.val}
        backgroundColor="#020617"
        d3VelocityDecay={0.35}
      />
    </div>
  );
}
