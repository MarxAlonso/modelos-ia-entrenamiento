import { useState, useEffect } from "react";
import ForceGraph3D from "react-force-graph-3d";

export default function Word2Vec3D() {
  const [datos, setDatos] = useState(null);
  const [cargando, setCargando] = useState(true);
  const [filtro, setFiltro] = useState("todos");
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });

  useEffect(() => {
    fetch("/src/data/word2vec.json")
      .then(r => r.json())
      .then(d => { 
        setDatos(d); 
        // Convert to ForceGraph3D format with explicit positions
        const nodes = d.nodes.map(n => ({
          id: n.id,
          name: n.nombre,
          category: n.category,
          val: 8,
          x: n.x * 100,  // Scale up positions
          y: n.y * 100,
          z: n.z * 100,
        }));
        setGraphData({ nodes, links: [] });
        setCargando(false); 
      })
      .catch(() => {
        fetch("../src/data/word2vec.json")
          .then(r => r.json())
          .then(d => { 
            setDatos(d);
            const nodes = d.nodes.map(n => ({
              id: n.id,
              name: n.nombre,
              category: n.category,
              val: 8,
              x: n.x * 100,
              y: n.y * 100,
              z: n.z * 100,
            }));
            setGraphData({ nodes, links: [] });
            setCargando(false); 
          })
          .catch(() => setCargando(false));
      });
  }, [filtro]);

  const PALETA = {
    "pasabocas": "#f97316", "dulces-y-postres": "#ec4899", "bebidas": "#3b82f6",
    "carne-y-pollo": "#ef4444", "charcuteria": "#f43f5e", "vinos-y-licores": "#8b5cf6",
    "frutas-y-verduras": "#22c55e", "lacteos-huevos-y-refrigerados": "#06b6d4",
    "pescados-y-mariscos": "#14b8a6", "panaderia-y-pasteleria": "#f59e0b",
    "despensa": "#64748b", "cuidado-personal": "#a855f7",
  };

  const categorias = datos ? [...new Set(datos.nodes.map(n => n.category))].sort() : [];
  const nodosFiltrados = filtro === "todos" 
    ? graphData.nodes 
    : graphData.nodes.filter(n => n.category === filtro);

  if (cargando) return <div style={{ padding: 40, color: "#94a3b8" }}>Cargando datos Word2Vec...</div>;
  if (!datos) return <div style={{ padding: 40, color: "#f87171" }}>Error: ejecuta primero generar_grafos_3d.py --json</div>;

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <div style={{ position: "absolute", top: 16, left: 16, zIndex: 10, background: "rgba(15,23,42,0.9)", padding: "16px 20px", borderRadius: 8, color: "#e2e8f0", maxWidth: 300 }}>
        <h3 style={{ fontSize: 16, marginBottom: 8 }}>Embeddings Word2Vec</h3>
        <p style={{ fontSize: 12, color: "#94a3b8", marginBottom: 8 }}>
          Vocabulario: {datos.vocab_size?.toLocaleString()} palabras | Dimensión: {datos.vector_size}D
        </p>
        <p style={{ fontSize: 11, color: "#64748b", marginBottom: 12 }}>
          PCA 3D de embeddings entrenados con 107K productos
        </p>
        <select value={filtro} onChange={e => setFiltro(e.target.value)} style={{
          width: "100%", padding: "6px 10px", borderRadius: 4,
          background: "#1e293b", color: "#e2e8f0", border: "1px solid #334155", fontSize: 12,
        }}>
          <option value="todos">Todas las categorías ({graphData.nodes.length})</option>
          {categorias.map(c => (
            <option key={c} value={c}>{c} ({graphData.nodes.filter(n => n.category === c).length})</option>
          ))}
        </select>
        <div style={{ marginTop: 12, display: "flex", flexWrap: "wrap", gap: 4 }}>
          {categorias.slice(0, 12).map(c => (
            <span key={c} style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 9, color: "#94a3b8" }}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: PALETA[c], display: "inline-block" }} />
              {c.split("-").slice(0, 2).join("-")}
            </span>
          ))}
        </div>
      </div>
      <ForceGraph3D
        graphData={{ nodes: nodosFiltrados, links: [] }}
        nodeLabel={n => `<b>${n.name}</b><br>${n.category}`}
        nodeColor={n => PALETA[n.category] || "#94a3b8"}
        nodeVal={n => n.val}
        backgroundColor="#020617"
        warmupTicks={100}
        cooldownTime={3000}
      />
    </div>
  );
}
