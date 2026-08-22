import { useState, useCallback } from "react";
import ForceGraph3D from "react-force-graph-3d";
import SpriteText from "three-spritetext";

const PALETA = {
  "Herramientas Manuales": "#ef4444", "Herramientas Electricas": "#f97316",
  "Jardineria": "#22c55e", "Pinturas": "#3b82f6", "Plomeria": "#8b5cf6",
  "Electricidad": "#eab308", "Seguridad": "#ec4899", "Adhesivos y Selladores": "#06b6d4",
  "Tornilleria": "#a855f7", "Materiales de Construccion": "#f43f5e",
  "Accesorios para Herramientas": "#14b8a6", "Equipos de Medición": "#64748b",
};

function colorNodo(cat) { return PALETA[cat] || "#94a3b8"; }

export default function Embeddings3D() {
  const [datos, setDatos] = useState(null);
  const [cargando, setCargando] = useState(true);
  const [filtro, setFiltro] = useState("todos");

  useState(() => {
    fetch("/src/data/embeddings.json")
      .then(r => r.json())
      .then(d => { setDatos(d); setCargando(false); })
      .catch(() => {
        fetch("../src/data/embeddings.json")
          .then(r => r.json())
          .then(d => { setDatos(d); setCargando(false); })
          .catch(() => setCargando(false));
      });
  }, []);

  const categorias = datos ? [...new Set(datos.nodes.map(n => n.category))].sort() : [];

  const nodosFiltrados = datos
    ? filtro === "todos" ? datos.nodes : datos.nodes.filter(n => n.category === filtro)
    : [];

  const nodoObject = useCallback(node => {
    const txt = new SpriteText(node.nombre, 2, colorNodo(node.category));
    txt.textHeight = 3;
    txt.position.y = 6;
    return txt;
  }, []);

  if (cargando) return <div style={{ padding: 40, color: "#94a3b8" }}>Cargando datos...</div>;
  if (!datos) return <div style={{ padding: 40, color: "#f87171" }}>Error: ejecuta primero generar_grafos_3d.py --json</div>;

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <div style={{ position: "absolute", top: 16, left: 16, zIndex: 10, background: "rgba(15,23,42,0.9)", padding: "16px 20px", borderRadius: 8, color: "#e2e8f0" }}>
        <h3 style={{ fontSize: 16, marginBottom: 8 }}>Espacio 3D de Embeddings</h3>
        <p style={{ fontSize: 12, color: "#94a3b8", marginBottom: 12 }}>PCA de 32 dimensiones a 3 — productos que aprendio la red</p>
        <select value={filtro} onChange={e => setFiltro(e.target.value)} style={{
          width: "100%", padding: "6px 10px", borderRadius: 4,
          background: "#1e293b", color: "#e2e8f0", border: "1px solid #334155", fontSize: 12,
        }}>
          <option value="todos">Todas las categorias ({datos.nodes.length})</option>
          {categorias.map(c => (
            <option key={c} value={c}>{c} ({datos.nodes.filter(n => n.category === c).length})</option>
          ))}
        </select>
        <div style={{ marginTop: 12, display: "flex", flexWrap: "wrap", gap: 4 }}>
          {categorias.map(c => (
            <span key={c} style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 9, color: "#94a3b8" }}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: PALETA[c], display: "inline-block" }} />
              {c.split(" ").slice(0, 2).join(" ")}
            </span>
          ))}
        </div>
      </div>
      <ForceGraph3D
        graphData={{ nodes: nodosFiltrados, links: [] }}
        nodeLabel={n => `${n.nombre}\n${n.category}`}
        nodeColor={n => colorNodo(n.category)}
        nodeVal={3}
        nodeThreeObject={nodoObject}
        backgroundColor="#020617"
        d3VelocityDecay={0.4}
        warmupTicks={50}
        cooldownTicks={150}
        cameraPosition={{ x: 300, y: 200, z: 300 }}
      />
    </div>
  );
}
