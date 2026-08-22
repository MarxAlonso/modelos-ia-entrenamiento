import { useState, useCallback } from "react";
import ForceGraph3D from "react-force-graph-3d";
import SpriteText from "three-spritetext";

const COLORES_CAT = [
  "#ef4444", "#f97316", "#eab308", "#22c55e", "#14b8a6",
  "#06b6d4", "#3b82f6", "#8b5cf6", "#a855f7", "#ec4899",
  "#f43f5e", "#64748b",
];

export default function UsuariosCategorias3D() {
  const [datos, setDatos] = useState(null);
  const [cargando, setCargando] = useState(true);

  useState(() => {
    fetch("/src/data/usuarios_categorias.json")
      .then(r => r.json())
      .then(d => { setDatos(d); setCargando(false); })
      .catch(() => {
        fetch("../src/data/usuarios_categorias.json")
          .then(r => r.json())
          .then(d => { setDatos(d); setCargando(false); })
          .catch(() => setCargando(false));
      });
  }, []);

  const nodoObject = useCallback(node => {
    if (node.type === "categoria") {
      const txt = new SpriteText(node.id, 8, node.color);
      txt.textHeight = 12;
      txt.position.y = 18;
      return txt;
    }
    return undefined;
  }, []);

  if (cargando) return <div style={{ padding: 40, color: "#94a3b8" }}>Cargando datos...</div>;
  if (!datos) return <div style={{ padding: 40, color: "#f87171" }}>Error: ejecuta primero generar_grafos_3d.py --json</div>;

  const cats = datos.nodes.filter(n => n.type === "categoria");

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <div style={{ position: "absolute", top: 16, left: 16, zIndex: 10, background: "rgba(15,23,42,0.9)", padding: "16px 20px", borderRadius: 8, color: "#e2e8f0", maxWidth: 280 }}>
        <h3 style={{ fontSize: 16, marginBottom: 8 }}>Red Usuario-Categoria</h3>
        <p style={{ fontSize: 12, color: "#94a3b8", marginBottom: 12 }}>
          {datos.nodes.filter(n => n.type === "usuario").length} usuarios conectados a {cats.length} categorias
        </p>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {cats.map((c, i) => (
            <span key={c.id} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "#94a3b8" }}>
              <span style={{ width: 10, height: 10, borderRadius: "50%", background: COLORES_CAT[i], display: "inline-block" }} />
              {c.id}
            </span>
          ))}
        </div>
      </div>
      <ForceGraph3D
        graphData={datos}
        nodeLabel={n => n.type === "categoria" ? n.id : `Usuario ${n.id}`}
        nodeColor={n => n.color || "#64748b"}
        nodeVal={n => n.type === "categoria" ? 18 : 4}
        nodeThreeObject={nodoObject}
        linkWidth={l => Math.max(0.5, l.weight * 3)}
        linkColor={l => {
          const cat = datos.nodes.find(n => n.id === l.target);
          return cat ? cat.color : "#334155";
        }}
        linkOpacity={0.35}
        backgroundColor="#020617"
        d3VelocityDecay={0.4}
        warmupTicks={100}
        cooldownTicks={200}
      />
    </div>
  );
}
