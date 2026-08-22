import { useState, useCallback, useRef } from "react";
import ForceGraph3D from "react-force-graph-3d";
import SpriteText from "three-spritetext";

const COLORES = {
  embedding: "#22d3ee",
  flatten: "#a78bfa",
  dot: "#facc15",
  multiply: "#f97316",
  activation: "#34d399",
  dropout: "#f87171",
  batchnorm: "#60a5fa",
  merge: "#c084fc",
  linear: "#fb923c",
  default: "#94a3b8",
};

const COLORES_TORRE = { user: "#38bdf8", item: "#fb923c", fusion: "#34d399" };

function colorCapa(layer) {
  if (layer.torre && COLORES_TORRE[layer.torre]) return COLORES_TORRE[layer.torre];
  const t = (layer.type || "").toLowerCase();
  if (t.includes("embedding")) return COLORES.embedding;
  if (t.includes("flatten") || t.includes("reshape")) return COLORES.flatten;
  if (t.includes("dot")) return COLORES.dot;
  if (t.includes("multiply") || t.includes("mul")) return COLORES.multiply;
  if (t.includes("relu") || t.includes("activation") || t.includes("sigmoid") || t.includes("softmax")) return COLORES.activation;
  if (t.includes("dropout")) return COLORES.dropout;
  if (t.includes("batch")) return COLORES.batchnorm;
  if (t.includes("concat") || t.includes("merge") || t.includes("add")) return COLORES.merge;
  if (t.includes("dense") || t.includes("linear")) return COLORES.linear;
  return COLORES.default;
}

export default function RedNeuronal3D() {
  const [datos, setDatos] = useState(null);
  const [cargando, setCargando] = useState(true);
  const fgRef = useRef();

  useState(() => {
    fetch("/src/data/red_neuronal.json")
      .then(r => r.json())
      .then(d => { setDatos(d); setCargando(false); })
      .catch(() => {
        fetch("../src/data/red_neuronal.json")
          .then(r => r.json())
          .then(d => { setDatos(d); setCargando(false); })
          .catch(() => setCargando(false));
      });
  }, []);

  const nodoObject = useCallback(node => {
    const bg = new SpriteText(node.name, 6, colorCapa(node));
    bg.textHeight = 8;
    bg.position.y = 14;
    return bg;
  }, []);

  if (cargando) return <div style={{ padding: 40, color: "#94a3b8" }}>Cargando datos...</div>;
  if (!datos) return <div style={{ padding: 40, color: "#f87171" }}>Error: ejecuta primero generar_grafos_3d.py --json</div>;

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <div style={{ position: "absolute", top: 16, left: 16, zIndex: 10, background: "rgba(15,23,42,0.9)", padding: "16px 20px", borderRadius: 8, color: "#e2e8f0", maxWidth: 300 }}>
        <h3 style={{ fontSize: 16, marginBottom: 8 }}>Arquitectura Torre Doble (Two-Tower)</h3>
        <p style={{ fontSize: 12, color: "#94a3b8", marginBottom: 12 }}>{datos.descripcion || "User Tower + Item Tower con similitud coseno"}</p>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 10 }}>
          {[["Torre Usuario", COLORES_TORRE.user], ["Torre Producto", COLORES_TORRE.item], ["Fusión/Salida", COLORES_TORRE.fusion]].map(([k, c]) => (
            <span key={k} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 10, color: "#94a3b8" }}>
              <span style={{ width: 8, height: 8, borderRadius: "50%", background: c, display: "inline-block" }} />
              {k}
            </span>
          ))}
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          {Object.entries(COLORES).filter(([k]) => k !== "default").map(([k, c]) => (
            <span key={k} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 10, color: "#94a3b8" }}>
              <span style={{ width: 8, height: 8, borderRadius: "50%", background: c, display: "inline-block" }} />
              {k}
            </span>
          ))}
        </div>
      </div>
      <ForceGraph3D
        ref={fgRef}
        graphData={{
          nodes: datos.nodes.map(n => ({ ...n, id: n.name || n.id })),
          links: datos.links,
        }}
        nodeLabel="name"
        nodeColor={colorCapa}
        nodeVal={n => Math.max(4, Math.log2((n.params || 1) + 1) * 2)}
        nodeThreeObject={nodoObject}
        linkDirectionalParticles={4}
        linkDirectionalParticleWidth={1.5}
        linkDirectionalParticleColor={() => "#38bdf8"}
        linkColor={() => "#334155"}
        linkOpacity={0.6}
        backgroundColor="#020617"
        d3VelocityDecay={0.3}
        warmupTicks={100}
        cooldownTicks={200}
      />
    </div>
  );
}
