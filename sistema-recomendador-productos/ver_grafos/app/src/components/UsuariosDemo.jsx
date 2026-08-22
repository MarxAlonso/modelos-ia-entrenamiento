import { useState, useEffect } from "react";
import Plot from "react-plotly.js";

const COLORES = ["#f97316", "#ef4444", "#8b5cf6", "#22c55e", "#f59e0b"];

export default function UsuariosDemo() {
  const [datos, setDatos] = useState(null);
  const [usuarioSeleccionado, setUsuarioSeleccionado] = useState("u0007");

  useEffect(() => {
    fetch("/src/data/usuarios_demo.json")
      .then(r => r.json())
      .then(setDatos)
      .catch(() => fetch("../src/data/usuarios_demo.json").then(r => r.json()).then(setDatos));
  }, []);

  if (!datos) return <div style={{ padding: 40, color: "#94a3b8" }}>Cargando...</div>;

  const usuarios = Object.keys(datos);
  const usuario = datos[usuarioSeleccionado];

  if (!usuario) return <div style={{ padding: 40, color: "#f87171" }}>Usuario no encontrado</div>;

  return (
    <div style={{ width: "100%", height: "100%", overflow: "auto", padding: 20 }}>
      <h2 style={{ color: "#e2e8f0", marginBottom: 20 }}>Perfiles de Usuarios Demo</h2>
      
      <div style={{ display: "flex", gap: 20, marginBottom: 20 }}>
        {usuarios.map((uid, i) => (
          <button
            key={uid}
            onClick={() => setUsuarioSeleccionado(uid)}
            style={{
              padding: "12px 20px",
              borderRadius: 8,
              border: usuarioSeleccionado === uid ? `2px solid ${COLORES[i]}` : "1px solid #334155",
              background: usuarioSeleccionado === uid ? "#1e293b" : "transparent",
              color: usuarioSeleccionado === uid ? COLORES[i] : "#94a3b8",
              cursor: "pointer",
              fontSize: 14,
              fontWeight: usuarioSeleccionado === uid ? "bold" : "normal",
            }}
          >
            {uid}
          </button>
        ))}
      </div>

      <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
        <div style={{ flex: 1, minWidth: 300, background: "#111827", borderRadius: 12, padding: 20 }}>
          <h3 style={{ color: "#e2e8f0", marginBottom: 16 }}>Historial de Compras</h3>
          <p style={{ color: "#94a3b8", fontSize: 12, marginBottom: 12 }}>
            Total: {usuario.total_compras} compras
          </p>
          <div style={{ maxHeight: 300, overflowY: "auto" }}>
            {usuario.historial.map((prod, i) => (
              <div key={i} style={{ 
                padding: "8px 12px", 
                borderBottom: "1px solid #1e293b",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}>
                <div>
                  <div style={{ color: "#e2e8f0", fontSize: 13 }}>{prod.nombre}</div>
                  <div style={{ color: "#64748b", fontSize: 11 }}>{prod.categoria}</div>
                </div>
                <div style={{ 
                  color: prod.rating >= 4 ? "#22c55e" : prod.rating >= 3 ? "#f59e0b" : "#ef4444",
                  fontWeight: "bold",
                  fontSize: 14,
                }}>
                  {prod.rating.toFixed(1)}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div style={{ flex: 1, minWidth: 300 }}>
          <div style={{ background: "#111827", borderRadius: 12, padding: 20, marginBottom: 20 }}>
            <h3 style={{ color: "#e2e8f0", marginBottom: 16 }}>Categorías Preferidas</h3>
            <Plot
              data={[{
                type: "bar",
                x: usuario.categorias_preferidas.map(c => c.cat),
                y: usuario.categorias_preferidas.map(c => c.count),
                marker: { color: COLORES.slice(0, usuario.categorias_preferidas.length) },
                text: usuario.categorias_preferidas.map(c => `${c.count} compras`),
                textposition: "outside",
              }]}
              layout={{
                height: 250,
                margin: { l: 40, r: 20, t: 10, b: 80 },
                xaxis: { tickangle: -45 },
                yaxis: { title: "Compras" },
              }}
              config={{ responsive: true }}
            />
          </div>

          <div style={{ background: "#111827", borderRadius: 12, padding: 20 }}>
            <h3 style={{ color: "#e2e8f0", marginBottom: 16 }}>Distribución de Ratings</h3>
            <Plot
              data={[{
                type: "pie",
                labels: ["5★", "4★", "3★", "2★", "1★"],
                values: [
                  usuario.historial.filter(p => p.rating >= 4.5).length,
                  usuario.historial.filter(p => p.rating >= 3.5 && p.rating < 4.5).length,
                  usuario.historial.filter(p => p.rating >= 2.5 && p.rating < 3.5).length,
                  usuario.historial.filter(p => p.rating >= 1.5 && p.rating < 2.5).length,
                  usuario.historial.filter(p => p.rating < 1.5).length,
                ],
                marker: { colors: ["#22c55e", "#86efac", "#fde047", "#fb923c", "#ef4444"] },
                hole: 0.4,
              }]}
              layout={{
                height: 250,
                margin: { l: 20, r: 20, t: 10, b: 10 },
                showlegend: true,
                legend: { font: { color: "#94a3b8", size: 10 } },
              }}
              config={{ responsive: true }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
