import { useState } from "react";
import {
  FaBrain, FaCube, FaCircleNodes, FaBullseye,
  FaChartColumn, FaUser, FaMapLocationDot, FaBolt,
} from "react-icons/fa6";
import RedNeuronal3D from "./components/RedNeuronal3D.jsx";
import EspacioLatente from "./components/EspacioLatente.jsx";
import UsuariosCategorias3D from "./components/UsuariosCategorias3D.jsx";
import Metricas2D from "./components/Metricas2D.jsx";
import AfinidadCategorias from "./components/AfinidadCategorias.jsx";
import FlujoHibrido from "./components/FlujoHibrido.jsx";
import UsuariosDemo from "./components/UsuariosDemo.jsx";
import ConsultarRecomendaciones from "./components/ConsultarRecomendaciones.jsx";

const GRAFOS = [
  { id: "red3d", label: "Red Neuronal 3D (Torre Doble)", icon: FaBrain, grupo: "3d" },
  { id: "espacio", label: "Espacio Latente 3D", icon: FaCube, grupo: "3d" },
  { id: "uc3d", label: "Grafo Usuario-Producto", icon: FaCircleNodes, grupo: "3d" },
  { id: "consultar", label: "Consultar Recomendaciones", icon: FaBullseye, grupo: "analisis" },
  { id: "metricas", label: "Métricas de Modelos", icon: FaChartColumn, grupo: "analisis" },
  { id: "usuarios", label: "Perfiles de Usuarios", icon: FaUser, grupo: "analisis" },
  { id: "afinidad", label: "Afinidad Categorías", icon: FaMapLocationDot, grupo: "analisis" },
  { id: "flujo", label: "Flujo del Híbrido", icon: FaBolt, grupo: "analisis" },
];

export default function App() {
  const [activo, setActivo] = useState("red3d");

  const renderComponente = () => {
    switch (activo) {
      case "red3d": return <RedNeuronal3D />;
      case "espacio": return <EspacioLatente />;
      case "uc3d": return <UsuariosCategorias3D />;
      case "consultar": return <ConsultarRecomendaciones />;
      case "metricas": return <Metricas2D />;
      case "usuarios": return <UsuariosDemo />;
      case "afinidad": return <AfinidadCategorias />;
      case "flujo": return <FlujoHibrido />;
      default: return <RedNeuronal3D />;
    }
  };

  return (
    <div style={{ display: "flex", width: "100%", height: "100%" }}>
      <nav style={{
        width: 240, background: "#0f172a", color: "#e2e8f0",
        display: "flex", flexDirection: "column", padding: "20px 0",
        flexShrink: 0, overflowY: "auto",
      }}>
        <h2 style={{ fontSize: 14, padding: "0 16px 16px", borderBottom: "1px solid #1e293b", color: "#94a3b8" }}>
          SISTEMA RECOMENDADOR
        </h2>
        {["3d", "analisis"].map(grupo => (
          <div key={grupo}>
            <div style={{ padding: "8px 16px", fontSize: 11, color: "#64748b",
              borderBottom: "1px solid #1e293b", margin: "8px 0" }}>
              {grupo === "3d" ? "VISUALIZACIONES 3D" : "ANÁLISIS Y MÉTRICAS"}
            </div>
            {GRAFOS.filter(g => g.grupo === grupo).map(g => (
              <button key={g.id} onClick={() => setActivo(g.id)} style={{
                display: "flex", alignItems: "center", gap: 10,
                background: activo === g.id ? "#1e293b" : "transparent",
                border: "none", color: activo === g.id ? "#38bdf8" : "#94a3b8",
                padding: "10px 16px", cursor: "pointer", fontSize: 13,
                textAlign: "left",
                borderLeft: activo === g.id ? "3px solid #38bdf8" : "3px solid transparent",
              }}>
                <g.icon size={14} style={{ flexShrink: 0 }} /> {g.label}
              </button>
            ))}
          </div>
        ))}
        <div style={{ marginTop: "auto", padding: "16px", fontSize: 11, color: "#475569" }}>
          NCF · Word2Vec · Transformer<br />Two-Tower · LightGCN
        </div>
      </nav>
      <main style={{ flex: 1, position: "relative" }}>
        {renderComponente()}
      </main>
    </div>
  );
}
