import { useEffect, useMemo, useState } from "react";

const COMPONENTES = [
  { id: "two_tower", label: "Two-Tower", color: "#38bdf8" },
  { id: "ncf", label: "NCF", color: "#a78bfa" },
  { id: "knn", label: "KNN", color: "#34d399" },
  { id: "nlp_resenas", label: "NLP", color: "#fb923c" },
  { id: "lightgcn", label: "GNN", color: "#f472b6" },
];

export default function ConsultarRecomendaciones() {
  const [datos, setDatos] = useState(null);
  const [busqueda, setBusqueda] = useState("");
  const [uid, setUid] = useState("u0007");

  useEffect(() => {
    fetch("/src/data/recomendaciones.json")
      .then(r => r.json())
      .then(setDatos)
      .catch(() => {});
  }, []);

  const usuarios = useMemo(() => (datos ? Object.keys(datos.usuarios) : []), [datos]);
  const filtrados = useMemo(() => {
    const q = busqueda.toLowerCase();
    return q ? usuarios.filter(x => x.includes(q)) : usuarios;
  }, [busqueda, usuarios]);

  if (!datos) return <div style={{ padding: 40, color: "#94a3b8" }}>Cargando motor de recomendaciones...</div>;

  const u = datos.usuarios[uid];

  if (!u) return <div style={{ padding: 40, color: "#f87171" }}>Usuario no encontrado</div>;

  return (
    <div style={{ width: "100%", height: "100%", overflow: "auto", padding: 20 }}>
      <h2 style={{ color: "#e2e8f0", marginBottom: 4 }}>Consultar Recomendaciones</h2>
      <p style={{ fontSize: 12, color: "#64748b", marginBottom: 14 }}>
        Modelo HÍBRIDO {datos.version ? datos.version.toUpperCase() : "V4"} entrenado · generado {datos.generado} · pesos:
        {COMPONENTES.map(c => ` ${c.label} ${(((datos.pesos[c.id] ?? datos.pesos[c.id === "ncf" ? "ncf_parcial" : c.id]) ?? 0) * 100) | 0}%`).join(" ·")}
      </p>

      <div style={{ display: "flex", gap: 12, marginBottom: 16, alignItems: "center", flexWrap: "wrap" }}>
        <input
          value={busqueda}
          onChange={e => setBusqueda(e.target.value)}
          placeholder="Buscar usuario (ej: u0007)..."
          style={{
            padding: "9px 14px", borderRadius: 8, border: "1px solid #334155",
            background: "#1e293b", color: "#e2e8f0", fontSize: 14, width: 260,
            outline: "none",
          }}
        />
        <select value={uid} onChange={e => setUid(e.target.value)} style={{
          padding: "9px 12px", borderRadius: 8, border: "1px solid #334155",
          background: "#1e293b", color: "#e2e8f0", fontSize: 14,
        }}>
          {filtrados.slice(0, 500).map(x => (
            <option key={x} value={x}>
              {datos.usuarios[x].nombre ? `${datos.usuarios[x].nombre} — ${x}` : x}
            </option>
          ))}
        </select>
        <span style={{ fontSize: 12, color: "#64748b" }}>{u.total_compras} compras en train</span>
      </div>

      <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
        {/* Columna izquierda: perfil e historial */}
        <div style={{ flex: "0 0 320px" }}>
          <div style={{ background: "#111827", borderRadius: 12, padding: 16, marginBottom: 14 }}>
            {u.nombre && (
              <>
                <h2 style={{ color: "#e2e8f0", fontSize: 20, marginBottom: 2 }}>{u.nombre}</h2>
                <p style={{ fontSize: 11, color: "#64748b", marginBottom: 10 }}>usuario {uid}</p>
              </>
            )}
            <h3 style={{ color: "#7dd3fc", fontSize: 14, marginBottom: 10 }}>Perfil{u.nombre ? "" : ` de ${uid}`}</h3>
            {Object.entries(u.perfil).map(([cat, n]) => (
              <div key={cat} style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", fontSize: 13, borderBottom: "1px solid #1e293b" }}>
                <span style={{ color: "#94a3b8" }}>{cat}</span>
                <span style={{ color: "#e2e8f0" }}>{n} compras</span>
              </div>
            ))}
          </div>
          <div style={{ background: "#111827", borderRadius: 12, padding: 16 }}>
            <h3 style={{ color: "#7dd3fc", fontSize: 14, marginBottom: 10 }}>Últimas compras</h3>
            {(u.historial || []).map((h, i) => (
              <div key={i} style={{ padding: "6px 0", borderBottom: "1px solid #1e293b", fontSize: 12 }}>
                <div style={{ color: "#e2e8f0" }}>{h.nombre}</div>
                <div style={{ display: "flex", justifyContent: "space-between", gap: 6 }}>
                  <span style={{ color: "#64748b" }}>
                    {h.categoria}{h.extra ? ` · ${h.extra}` : ""}
                  </span>
                  <span style={{
                    color: h.rating >= 4 ? "#22c55e" : h.rating >= 3 ? "#f59e0b" : "#ef4444",
                    flexShrink: 0,
                  }}>
                    {"★".repeat(Math.round(h.rating))}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Recomendaciones con explicacion */}
        <div style={{ flex: 1, minWidth: 480 }}>
          <div style={{ background: "#111827", borderRadius: 12, padding: 16 }}>
            <h3 style={{ color: "#34d399", fontSize: 15, marginBottom: 2 }}>
              Top-10 recomendado para {u.nombre || uid} {u.nombre && <span style={{ color: "#64748b", fontWeight: 400 }}>({uid})</span>}
            </h3>
            <p style={{ fontSize: 11, color: "#64748b", marginBottom: 12 }}>
              Cada barra muestra cuánto votó cada componente del híbrido por ese producto.
            </p>
            {(u.recomendaciones || []).map((r, i) => {
              const contrib = r.contribucion && Object.keys(r.contribucion).length
                ? r.contribucion : null;
              const total = contrib
                ? Math.max(Object.values(contrib).reduce((a, b) => a + b, 0), 1e-9)
                : 0;
              return (
                <div key={i} style={{
                  display: "flex", gap: 14, alignItems: "center",
                  padding: "10px 12px", marginBottom: 8,
                  background: "#1e293b", borderRadius: 10,
                }}>
                  <div style={{
                    width: 30, height: 30, borderRadius: "50%",
                    background: "#0ea5e9", color: "#04283d", fontWeight: 700,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    flexShrink: 0,
                  }}>{i + 1}</div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ color: "#e2e8f0", fontSize: 13, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                      {r.producto}
                    </div>
                    <div style={{ fontSize: 11, color: "#64748b", marginBottom: 4 }}>
                      {r.categoria}{r.extra ? ` · ${r.extra}` : ""}
                    </div>
                    {/* barras apiladas por componente (o barra simple en v5) */}
                    <div style={{ display: "flex", height: 8, borderRadius: 4, overflow: "hidden", background: "#0ea5e9" }}>
                      {contrib && COMPONENTES.map(c => (
                        <div key={c.id} title={`${c.label}: ${contrib[c.id]}`}
                          style={{ width: `${(contrib[c.id] / total) * 100}%`, background: c.color }} />
                      ))}
                    </div>
                  </div>
                  <div style={{ textAlign: "right", flexShrink: 0 }}>
                    <div style={{ color: "#34d399", fontWeight: 700, fontSize: 15 }}>{(r.score * 100).toFixed(1)}%</div>
                    <div style={{ fontSize: 10, color: "#64748b" }}>afinidad</div>
                  </div>
                </div>
              );
            })}
            <div style={{ display: "flex", gap: 12, marginTop: 10, flexWrap: "wrap" }}>
              {COMPONENTES.map(c => (
                <span key={c.id} style={{ fontSize: 11, color: "#94a3b8", display: "flex", alignItems: "center", gap: 4 }}>
                  <span style={{ width: 8, height: 8, borderRadius: 2, background: c.color }} />
                  {c.label}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
