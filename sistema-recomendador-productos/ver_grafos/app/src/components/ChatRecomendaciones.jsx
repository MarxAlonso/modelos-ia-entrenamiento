import { useEffect, useRef, useState } from "react";
import { FaCircleNotch, FaPaperPlane, FaRobot, FaUser } from "react-icons/fa6";

const SUGERENCIAS = [
  "¿Qué me recomiendas?",
  "¿Cuál es mi perfil de compra?",
  "¿Por qué me recomiendas eso?",
  "¿Quién eres?",
];

export default function ChatRecomendaciones({ uid, nombre }) {
  const [mensajes, setMensajes] = useState([
    {
      rol: "asistente",
      texto: `Hola${nombre ? ` ${nombre.split(" ")[0]}` : ""}. Soy un mini-GPT entrenado desde cero y afinado con las respuestas del motor híbrido. Pregúntame qué comprar, por qué te lo recomiendo o cómo funciono.`,
      fuente: null,
    },
  ]);
  const [entrada, setEntrada] = useState("");
  const [enviando, setEnviando] = useState(false);
  const finRef = useRef(null);

  useEffect(() => {
    finRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [mensajes, enviando]);

  const enviar = async texto => {
    const mensaje = (texto ?? entrada).trim();
    if (!mensaje || enviando) return;
    setEntrada("");
    setMensajes(m => [...m, { rol: "usuario", texto: mensaje }]);
    setEnviando(true);
    try {
      const r = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mensaje, uid }),
      });
      const d = await r.json();
      setMensajes(m => [...m, {
        rol: "asistente",
        texto: d.respuesta || d.error || "No tengo respuesta para eso.",
        fuente: d.fuente,
        intencion: d.intencion,
        tiempoMs: d.tiempo_ms,
      }]);
    } catch {
      setMensajes(m => [...m, {
        rol: "asistente",
        texto: "No pude contactar al servidor del asistente. Arráncalo con:\n" +
          "wsl -d Ubuntu-22.04 -- ~/run-tf.sh src/servicio/servidor_chat.py",
        error: true,
      }]);
    } finally {
      setEnviando(false);
    }
  };

  return (
    <div style={{
      background: "#111827", borderRadius: 12, padding: 16,
      display: "flex", flexDirection: "column", height: 520,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
        <FaRobot color="#34d399" size={16} />
        <h3 style={{ color: "#34d399", fontSize: 15, margin: 0 }}>
          Asistente GPT-Híbrido
        </h3>
        <span style={{
          fontSize: 10, color: "#94a3b8", border: "1px solid #334155",
          borderRadius: 999, padding: "2px 8px",
        }}>
          mini-GPT {`{`}SFT{`}`} · consulta a {uid}
        </span>
      </div>
      <p style={{ fontSize: 11, color: "#64748b", marginBottom: 10 }}>
        Transformer afinado por instrucciones que redacta respuestas con los datos reales del motor híbrido.
      </p>

      {/* historial de mensajes */}
      <div style={{
        flex: 1, overflowY: "auto", display: "flex", flexDirection: "column",
        gap: 10, padding: "4px 2px", marginBottom: 10,
      }}>
        {mensajes.map((m, i) => (
          <div key={i} style={{
            display: "flex", gap: 8,
            flexDirection: m.rol === "usuario" ? "row-reverse" : "row",
            maxWidth: "100%",
          }}>
            <div style={{
              width: 26, height: 26, borderRadius: "50%", flexShrink: 0,
              display: "flex", alignItems: "center", justifyContent: "center",
              background: m.rol === "usuario" ? "#0ea5e9" : m.error ? "#ef4444" : "#065f46",
              color: m.rol === "usuario" ? "#04283d" : "#d1fae5",
            }}>
              {m.rol === "usuario" ? <FaUser size={11} /> : <FaRobot size={12} />}
            </div>
            <div style={{
              background: m.rol === "usuario" ? "#164e63" : m.error ? "#7f1d1d" : "#1e293b",
              borderRadius: 10, padding: "8px 12px", maxWidth: "78%",
            }}>
              <div style={{ color: "#e2e8f0", fontSize: 13, whiteSpace: "pre-wrap" }}>
                {m.texto}
              </div>
              {m.fuente && (
                <div style={{ marginTop: 5, display: "flex", gap: 8, alignItems: "center" }}>
                  <span style={{
                    fontSize: 9, color: m.fuente === "gpt+hibrido" ? "#34d399" : "#fbbf24",
                    border: `1px solid ${m.fuente === "gpt+hibrido" ? "#065f46" : "#92400e"}`,
                    borderRadius: 999, padding: "1px 7px",
                  }}>
                    {m.fuente === "gpt+hibrido" ? "redactado por el GPT" : "datos del híbrido"}
                  </span>
                  {m.intencion && (
                    <span style={{ fontSize: 9, color: "#64748b" }}>intención: {m.intencion}</span>
                  )}
                  {m.tiempoMs != null && (
                    <span style={{ fontSize: 9, color: "#475569" }}>{m.tiempoMs} ms</span>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
        {enviando && (
          <div style={{ display: "flex", gap: 8, alignItems: "center", color: "#64748b", fontSize: 12 }}>
            <div style={{
              width: 26, height: 26, borderRadius: "50%", flexShrink: 0,
              display: "flex", alignItems: "center", justifyContent: "center",
              background: "#065f46", color: "#d1fae5",
            }}>
              <FaCircleNotch size={12} className="animate-spin" />
            </div>
            El asistente está generando la respuesta...
          </div>
        )}
        <div ref={finRef} />
      </div>

      {/* sugerencias rapidas */}
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 8 }}>
        {SUGERENCIAS.map(s => (
          <button key={s} onClick={() => enviar(s)} disabled={enviando} style={{
            background: "#1e293b", color: "#94a3b8", border: "1px solid #334155",
            borderRadius: 999, padding: "4px 10px", fontSize: 11, cursor: "pointer",
          }}>
            {s}
          </button>
        ))}
      </div>

      {/* caja de entrada */}
      <div style={{ display: "flex", gap: 8 }}>
        <input
          value={entrada}
          onChange={e => setEntrada(e.target.value)}
          onKeyDown={e => e.key === "Enter" && enviar()}
          placeholder={`Pregunta algo sobre ${uid}...`}
          disabled={enviando}
          style={{
            flex: 1, padding: "9px 14px", borderRadius: 8,
            border: "1px solid #334155", background: "#1e293b",
            color: "#e2e8f0", fontSize: 13, outline: "none",
          }}
        />
        <button onClick={() => enviar()} disabled={enviando || !entrada.trim()} style={{
          background: enviando ? "#334155" : "#059669", border: "none",
          borderRadius: 8, width: 42, cursor: enviando ? "default" : "pointer",
          display: "flex", alignItems: "center", justifyContent: "center",
          color: "#d1fae5",
        }} title="Enviar">
          <FaPaperPlane size={14} />
        </button>
      </div>
    </div>
  );
}
