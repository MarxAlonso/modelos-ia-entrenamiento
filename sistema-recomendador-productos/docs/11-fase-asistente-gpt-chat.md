# Fase 11 — Asistente GPT-Híbrido (mini-GPT afinado + RAG)

El modelo base v6 (Mini-GPT de caracteres) se convirtió en un **asistente
consultable** aplicando el mismo ciclo de los LLM comerciales:

```
modelo base (preentrenamiento) ──> SFT por instrucciones ──> producto (chat)
        etapa A                        etapa B                servidor + UI
```

La diferencia con un chatbot normal: el asistente **no inventa datos**. Cada
pregunta viaja con una línea `### Contexto:` que contiene los Top-10 REALES del
motor híbrido (el mismo `recomendaciones.json` que consume la app), y toda
respuesta generada se **valida contra esos datos** antes de mostrarse. Si el
GPT garabatea, responde la plantilla basada en datos.

---

## 1. Especificaciones del modelo

| Parámetro | Valor |
|---|---|
| Arquitectura | Transformer decodificador (GPT): 6 bloques, d=256, 8 cabezas, FFN 768 |
| Contexto | 512 caracteres (pregunta + contexto RAG + respuesta) |
| Vocabulario | 147 símbolos (nivel carácter) |
| **Parámetros** | **4 157 588 (~4.2 M)** |
| **Peso en disco** | **16.06 MB** (`models/mini_gpt_chat.weights.h5`) |
| Checkpoint base (etapa A) | 16.06 MB (`models/mini_gpt_chat_base.weights.h5`) |
| Dataset SFT | 12 399 parejas QA — `data/entrenamiento_qa.jsonl` (4.7 MB) |
| Entrenamiento | Etapa A: 900 pasos (ppl 7.23) · Etapa B: 6 000 pasos (ppl 1.19) |

## 2. Consumo medido (GTX 1650 4 GB, TensorFlow vía WSL)

| Momento | RAM (RSS del proceso) | VRAM |
|---|---|---|
| Modelo cargado, sin tráfico | ~1.05 GB | ~2.1 GB |
| Durante inferencia (3 candidatos en paralelo) | ~1.25 GB | ~2.2 GB (pico) |
| Entrenamiento SFT (batch 8 × 512 tokens) | ~2 GB | ~2.15 GB estables |

Latencias típicas por respuesta: **~7 s** la primera consulta de cada tipo
(compila el paso de inferencia y genera 3 candidatos), **instantánea** después
(caché en memoria por intención+usuario). El peso del modelo en VRAM es mínimo
(16 MB); lo que consume son las activaciones de atención (8 cabezas × 512×512)
y el contexto de CUDA (~400 MB fijos).

## 3. Comandos — modelo LLM

Todos desde la raíz del proyecto (`sistema-recomendador-productos/`).

### 3.1 Regenerar el dataset de instrucciones

```powershell
.\venv\Scripts\python.exe src\modelos\v6_mini_gpt\datos_entrenamiento_qa.py
```

Requiere `ver_grafos/app/src/data/recomendaciones.json`
(se regenera con `src\servicio\recomendaciones_app.py` si hace falta).

### 3.2 Entrenar (GPU vía WSL — recomendado)

```powershell
# Pipeline completo: etapa A (900 pasos) + SFT (2000 pasos), ~20 min
wsl -d Ubuntu-22.04 -- bash /root/arrancar-rag.sh

# Solo SFT reutilizando el checkpoint base (~10 min)
wsl -d Ubuntu-22.04 -- bash /root/arrancar-sftrag.sh

# Continuar el SFT desde los pesos finales (más memorización)
wsl -d Ubuntu-22.04 -- bash /root/arrancar-sftrag2.sh
```

Equivalente manual (permite cambiar pasos):

```powershell
wsl -d Ubuntu-22.04 -- bash -lc "cd '/mnt/c/developer-marx/Proyectos UTP/modelos-ia-entrenamiento/sistema-recomendador-productos' && ~/run-tf.sh src/modelos/v6_mini_gpt/afinar_gpt_chat.py --pasos-base 900 --pasos-sft 2000"
```

Seguimiento del progreso:

```powershell
wsl -d Ubuntu-22.04 -- tail -f /root/entrenamiento_rag.log
```

En CPU (Windows, lento — solo para pruebas cortas):

```powershell
.\venv\Scripts\python.exe src\modelos\v6_mini_gpt\afinar_gpt_chat.py --pasos-base 50 --pasos-sft 50
```

### 3.3 Probar el modelo sin servidor (CLI)

```powershell
# Pregunta suelta (sin datos RAG)
wsl -d Ubuntu-22.04 -- bash -lc "cd '/mnt/c/developer-marx/Proyectos UTP/modelos-ia-entrenamiento/sistema-recomendador-productos' && ~/run-tf.sh src/modelos/v6_mini_gpt/afinar_gpt_chat.py --generar 'Cual es mi perfil de compra?'"

# Con inyeccion RAG real para un usuario (--con-contexto)
wsl -d Ubuntu-22.04 -- bash -lc "cd '/mnt/c/developer-marx/Proyectos UTP/modelos-ia-entrenamiento/sistema-recomendador-productos' && ~/run-tf.sh src/modelos/v6_mini_gpt/afinar_gpt_chat.py --generar 'Que me recomiendas?' --uid u0007 --con-contexto"
```

Métricas del último entrenamiento: `models/mini_gpt_chat_metricas.json`.

## 4. Comandos — backend y frontend

### 4.1 Backend del asistente (puerto 8000)

```powershell
# GPU via WSL (recomendado, deja la terminal abierta)
wsl -d Ubuntu-22.04 -- bash /root/arrancar-chat.sh

# Alternativa CPU nativa (funciona, ~40 s por respuesta)
.\venv\Scripts\python.exe src\servicio\servidor_chat.py
```

Verificación:

```powershell
curl http://localhost:8000/api/salud
# {"ok": true, "modelo_cargado": true, "usuarios": 500}

curl -X POST http://localhost:8000/api/chat `
  -H "Content-Type: application/json" `
  -d '{\"mensaje\": \"Que me recomiendas?\", \"uid\": \"u0007\"}'
```

### 4.2 Frontend (puerto 5173)

```powershell
cd ver_grafos\app
npm install     # solo la primera vez
npm run dev
```

El panel **Asistente GPT-Híbrido** aparece al final de la vista
*Consultar Recomendaciones*. Las llamadas `fetch("/api/chat")` pasan por el
proxy de Vite (`vite.config.js` → `http://localhost:8000`).

## 5. Problemas frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| `Address already in use` al arrancar backend | otro servidor vivo en 8000 | `wsl -d Ubuntu-22.04 -- bash -lc "pkill -f servidor_chat.py"` |
| `OOM when allocating tensor [16,8,512,512]` | VRAM insuficiente entrenando | ya está mitigado (`BATCH_SFT=8`); si reaparece, baja a 4 en `afinar_gpt_chat.py` |
| El chat responde *“No pude contactar al servidor”* | backend apagado | levanta el backend (4.1) y recarga la vista |
| Respuestas lentas en CPU | TF Windows no usa GPU | usa la vía WSL (4.1) |
| `No encuentro ese usuario` | uid fuera de rango | usuarios válidos: `u0000` … `u0499` |

## 6. Archivos de la fase

```
src/modelos/v6_mini_gpt/datos_entrenamiento_qa.py   generador del dataset QA+RAG
src/modelos/v6_mini_gpt/afinar_gpt_chat.py          arquitectura + SFT + inferencia
src/servicio/servidor_chat.py                       backend HTTP (intencion+RAG+validacion)
ver_grafos/app/src/components/ChatRecomendaciones.jsx   UI del chat
data/entrenamiento_qa.jsonl                          12 399 parejas pregunta-respuesta
models/mini_gpt_chat.weights.h5                      pesos finales (16 MB)
models/mini_gpt_chat_vocab.json                      vocabulario de caracteres
models/mini_gpt_chat_metricas.json                   metricas del entrenamiento
```
