# Fase 7 — Mini-GPT: modelo de lenguaje Transformer entrenado desde cero

## Objetivo

Completar la línea de aprendizaje NLP del proyecto: tokenización (Fase 2) → embeddings Word2Vec (Fase 6) → **modelo de lenguaje autorregresivo tipo GPT** entrenado desde cero sobre el corpus propio, en escala educativa (~633K parámetros) y ejecutable en CPU.

No es un LLM comercial ni pretende serlo: es **el mismo mecanismo de GPT** (self-attention causal + predicción del siguiente token) a escala mínima, propiedad total del proyecto.

## Pipeline (`src/lenguaje_mini_gpt.py`)

```
corpus (descripciones + titulos ES, ~7M caracteres)
        │
[1-2] Vocabulario a nivel de CARACTER (147 simbolos)
        │
[3]   Codificacion del corpus
        │
[4]   Mini-GPT: emb_token + emb_posicion + 3 bloques transformer + cabeza LM
        │      (cada bloque: MultiHeadAttention causal(4 cabezas)
        │            -> residual+LayerNorm -> FFN 128->512->128 -> residual+LN)
[5]   Entrenamiento auto-supervisado: predecir el siguiente caracter
        │      (Adam 3e-4, batch 64, contexto 96 caracteres)
[6]   Pesos + vocabulario + metricas
        │
[7]   Generacion autorregresiva con temperatura
```

## Decisiones técnicas

| Decisión | Justificación |
|----------|---------------|
| Tokenización por caracter | Sin palabras fuera de vocabulario; vocab de 147 vs ~24K palabras; los LLM reales usan BPE — es donde crecería este diseño |
| Contexto de 96 caracteres | La atención cuesta O(n²); en CPU mantiene ~0.28 s/paso |
| 3 bloques · d=128 · 4 cabezas | 632,852 parámetros (2.4 MB): entrena en minutos, no horas |
| `use_causal_mask=True` | Máscara causal nativa de Keras: cada posición solo ve su pasado — el corazón de un GPT |
| Pesos + vocab JSON (no `.keras`) | El modelo se reconstruye con la misma función y carga pesos: evita problemas de serialización |

## Curva de entrenamiento

### Primera versión — CPU Windows (1500 pasos ≈ 8 min)

| Paso | Pérdida | Perplejidad |
|------|---------|-------------|
| 100 | 3.057 | 21.3 |
| 1500 | 1.865 | 6.45 |

### Versión final — GPU vía WSL2 (5000 pasos ≈ 7 min)

| Paso | Pérdida | Perplejidad |
|------|---------|-------------|
| 500 | 2.456 | 11.7 |
| 2000 | 1.835 | 6.3 |
| 3500 | 1.634 | 5.1 |
| 5000 | **1.544** | **4.68** ✅ |

## Aceleración GPU: WSL2 + CUDA

TensorFlow ≥ 2.11 no soporta GPU en Windows nativo; la solución es entrenar dentro de Ubuntu sobre WSL2:

| Componente | Configuración |
|------------|---------------|
| GPU | GeForce GTX 1650 (Turing, CC 7.5, 4 GB) |
| Driver Windows | NVIDIA Studio 610.88 (≥ 525 requerido para passthrough CUDA-WSL) |
| Distro | Ubuntu 22.04 LTS (`wsl --install -d Ubuntu-22.04`) |
| Python | 3.10.12 + venv en `/root/gpt-gpu` |
| Framework | `pip install tensorflow[and-cuda]` → TF 2.21 + CUDA 12 vía pip (sin toolkit manual) |
| Variable clave | `LD_LIBRARY_PATH` apuntando a `site-packages/nvidia/*/lib` |

### Benchmark medido (mismo modelo y datos)

| Entorno | s/paso | 1500 pasos | 5000 pasos |
|---------|--------|------------|------------|
| CPU Windows nativo | 0.278 | ~7 min | ~23 min |
| CPU WSL (driver viejo) | 1.096 | ~27 min | ~91 min |
| **GPU GTX 1650 (WSL2)** | **0.068** | **~1.7 min** | **~5.7 min** |

Aceleración efectiva: **4.1× frente a CPU nativo**, lo que hizo viable cuadruplicar el entrenamiento.

## Generación real comparada (temperatura 0.8)

```
Prompt: 'Bolsa con capacidad para'
CPU 1500 pasos : 'Bolsa con capacidad para con una carro. La cantidad de una bebendele...'
GPU 5000 pasos : 'Bolsa con capacidad para estaño contenido o para compartir con una
                  creagulada y especial... el paquete en grande... parece ser maíz'
```

Con 5000 pasos aparecen frases completas coherentes ("lo que facilita su…", "listo para…", "lo que sugiere que…"), formato de presentación correcto (`x750ml`, `x5und`, "paquete que contiene") y conectores discursivos naturales. Persisten palabras inventadas ocasionales — esperable a esta escala.

## Uso

```powershell
# Generar (Windows, CPU rapido para inferencia)
venv\Scripts\python.exe src\lenguaje_mini_gpt.py --generar "Arroz Diana"

# Reentrenar en GPU (WSL)
wsl -d Ubuntu-22.04 -u root -- bash <script_env> "/mnt/c/.../src/lenguaje_mini_gpt.py" --pasos 5000
```

Los pesos se guardan directamente en `models/` del proyecto (ruta montada `/mnt/c`), por lo que la inferencia sigue funcionando desde Windows sin cambios.

## Artefactos

| Archivo | Contenido |
|---------|-----------|
| `models/mini_gpt.weights.h5` | Pesos del Transformer |
| `models/mini_gpt_vocab.json` | Vocabulario carácter→índice |
| `models/mini_gpt_metricas.json` | Parámetros, curva final, perplejidad |

## Resultados de la fase

- [x] Transformer decoder implementado desde cero en Keras (sin modelos preentrenados)
- [x] Entrenamiento auto-supervisado sobre el corpus propio del proyecto
- [x] Perplejidad 21.3 → 6.45 y generación con estilo del dominio verificada
- [x] CLI de generación y reentrenamiento (`--generar`, `--pasos`)
- [x] Limitaciones documentadas con honestidad (escala, CPU-Windows, calidad de texto)

## Conexión con el recomendador (idea a futuro)

El mini-GPT habilita **cold-start de contenido**: productos nuevos sin descripción podrían recibir una generada por este modelo y vectorizarse con TF-IDF/W2V para entrar al híbrido sin esperar interacciones.
