# Aprendizaje Fase 8 — Interpretabilidad: Grad-CAM

Material de estudio basado en la ejecución REAL sobre v003. La fase donde dejamos de preguntar "¿cuánto acierta?" para preguntar "**¿por qué acierta?**" — la pregunta que separa un juguete de una herramienta médica.

---

## 1. Por qué interpretabilidad no es opcional en medicina

Un clasificador con f1=0.84 que NO podemos auditar es inutilizable clínicamente:

- Un radiólogo jamás confiará en una caja negra ("confía en mí" no es argumento clínico).
- Si el modelo aprendió ATAJOS (marcadores laterales "L/R", texto de la placa, brillo del hospital) sus métricas mienten sobre su capacidad real.
- La regulación (FDA/CE) exige trazabilidad de decisiones en software médico.

Grad-CAM responde visualmente: **"estas son las zonas de la imagen que empujaron tu predicción"**.

## 2. Cómo funciona, paso a paso

```text
imagen 224×224
   ↓ base convolucional
activaciones última conv: 7×7×1280        ← "lo que vio" cada detector
   ↓ resto de la red
probabilidad P(PNEUMONIA)

Grad-CAM:
1. Forward hasta la última conv → feats (7×7×1280)
2. d(probabilidad)/d(feats) = gradientes         ← ¿cuánto cambia la salida
                                                    si cambiara cada activación?
3. pesos_canal = promedio espacial del gradiente  ← importancia global de cada
                                                    uno de los 1280 detectores
4. mapa = Σ_canal peso·feats  (7×7)
5. mapa = ReLU(mapa)      ← solo nos interesa lo que SUMA a favor de la clase
6. normalizar + upsample bilinear a 224×224
7. overlay semitransparente sobre la radiografía
```

Ideas clave por paso:

- **Paso 2**: el gradiente es la conexión matemática entre "región de la imagen" y "decisión". Sin él no sabemos qué activación influyó.
- **Paso 3**: promediar el gradiente en el espacio condensa "este canal fue relevante en general" en UN número.
- **Paso 5 ReLU**: los valores negativos significan evidencia EN CONTRA de la clase; para explicar la clase positiva solo conservamos lo positivo.
- **Paso 6**: el heatmap nace a resolución 7×7 (la imagen ya pasó 5 reducciones); al estirarlo se ve borroso — es normal y honesto: Grad-CAM da REGIONES, no píxeles.

Familia CAM: `CAM (2016)` exigía arquitectura especial → `Grad-CAM (2017)` generalizó con gradientes → `Grad-CAM++` mejora con múltiples instancias de la clase. Nosotros implementamos Grad-CAM clásico (~30 líneas).

---

## 3. Lo que encontramos en v003 (con números, no con impresiones)

Generamos 23 mapas: los casos más representativos de TP, TN, FP y FN del test. Además, un **chequeo objetivo de plausibilidad**: como los campos pulmonares ocupan el centro de una radiografía AP y los marcadores/textos los bordes, medimos dónde cae la energía del heatmap:

```json
{
  "energia_media_en_centro": 0.6396,
  "energia_media_en_borde": 0.2734,
  "interpretacion": "Revisar: parte relevante cae fuera de la zona pulmonar"
}
```

Lectura honesta (así se reporta en medicina):

1. ✔ **~64% de la energía está en el centro** → el grueso de la decisión mira anatomía relevante.
2. ⚠ **~27% en bordes** → hay evidencia de atención a zonas sospechosas de contener marcadores/texto de placa. Es un problema DOCUMENTADO de este dataset (los PNG traen letras L/R y datos quemados).
3. El chequeo automático exige >75% para aprobar: v003 queda "en observación". Esto NO invalida sus métricas — las hace merecedoras de escepticismo sano.

Posibles causas y caminos si el problema se confirmara: recortar bordes en preprocesamiento, dataset v2 sin artefactos, o entrenar con marcadores ocultos y comparar (experimento de control).

---

## 4. Limitaciones de Grad-CAM (para no sobrevender)

| Limitación | Consecuencia |
|---|---|
| Es una aproximación lineal local | Muestra correlación espacial, no causalidad |
| Resolución nativa 7×7 | Regiones gruesas; no sirve para lesiones milimétricas |
| Solo explica LA clase elegida | Un mapa distinto por clase en multiclase |
| Depende de la última conv | Elegir otra capa cuenta otra historia |
| Puede "mirar bien por razones malas" | Overlay bonito ≠ modelo correcto |

Regla profesional: Grad-CAM genera HIPÓTESIS auditables (ej. "¿mira los marcadores?"), que luego se prueban con experimentos controlados — nunca se publica como prueba definitiva.

---

## 5. Glosario rápido

| Término | Significado |
|---|---|
| **Explicabilidad / XAI** | Técnicas que hacen inspeccionables las decisiones de un modelo |
| **Mapa de saliencia** | Imagen de calor con la relevancia espacial para una predicción |
| **Gradiente** | Derivada de la salida respecto a un tensor interno |
| **ReLU sobre el mapa** | Conservar solo evidencia a favor de la clase explicada |
| **Atajo (shortcut learning)** | Aprender correlaciones espurias (marcadores) en vez del fenómeno (patología) |

---

## 6. Ejercicios sugeridos

1. ¿Por qué el paso ReLU es innecesario si explicamos la clase NORMAL con salida sigmoide invertida? Piensa qué representa score<0.5.
2. Diseña el "experimento de control" para confirmar/refutar que v003 usa los marcadores laterales: ¿qué harías con las imágenes, qué compararías?
3. Explica en 3 líneas a un médico qué significa el overlay, sin usar las palabras gradiente ni tensor.

---
*Referencias: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization", ICCV 2017 · Zhou et al., "Learning Deep Features for Discriminative Localization" (CAM), CVPR 2016 · DeGrave et al., "AI for radiographic COVID-19 detection selects shortcuts over signal", Nature MI 2021 (shortcuts).*

---

## 7. ADDENDUM — Resultado real del experimento de control (v006)

La hipótesis del punto 3 se puso a prueba recortando 24 px de cada borde y reentrenando con receta idéntica (`docs/12-v006-experimento-control.md`):

```text
                v003 (bordes)   v006 (sin bordes)
f1_macro        0.8442          0.8350      → diferencia dentro de ruido
recall PNEU     0.9872          0.9923      → igual o mejor sin bordes
energía borde   27.3%           19.6%       ↓ como esperaba el diagnóstico
```

**Veredicto: atajo refutado.** La energía en bordes era correlación incidental; el modelo decide mirando anatomía. Lección metodológica final: un heatmap sospechoso NO prueba nada por sí solo — se convierte en ciencia cuando genera un experimento controlado cuyo resultado queda registrado pase lo que pase.
