# Aprendizaje Fase 10 — Dominios distintos, etiquetas débiles y expectativas realistas

Material de estudio basado en la versión REAL v007 (NIH adulto), cuyo resultado modesto enseña más que un buen número: **cómo se razona cuando el modelo "no funciona"**.

---

## 1. Lo que pasó en v007

| Métrica | v003 (v1 pediátrico) | v007 (v2 NIH adulto) |
|---|---|---|
| F1 macro | 0.844 | **0.334** |
| ROC-AUC | 0.967 | 0.684 |
| FP | 79 | **6 233** (de 9 861 normales) |
| Precision PNEUMONIA | 0.83 | **0.07** |

El modelo grita "PNEUMONIA" por todo. Tres causas reales, en orden de importancia:

### a) Etiquetas débiles (la causa raíz)

Las etiquetas de NIH ChestX-ray14 fueron extraídas **por minería de texto de informes radiológicos** (NLP sobre reportes), no por doble lectura de radiólogos. "Pneumonia" en NIH significa *el informe mencionó neumonía* — con falsos positivos y negativos propios del NLP. Estás entrenando contra una verdad imperfecta; ningún ajuste de hiperparámetros arregla eso.

> Concepto: **label noise**. Con ~8-20% de ruido en las etiquetas positivas, el techo práctico de AUC baja incluso para modelos enormes.

### b) La tarea que construimos es artificialmente dura

Nuestro mapeo binario estricto excluyó 50 328 imágenes con hallazgos NO-neumonía (Infiltration, Atelectasis...). Muchas de esas se VEN como neumonía (infiltrados). El modelo solo vio pulmones perfectamente sanos vs neumonía — pero la neumonía NIH incluye cuadros leves que parecen "casi normales". La frontera aprendida es frágil.

### c) Umbral 0.5 + weights extremos + pocos positivos

Con 754 positivos y weights PNEU=2.0, el modelo compensa subiendo probabilidades globales → a corte 0.50 dispara FP masivos. El ROC-AUC 0.68 muestra que ORDENA regular, no que sea inútil; el punto de operación está mal elegido para esta distribución (lección de v005 aplicándose).

### Contexto de literatura (para calibrar expectativas)

CheXNet y derivados reportan AUC ~0.76-0.84 para Pneumonia en ChestX-ray14 usando TODO el dataset y arquitecturas DenseNet121 grandes. Nuestro v007 (754 positivos, MobileNetV2 congelada, 20 min CPU) sacó 0.684. **No estamos lejos del estado del arte dadas las condiciones** — pero tampoco cerca de lo que v003 logró en v1, porque v2 ES más difícil.

---

## 2. Domain shift: por qué prohibimos comparar v1 vs v2

```text
v1: niños guangzhoneses, AP pediátrico, etiquetado por consenso médico, 2 clases limpias
v2: adultos EE.UU., variabilidad de equipos enorme, etiquetas por NLP, multi-etiqueta recortada
```

Son problemas DIFERENTES que comparten nombre. Por eso:

- El registro ahora tiene `mejor_por_dataset` (campeón por dominio): v003 gobierna v1, v007 gobierna v2.
- Los estados vigente/superada se calculan SOLO dentro del mismo dataset.
- Si algún día quieres medir el shift: entrena en v1 → evalúa en test v2 (y viceversa); esa caída de métricas ES la medida del shift.

---

## 3. Caminos legítimos para mejorar en NIH (v008+)

| Camino | Idea | Costo |
|---|---|---|
| Incluir hallazgos similares como NEGATIVOS explícitos | Entrenar con Infiltration/etc. como clase "no-neumonía" para endurecer la frontera | medio |
| Umbral/calibración específica v2 | Repetir análisis tipo v005 pero con validación cruzada | bajo |
| Más épocas + datos positivos aumentados | Oversampling de los 754 positivos | bajo-medio |
| Fine tuning completo en NIH | Descongelar base completa con lr 1e-6..1e-5 y muchas épocas | alto en CPU |

---

## 4. Glosario rápido

| Término | Significado |
|---|---|
| **Etiqueta débil / label noise** | Etiquetas generadas indirectamente (NLP sobre informes) con errores inherentes |
| **Multi-etiqueta → binario** | Reducir 14 hallazgos a NORMAL/PNEUMONIA decide qué casos "duros" se descartan o confunden |
| **Domain shift** | Cambio de distribución entre datasets (pediátrico↔adulto, hospital↔hospital) |
| **Campeón por dominio** | Comparar versiones solo dentro del mismo dataset/test (regla §4 de nuestra política) |
| **Techo por ruido** | Máxima calidad alcanzable limitada por la calidad de las etiquetas, no por el modelo |

---

## 5. Ejercicios sugeridos

1. Con precision=0.0719 y recall=0.8703 de v007: calcula cuántos FP hay por cada TP y explica por qué este modelo no es desplegable con t=0.5 aunque su recall sea alto.
2. Diseña el experimento "entrenar en v1, evaluar en v2": ¿qué esperas que pase con recall_pneumonia? ¿Por qué sería una medición válida del shift y no un fracaso?
3. Investiga: ¿qué diferencia hay entre las etiquetas de ChestX-ray14 y las del dataset CheXpert (también débil, pero con reglas de incertidumbre)?

---
*Referencias: Wang et al., "ChestX-ray8/ChestX-ray14", CVPR 2017 · Rajpurkar et al., "CheXNet" arXiv:1711.05225 · Irvin et al., "CheXpert" AAAI 2019 (label uncertainty).*
