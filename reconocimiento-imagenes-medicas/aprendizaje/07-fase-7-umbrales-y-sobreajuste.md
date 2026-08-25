# Aprendizaje Fase 7 — Umbrales de decisión y el sobreajuste invisible

Material de estudio basado en la versión REAL v005, cuya hipótesis quedó **refutada por segunda vez consecutiva** — y esa es precisamente la lección más valiosa del proyecto hasta ahora.

---

## 1. El umbral: la decisión escondida después del modelo

El modelo no decide "NORMAL" o "PNEUMONIA". Emite una **probabilidad continua** y alguien corta:

```text
sigmoid(modelo) = 0.83  ──¿es neumonía?──►  depende del UMBRAL

p ≥ t → PNEUMONIA        p < t → NORMAL
```

`t = 0.5` NO tiene nada de mágico: es una convención heredada. Con class_weights (que deforman las probabilidades a propósito) y clases desbalanceadas, el óptimo casi nunca está en 0.5.

### La matriz de costos clínica

Elegir t es elegir qué error prefieres pagar:

| Umbral bajo (t=0.09) | Umbral alto (t=0.8) |
|---|---|
| casi todo → PNEUMONIA | casi todo → NORMAL |
| FN ≈ 0 (no escapa enfermo) ✔ | FP ≈ 0 |
| muchos FP: sanos a estudios extra ✘ | FN peligrosos ✘ |

En triaje pediátrico se tolera más el FP que el FN → umbral bajo. En confirmación diagnóstica, al revés. **El umbral es una decisión de negocio/clínica, no matemática.**

---

## 2. Lo que intentamos en v005 (y por qué era prometedor)

v004 tenía el mejor ROC-AUC histórico (0.972) pero f1_macro pobre con t=0.5. Hipótesis: el conocimiento está bien ordenado; solo hay que cortar donde corresponde. Protocolo correcto en teoría:

```text
1. Barrer t ∈ [0.05..0.95] sobre VALIDATION   ← jamás test
2. t* = argmax f1_macro con recall_pneu ≥ 0.97
3. Evaluar UNA vez en test con t* fijo
```

Resultado real:

```text
validation:  t* = 0.09  →  f1_macro = 0.9378   🎉
test:        t  = 0.09  →  f1_macro = 0.7372   💀  (recall_normal cayó a 0.449)
```

## 3. ¿Qué falló? Tres nombres para el mismo fantasma

### a) Sobreajuste de la capa de decisión

Con solo 519 imágenes de validation, el barrido de 91 umbrales acaba "memorizando" los caprichos de esas 519. Es el mismo overfitting de siempre, pero aplicado a UN parámetro en lugar de a millones — tan traicionero que no lo ves en curvas de loss.

### b) Shift validation ↔ test

Las probabilidades de v004 estaban polarizadísimas EN VALIDATION (val_precision=1.0000 época tras época): ahí, casi todo NORMAL quedaba bajo 0.09 y todo PNEUMONIA encima. Pero el test (otras imágenes, otro subconjunto de pacientes) no reproduce esa separación tan limpia: al cortar en 0.09, media humanidad cae en PNEUMONIA → recall_normal 0.449.

Lección profunda: **validation y test son muestras distintas**. Optimizar agresivamente contra validation asume que ambas vienen de la misma distribución exacta — supuesto frágil con datasets médicos pequeños.

### c) Probabilidades no calibradas

class_weights entrena para ORDENAR bien (y lo logra: AUC alto), pero deja las probabilidades "infladas". Un modelo calibrado diría 0.7 cuando acierta el 70%; este dice 0.99 y 0.04 sin significado probabilístico literal. Ajustar umbrales sobre probabilidades mal calibradas es construir sobre arena. Herramientas reales: calibration curve, Platt scaling / isotonic (temas futuros).

---

## 4. Lo que el registro hizo por nosotros (de nuevo)

```json
"mejor_version": "v003"     ← sigue vigente; v004 y v005 superadas
```

Dos experimentos refutados seguidos (v004 fine tuning, v005 umbral) y el proyecto NO perdió nada: cada uno costó minutos, dejó artefactos, config exacto y una lección escrita. Esto es ingeniería experimental madura:

> Un experimento negativo documentado elimina un camino para siempre. Uno positivo sin registro es in reproducible y no sirve.

---

## 5. Caminos reales para v006 (hipótesis nuevas, ahora mejor informadas)

| Camino | Idea | Riesgo |
|---|---|---|
| **Umbral robusto** | elegir t por validación cruzada k-fold dentro del pool train+val, no un único split | costo moderado |
| **Recalibrar probabilidades** | Platt/isotonic sobre validation, luego t=0.5 sobre probabilidad calibrada | requiere cuidado |
| **Rebalancear pesos** | reentrenar desde v003 con class_weights suavizados (ej. 1.4/0.8) buscando equilibrio | costo de entrenamiento |
| **Más datos (dataset v2)** | NIH ChestX-ray14: más variedad ⇒ menos shift entre splits | descarga grande |

---

## 6. Glosario rápido

| Término | Significado |
|---|---|
| **Umbral de decisión** | Corte sobre la probabilidad que convierte score continuo en clase |
| **Matriz de costos** | Cuánto cuesta cada tipo de error; define el umbral óptimo |
| **Calibración** | Correspondencia entre probabilidad dicha y frecuencia real de acierto |
| **Platt / isotonic** | Métodos para recalibrar probabilidades post-hoc |
| **Distribution shift** | Cambio de distribución entre conjuntos (aquí: validation vs test) |
| **Sobreajuste de decisión** | Memorizar los caprichos del split de ajuste vía hiperparámetros de salida |

---

## 7. Ejercicios sugeridos

1. Con la matriz de v005-test (TN=105, recall_normal=0.4487): reconstruye FP, y explica qué habría pasado clínicamente desplegando t=0.09 sin la fase de test.
2. Propón un protocolo de validación cruzada para elegir umbral usando SOLO train+validation, dejando test intacto. ¿Cuántos folds y por qué?
3. Si las probabilidades estuvieran perfectamente calibradas, ¿t=0.5 sería razonable? ¿Qué relación tendría con la prevalencia de la enfermedad?

---
*Referencias: Flach, "Machine Learning: The Art and Science" (cap. calibración y umbrales) · Platt 1999 · Niculescu-Mizil & Caruana, "Predicting Good Probabilities with Supervised Learning", ICML 2005.*
