# Aprendizaje Fase 4 — El ciclo de mejora: learning rate y early stopping bien hechos

Material de estudio basado en la comparación REAL v001 vs v002 de este proyecto. Aquí se ve cómo una decisión de entrenamiento vale tanto como la arquitectura.

---

## 1. El experimento controlado

Cambiamos UNA sola cosa a la vez (metodología científica básica):

```text
                    v001                          v002
arquitectura        idéntica                      idéntica
dataset/splits      idénticos                     idénticos
class_weights       idénticos                     idénticos
early stopping      val_recall, patience 5   →    val_auc, patience 8
epochs máx          25                       →    35
LR schedule         fijo 1e-3                →    ReduceLROnPlateau(÷3)
─────────────────────────────────────────────────────────────
RESULTADO test      f1_macro 0.519           →    0.801  (+0.28!)
                    ROC-AUC  0.579           →    0.891
```

Mismo modelo, mismos datos: **+54% de f1_macro solo por calibrar mejor el entrenamiento**. En ML, el "tuning" no es cosmética; es ingeniería.

---

## 2. Learning Rate: el acelerador del aprendizaje

El learning rate (lr) es el tamaño del paso con que el gradiente ajusta los pesos:

```text
peso_nuevo = peso - lr × gradiente

lr muy grande → pasos gigantes → saltas sobre el mínimo, oscilas (v001-v002 época 1-5)
lr muy pequeño → pasos diminutos → aprendes, pero tardarías cientos de épocas
```

### ReduceLROnPlateau: freno progresivo

```python
ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.33, patience=2)
# si val_auc no mejora en 2 épocas → lr *= 0.33
```

La estrategia clásica "empieza rápido, afina despacio". Lo que pasó en v002 (log real):

| Época | Evento | Efecto visible |
|---|---|---|
| 1-5 | lr=1e-3, oscilando | val_auc estancado ~0.76 |
| 5 | **lr ÷3 → 3.3e-4** | época 6: salto a 0.861 |
| 7-14 | mejora constante | val_auc 0.93→0.95 |
| 15 | **lr ÷3 → 1.09e-4** | época 17: val_accuracy 0.846 |
| 22 | **lr ÷3 → 3.6e-5** | época 23: val_loss baja a 0.41 |
| 32 | **lr ÷3 → 1.2e-5** | pulido final, época 35: mejor de todas |

Cada reducción desbloqueó una meseta (*plateau*): con pasos grandes la red rebotaba alrededor del mínimo; al reducirlos, pudo entrar en él.

---

## 3. Por qué `val_auc` sí y `val_recall` no

Con class_weights y 519 imágenes de validation:

```text
val_recall por época (v002): 0.75, 0.55, 0.55, ... 0.45!, 0.40!, ... 0.71
val_auc   por época (v002):  0.72, 0.73, 0.77, ... sube monótona hasta 0.959
```

- **Recall@umbral 0.5** depende de dónde caen las probabilidades respecto a UN punto exacto. Un lote raro de augmentación puede empujar decenas de imágenes al otro lado del corte → métrica montaña rusa.
- **AUC** integra sobre TODOS los umbrales posibles: mide el ordenamiento global enfermo> sano. Mucho más estable época a época.

Regla práctica: para early stopping usa métricas suaves/continuas (loss, auc); deja las métricas de umbral (recall/precision/f1) para la evaluación final.

---

## 4. Leyendo el log como ingeniero: señales de progreso y alerta

Progreso sano (épocas 6-18): train y validation suben juntos, gap moderado.

Alerta de overfitting (visible desde ~época 9): train accuracy sigue subiendo (0.82→0.88) mientras val_loss oscila fuerte (0.34↔0.88). La red empieza a memorizar particularidades del train (augmentation incluida). Defensas ya activas que evitó que explotara: dropout 0.5 + augmentación + early stopping con restore_best_weights.

Detalle curioso: la época 35 fue LA MEJOR (val_auc 0.9589) — el modelo aún tenía margen. Para v003+: epochs 50 o cosine decay.

---

## 5. La comparación entre versiones, formalizada

Gracias al registro (`registro_versiones.json`), la comparación es automática y auditada:

```json
"mejor_version": "v002",
"versiones": [
  { "id": "v001", ..., "f1_macro": 0.5191, "estado": "superada" },
  { "id": "v002", ..., "f1_macro": 0.8013, "estado": "vigente" }
]
```

Preguntas que el registro responde sin abrir modelos:

1. ¿Qué versión es la mejor ahora? → campo `mejor_version`.
2. ¿Por qué? → compara `config.json` de ambas (diff de 3 líneas).
3. ¿Cuánto cuesta? → `tiempo_entrenamiento_min` (13.7 vs 83.7 min).
4. ¿Cuál despliego? → la int8 vigente (0.12 MB, 11 ms).

---

## 6. Glosario rápido

| Término | Significado |
|---|---|
| **Learning rate (lr)** | Tamaño del paso de ajuste de pesos en cada gradiente |
| **Plateau** | Meseta: épocas sin mejora porque el paso es inadecuado |
| **ReduceLROnPlateau** | Callback: reduce lr automáticamente ante estancamiento |
| **Cosine decay** | Alternativa: lr decrece siguiendo una curva coseno predefinida |
| **Overfitting** | Memorizar train perdiendo generalización (train↑, val_loss↑) |
| **Gap train-val** | Diferencia entre métricas de train y validation; indicador de overfitting |
| **Experimento controlado** | Cambiar una variable por vez para atribuir causas |

---

## 7. Ejercicios sugeridos

1. Con el log de v002: ¿qué habría pasado si patience=2 con monitor val_auc? (Pista: época 3→4 bajó levemente.)
2. Calcula el costo total del ciclo v001+v002 (13.7+83.7 min). ¿Valió la pena vs haber empezado directo con MobileNetV2? Argumenta con la filosofía "baseline primero".
3. Diseña v003 en papel: manteniendo arquitectura, propón 2 cambios y su hipótesis medible.

---
*Referencias: Keras callbacks (ReduceLROnPlateau, EarlyStopping) · Smith, "Cyclical Learning Rates for Training Neural Networks", arXiv:1506.01186.*
