# Aprendizaje Fase 6 — Fine Tuning (y cuando el resultado NO mejora)

Material de estudio basado en la versión REAL v004. La lección central: **un experimento que "falla" pero está bien documentado vale más que uno exitoso sin explicación**.

---

## 1. Qué es fine tuning

Feature extraction congela TODA la base y solo entrena una cabeza nueva. Fine tuning da un paso más:

```text
Feature extraction (v003)          Fine tuning (v004)
──────────────────────────         ──────────────────────────────
base 100% congelada 🔒             últimas N capas DESBLOQUEADAS 🔓
solo aprende "a decidir"           también pule "cómo mirar"
rápido y seguro                    potencial mayor, riesgo de romper
```

Analogía: el oftalmólogo entrenado (ImageNet) ya sabe ver; feature extraction le enseñó el criterio de neumonía. Fine tuning ahora le permite **ajustar su propia percepción** al dominio de radiografías pediátricas — despacio, con lr diminuto, porque años de experiencia visual no se deben reescribir de golpe.

### Las tres reglas de oro del fine tuning

1. **Learning rate ~100× menor** (1e-5 vs 1e-3). Los pesos preentrenados están en un mínimo excelente para características generales; pasos grandes los destrozan (catastrophic forgetting).
2. **Descongelar poco**: solo las capas superiores (nosotros: últimas 30 → quedaron 19 efectivas; las inferiores detectan bordes/texturas universales que no conviene tocar).
3. **BatchNorm SIEMPRE congelada**: sus medias/varianzas vienen de millones de imágenes; recalibrarlas con 4 671 radiografías corrompe las estadísticas. En nuestro grafo además `training=False` quedó fijado al guardar v003 — doble protección.

Resultado mecánico en v004:

```text
params entrenables: 164,097 → 1,674,817   (6.8% → 69% del modelo)
lr: 1e-5 · 10 épocas · 15.3 min
```

---

## 2. Lo que pasó: mejora en un eje, retroceso en otro

| Métrica | v003 | v004 | Lectura |
|---|---|---|---|
| Recall PNEUMONIA | 0.9872 | **0.9923** | ✔ mejor: FN 5→3 |
| ROC-AUC | 0.9672 | **0.9719** | ✔ mejor ordenamiento global |
| Recall NORMAL | 0.6624 | 0.5983 | ✘ peor: FP 79→94 |
| **F1 macro** | **0.8442** | 0.8157 | ✘ empeora ⇒ registro marca v004 `superada` |

Interpretación como ingeniero:

1. El modelo tenía un sesgo hacia PNEUMONIA heredado de v003 (class_weights 1.94 contra NORMAL + umbral fijo).
2. Darle más capacidad entrenable SIN cambiar ese sesgo hizo que lo amplificara: encontró aún más formas de decir PNEUMONIA.
3. Pero el ROC-AUC subió: **el ordenamiento interno enfermo> sano es el mejor de todas las versiones**. Los errores no son de conocimiento sino de UMBRAL.

### La conclusión técnica (para v005)

El siguiente paso barato no es reentrenar: es **ajustar el umbral de decisión sobre validation** (ej. elegir t que maximice f1_macro) y documentar el umbral elegido en config. ROC-AUC 0.972 promete mucho si se corta en el lugar correcto.

### La conclusión metodológica

- El registro comparó objetivamente y mantuvo a v003 como vigente. Sin ese mecanismo, habríamos "sentido" que fine tuning = siempre mejor.
- Hipótesis previa ("subirá recall_normal") quedó REFUTADA y documentada. Refutar hipótesis con datos es progreso, no derrota.

---

## 3. Detalle de tamaño: ¿por qué v004 pesa 23.7 MB y v003 11.6 MB?

Mismos 2.42M parámetros... pero `.keras` guarda también el **estado del optimizador Adam** (momentos m y v por parámetro entrenable). Con 10× más params entrenables, el estado creció ~13 MB. Para desplegar usamos TFLite (que NO guarda optimizador): int8 quedó en 2.90 MB igual que v003.

---

## 4. Glosario rápido

| Término | Significado |
|---|---|
| **Fine tuning** | Reentrenar con lr pequeño parte de una base preentrenada |
| **Catastrophic forgetting** | Destruir conocimiento preentrenado por lr demasiado alto |
| **Umbral de decisión** | Corte sobre la probabilidad (0.5 por defecto); ajustable según costo FN/FP |
| **Sensibilidad vs especificidad** | Detectar enfermos vs confirmar sanos; se intercambian moviendo el umbral |
| **Estado del optimizador** | Momentos Adam guardados junto al modelo (inflan .keras, no van a TFLite) |

---

## 5. Ejercicios sugeridos

1. Con ROC-AUC 0.9719: explica por qué existe un umbral t tal que f1_macro(v004,t) > 0.8442 aunque con t=0.5 sea 0.8157.
2. ¿Qué pasaría si descongeláramos TODA la base con lr 1e-5? ¿Y con lr 1e-3? Justifica con catastrophic forgetting.
3. Diseña v005: ¿ajustas umbral o intentas class_weights distintos? Define cómo medirías el éxito sin tocar el test antes de decidir.

---
*Referencias: Yosinski et al. 2014 (transferability) · Keras fine-tuning guide · Howard & Ruder, "Universal Language Model Fine-tuning" (reglas de lr discriminativo), ACL 2018.*
