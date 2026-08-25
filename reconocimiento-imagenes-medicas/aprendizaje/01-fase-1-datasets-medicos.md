# Aprendizaje Fase 1 — Imágenes médicas y datasets

Material de estudio para estudiantes de ingeniería de software. Lee esto ANTES de mirar los scripts de la fase: cada script tiene sentido solo si entiendes estos conceptos.

---

## 1. ¿Qué es una radiografía de tórax (y por qué es un buen primer problema)?

Una radiografía es una imagen creada al hacer pasar rayos X a través del cuerpo: los tejidos densos (huesos) absorben más radiación y salen **blancos**; el aire absorbe casi nada y sale **negro**; los tejidos blandos quedan en grises.

En una radiografía normal ves las costillas, la columna, el corazón en el centro y **dos campos pulmonares oscuros** (llenos de aire).

La **neumonía** es una infección que llena los alvéolos pulmonares con líquido/pus. En la imagen eso aparece como **opacidades blancas o velos grises** donde debería haber negro.

```text
NORMAL                    PNEUMONIA
   ___________               ___________
  /           \             /    ████   \
 |   oscuro    |           |  oscuro ██  |   ███ = zonas de infiltrado
 |             |           |     ████   |          (líquido en alvéolos)
  \___________/             \___________/
```

Por qué es buen problema inicial:

- Solo 2 clases (binario) → métricas fáciles de interpretar.
- El patrón visual (oscuro vs opaco) es aprendible por una CNN pequeña.
- Es EL dataset clásico de la literatura (Kermany et al., Cell 2018): puedes comparar tus resultados con papers.

⚠️ Contexto ético: este dataset son **pacientes pediátricos reales** (1-5 años, Guangzhou). Trátalo con seriedad académica: nunca "para diagnosticar", siempre "para aprender".

---

## 2. Formatos: DICOM vs JPEG/PNG

| Formato | Uso real en hospitales | Este dataset |
|---|---|---|
| **DICOM** (.dcm) | Estándar médico: imagen + metadatos (paciente, intensidad real en Hounsfield Units, ángulo del equipo) | ❌ No usa |
| **JPEG/PNG** | Consumo general | ✅ Ya convertido |

El dataset original nació en DICOM y fue convertido a JPEG. Consecuencias prácticas:

1. Perdimos metadatos clínicos → no podemos agrupar por paciente real (solo por nombre de archivo).
2. La escala de grises quedó comprimida a 8 bits → suficiente para clasificar, insuficiente para medir densidades.
3. Algunas imágenes vienen en modo `L` (grises) y otras `RGB` (3 canales) → **nuestro preprocesamiento deberá uniformarlas** (convertir todo a RGB o a L).

Lección de ingeniería: **siempre inspecciona `modos_color` en la exploración antes de asumir**. Nuestro resumen detectó 5 573 imágenes `L` y 283 `RGB`.

---

## 3. Anatomía de este dataset (y sus problemas conocidos)

```text
5 856 imágenes totales
├── train       5 216   (1341 NORMAL / 3875 PNEUMONIA)  ← desbalance 2.89 : 1
├── validation     16   (8 / 8)                          ← ¡INÚTIL! demasiado pequeño
└── test          624   (234 / 390)
```

Tres defectos clásicos que nuestra exploración confirmó con datos:

### a) Desbalance de clases (2.89 : 1)

Si un modelo "predice siempre PNEUMONIA" acierta el 74% en train sin aprender nada. Defensas:

- **class_weights**: penalizar más los errores en la clase minoritaria.
  Fórmula: `peso_c = n_total / (2 × n_c)` → NORMAL=1.937, PNEUMONIA=0.674 (calculados en `generar_manifiesto.py`).
- **Métricas correctas**: reportar Recall y F1 por clase, nunca solo accuracy.

### b) Validation oficial de 16 imágenes

Una época de validación con 16 imágenes produce métricas de ruido puro. Solución aplicada: **regeneramos nuestra propia validación estratificada del 10%** (134/385 = 519 imágenes) desde el pool de train.

### c) Duplicados internos

Encontramos 32 imágenes con MD5 idéntico (26 en train, 6 en test). Son típicamente **dos radiografías del mismo paciente** guardadas dos veces, o errores del pipeline original. Los eliminamos del pool de train para no inflar el aprendizaje.

---

## 4. Data leakage (fuga de datos) — el error #1 en ML médico

**Definición**: información que NO estará disponible en producción se filtra al entrenamiento, produciendo métricas irreales.

Caso típico en radiografías: el mismo paciente aparece en train y en test. El modelo memoriza *al paciente* (su anatomía, el equipo con que se tomó) en vez de aprender *la enfermedad*. En hospitales reales esto produce sistemas que fallan estrepitosamente.

Nuestra verificación (`explorar_dataset.py`, función `duplicados()`):

```text
md5 distintos compartidos entre train y test: 0  ← test limpio ✔
```

Regla de oro para el resto del proyecto:

> El test se toca UNA vez por versión, al final. Nunca se entrena con él, nunca se ajustan hiperparámetros mirándolo.

---

## 5. Por qué un manifiesto CSV fijo de splits

Problema que evitamos: si cada script hace su propio `train_test_split`, cada versión evalúa sobre un test distinto → **comparaciones inválidas**.

Solución: `data/manifiestos/v1_splits.csv` congela la asignación exacta:

```csv
ruta_relativa,clase,split_oficial,split_final
train/NORMAL/IM-0115-0001.jpeg,NORMAL,train,validation
...
test/PNEUMONIA/person1_bacteria_1.jpeg,PNEUMONIA,test,test
```

Propiedades:

- Determinista: semilla 42, estratificado por clase.
- Commiteado a git → cualquier persona/máquina entrena y evalúa EXACTAMENTE lo mismo.
- El dashboard Astro puede auditar cuántas imágenes vio cada versión.

---

## 6. Resoluciones variables → resize obligatorio

La exploración midió resoluciones entre **384×127 y 2916×2713 px** (media 1328×971). Las CNN exigen entrada fija, así que en Fase 2 redimensionaremos todo a **224×224**:

- 224² es el tamaño nativo de ResNet50/MobileNetV2 (ImageNet).
- Bajar de ~1328px a 224px pierde detalle fino, pero conserva los patrones globales de opacidad que definen la neumonía.
- 224² mantiene el costo de cómputo razonable en CPU (tu laptop, sin GPU nativa).

---

## 7. Glosario rápido

| Término | Significado |
|---|---|
| **AP (antero-posterior)** | Proyección de la radiografía: rayos entran por delante. La mayoría de este dataset es AP pediátrico. |
| **Infiltrado** | Zona del pulmón que debería verse negra y aparece gris/blanca por líquido. |
| **Estratificado** | Split que preserva la proporción de clases en cada subconjunto. |
| **MD5** | Hash del contenido del archivo: igual MD5 ⇒ bytes idénticos ⇒ duplicado seguro. |
| **Class weight** | Factor que multiplica la loss de cada clase para compensar su escasez. |
| **ROC-AUC** | Capacidad del modelo de ordenar enfermo > sano independientemente del umbral. |
| **Recall (sensibilidad)** | De los realmente enfermos, ¿a cuántos detectamos? La métrica crítica aquí: un FN = niño enfermo enviado a casa. |

---

## 8. Ejercicios sugeridos (para fijar conceptos)

1. Abre `data/reportes/v1_exploracion/distribucion_clases.png`: calcula mentalmente qué accuracy tendría el predictor trivial "siempre PNEUMONIA" en cada split.
2. ¿Por qué eliminamos duplicados de train pero dejamos el test oficial intacto? (Pista: comparabilidad con la literatura vs pureza del entrenamiento.)
3. Si mañana añadimos el dataset NIH ChestX-ray14 como v2, ¿qué riesgo nuevo de data leakage aparece? (Pista: mismos hospitales, mismos pacientes.)

---
*Fuentes: Kermany et al., "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning", Cell 172(5), 2018 · Documentación scikit-learn sobre class_weight.*
