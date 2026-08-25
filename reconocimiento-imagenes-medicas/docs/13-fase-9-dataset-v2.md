# Fase 9 — Dataset v2: NIH ChestX-ray14

Fecha: 2026-08-25 · Estado: ✅ Completada — dataset v2 real, explorado y con manifiesto fijo

---

## 1. Qué se incorporó

**NIH ChestX-ray14**: 112 120 radiografías PA de **30 805 pacientes adultos** del NIH Clinical Center (Wang et al., CVPR 2017) — el estándar académico de rayos X de tórax. Espejo usado: `arudaev/chest-xray-14-320` (pre-redimensionado a 320², 7.4 GB, splits oficiales incluidos).

| Script | Rol |
|---|---|
| `src/preparar_datos_v2.py` | Descarga parquet → imágenes + `etiquetas_crudas.csv` (multi-etiqueta cruda) |
| `src/explorar_dataset_v2.py` | Frecuencia de hallazgos, mapeo binario, pacientes, integridad muestral (3 000 img) |
| `src/generar_manifiesto_v2.py` | Manifiesto fijo con las decisiones de la sección 3 |

```text
data/versiones/v2_nih_chest_xray14/
├── images/{train,validation,test}/   112 120 PNG
└── etiquetas_crudas.csv              split · filename · labels NIH
```

## 2. Hallazgos de la exploración

1. **Neumonía es rara**: 1 431 positivos en todo el dataset (desbalance original 58.6:1 en train).
2. **Fuga real detectada**: la validation oficial compartía **4 571 pacientes** con train. El test sí estaba limpio (0 pacientes compartidos).
3. Integridad: 0 dañadas en muestra de 3 000; resolución uniforme 320×320 RGB.
4. Multi-etiqueta: 50 328 imágenes tienen hallazgos SIN neumonía (Infiltration, Atelectasis...) → fuera de la tarea binaria.

## 3. Decisiones de diseño (registradas en el manifiesto)

| # | Decisión | Justificación |
|---|---|---|
| A | Test oficial intacto | Único split limpio de fábrica |
| B | Validation regenerada **por paciente** (8% del pool) desde train+val oficial | Elimina la fuga detectada; validación honesta |
| C | NORMAL submuestreado 1:3 en train (−43 846 filas, marcadas `incluir=false`) | Épocas entrenables en CPU (~3 000 imgs ≈ escala v1); filas excluidas quedan auditables |
| D | Mapeo estricto: `No Finding`→NORMAL · contiene `Pneumonia`→PNEUMONIA · resto EXCLUIDA | Evita etiquetar "Infiltración" como neumonía (ruido clínico) |

## 4. Splits finales (congelados)

```text
                 NORMAL   PNEUMONIA   total
train              2262         754    3016   (submuestreado 1:3)
validation         4392         122    4514   (patient-wise, completa)
test               9861         555   10416   (oficial intacto)

fuga de pacientes entre CUALQUIER par de splits: 0 ✔
class_weights: NORMAL=0.667 · PNEUMONIA=2.000
```

## 5. Advertencia de dominio (crítica)

v1 = pediátricos (Guangzhou, 1-5 años) · v2 = adultos (NIH, EE.UU.). Las métricas NO son comparables directamente:

- ❌ No decir "el modelo mejoró porque f1(v2) > f1(v1)".
- ✔ Usos válidos: (a) modelo entrenado y evaluado SOLO en v2; (b) entrenar en uno y evaluar en otro para MEDIR el domain shift; (c) combinados con test por-dominio separado.

## 6. Repos descartados (documentados)

`mmenendezg/raw_pneumonia_x_ray` (mismo Kermany re-splitteado) y `MadElf1337/Pneumonia_Images` (muestra de 416): veredicto y motivo en `registro_datasets.json → descartados_evaluados`.

---

**Siguiente paso → primera versión entrenada en v2 (v007), o dashboard Astro antes de seguir.**
