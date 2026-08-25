# Fase 5 — Versión v004: fine tuning de v003 (resultado mixto, documentado)

Fecha: 2026-08-23 · Estado: ✅ Completada (v004 NO supera a v003; el registro la marca `superada`) · Material de estudio: [`aprendizaje/06-fase-6-fine-tuning.md`](../aprendizaje/06-fase-6-fine-tuning.md)

---

## 1. Qué se construyó

- `src/afinar.py`: script de fine tuning que carga el `modelo.keras` de una versión anterior, descongela las últimas N capas de su base (BatchNorm siempre congeladas), recompila con lr pequeño y guarda una versión nueva con `base_anterior` declarado.

```powershell
python src/afinar.py --version-id v004 \
  --desde models/versiones/v003_20260823-2049_mobilenetv2_fe/modelo.keras \
  --capas-descongeladas 30 --epochs 10 --patience 4
```

Carpeta: `models/versiones/v004_20260823-2126_ft/`

```text
base:            mobilenetv2_1.00_224
descongeladas:   19 capas efectivas (de las últimas 30 pedidas; BN excluidas)
entrenables:     164,097 -> 1,674,817 (69%)
lr:              1e-5 · 10 épocas · 15.3 min
best época:      8 (val_auc)
```

## 2. Resultados (test congelado)

| Métrica | v003 | **v004** | Δ |
|---|---|---|---|
| Recall PNEUMONIA | 0.9872 | **0.9923** | ✔ +0.005 (FN: 5→3) |
| ROC-AUC | 0.9672 | **0.9719** | ✔ mejor ordenamiento histórico |
| Recall NORMAL | 0.6624 | 0.5983 | ✘ −0.064 (FP: 79→94) |
| F1 macro | **0.8442** | 0.8157 | ✘ −0.028 |
| Tamaño .keras | 11.6 MB | 23.7 MB* | estado del optimizador Adam incluido |

Matriz fp32: TN=140 · FP=94 · FN=**3** · TP=387

### Cuantización

int8: 2.90 MB (8.2×↓) · accuracy 0.8446 · f1_macro 0.8145 · latencia 7.2 ms

## 3. Veredicto del registro

**v003 sigue vigente** (f1_macro 0.844 > 0.816). El ciclo funcionó exactamente como fue diseñado: la hipótesis "fine tuning subiría recall_normal" quedó refutada con datos y el mecanismo de comparación lo capturó automáticamente.

Análisis:
1. Más capacidad entrenable sin cambiar el sesgo (class_weights + umbral 0.5) amplificó el sesgo hacia PNEUMONIA.
2. El ROC-AUC récord (0.972) indica que el conocimiento está; el problema es dónde se corta la decisión.

## 4. Mejora definida para v005

Ajuste de umbral sobre validation (NO tocar test para elegirlo): buscar t∈[0.3..0.8] que maximice f1_macro en validation, fijarlo en config.json y evaluar UNA vez en test. Hipótesis: f1_macro ≥ 0.87 manteniendo recall_pneumonia ≥ 0.97. Costo: minutos, sin GPU ni reentrenamiento — la mejora más barata pendiente del proyecto.

---

**Siguiente paso → v005 (ajuste de umbral) o continuar mapa (Fase 6 ResNet50 / Grad-CAM).**
