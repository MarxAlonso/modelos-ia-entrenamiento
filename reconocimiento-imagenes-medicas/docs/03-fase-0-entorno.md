# Fase 0 — Entorno de desarrollo

Fecha: 2026-08-23 · Estado: ✅ Completada

---

## 1. Software instalado

```text
Python      3.12.10   (venv propio del proyecto)
TensorFlow  2.21.0
numpy       2.5.2
pandas      3.0.5
scikit-learn 1.9.0
matplotlib  3.11.1
seaborn     0.13.2
Pillow / OpenCV / TensorBoard  (últimas compatibles)
```

Versiones clavadas en [`requirements.txt`](../requirements.txt). Son las mismas que ya funcionan en el proyecto recomendador de esta laptop → entorno probado, cero sorpresas.

## 2. Cómo activar el entorno

```powershell
cd reconocimiento-imagenes-medicas
.\venv\Scripts\Activate.ps1
python -c "import tensorflow as tf; print(tf.__version__)"
```

## 3. Hardware registrado

| Componente | Detalle | Implicación |
|---|---|---|
| CPU | Intel i5-10300H @ 2.50 GHz (4C/8T) | Entrenamiento viable con oneDNN activo |
| RAM | 15.9 GB | Batch 16-32 en 224×224 sin problema |
| GPU | NVIDIA GTX 1650 4 GB VRAM | **No usable por TF nativo en Windows** (TF ≥ 2.11) |
| Disco C | 708 GB libres | Espacio de sobra para datasets/versiones |

Salida real de la verificación:

```text
TF: 2.21.0
GPU: []
WARNING: TensorFlow GPU support is not available on native Windows
         for TensorFlow >= 2.11. Please use WSL2 or DirectML plugin.
```

## 4. Decisión de cómputo

1. **Ahora: CPU nativa con oneDNN.** Los modelos del proyecto son livianos a propósito:
   - CNN propia: minutos por época.
   - MobileNetV2 feature extraction: rápido en CPU.
   - ResNet50: más lento pero factible con early stopping.
2. **Si un fine tuning pesado lo requiere: WSL2 + CUDA.** Ya existe experiencia probada en esta laptop (scripts `src/mlops/` del proyecto recomendador: `entrenar_cadena_wsl.sh`, `entrenar_gpu.ps1`). No se instala nada de CUDA hasta que esa necesidad sea real, como establece el mapa original.
3. **La cuantización es parte de la estrategia**, no un extra: inferencia siempre en `.tflite` int8/dynamic sobre CPU.

## 5. Reproducibilidad

Semilla global acordada para TODAS las versiones:

```python
SEED = 42   # numpy, random, tensorflow (tf.random.set_seed)
```

Se declara en el `config.json` de cada versión. Cambiarla solo si se documenta en `notas`.

## 6. Checklist Fase 0

- [x] Crear `venv` propio del proyecto
- [x] Instalar dependencias (`requirements.txt`)
- [x] Verificar TensorFlow 2.21.0
- [x] Comprobar CPU/RAM/GPU/disco
- [x] Definir semillas aleatorias (SEED=42)
- [ ] (Opcional, futuro) WSL2+CUDA si fine tuning lo exige

---

**Siguiente paso → Fase 1:** descargar Dataset v1 (Kaggle Chest X-Ray Pneumonia) y exploración.
