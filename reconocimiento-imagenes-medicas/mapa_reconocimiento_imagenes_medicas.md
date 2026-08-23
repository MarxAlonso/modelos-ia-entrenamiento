# 🩺 Reconocimiento de Imágenes Médicas con CNN

## 1. Objetivo del proyecto

Desarrollar un sistema de Deep Learning capaz de **clasificar imágenes médicas** en diferentes categorías utilizando Redes Neuronales Convolucionales (CNN).

### Objetivo académico

Comparar:

1. Una CNN construida manualmente.
2. ResNet50 mediante Transfer Learning.
3. MobileNetV2 mediante Transfer Learning.

Finalmente seleccionar el mejor modelo según métricas de clasificación y añadir una técnica de interpretabilidad como **Grad-CAM**.

> **Importante:** el proyecto tiene finalidad académica y experimental. No debe utilizarse para realizar diagnósticos médicos reales.

---

# 2. Arquitectura general

```text
                    DATASET MÉDICO
                          │
                          ▼
                 Exploración de datos
                          │
                          ▼
                 Limpieza / validación
                          │
                          ▼
                 Preprocesamiento
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
              TRAIN              TEST
                 │
                 ▼
         ┌─────────────────┐
         │ CNN propia      │
         └────────┬────────┘
                  │
                  ▼
             Evaluación
                  │
         ┌────────┴────────┐
         ▼                 ▼
      ResNet50         MobileNetV2
   Transfer Learning  Transfer Learning
         │                 │
         └────────┬────────┘
                  ▼
             Comparación
                  │
                  ▼
            Mejor modelo
                  │
                  ▼
             Grad-CAM
                  │
                  ▼
        Visualización / Demo
```

---

# 3. Dataset

## Primera opción recomendada

Empezar con un problema de clasificación relativamente sencillo, por ejemplo:

```text
Radiografías de tórax
├── NORMAL
└── PNEUMONIA
```

No empezar con demasiadas clases.

## Estructura esperada

```text
data/
├── raw/
├── processed/
├── train/
│   ├── normal/
│   └── pneumonia/
├── validation/
│   ├── normal/
│   └── pneumonia/
└── test/
    ├── normal/
    └── pneumonia/
```

## Antes de entrenar

Comprobar:

- Número total de imágenes.
- Número de imágenes por clase.
- Resolución de imágenes.
- Imágenes dañadas.
- Clases desbalanceadas.
- Posibles duplicados.
- Distribución train/validation/test.

---

# 4. Entorno local

## Software principal

```text
Windows
Python
VS Code
Jupyter Notebook
Git
```

## Entorno virtual

```powershell
python -m venv venv
venv\Scripts\activate
```

## Librerías principales

```text
tensorflow
numpy
pandas
scikit-learn
matplotlib
seaborn
Pillow
opencv-python
jupyter
tensorboard
```

Opcional:

```text
grad-cam
```

o una implementación propia de Grad-CAM.

---

# 5. Comprobar hardware

Antes de entrenar:

```python
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))
```

Registrar también:

```text
CPU
RAM
GPU
VRAM
espacio disponible
```

## Objetivo

Determinar si el entrenamiento se realizará:

- Solo CPU.
- GPU local.
- GPU mediante WSL2/Linux si fuera necesario.

No instalar CUDA/cuDNN de forma automática sin comprobar primero qué GPU y versión de TensorFlow se utilizarán.

---

# 6. Fase 1 — Exploración del dataset

Crear:

```text
notebooks/01_exploracion.ipynb
```

Analizar:

- Cantidad de imágenes.
- Clases.
- Imágenes por clase.
- Tamaños.
- Histogramas.
- Ejemplos visuales.

Crear gráficos:

```text
Distribución por clase
Muestras de cada clase
Resolución de imágenes
```

### Resultado esperado

Un informe que permita conocer el dataset antes de entrenar.

---

# 7. Fase 2 — Preprocesamiento

Crear:

```text
src/preprocesamiento.py
```

Procesos:

```text
Imagen
  ↓
Validación
  ↓
Resize
  ↓
Normalización
  ↓
Data augmentation
  ↓
Tensor
```

## Resize

Para comenzar:

```text
224 × 224
```

## Normalización

Adaptar la escala de píxeles al modelo utilizado.

## Data augmentation

Utilizar transformaciones razonables:

```text
Rotación pequeña
Zoom pequeño
Traslación
Volteo horizontal cuando sea apropiado
```

No aplicar transformaciones médicamente poco realistas.

---

# 8. Fase 3 — CNN propia

Crear:

```text
src/train_cnn.py
```

Arquitectura inicial:

```text
Input
 ↓
Conv2D
 ↓
ReLU
 ↓
MaxPooling
 ↓
Conv2D
 ↓
ReLU
 ↓
MaxPooling
 ↓
Conv2D
 ↓
ReLU
 ↓
GlobalAveragePooling
 ↓
Dense
 ↓
Dropout
 ↓
Output
```

Ejemplo conceptual:

```text
224×224×3
     ↓
Conv2D 32
     ↓
MaxPooling
     ↓
Conv2D 64
     ↓
MaxPooling
     ↓
Conv2D 128
     ↓
GlobalAveragePooling
     ↓
Dense
     ↓
Output
```

## Entrenamiento

Registrar:

```text
loss
accuracy
precision
recall
F1
```

Guardar:

```text
models/cnn_basica.keras
```

---

# 9. Fase 4 — ResNet50

Crear:

```text
src/train_resnet.py
```

## Primera etapa: Feature Extraction

```text
ResNet50 preentrenada
        ↓
Capas congeladas
        ↓
GlobalAveragePooling
        ↓
Dense
        ↓
Output
```

Objetivo:

- Aprovechar características aprendidas previamente.
- Entrenar solamente el clasificador final.

Guardar:

```text
models/resnet50_feature_extraction.keras
```

---

# 10. Fase 5 — Fine Tuning de ResNet50

Después del primer entrenamiento:

```text
ResNet50
   ↓
Descongelar algunas capas superiores
   ↓
Learning rate pequeño
   ↓
Reentrenamiento
```

Guardar:

```text
models/resnet50_finetuned.keras
```

Comparar:

```text
ResNet Feature Extraction
vs
ResNet Fine Tuning
```

---

# 11. Fase 6 — MobileNetV2

Crear:

```text
src/train_mobilenet.py
```

Arquitectura:

```text
MobileNetV2 preentrenada
          ↓
GlobalAveragePooling
          ↓
Dense
          ↓
Output
```

Repetir:

1. Feature Extraction.
2. Fine Tuning.
3. Evaluación.

Guardar:

```text
models/mobilenetv2.keras
```

---

# 12. Fase 7 — Comparación de modelos

Crear:

```text
notebooks/06_evaluacion.ipynb
```

Tabla final:

| Modelo | Accuracy | Precision | Recall | F1 | Tiempo |
|---|---:|---:|---:|---:|---:|
| CNN propia | | | | | |
| ResNet50 | | | | | |
| ResNet50 Fine-Tuning | | | | | |
| MobileNetV2 | | | | | |
| MobileNetV2 Fine-Tuning | | | | | |

No usar solamente Accuracy, especialmente si las clases están desbalanceadas.

---

# 13. Matriz de confusión

Para el mejor modelo generar:

```text
                 Predicción
              Normal  Pneumonia
Real Normal      TN       FP
Real Pneumonia  FN       TP
```

Mostrar:

- True Positive.
- True Negative.
- False Positive.
- False Negative.

Esto permitirá explicar dónde se equivoca el modelo.

---

# 14. Fase 8 — Grad-CAM

Esta es una de las mejoras más interesantes.

Objetivo:

Mostrar aproximadamente **qué región de la imagen influyó en la predicción**.

Pipeline:

```text
Imagen
  ↓
Modelo CNN
  ↓
Predicción
  ↓
Grad-CAM
  ↓
Mapa de activación
  ↓
Superponer sobre imagen
```

Resultado visual:

```text
Radiografía original
        +
Mapa de calor
        ↓
Imagen explicable
```

Esto permite analizar el comportamiento del modelo desde el punto de vista de interpretabilidad.

---

# 15. Fase 9 — Inferencia local

Crear:

```text
src/predict.py
```

Entrada:

```text
imagen.jpg
```

Salida:

```text
Clase predicha: PNEUMONIA
Probabilidad: XX.XX%
```

Además:

```text
imagen original
+
Grad-CAM
```

---

# 16. Fase 10 — Demo

Se puede crear una pequeña interfaz local.

## Opción sencilla

```text
Streamlit
```

Flujo:

```text
Usuario carga imagen
       ↓
Modelo
       ↓
Predicción
       ↓
Probabilidades
       ↓
Grad-CAM
```

Pantalla:

```text
┌─────────────────────────────┐
│ Reconocimiento de Imágenes  │
│ Médicas                     │
├─────────────────────────────┤
│ [ Subir imagen ]            │
│                             │
│ Predicción: NORMAL          │
│ Probabilidad: 94.2%         │
│                             │
│ [Imagen + Grad-CAM]         │
└─────────────────────────────┘
```

---

# 17. Estructura del proyecto

```text
reconocimiento-imagenes-medicas/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── models/
│   ├── cnn_basica.keras
│   ├── resnet50_feature_extraction.keras
│   ├── resnet50_finetuned.keras
│   └── mobilenetv2.keras
│
├── notebooks/
│   ├── 01_exploracion.ipynb
│   ├── 02_preprocesamiento.ipynb
│   ├── 03_cnn.ipynb
│   ├── 04_resnet.ipynb
│   ├── 05_mobilenet.ipynb
│   └── 06_evaluacion.ipynb
│
├── src/
│   ├── preparar_datos.py
│   ├── preprocesamiento.py
│   ├── train_cnn.py
│   ├── train_resnet.py
│   ├── train_mobilenet.py
│   ├── evaluate.py
│   ├── gradcam.py
│   └── predict.py
│
├── logs/
│   └── tensorboard/
│
├── app/
│   └── app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 18. Orden exacto de desarrollo

## Etapa A — Preparación

- [ ] Instalar Python.
- [ ] Crear entorno virtual.
- [ ] Instalar librerías.
- [ ] Configurar VS Code/Jupyter.
- [ ] Comprobar TensorFlow.
- [ ] Comprobar CPU/RAM/GPU.

## Etapa B — Datos

- [ ] Conseguir dataset.
- [ ] Descargar dataset.
- [ ] Organizar clases.
- [ ] Validar imágenes.
- [ ] Separar train/validation/test.
- [ ] Analizar balance de clases.

## Etapa C — CNN

- [ ] Crear CNN básica.
- [ ] Entrenar.
- [ ] Guardar modelo.
- [ ] Graficar loss/accuracy.
- [ ] Evaluar.

## Etapa D — ResNet50

- [ ] Cargar modelo preentrenado.
- [ ] Congelar capas.
- [ ] Entrenar clasificador.
- [ ] Evaluar.
- [ ] Realizar fine tuning.
- [ ] Volver a evaluar.

## Etapa E — MobileNetV2

- [ ] Cargar modelo.
- [ ] Feature extraction.
- [ ] Fine tuning.
- [ ] Evaluar.

## Etapa F — Comparación

- [ ] Accuracy.
- [ ] Precision.
- [ ] Recall.
- [ ] F1.
- [ ] Matriz de confusión.
- [ ] Tiempo de entrenamiento.
- [ ] Tamaño del modelo.

## Etapa G — Interpretabilidad

- [ ] Implementar Grad-CAM.
- [ ] Generar mapas de activación.
- [ ] Analizar ejemplos correctos.
- [ ] Analizar errores.

## Etapa H — Demo

- [ ] Crear predictor.
- [ ] Crear interfaz Streamlit.
- [ ] Cargar imagen.
- [ ] Mostrar predicción.
- [ ] Mostrar probabilidad.
- [ ] Mostrar Grad-CAM.

---

# 19. Métricas finales

El informe debería incluir como mínimo:

```text
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
ROC-AUC (si aplica)
Tiempo de entrenamiento
Tamaño del modelo
```

Si existe desequilibrio entre clases, dar especial importancia a:

```text
Recall
Precision
F1
ROC-AUC
```

---

# 20. Tecnologías finales

## Lenguaje

```text
Python
```

## Deep Learning

```text
TensorFlow
Keras
```

## Modelos

```text
CNN
ResNet50
MobileNetV2
Transfer Learning
Fine Tuning
```

## Procesamiento

```text
NumPy
Pandas
Pillow
OpenCV
```

## Machine Learning / evaluación

```text
Scikit-learn
```

## Visualización

```text
Matplotlib
Seaborn
TensorBoard
```

## Interpretabilidad

```text
Grad-CAM
```

## Demo

```text
Streamlit
```

---

# 21. Resultado final esperado

El proyecto debería demostrar:

```text
1. Entendimiento de CNN
          ↓
2. Entrenamiento de modelo propio
          ↓
3. Uso de Transfer Learning
          ↓
4. Comparación ResNet vs MobileNet
          ↓
5. Fine Tuning
          ↓
6. Evaluación con métricas
          ↓
7. Interpretabilidad con Grad-CAM
          ↓
8. Demo local
```

## ⭐ Nivel recomendado

Para un proyecto académico, la combinación ideal sería:

```text
CNN propia
+
ResNet50
+
MobileNetV2
+
Transfer Learning
+
Fine Tuning
+
Grad-CAM
+
Comparación de métricas
+
Demo con Streamlit
```

Eso es suficiente para construir un proyecto sólido sin intentar entrenar modelos gigantes desde cero.
