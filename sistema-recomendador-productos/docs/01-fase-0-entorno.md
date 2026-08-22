# Fase 0 — Entorno de desarrollo

## Objetivo

Preparar un entorno Python aislado y reproducible con todas las librerías del proyecto: ML clásico (Scikit-learn), Deep Learning (TensorFlow) y NLP (spaCy).

## Decisiones técnicas tomadas

### 1. Problema de versión de Python

La laptop tenía **Python 3.14.6**, pero TensorFlow aún **no publica binarios para 3.14** (`pip install tensorflow` falla con "No matching distribution found").

| Opción evaluada | Resultado |
|-----------------|-----------|
| Usar PyTorch en Python 3.14 | Funcionaba, pero se quería TensorFlow |
| **Instalar Python 3.12 y recrear el venv** | ✅ Elegida: TensorFlow sí soporta 3.12 |

Se instaló **Python 3.12.10** vía `winget` (convive sin conflicto con la versión 3.14 existente, gestionadas por el launcher `py`).

```powershell
winget install Python.Python.3.12
py -3.12 -m venv venv
```

### 2. Entorno virtual

El `venv` aísla las dependencias del proyecto de las globales del sistema. Cualquiera puede reproducirlas con:

```powershell
venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Librerías instaladas y verificadas

| Librería | Versión | Rol en el proyecto |
|----------|---------|--------------------|
| Python | 3.12.10 | Lenguaje base |
| pandas | 3.0.5 | Manipulación de tablas/datasets |
| numpy | 2.5.2 | Cálculo numérico, matrices |
| scikit-learn | 1.9.0 | TF-IDF, TruncatedSVD (colaborativo), métricas |
| tensorflow | 2.21.0 | Red neuronal con embeddings (Keras 3.15.1) |
| spacy | 3.8.15 | Tokenización, stopwords y lematización en español |
| es_core_news_sm | 3.8.0 | Modelo estadístico de español para spaCy |
| matplotlib | 3.11.1 | Gráficas de análisis |
| jupyter | 1.1.1 | Notebooks interactivos |
| huggingface_hub | 1.28.0 | Descarga del dataset desde HF Hub |
| joblib / scipy | — | Serialización de modelos / matrices dispersas |

Verificación ejecutada:

```
pandas 3.0.5 | numpy 2.5.2 | scikit-learn 1.9.0
tensorflow 2.21.0 | spacy 3.8.15 | matplotlib 3.11.1 → TODO OK
```

## Resultados de la fase

- [x] Python 3.12.10 instalado junto a 3.14 (sin romper nada)
- [x] `venv/` creado con la versión correcta
- [x] Todas las librerías instaladas e importables
- [x] `requirements.txt` generado con `pip freeze`
- [x] Estructura de carpetas creada: `data/`, `docs/`, `models/`, `notebooks/`, `src/`
