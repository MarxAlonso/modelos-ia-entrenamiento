#!/bin/bash
# Cadena completa de reentrenamiento en GPU (WSL)
set -e
export LD_LIBRARY_PATH=$(find /root/gpt-gpu/lib/python3.10/site-packages/nvidia -type d -name lib | tr '\n' ':')$LD_LIBRARY_PATH
export TF_CPP_MIN_LOG_LEVEL=2
cd "/mnt/c/developer-marx/Proyectos UTP/modelos-ia-entrenamiento/sistema-recomendador-productos"
PY=/root/gpt-gpu/bin/python
echo "=== GPU detectada ==="
$PY -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
echo "=== [1/3] Colaborativo (SVD + KNN) ==="
$PY src/colaborativo.py 2>&1 | tail -4
echo "=== [2/3] NCF en GPU (red_neuronal) ==="
$PY src/red_neuronal.py 2>&1 | grep -E "Comparacion|init aleatoria|init Word2Vec|Popularidad|KNN item|NCF red|Fase 4"
echo "=== [3/3] Hibrido de 4 motores ==="
$PY src/hibrido.py 2>&1 | grep -E "Pesos seleccionados|HIBRIDO|Popularidad|Word2Vec semantico|NCF red|KNN item|Contenido TF|Recall@10|completada"
