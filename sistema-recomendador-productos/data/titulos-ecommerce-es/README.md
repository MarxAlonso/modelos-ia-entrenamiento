---
language:
  - es
license: apache-2.0
size_categories:
  - 100K<n<1M
task_categories:
  - text-classification
  - sentence-similarity
tags:
  - e-commerce
  - product-titles
  - spanish
  - mercadolibre
dataset_info:
  features:
  - name: title
    dtype: string
  - name: category
    dtype: string
  splits:
  - name: train
    num_bytes: 5861175
    num_examples: 85000
  - name: test
    num_bytes: 1032905
    num_examples: 15000
  download_size: 4726554
  dataset_size: 6894080
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---

# Títulos de Productos E-commerce en Español (100K)

Subconjunto **procesado y balanceado** de títulos de productos reales en español, listo
para clasificación de categorías, búsqueda semántica o entrenamiento de embeddings.

## Contenido
- **100.000 títulos** en español con su categoría (train 85.000 · test 15.000)
- 682 categorías de productos

## Procesamiento aplicado
Derivado del [MeLi Data Challenge 2019](https://www.kaggle.com/datasets/abugim/meli-data-challenge-2019).
Transformaciones propias: filtrado por idioma español y calidad de etiqueta (`reliable`),
eliminación de duplicados y títulos poco informativos, y muestreo balanceado con tope por
categoría para evitar el dominio de categorías mayoritarias.

## Uso
```python
from datasets import load_dataset
ds = load_dataset("Mateo-Rua/titulos-productos-ecommerce-es")
```

## Fuente
Datos originales: MeLi Data Challenge 2019 (Mercado Libre). Subconjunto transformado con
fines educativos y de portafolio.

**Autor:** [Mateo Rúa](https://huggingface.co/Mateo-Rua)