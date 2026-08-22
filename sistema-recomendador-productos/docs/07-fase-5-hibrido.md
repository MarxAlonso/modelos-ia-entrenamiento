# Fase 5 — Sistema híbrido y evaluación final

## Objetivo

Combinar los tres motores de puntaje desarrollados en las fases anteriores en un único sistema recomendador, seleccionar los pesos de fusión sin fuga de datos, medir la mejora frente a cada modelo individual y entregar la función `recomendar(usuario)` como producto final.

## 1. Arquitectura del sistema híbrido

```
                    ┌──────────────────────────────┐
 Historial compras ─►│ Fase 3: KNN ítem-ítem coseno │──┐ S_knn   (500×6557)
                    └──────────────────────────────┘  │
                    ┌──────────────────────────────┐  │
 IDs + texto ES    ─►│ Fase 4: Red NCF + contenido  │──┤ S_ncf   (500×6557)
                    └──────────────────────────────┘  │      min-max por usuario
                    ┌──────────────────────────────┐  │
 Descripciones ES  ─►│ Content-based: perfil TF-IDF │──┘ S_texto (500×6557)
                    └──────────────────────────────┘
                                   │
        score_final = w1·KNN + w2·NCF + w3·Contenido
                                   │
                    ┌──────────────────────────────┐
                    │  recomendar(usuario) → Top-N │
                    └──────────────────────────────┘
```

### Componente de contenido (nuevo en esta fase)

Perfil del usuario = media ponderada por rating de los vectores TF-IDF (normalizados L2) de sus productos comprados en train; puntaje del producto = similitud coseno perfil↔producto.

## 2. Metodología anti-fuga de datos

El test original se dividió en dos mitades disjuntas (semilla 42):

| Mitad | Uso | Usuarios con relevantes |
|-------|-----|-------------------------|
| Ajuste | Búsqueda de pesos | 463 |
| Reporte | Métricas finales (jamás usadas para decidir) | 457 |

Los modelos base ya estaban entrenados solo con train; los pesos del híbrido se eligieron exclusivamente en la mitad de ajuste.

## 3. Búsqueda de pesos (mitad de ajuste)

Rejilla sobre el simplex {0, 0.25, 0.5, 0.75, 1}³ (15 combinaciones válidas). Mejores resultados:

| Pesos (KNN/NCF/Texto) | Precision@10 ajuste |
|----------------------|---------------------|
| **(0.00, 0.25, 0.75)** ← seleccionado | **0.0069** |
| (1.00, 0.00, 0.00) KNN puro | 0.0065 |
| (0.25, 0.25, 0.50) | 0.0065 |
| (0.00, 0.25, 0.75) vs peor (NCF puro) | 0.0069 vs 0.0048 |

Lectura: el texto español aporta la mayor señal individual, la red neuronal lo complementa, y el KNN —aunque potente en su mitad— no añade valor incremental sobre esa mezcla en este dataset.

## 4. Evaluación final (mitad de reporte)

| Modelo | Precision@10 | Recall@10 |
|--------|--------------|-----------|
| Popularidad (base) | 0.0024 | — |
| Contenido TF-IDF | 0.0042 | — |
| NCF red neuronal | 0.0044 | — |
| KNN ítem-ítem | 0.0061 | — |
| **HÍBRIDO (0, 0.25, 0.75)** | **0.0070** ✅ | **0.0221** |

**Mejora del híbrido: +15 % sobre el mejor modelo individual (KNN) y ~3× sobre popularidad**, validada fuera de la muestra de decisión.

## 5. Validación cualitativa — `recomendar()` sobre usuarios reales

| Usuario | Preferencias declaradas | Top-5 del híbrido | ¿En favoritas? |
|---------|------------------------|-------------------|----------------|
| u0007 | pasabocas, frutas-y-verduras, despensa | Galletas Saltinas · Galletas digestive · Tartaletas Greco · Piña chocolateada · Plátano chips | **5/5** (pasabocas/dulces) |
| u0100 | vinos-y-licores, carne-y-pollo | Muchacho · Churrasco · Pulpa · Pal sancocho · Carne molida | **5/5** (carne-y-pollo) |
| u0250 | dulces-y-postres, vinos-y-licores, panadería | Ron Viejo de Caldas · Vino Rioja rosado · Tequila Don Julio · Cerveza La Chouffe · Vino Chardonnay | **5/5** (vinos-y-licores) |

**15/15 recomendaciones dentro de categorías preferidas declaradas.**

El desglose por componente muestra el funcionamiento interno, p. ej. para u0100:
`Muchacho en bandeja [knn=0.17 | ncf=0.95 | texto=1.00]` → el contenido español domina y la red confirma.

## 6. Resumen integral del proyecto

| Fase | Técnica | Resultado clave |
|------|---------|-----------------|
| 0 | Entorno Python 3.12 + TF/PyTorch/spaCy | Reproducible (`requirements.txt`) |
| 1 | Catálogo real supermercado + interacciones simuladas | 7449 productos · 500 usuarios · 25 854 compras |
| 2 | NLP 5 etapas (spaCy es) + TF-IDF | Matriz 7449×5000 · similitud semántica validada |
| 3 | Filtrado colaborativo (sesgos+SVD / KNN ítem-ítem) | RMSE 0.77 · P@10 0.0118 (~15× azar) |
| 3b | Amazon real: reseñas EN→ES locales + NLP + CF real | r sentimiento-rating 0.617 · comparación honesta sintético vs real |
| 4 | NCF con embeddings + muestreo negativo | P@10 0.0086 (2× popularidad), 3 iteraciones documentadas |
| 5 | **Híbrido ponderado anti-fuga** | **P@10 0.0070 > todos los individuales · 15/15 demo** |

> Nota: el P@10 del híbrido (0.0070) y del KNN (0.0061) corresponden a la mitad de reporte; las cifras de las Fases 3-4 (0.0118 / 0.0086) corresponden al test completo — por eso no deben compararse entre sí directamente, sino dentro de su propia tabla.

## Artefactos finales

| Archivo | Contenido |
|---------|-----------|
| `models/hibrido.pkl` | Pesos seleccionados, métricas finales, índices usuario/producto |
| `src/hibrido.py` | Pipeline completo: carga modelos → contenido → normalización → búsqueda de pesos → evaluación → `recomendar()` |
| `docs/*.md` | Documentación completa fase por fase |

## Resultados de la fase

- [x] Fusión ponderada de 3 motores con normalización min-max por usuario
- [x] Selección de pesos sin fuga (ajuste/reporte)
- [x] Híbrido superior a todos los modelos individuales en datos no vistos
- [x] Función `recomendar()` operativa con desglose explicable por componente
- [x] Proyecto completo documentado de extremo a extremo
