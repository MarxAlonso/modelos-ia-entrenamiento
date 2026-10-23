"""
Rasgos del re-ranker: una sola definicion para el entrenamiento y para el servicio.

El backend importa este archivo (src/publicar.py lo copia a models/servir/rasgos.py) para que
los rasgos que ve en produccion sean exactamente los que el modelo aprendio. Si se cambia
una formula hay que volver a publicar: el orden de RASGOS es el que espera el modelo.

Los rasgos se calculan en tres bloques:
  1. senales del modelo   : SASRec, similitud two-tower, popularidad, rating, resenas
  2. contexto del cliente  : historial (categoria, publico, tienda, precio, estrellas)
  3. posicion en el pool   : rangos y el puesto de popularidad global
"""
import numpy as np

# ----------------------------------------------------------------- senales (11 originales)
RASGOS_BASE = ["sasrec", "log_pop", "tt_media", "tt_max", "tt_ultima", "cat_frac", "cat_ultima",
               "pub_frac", "tienda", "rating", "log_resenas"]
# ----------------------------------------------------------------- contexto y posicion
RASGOS_EXTRA = ["rango_sasrec", "rango_pop", "rango_tt", "rango_rrf",
                "precio_rel", "precio_z_cat", "rating_z_cat", "share_pop_cat", "es_precio_real",
                "tienda_freq", "n_hist", "hist_rating"]
# ----------------------------------------------------------------- categorias (one-hot)
CATEGORIAS = ["accesorios", "bolsos", "calzado", "joyeria", "lentes", "otros", "relojes",
              "ropa", "ropa_interior"]
PUBLICOS = ["hombre", "mujer", "ninos", "unisex"]
RASGOS_CAT = [f"cat={c}" for c in CATEGORIAS] + [f"pub={p}" for p in PUBLICOS]

RASGOS = RASGOS_BASE + RASGOS_EXTRA + RASGOS_CAT
IDX = {n: i for i, n in enumerate(RASGOS)}


def _rank01(x):
    """Puesto normalizado dentro de la fila: 1 = el mejor. np.argsort es estable."""
    orden = np.argsort(-x, kind="stable")
    r = np.empty(len(x), dtype=np.float32)
    r[orden] = np.arange(len(x), dtype=np.float32)
    return 1.0 - r / max(len(x) - 1, 1)


def rasgos(historias, cand, s_sas, items, popularidad, catalogo, estrellas_hist=None):
    """Devuelve [n_usuarios, n_candidatos, n_rasgos] en float32.

    historias : lista de arrays con los item_idx comprados (en orden, del mas antiguo al mas nuevo)
    cand     : [n_usuarios, n_candidatos] con los item_idx del pool
    s_sas    : [n_usuarios, n_candidatos] puntaje de SASRec
    items    : [n_items, dim] embeddings de producto (L2 normalizados)
    Items del catalogo: categoria, publico, tienda, rating_promedio, numero_resenas,
    precio_final, precio_faltante.
    """
    n, m = cand.shape
    cat = catalogo["categoria"].to_numpy()
    pub = catalogo["publico"].to_numpy()
    tienda = catalogo["tienda"].fillna("").to_numpy()
    rating = catalogo["rating_promedio"].fillna(0).to_numpy(np.float32)
    n_res = catalogo["numero_resenas"].fillna(0).to_numpy(np.float32)
    precio = catalogo["precio_final"].fillna(0).to_numpy(np.float32)
    sin_precio = catalogo["precio_faltante"].fillna(1).to_numpy(np.float32)

    log_precio = np.log1p(np.maximum(precio, 0)).astype(np.float32)
    # estadisticas por categoria (mediana del precio, media del rating, cuota de popularidad)
    med_cat, rat_cat, share = {}, {}, np.zeros(len(cat), dtype=np.float32)
    for c in np.unique(cat):
        sel = cat == c
        med_cat[c] = float(np.median(log_precio[sel]))
        rat_cat[c] = float(rating[sel].mean())
        tot = float(popularidad[sel].sum()) or 1.0
        share[sel] = np.minimum(popularidad[sel] / tot, 1.0)

    F = np.zeros((n, m, len(RASGOS)), dtype=np.float32)
    cat_idx = {c: i for i, c in enumerate(CATEGORIAS)}
    pub_idx = {p: i for i, p in enumerate(PUBLICOS)}

    for i, h in enumerate(historias):
        c = cand[i]
        ec, eh = items[c], items[h]
        if len(h):
            sims = ec @ eh.T
            media = eh.mean(0)
            media = media / max(np.linalg.norm(media), 1e-8)
            F[i, :, 2] = ec @ media
            F[i, :, 3] = sims.max(1)
            F[i, :, 4] = sims[:, -1]
            F[i, :, 5] = (cat[c][:, None] == cat[h][None]).mean(1)
            F[i, :, 6] = (cat[c] == cat[h[-1]]).astype(np.float32)
            F[i, :, 7] = (pub[c][:, None] == pub[h][None]).mean(1)
            tiendas = [t for t in tienda[h] if t]
            F[i, :, 8] = np.isin(tienda[c], tiendas).astype(np.float32)
            F[i, :, IDX["precio_rel"]] = log_precio[c] - log_precio[h].mean()
            F[i, :, IDX["n_hist"]] = np.log1p(len(h))
            if estrellas_hist is not None and len(estrellas_hist[i]):
                F[i, :, IDX["hist_rating"]] = float(np.mean(estrellas_hist[i]))
        else:
            F[i, :, IDX["n_hist"]] = 0.0
        F[i, :, 0] = s_sas[i]
        F[i, :, 1] = np.log1p(popularidad[c])
        F[i, :, 9] = rating[c]
        F[i, :, 10] = np.log1p(n_res[c])
        F[i, :, IDX["rango_sasrec"]] = _rank01(s_sas[i])
        F[i, :, IDX["rango_pop"]] = _rank01(popularidad[c])
        F[i, :, IDX["rango_tt"]] = _rank01(F[i, :, 2])
        # RRF de similitud + SASRec (mismo criterio que usa el backend)
        rrf = 0.0
        for x in (F[i, :, 2], s_sas[i]):
            r = np.empty(m, dtype=np.float32)
            r[np.argsort(-x, kind="stable")] = np.arange(m)
            rrf = rrf + 1.0 / (60.0 + r + 1)
        F[i, :, IDX["rango_rrf"]] = _rank01(rrf)
        # desvío del precio respecto a la mediana de su categoria
        med = np.array([med_cat[x] for x in cat[c]], dtype=np.float32)
        F[i, :, IDX["precio_z_cat"]] = log_precio[c] - med
        rat = np.array([rat_cat[x] for x in cat[c]], dtype=np.float32)
        F[i, :, IDX["rating_z_cat"]] = rating[c] - rat
        F[i, :, IDX["share_pop_cat"]] = share[c]
        F[i, :, IDX["es_precio_real"]] = (sin_precio[c] == 0).astype(np.float32)
        # veces que el cliente compro en esa tienda
        if len(h):
            hist_tiendas = tienda[h]
            unicas, conteo = np.unique(hist_tiendas, return_counts=True)
            mapa = dict(zip(unicas.tolist(), conteo.tolist()))
            F[i, :, IDX["tienda_freq"]] = np.array([mapa.get(t, 0) / len(h) for t in tienda[c]],
                                                    dtype=np.float32)
        for c_ in np.unique(cat[c]):
            j = cat_idx.get(c_)
            if j is not None:
                F[i, cat[c] == c_, IDX[f"cat={c_}"]] = 1.0
        for p_ in np.unique(pub[c]):
            j = pub_idx.get(p_)
            if j is not None:
                F[i, pub[c] == p_, IDX[f"pub={p_}"]] = 1.0
    return F


def apply_modelo(m, F):
    """Puntaje del modelo lineal guardado en pesos.json (media/desv/w)."""
    return ((F - m["media"]) / m["desv"]) @ m["w"]
