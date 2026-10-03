"""
Rutas, constantes y reglas compartidas por todos los scripts de src/.

El backend no importa este modulo: recibe la categoria y el publico ya calculados en
models/servir/catalogo.csv.
"""
import os
import re

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(RAIZ, "data", "raw")
DATA_AZ = os.path.join(RAIZ, "data", "processed_az")
DATA_SPLITS = os.path.join(RAIZ, "data", "splits")
DATA_LAYA = os.path.join(RAIZ, "data", "laya")
MODELOS = os.path.join(RAIZ, "models")
SERVIR = os.path.join(MODELOS, "servir")

SEMILLA = 42

# Encoder base del two-tower: multilingue (50+ idiomas), 384 dimensiones. El catalogo esta
# en ingles y los clientes escriben en espanol: un encoder multilingue los pone en el
# mismo espacio sin traducir nada.
ENCODER_BASE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PREFIJO_CONSULTA = "consulta: "
PREFIJO_PRODUCTO = "producto: "
MAX_LEN_CONSULTA = 64
MAX_LEN_PRODUCTO = 96

# Protocolo de evaluacion de ranking (He et al. 2017, NCF): 1 positivo + 99 negativos.
K = 10
CANDIDATOS_EVAL = 100

# Categorias derivadas por reglas sobre el titulo: el catalogo processed_az no trae
# categoria, y Laya necesita un conjunto cerrado de opciones para su pregunta `choice`.
# El orden importa: gana la primera regla que encaje ("watch band" -> relojes, no ropa).
CATEGORIAS = {
    "relojes": r"\b(watch|watches|smartwatch|fitbit|apple watch|watch band)\b",
    "joyeria": r"\b(earrings?|necklaces?|bracelets?|rings?|pendants?|lockets?|charms?|anklets?|jewelry|brooch)\b",
    "lentes": r"\b(sunglasses|glasses|eyewear|goggles)\b",
    "calzado": r"\b(shoes?|boots?|sneakers?|sandals?|slippers?|heels|loafers?|flats|clogs|mules)\b",
    "bolsos": r"\b(bags?|backpacks?|wallets?|purses?|handbags?|totes?|clutch|crossbody|luggage)\b",
    "ropa_interior": r"\b(socks?|underwear|bras?|panties|briefs|boxers?|lingerie|thong|pajamas?|sleepwear)\b",
    "accesorios": r"\b(belts?|hats?|caps?|beanies?|scarf|scarves|gloves|mittens|ties|headband|umbrella|keychain)\b",
    "ropa": r"\b(dress|dresses|shirts?|t-shirt|tops?|blouses?|sweaters?|hoodies?|jackets?|coats?|jeans|pants|shorts|skirts?|leggings|swimsuits?|bikini|tunic|cardigan|vest|jumpsuit|romper|tank|sweatshirt|capris|joggers?|outfits?|costume|uniform)\b",
}
CATEGORIA_OTROS = "otros"

PUBLICOS = {
    "ninos": r"\b(baby|babies|toddlers?|kids?|boys?|girls?|infant|newborn|children|child)\b",
    "mujer": r"\b(women|womens|woman|ladies|lady|female|maternity)\b",
    "hombre": r"\b(men|mens|man|male|gentlemen)\b",
}
PUBLICO_UNISEX = "unisex"

_REGEX_CAT = {k: re.compile(v) for k, v in CATEGORIAS.items()}
_REGEX_PUB = {k: re.compile(v) for k, v in PUBLICOS.items()}


def categoria_de(titulo: str) -> str:
    t = str(titulo).lower().replace("'", "")
    for cat, rx in _REGEX_CAT.items():
        if rx.search(t):
            return cat
    return CATEGORIA_OTROS


def publico_de(titulo: str) -> str:
    t = str(titulo).lower().replace("'", "")
    for pub, rx in _REGEX_PUB.items():  # ninos antes que mujer/hombre ("baby girls")
        if rx.search(t):
            return pub
    return PUBLICO_UNISEX


def hr_ndcg(puntajes, k=K):
    """puntajes [n, 100] con la columna 0 = producto real. Devuelve (HR@k, NDCG@k)."""
    import numpy as np
    rango = (puntajes[:, 1:] > puntajes[:, :1]).sum(1)  # cuantos negativos le ganan
    hit = rango < k
    return float(hit.mean()), float(np.where(hit, 1.0 / np.log2(rango + 2), 0.0).mean())
