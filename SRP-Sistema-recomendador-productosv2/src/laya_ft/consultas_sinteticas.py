"""
Consultas de compra sinteticas en espanol, con su respuesta correcta conocida, para afinar
Laya en las preguntas de backend/app/preguntas_laya.json["consulta"].

Por que hace falta: zero-shot, laya-multilingual respondio "es_regalo = si" y
"sensibilidad_precio ~ 0.9" a TODAS las consultas de prueba, y eligio publico=hombre para
"un vestido elegante para una boda". Como las consultas se arman pieza por pieza, el gold
sale gratis: si la plantilla puso "para mi mamá", el publico es mujer y es regalo.

Para que test mida generalizacion y no memoria, las plantillas de test no se usan en train.
"""
import random

PRODUCTOS = {
    "ropa": ["un vestido", "una blusa", "un polo", "una camisa", "un pantalón", "unos jeans", "una falda",
             "una casaca", "una chompa", "un short", "un traje de baño", "una polera", "un enterizo", "un saco"],
    "calzado": ["zapatos", "zapatillas", "botas", "sandalias", "unas pantuflas", "tacones", "mocasines", "botines"],
    "joyeria": ["aretes", "un collar", "una pulsera", "un anillo", "un dije", "una cadena", "unos pendientes"],
    "relojes": ["un reloj", "una correa para mi reloj", "un reloj deportivo", "un smartwatch", "una correa para fitbit"],
    "lentes": ["lentes de sol", "gafas de sol", "unos anteojos", "lentes polarizados"],
    "bolsos": ["una cartera", "un bolso", "una mochila", "una billetera", "una maleta", "un monedero"],
    "ropa_interior": ["medias", "calcetines", "ropa interior", "un pijama", "un brasier", "boxers"],
    "accesorios": ["un cinturón", "un gorro", "una gorra", "una bufanda", "guantes", "una corbata", "un sombrero"],
}
SIN_PRODUCTO = ["algo bonito", "algo para estrenar", "lo que esté de moda", "una sorpresa", "algo útil",
                "cualquier cosa linda"]
PUBLICO = {
    "mujer": ["para mujer", "para dama", "para mi esposa", "para mi novia", "para mi hermana", "para una señora"],
    "hombre": ["para hombre", "para caballero", "para mi esposo", "para mi novio", "para mi hermano", "para un señor"],
    "ninos": ["para niño", "para niña", "para mi bebé", "para mi hijo de 5 años", "para mi sobrina pequeña"],
    "unisex": [""],
}
# Frases de publico que ademas implican regalo (se compra para otra persona)
REGALO = ["para regalar", "de regalo", "para el cumpleaños de", "como obsequio", "para sorprender a"]
PRECIO = {0: [""], 1: ["que no sea tan caro", "de precio razonable", "a buen precio"],
          2: ["barato", "económico", "lo más barato posible", "que cueste poco", "con poco presupuesto"]}
DETALLES = ["", "", "negro", "de color rojo", "azul", "blanco", "de cuero", "de algodón", "de plata", "dorado",
            "para el verano", "para el invierno", "para una boda", "para el gimnasio", "para la oficina",
            "talla M", "talla grande", "elegante", "casual", "deportivo", "cómodo"]

PLANTILLAS_TRAIN = [
    "busco {prod} {det} {pub} {reg} {pre}", "quiero comprar {prod} {pub} {det} {pre}",
    "necesito {prod} {det} {reg} {pub}", "{prod} {det} {pub} {pre}", "hola, tienen {prod} {det} {pub}? {pre}",
    "me recomiendas {prod} {pub} {reg}? {pre}", "estoy buscando {prod} {det} {pre} {pub}",
    "qué {prod} me recomiendan {pub} {reg} {pre}", "ando buscando {prod} {pub} {det}, {pre}",
]
PLANTILLAS_TEST = [
    "quisiera ver {prod} {det} {pub} {reg} {pre}", "me gustaría encontrar {prod} {pub} {pre} {det}",
    "tendrán {prod} {det} {reg} {pub}? {pre}",
]


def _prob(etiquetas, label, p=0.9):
    resto = (1 - p) / (len(etiquetas) - 1)
    return {e: round(p if e == label else resto, 4) for e in etiquetas}


def generar(n, plantillas, rng: random.Random):
    casos = []
    for _ in range(n):
        cat = rng.choice(list(PRODUCTOS) + ["otros"])
        prod = rng.choice(SIN_PRODUCTO if cat == "otros" else PRODUCTOS[cat])
        pub = rng.choices(list(PUBLICO), weights=[3, 3, 2, 2])[0]
        frase_pub = rng.choice(PUBLICO[pub])
        regalo = pub != "unisex" and rng.random() < 0.4
        frase_reg = ""
        if regalo:
            opciones = REGALO if "mi " in frase_pub else [r for r in REGALO if not r.endswith((" de", " a"))]
            frase_reg = rng.choice(opciones)
            if frase_reg.endswith((" de", " a")):  # "para el cumpleaños de mi esposa"
                frase_pub, frase_reg = frase_reg + " " + frase_pub.replace("para ", "", 1), ""
        nivel = rng.choices([0, 1, 2], weights=[5, 2, 3])[0]
        plantilla = rng.choice(plantillas)
        if "{reg}" not in plantilla:  # el regalo no puede quedar en la etiqueta y no en el texto
            plantilla += " {reg}"
        texto = plantilla.format(prod=prod, det=rng.choice(DETALLES), pub=frase_pub,
                                 reg=frase_reg, pre=rng.choice(PRECIO[nivel]))
        texto = " ".join(texto.split()).replace(" ,", ",").replace(" ?", "?").strip(" ,")
        p_niveles = [0.075, 0.075, 0.075]
        p_niveles[nivel] = 0.85
        s = sum(p_niveles)
        p_niveles = [x / s for x in p_niveles]
        gold = {
            "categoria": {"probabilities": _prob(list(PRODUCTOS) + ["otros"], cat), "label": cat},
            "publico": {"probabilities": _prob(list(PUBLICO), pub), "label": pub},
            "es_regalo": {"probabilities": {"true": 0.92 if regalo else 0.08, "false": 0.08 if regalo else 0.92},
                          "label": "true" if regalo else "false", "noul": 0.92 if regalo else 0.08},
            "sensibilidad_precio": {"probabilities": {str(i): round(x, 4) for i, x in enumerate(p_niveles)},
                                    "label": nivel, "score": round(sum(i * x for i, x in enumerate(p_niveles)), 4)},
        }
        casos.append((texto, gold))
    return casos
