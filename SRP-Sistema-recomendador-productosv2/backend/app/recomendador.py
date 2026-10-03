"""
Carga models/servir/ y responde recomendaciones con ONNX Runtime (sin PyTorch).

Flujo:
  1. recuperacion: encoder Transformer (two-tower) -> coseno contra los 71k productos
  2. re-ranking:   SASRec (Transformer secuencial con sesgo de popularidad) para usuarios con
                   historial. Decisiones tomadas con las metricas de test (models/*/metricas.json):
                   - usuario: el usuario es el promedio de embeddings de lo que compro (item-to-item,
                     HR@10 0.255 vs 0.186 codificando el historial como texto) fusionado por RRF con
                     el SASRec. En recall@10 sobre los 71k productos (la metrica justa: la de 1+99
                     negativos al azar sobrevalora la popularidad) la fusion es lo mejor: 0.0445 vs
                     0.0411 popularidad, 0.0401 SASRec, 0.0173 item-to-item solo
                   - consulta: manda la similitud de texto (para eso se afino el two-tower) y el
                     SASRec solo desempata con peso 0.3
  3. ajustes:      filtros y preferencias que decidio Laya (categoria, publico, regalo, precio)
"""
import hashlib
import json
import os

import numpy as np
import onnxruntime as ort
import pandas as pd
from tokenizers import Tokenizer

# Deben coincidir con src/comun.py y src/sasrec.py
PREFIJO_CONSULTA = "consulta: "
MAX_LEN_CONSULTA = 64
LARGO_SASREC = 10
HIST_MAX = 5
POOL = 300
PESO_SASREC_EN_CONSULTA = 0.3


def rrf(puntajes, pesos, k=60):
    """Reciprocal Rank Fusion ponderada: suma peso / (k + rango) de cada senal."""
    total = 0
    for s, w in zip(puntajes, pesos):
        rango = (-s).argsort().argsort()
        total = total + w / (k + rango + 1)
    return total


class Recomendador:
    def __init__(self, carpeta: str):
        leer = lambda a: json.load(open(os.path.join(carpeta, a), encoding="utf-8"))
        self.manifiesto = leer("manifiesto.json")
        self.version = self.manifiesto["version"]
        self.usuarios = leer("usuarios.json")
        self.catalogo = pd.read_csv(os.path.join(carpeta, "catalogo.csv"))
        self.titulos = self.catalogo["titulo"].astype(str).tolist()
        self.idx_producto = {pid: k for k, pid in enumerate(self.catalogo["product_id"])}
        self.items = np.load(os.path.join(carpeta, "items.npy")).astype(np.float32)
        h = np.load(os.path.join(carpeta, "historial.npz"))
        self.ptr, self.hist, self.popularidad = h["ptr"], h["items"], h["popularidad"]
        # estrellas, fecha y resena de cada compra (publicaciones anteriores no los traen)
        self.ratings = h["ratings"] if "ratings" in h.files else None
        self.fechas = h["fechas"] if "fechas" in h.files else None
        ruta_res = os.path.join(carpeta, "resenas.parquet")
        self.resenas = pd.read_parquet(ruta_res)["resena"].tolist() if os.path.exists(ruta_res) else None
        # puesto de cada producto en el ranking de mas comprados (0 = el mas comprado)
        self.puesto_popular = np.empty(len(self.popularidad), dtype=np.int64)
        self.puesto_popular[np.argsort(-self.popularidad, kind="stable")] = np.arange(len(self.popularidad))

        self.tok = Tokenizer.from_file(os.path.join(carpeta, "tokenizer.json"))
        self.tok.enable_truncation(MAX_LEN_CONSULTA)
        self.tok.enable_padding(pad_id=self.tok.token_to_id("<pad>") or 1, pad_token="<pad>")
        opciones = ort.SessionOptions()
        opciones.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
        self.encoder = ort.InferenceSession(os.path.join(carpeta, "encoder.onnx"), opciones,
                                            providers=["CPUExecutionProvider"])
        self.sasrec = ort.InferenceSession(os.path.join(carpeta, "sasrec.onnx"), opciones,
                                           providers=["CPUExecutionProvider"])
        # re-ranker aprendido (src/reranker.py); publicaciones anteriores no lo traen -> RRF
        ruta_rr = os.path.join(carpeta, "reranker.json")
        self.reranker = leer("reranker.json") if os.path.exists(ruta_rr) else None
        if self.reranker:
            for c in ("media", "desv", "w"):
                self.reranker[c] = np.asarray(self.reranker[c], dtype=np.float32)
            self._cat = self.catalogo["categoria"].to_numpy()
            self._pub = self.catalogo["publico"].to_numpy()
            self._tienda = self.catalogo["tienda"].fillna("").to_numpy()
            self._rating = self.catalogo["rating_promedio"].fillna(0).to_numpy(np.float32)
            self._log_resenas = np.log1p(self.catalogo["numero_resenas"].fillna(0).to_numpy(np.float32))

    # ------------------------------------------------------------------ piezas
    def codificar(self, textos):
        enc = self.tok.encode_batch(textos)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        return self.encoder.run(None, {"input_ids": ids, "attention_mask": mask})[0]

    def historial(self, user_id):
        f = self.usuarios.get(user_id) if user_id else None
        return [] if f is None else self.hist[self.ptr[f]:self.ptr[f + 1]].tolist()

    # ------------------------------------------------- historial ponderado
    DECAY_RECENCIA = 0.85  # la compra mas reciente pesa 1, la anterior 0.85, etc.

    def _pesos(self, hist, estrellas=None, textos=None):
        """Peso de cada compra en el perfil (suma 1). Recencia x estrellas x resena:
        lo ultimo pesa mas, lo bien calificado pesa mas, lo comentado con detalle pesa mas."""
        n = len(hist)
        w = np.array([self.DECAY_RECENCIA ** (n - 1 - i) for i in range(n)], dtype=np.float64)
        if estrellas is not None:
            e = np.asarray(estrellas, dtype=np.float64)
            w = w * (0.5 + e / 5.0)  # 1★ -> x0.7, 5★ -> x1.5
        if textos is not None:
            largos = np.array([len(t or "") for t in textos], dtype=np.float64)
            w = w * np.where(largos > 80, 1.15, np.where(largos > 0, 1.05, 1.0))
        s = w.sum()
        return (w / s) if s > 0 else np.full(n, 1.0 / max(n, 1))

    def _ctx_usuario(self, user_id, m):
        """Estrellas y textos de las m primeras compras del usuario (el historial va de
        la mas antigua a la mas reciente; en la prueba se usan las m anteriores)."""
        f = self.usuarios.get(user_id) if user_id else None
        if f is None:
            return None, None
        ini = int(self.ptr[f])
        est = self.ratings[ini:ini + m].tolist() if self.ratings is not None else None
        txt = self.resenas[ini:ini + m] if self.resenas is not None else None
        return est, txt

    def perfil_vector(self, hist, estrellas=None, textos=None):
        hist = [int(i) for i in hist]
        if not hist:
            return np.zeros(self.items.shape[1], dtype=np.float32)
        w = self._pesos(hist, estrellas, textos)
        q = (self.items[np.array(hist)] * w[:, None]).sum(0)
        return (q / max(np.linalg.norm(q), 1e-8)).astype(np.float32)

    def afinidad_historial(self, hist):
        """Categoria/publico dominantes del historial para el boost y la traza."""
        if not hist:
            return None
        cats = self.catalogo["categoria"].to_numpy()[np.array(hist)]
        pubs = self.catalogo["publico"].to_numpy()[np.array(hist)]
        vc, cc = np.unique(cats, return_counts=True)
        top = vc[int(np.argmax(cc))]
        vp, cp = np.unique(pubs, return_counts=True)
        topp = vp[int(np.argmax(cp))]
        return {"categoria": str(top), "frac_categoria": round(float(cc.max() / len(hist)), 3),
                "publico": str(topp), "frac_publico": round(float(cp.max() / len(hist)), 3)}

    def puntuar_sasrec(self, hist, cand):
        s = [i + 1 for i in hist[-LARGO_SASREC:]]
        seq = np.array([[0] * (LARGO_SASREC - len(s)) + s], dtype=np.int64)
        return self.sasrec.run(None, {"secuencia": seq, "candidatos": (cand + 1)[None].astype(np.int64)})[0][0]

    def puntuar_reranker(self, hist, pool, sas):
        """Mismos rasgos y orden que src/reranker.py:rasgos. Devuelve (puntaje, P(acierto top-10)):
        la probabilidad incluye la clase "ninguno" (el producto que busca no esta en el pool)."""
        h, c = np.array(hist, dtype=np.int64), pool
        ec, eh = self.items[c], self.items[h]
        sims = ec @ eh.T
        media = eh.mean(0); media /= max(np.linalg.norm(media), 1e-8)
        tiendas = [t for t in self._tienda[h] if t]
        F = np.stack([sas, np.log1p(self.popularidad[c]), ec @ media, sims.max(1), sims[:, -1],
                      (self._cat[c][:, None] == self._cat[h][None]).mean(1), self._cat[c] == self._cat[h[-1]],
                      (self._pub[c][:, None] == self._pub[h][None]).mean(1), np.isin(self._tienda[c], tiendas),
                      self._rating[c], self._log_resenas[c]], 1).astype(np.float32)
        rr = self.reranker
        s = ((F - rr["media"]) / rr["desv"]) @ rr["w"]
        z = np.append(s, rr["b_ninguno"])
        p = np.exp(z - z.max()); p /= p.sum()
        return s, float(np.sort(p[:-1])[::-1][:10].sum())

    def candidatos(self, categoria=None, publico=None):
        mask = np.ones(len(self.catalogo), dtype=bool)
        if categoria:
            mask &= (self.catalogo["categoria"] == categoria).to_numpy()
        if publico:
            mask &= self.catalogo["publico"].isin([publico, "unisex"]).to_numpy()
        return np.flatnonzero(mask)

    def _salida(self, items, puntaje, sim=None, sas=None):
        res = []
        for k, i in enumerate(items):
            f = self.catalogo.iloc[int(i)]
            d = {"product_id": f["product_id"], "item_idx": int(i), "titulo": f["titulo"], "tienda": f["tienda"],
                 "categoria": f["categoria"], "publico": f["publico"],
                 "precio": None if f["precio_faltante"] else round(float(f["precio_final"]), 2),
                 "rating_promedio": float(f["rating_promedio"]), "numero_resenas": int(f["numero_resenas"]),
                 "puntaje": round(float(puntaje[k]), 5)}
            if sim is not None:
                d["similitud"] = round(float(sim[k]), 4)
            if sas is not None:
                d["puntaje_sasrec"] = round(float(sas[k]), 4)
            res.append(d)
        return res

    def _rankear(self, q, cand, hist, sensibilidad_precio=0.0, es_regalo=False, k=10, modo="consulta"):
        """modo consulta: manda el coseno con la consulta (SASRec desempata); modo usuario:
        RRF coseno + SASRec, y al pool de similares se suman los populares.
        Devuelve (recomendaciones, traza): la traza cuenta cuantos productos sobreviven a cada
        etapa, para mostrar el embudo en el frontend."""
        traza = {"catalogo": len(self.catalogo), "tras_filtro": int(len(cand))}
        if hist:
            cand = cand[~np.isin(cand, hist)]
        sim_todo = self.items[cand] @ q
        orden = np.argsort(-sim_todo)[:POOL]
        if modo == "usuario":
            populares = np.argsort(-self.popularidad[cand])[:POOL]
            orden = np.unique(np.concatenate([orden, populares]))
        pool, sim = cand[orden], sim_todo[orden]
        traza.update(pool_similares=int(min(POOL, len(cand))), pool=int(len(pool)))
        sas = self.puntuar_sasrec(hist, pool) if hist else None
        confianza = None
        if modo == "usuario" and self.reranker:
            puntaje, confianza = self.puntuar_reranker(hist, pool, sas)
            # el softmax admite negativos; los boosts de abajo multiplican, asi que se pasa a (0, 1]
            puntaje = np.exp(puntaje - puntaje.max())
        elif modo == "usuario":
            puntaje = rrf([sim, sas], [1.0, 1.0])
        elif sas is not None:
            puntaje = rrf([sim, sas], [1.0, PESO_SASREC_EN_CONSULTA])
        else:
            puntaje = rrf([sim], [1.0])
        cat = self.catalogo.iloc[pool]
        afinidad = self.afinidad_historial(hist) if hist and modo == "usuario" else None
        if afinidad and afinidad["frac_categoria"] >= 0.5 and afinidad["categoria"] != "otros":
            # lo que la persona suele comprar sube un 8 %; la 2.a categoria no se boostea
            puntaje = puntaje * np.where(cat["categoria"].to_numpy() == afinidad["categoria"], 1.08, 1.0)
        if afinidad and afinidad["frac_publico"] >= 0.6 and afinidad["publico"] != "unisex":
            puntaje = puntaje * np.where(cat["publico"].to_numpy() == afinidad["publico"], 1.03, 1.0)
        if sensibilidad_precio > 0:  # solo productos con precio real (el 90 % esta imputado)
            real = cat["precio_faltante"].to_numpy() == 0
            caro = np.where(real, np.log1p(cat["precio_final"].to_numpy()) / np.log1p(200), 0.5).clip(0, 1)
            puntaje = puntaje * (1 - 0.15 * sensibilidad_precio * caro)
        if es_regalo:  # para regalar conviene lo probado
            confiable = (cat["rating_promedio"].to_numpy() >= 4) & (cat["numero_resenas"].to_numpy() >= 5)
            puntaje = puntaje * (1 + 0.1 * confiable)
        top = self._sin_repetidos(pool, np.argsort(-puntaje), hist, k)
        recs = self._salida(pool[top], puntaje[top], sim[top], None if sas is None else sas[top])
        self._explicar(recs, pool[top], hist, sim[top] if modo == "consulta" else None)
        traza.update(final=len(recs), sasrec=sas is not None, reranker=confianza is not None)
        if confianza is not None:
            # P(lo proximo que compre este en el top-10), calibrada en src/reranker.py
            traza["confianza_top10"] = round(confianza, 4)
        influencia = None
        if hist:
            parecidos = [bool(r.get("porque", {}).get("parecido_a") and
                              r["porque"]["parecido_a"]["similitud"] >= 0.6) for r in recs]
            misma_cat = 0
            if afinidad:
                misma_cat = int(sum(1 for r in recs if r["categoria"] == afinidad["categoria"]))
            influencia = {"compras_usadas": len(hist), "afinidad": afinidad,
                          "recs_parecidas_historial": int(sum(parecidos)),
                          "recs_en_categoria_favorita": misma_cat}
        traza["influencia_historial"] = influencia
        return recs, traza

    def _explicar(self, recs, items, hist, sim_consulta=None):
        """Por que se recomienda cada producto: a cual de tus compras se parece, si es de lo
        mas comprado y (en una busqueda) cuanto coincide con lo que escribiste."""
        if hist:
            parecidos = self.items[items] @ self.items[hist].T  # [k, n_hist]
        for j, r in enumerate(recs):
            razones = {}
            if sim_consulta is not None:
                razones["coincide_busqueda"] = round(float(sim_consulta[j]), 3)
            if hist:
                m = int(np.argmax(parecidos[j]))
                razones["parecido_a"] = {"posicion": m + 1, "titulo": self.titulos[hist[m]],
                                         "categoria": self.catalogo["categoria"].iat[hist[m]],
                                         "similitud": round(float(parecidos[j, m]), 3)}
            puesto = int(self.puesto_popular[items[j]])
            if puesto < 500:
                razones["popular"] = {"puesto": puesto + 1, "compras": int(self.popularidad[items[j]])}
            r["porque"] = razones

    def _sin_repetidos(self, pool, orden, hist, k):
        """Amazon publica el mismo producto con varios ids (tallas, colores): se deja una sola
        aparicion por titulo y nada que el usuario ya compro."""
        clave = lambda i: self.titulos[i][:60].lower()
        vistos = {clave(i) for i in hist}
        top = []
        for j in orden:
            c = clave(int(pool[j]))
            if c not in vistos:
                vistos.add(c); top.append(j)
                if len(top) == k:
                    break
        return np.array(top, dtype=np.int64)

    # ------------------------------------------------------------------ casos de uso
    def para_consulta(self, texto, k=10, user_id=None, categoria=None, publico=None,
                      sensibilidad_precio=0.0, es_regalo=False, hist=None):
        hist = [int(i) for i in hist] if hist is not None else self.historial(user_id)
        q = self.codificar([PREFIJO_CONSULTA + texto])[0]
        if hist:  # lo que pide ahora pesa mas que lo que compro antes
            est, txt = self._ctx_usuario(user_id, len(hist)) if user_id else (None, None)
            q = 0.75 * q + 0.25 * self.perfil_vector(hist, est, txt)
            q = q / max(np.linalg.norm(q), 1e-8)
        cand = self.candidatos(categoria, publico)
        if len(cand) == 0:
            cand = np.arange(len(self.catalogo))
        recs, traza = self._rankear(q, cand, hist, sensibilidad_precio, es_regalo, k)
        return {"usuario_con_historial": bool(hist), "candidatos_filtrados": int(len(cand)),
                "traza": traza, "recomendaciones": recs}

    def para_usuario(self, user_id, k=10, categoria=None, publico=None):
        hist = self.historial(user_id)
        est, txt = self._ctx_usuario(user_id, len(hist))
        r = self.para_historial(hist, k, categoria, publico, estrellas=est, textos=txt)
        r["historial"] = self.historial_detalle(user_id)
        return {"user_id": user_id, "usuario_conocido": user_id in self.usuarios, **r}

    # ------------------------------------------------------------------ perfil del cliente
    NOMBRES_MUJER = ["María", "Lucía", "Valeria", "Camila", "Sofía", "Daniela", "Andrea", "Gabriela", "Carmen",
                     "Rosa", "Ana", "Paola", "Fernanda", "Isabel", "Natalia", "Elena", "Mariana", "Patricia"]
    NOMBRES_HOMBRE = ["José", "Luis", "Carlos", "Jorge", "Miguel", "Diego", "Andrés", "Javier", "Ricardo",
                      "Fernando", "Manuel", "Raúl", "Sergio", "Pablo", "Héctor", "Martín", "Óscar", "Víctor"]
    APELLIDOS = ["García", "Rodríguez", "Flores", "Quispe", "Torres", "Ramírez", "Mendoza", "Castillo", "Vargas",
                 "Rojas", "Chávez", "Gutiérrez", "Herrera", "Díaz", "Morales", "Salazar", "Ríos", "Paredes"]

    def nombre_ficticio(self, user_id, hist):
        """El dataset es anonimo: nombre inventado, siempre el mismo para el mismo id. Se elige
        femenino/masculino segun para quien eran la mayoria de sus compras (si se sabe)."""
        h = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        pubs = self.catalogo["publico"].to_numpy()[hist] if len(hist) else []
        mujer, hombre = int((pubs == "mujer").sum()), int((pubs == "hombre").sum())
        lista = (self.NOMBRES_MUJER if mujer > hombre else self.NOMBRES_HOMBRE if hombre > mujer
                 else (self.NOMBRES_MUJER + self.NOMBRES_HOMBRE))
        return f"{lista[h % len(lista)]} {self.APELLIDOS[(h // 97) % len(self.APELLIDOS)]}"

    def historial_detalle(self, user_id):
        f = self.usuarios.get(user_id) if user_id else None
        if f is None:
            return []
        ini, fin = int(self.ptr[f]), int(self.ptr[f + 1])
        hist = self.hist[ini:fin]
        salida = self._salida(hist, np.zeros(len(hist)))
        est = self.ratings[ini:fin].tolist() if self.ratings is not None else None
        txt = self.resenas[ini:fin] if self.resenas is not None else None
        pesos = self._pesos(hist.tolist(), est, txt) if len(hist) else []
        for j, d in enumerate(salida):
            d["peso"] = round(float(pesos[j]), 3)
            if self.ratings is not None:
                d["estrellas"] = int(self.ratings[ini + j])
            if self.fechas is not None:
                x = str(int(self.fechas[ini + j]))
                d["fecha"] = f"{x[:4]}-{x[4:6]}-{x[6:]}"
            if self.resenas is not None:
                d["resena"] = self.resenas[ini + j]
        return salida

    def perfil_usuario(self, user_id):
        hist = self.historial(user_id)
        if not hist:
            return None
        cats = self.catalogo["categoria"].to_numpy()[hist]
        valores, cuentas = np.unique(cats, return_counts=True)
        orden = np.argsort(-cuentas)
        detalle = self.historial_detalle(user_id)
        est, txt = self._ctx_usuario(user_id, len(hist))
        w = self._pesos(hist, est, txt)
        con_resena = sum(1 for d in detalle if d.get("resena"))
        return {"user_id": user_id, "nombre": self.nombre_ficticio(user_id, hist), "nombre_es_ficticio": True,
                "compras": len(hist),
                "desde": detalle[0].get("fecha"), "hasta": detalle[-1].get("fecha"),
                "estrellas_promedio": round(float(np.mean([d["estrellas"] for d in detalle])), 2)
                if detalle and "estrellas" in detalle[0] else None,
                "categorias": [{"categoria": valores[i], "compras": int(cuentas[i])} for i in orden],
                "afinidad": self.afinidad_historial(hist),
                "resenas_escritas": con_resena,
                "peso_ultima_compra": round(float(w[-1]), 3),
                "compra_mas_influyente": int(np.argmax(w)) + 1}

    def clientes_con_acierto(self, k=12, n=8, revisar=400, semilla=7):
        """Busca clientes (3+ compras, al azar) en los que la prueba acierta, y cuenta en cuantos
        de los revisados paso: asi la demo puede mostrar un acierto real sin esconder su frecuencia."""
        if getattr(self, "_aciertos", None) is not None:
            return self._aciertos
        if not hasattr(self, "_usuario_de_fila"):
            self._usuario_de_fila = {f: u for u, f in self.usuarios.items()}
        filas = np.flatnonzero(np.diff(self.ptr) >= 3)
        filas = np.random.default_rng(semilla).permutation(filas)[:revisar]
        encontrados, revisados = [], 0
        for f in filas:
            uid = self._usuario_de_fila[int(f)]
            r = self.prueba(uid, k)
            revisados += 1
            if r and r["acerto_top_k"]:
                encontrados.append({"user_id": uid, "nombre": self.nombre_ficticio(uid, self.historial(uid)),
                                    "puesto": r["puesto"], "producto": r["oculta"]["titulo"]})
        self._aciertos = {"revisados": revisados, "aciertos": len(encontrados), "k": k,
                          "tasa": round(len(encontrados) / max(revisados, 1), 4), "clientes": encontrados[:n]}
        return self._aciertos

    def acierto_densos(self, minimo=5, k=12, revisar=300, semilla=7):
        """Tasa de acierto (leave-last-out) solo en clientes con >= `minimo` compras.
        Es la metrica honesta del modelo personalizado: con historial largo el sistema
        rinde mas, pero representa a una minoria (con 5 quedan ~274 de 186k). Se cachea."""
        clave = (minimo, k, revisar, semilla)
        cache = getattr(self, "_densos", {})
        if clave in cache:
            return cache[clave]
        if not hasattr(self, "_usuario_de_fila"):
            self._usuario_de_fila = {f: u for u, f in self.usuarios.items()}
        filas = np.flatnonzero(np.diff(self.ptr) >= minimo)
        filas = np.random.default_rng(semilla).permutation(filas)[:revisar]
        aciertos, puestos, revisados = 0, [], 0
        for f in filas:
            r = self.prueba(self._usuario_de_fila[int(f)], k)
            if r is None:
                continue
            revisados += 1
            if r["acerto_top_k"]:
                aciertos += 1
            if r["puesto"]:
                puestos.append(r["puesto"])
        import numpy as _np
        salida = {"minimo_compras": minimo, "k": k, "revisados": revisados, "aciertos": aciertos,
                  "tasa": round(aciertos / max(revisados, 1), 4),
                  "puesto_mediano": int(_np.median(puestos)) if puestos else None,
                  "usuarios_totales": int((np.diff(self.ptr) >= minimo).sum())}
        cache[clave] = salida
        self._densos = cache
        return salida

    def prueba(self, user_id, k=12):
        """Prueba de acierto: se oculta la ULTIMA compra real y se recomienda solo con las
        anteriores. Si el producto oculto aparece en las recomendaciones, el modelo acerto."""
        hist = self.historial(user_id)
        if len(hist) < 2:
            return None
        anterior, oculto = hist[:-1], hist[-1]
        est, txt = self._ctx_usuario(user_id, len(anterior))
        perfil = self.perfil_vector(anterior, est, txt)
        cand = np.arange(len(self.catalogo))
        recs, traza = self._rankear(perfil, cand, anterior, k=k, modo="usuario")
        # puesto del producto oculto en el orden completo del mismo pipeline
        cand = cand[~np.isin(cand, anterior)]
        sim = self.items[cand] @ perfil
        orden = np.unique(np.concatenate([np.argsort(-sim)[:300], np.argsort(-self.popularidad[cand])[:300]]))
        pool = cand[orden]
        sas = self.puntuar_sasrec(anterior, pool)
        punt = (self.puntuar_reranker(anterior, pool, sas)[0] if self.reranker
                else rrf([sim[orden], sas], [1.0, 1.0]))
        ranking = pool[np.argsort(-punt)]
        pos = np.flatnonzero(ranking == oculto)
        puesto_sim = int((sim > sim[np.flatnonzero(cand == oculto)[0]]).sum()) + 1
        detalle = self.historial_detalle(user_id)
        return {"user_id": user_id, "oculta": detalle[-1], "historial_usado": detalle[:-1],
                "puesto": int(pos[0]) + 1 if len(pos) else None, "puesto_por_similitud": puesto_sim,
                "candidatos": int(len(pool)), "catalogo": len(self.catalogo),
                "acerto_top_k": bool(len(pos) and pos[0] < k), "k": k,
                "recomendaciones": recs, "traza": traza}

    def para_historial(self, hist, k=10, categoria=None, publico=None, estrellas=None, textos=None):
        """hist = indices de productos comprados, del mas antiguo al mas reciente. Sirve para
        usuarios del dataset (con estrellas/textos) y para un historial armado a mano
        (cliente nuevo en la demo: solo recencia)."""
        hist = [int(i) for i in hist]
        cand = self.candidatos(categoria, publico)
        historial = self._salida(np.array(hist, dtype=np.int64), np.zeros(len(hist)))
        if hist:
            wh = self._pesos(hist, estrellas, textos)
            for j, d in enumerate(historial):
                d["peso"] = round(float(wh[j]), 3)
        if not hist:  # sin compras no hay nada que personalizar: lo mas popular
            top = self._sin_repetidos(cand, np.argsort(-self.popularidad[cand]), [], k)
            top = cand[top]
            recs = self._salida(top, self.popularidad[top].astype(float))
            self._explicar(recs, top, [])
            return {"usuario_con_historial": False, "estrategia": "popularidad (sin historial)",
                    "historial": historial, "recomendaciones": recs,
                    "traza": {"catalogo": len(self.catalogo), "tras_filtro": int(len(cand)),
                              "final": len(recs), "sasrec": False}}
        recs, traza = self._rankear(self.perfil_vector(hist, estrellas, textos), cand, hist, k=k, modo="usuario")
        w = self._pesos(hist, estrellas, textos)
        return {"usuario_con_historial": True,
                "estrategia": "candidatos: similares (two-tower item-to-item) + populares; orden: " + ("re-ranker aprendido (SASRec + similitud + categoria + tienda + popularidad)" if self.reranker else "RRF(item-to-item, SASRec)"),
                "historial": historial, "traza": traza, "recomendaciones": recs,
                "perfil": {"afinidad": self.afinidad_historial(hist),
                           "peso_ultima_compra": round(float(w[-1]), 3),
                           "peso_primera_compra": round(float(w[0]), 3) if len(w) > 1 else None}}

    def escenarios(self):
        """Clientes reales para la exposicion: el mas fiel a cada categoria (4 a 15 compras,
        la mayor fraccion en esa categoria) y uno de gustos variados. Se calcula una vez."""
        if hasattr(self, "_escenarios"):
            return self._escenarios
        usuario_de_fila = {f: u for u, f in self.usuarios.items()}
        cats = self.catalogo["categoria"].to_numpy()
        largos = np.diff(self.ptr)
        candidatas = np.flatnonzero((largos >= 4) & (largos <= 15))
        cats_de = {int(f): cats[self.hist[self.ptr[f]:self.ptr[f + 1]]] for f in candidatas}
        salida, usados = [], set()
        for cat, titulo in [("relojes", "Fan de los relojes"), ("joyeria", "Le encanta la joyería"),
                            ("calzado", "Busca calzado"), ("bolsos", "Siempre compra bolsos"),
                            ("lentes", "Lentes de sol")]:
            mejor = max(cats_de, key=lambda f: ((cats_de[f] == cat).mean(), largos[f]))
            frac = float((cats_de[mejor] == cat).mean())
            if frac >= 0.5:
                usados.add(mejor)
                salida.append({"titulo": titulo, "categoria": cat, "user_id": usuario_de_fila[mejor],
                               "nombre": self.nombre_ficticio(usuario_de_fila[mejor], self.historial(usuario_de_fila[mejor])),
                               "compras": int(largos[mejor]), "fraccion": round(frac, 2)})
        variado = max((f for f in cats_de if f not in usados), key=lambda f: (len(set(cats_de[f].tolist())), largos[f]))
        salida.append({"titulo": "Gustos variados", "categoria": "otros", "user_id": usuario_de_fila[variado],
                       "nombre": self.nombre_ficticio(usuario_de_fila[variado], self.historial(usuario_de_fila[variado])),
                       "compras": int(largos[variado]), "fraccion": None})
        self._escenarios = salida
        return salida

    def buscar(self, texto, k=10):
        """Busqueda directa con el two-tower (sin Laya ni filtros): para agregar productos."""
        q = self.codificar([PREFIJO_CONSULTA + texto])[0]
        sim = self.items @ q
        todos = np.arange(len(sim))
        top = self._sin_repetidos(todos, np.argsort(-sim)[: k * 5], [], k)
        return self._salida(top, sim[top], sim[top])

    def similares(self, product_id, k=10):
        i = self.idx_producto.get(product_id)
        if i is None:
            return None
        sim = self.items @ self.items[i]
        todos = np.arange(len(sim))
        top = self._sin_repetidos(todos, np.argsort(-sim)[: k * 5], [i], k)
        return self._salida(top, sim[top], sim[top])

    def usuarios_ejemplo(self, n=5, minimo=3, semilla=None):
        """Usuarios con al menos `minimo` compras; `semilla` elige otra muestra al azar."""
        if not hasattr(self, "_usuario_de_fila"):
            self._usuario_de_fila = {f: u for u, f in self.usuarios.items()}
        largos = np.diff(self.ptr)
        filas = np.flatnonzero(largos >= minimo)
        if semilla is not None:
            filas = np.random.default_rng(semilla).permutation(filas)
        salida = []
        for f in filas[:n]:
            items = self.hist[self.ptr[f]:self.ptr[f + 1]]
            cats = self.catalogo["categoria"].to_numpy()[items]
            salida.append({"user_id": self._usuario_de_fila[int(f)], "compras": int(largos[f]),
                           "categorias": sorted(set(cats.tolist())),
                           "ultimas": [self.titulos[i] for i in items[-3:]]})
        return salida
