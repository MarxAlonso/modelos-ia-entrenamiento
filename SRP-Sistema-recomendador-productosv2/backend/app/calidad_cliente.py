"""
Modelo de calidad de la resena, afinado por nosotros (src/calidad.py), servido en ONNX.

Responde las dos preguntas del cliente sobre una resena sin estrellas:
  - satisfaccion: cuantas estrellas (1..5) tiene la resena (RMSE 0.63 en test)
  - recomendaria: la probabilidad de que el cliente recomiende el producto (accuracy 96.8 %)

Usa el mismo tokenizador del two-tower (los dos modelos parten del mismo MiniLM multilingue),
asi que no hace falta guardar un segundo tokenizer.
"""
import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SERVIR = os.path.join(RAIZ, "models", "servir")
MAX_LEN = 96


class CalidadCliente:
    def __init__(self, carpeta=SERVIR):
        self.ruta = os.path.join(carpeta, "calidad.onnx")
        self.tokenizer = os.path.join(carpeta, "tokenizer.json")
        self.disponible = os.path.exists(self.ruta) and os.path.exists(self.tokenizer)
        if not self.disponible:
            return
        opciones = ort.SessionOptions()
        opciones.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
        self.sesion = ort.InferenceSession(self.ruta, opciones, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(self.tokenizer)
        self.tok.enable_truncation(MAX_LEN)
        self.tok.enable_padding(pad_id=self.tok.token_to_id("<pad>") or 1, pad_token="<pad>")

    def predecir(self, texto):
        """-> {'estrellas': 1..5, 'probabilidades': [...], 'recomendaria': 0..1, 'negativa': 0..1}"""
        if not self.disponible:
            return None
        enc = self.tok.encode_batch([texto or ""])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        estrellas, recomienda = self.sesion.run(None, {"input_ids": ids, "attention_mask": mask})
        p_est = np.exp(estrellas[0] - estrellas[0].max())
        p_est = p_est / p_est.sum()
        p_rec = float(_softmax(recomienda[0])[1])
        return {"estrellas": int(np.argmax(p_est) + 1),
                "probabilidades_estrellas": [round(float(x), 4) for x in p_est],
                "recomendaria": round(p_rec, 4),
                "negativa": round(1.0 - p_rec, 4),
                "confianza": round(float(max(p_est)), 4)}


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()
