
import sys
from pathlib import Path
_RUTA_MODULO = Path(__file__).resolve()
while _RUTA_MODULO.name != "src" and _RUTA_MODULO != _RUTA_MODULO.parent:
    _RUTA_MODULO = _RUTA_MODULO.parent
sys.path.insert(0, str(_RUTA_MODULO.parent))

from src.rutas import RUTA_RAIZ
import torch
from pathlib import Path
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer

MODELO = "Helsinki-NLP/opus-mt-en-es"
N_MUESTRA = 2000
SEMILLA = 42
LOTE = 32
MAX_CHARS = 450
N_POR_EJECUCION = 250

RUTA_ENTRADA = RUTA_RAIZ / "data/amazon/resenas.csv"
RUTA_SALIDA = RUTA_RAIZ / "data/amazon/resenas_traducidas.csv"


def construir_muestra() -> pd.DataFrame:
    resenas = pd.read_csv(RUTA_ENTRADA)
    resenas["longitud"] = resenas["texto"].fillna("").str.len()
    muestra = resenas[
        (resenas["longitud"] >= 20) & (resenas["longitud"] <= MAX_CHARS)
    ].sample(n=N_MUESTRA, random_state=SEMILLA).reset_index(drop=True)
    muestra["idx_muestra"] = muestra.index
    return muestra


def ya_traducidas() -> set[int]:
    if RUTA_SALIDA.exists():
        previo = pd.read_csv(RUTA_SALIDA)
        return set(previo["idx_muestra"].astype(int))
    return set()


def main() -> None:
    muestra = construir_muestra()
    pendientes_mask = ~muestra["idx_muestra"].isin(ya_traducidas())
    pendientes = muestra[pendientes_mask].reset_index(drop=True)

    print(f"Progreso actual: {N_MUESTRA - len(pendientes)}/{N_MUESTRA} traducidas")
    if len(pendientes) == 0:
        print("Todo el lote esta traducido. Nada por hacer.")
        return

    objetivo = pendientes.head(N_POR_EJECUCION).reset_index(drop=True)
    print(f"Esta ejecucion traduce las siguientes {len(objetivo)}...")

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = MarianTokenizer.from_pretrained(MODELO)
    modelo = MarianMTModel.from_pretrained(MODELO).to(dispositivo).eval()

    textos = objetivo["texto"].tolist()
    traducciones: list[str] = []
    with torch.no_grad():
        for inicio in range(0, len(textos), LOTE):
            lote = textos[inicio : inicio + LOTE]
            tokens = tokenizer(
                lote, return_tensors="pt", padding=True,
                truncation=True, max_length=128,
            )
            tokens = {k: v.to(dispositivo) for k, v in tokens.items()}
            salida = modelo.generate(**tokens, max_length=160, num_beams=2)
            traducciones.extend(tokenizer.batch_decode(salida, skip_special_tokens=True))
            print(f"      {min(inicio + LOTE, len(textos))}/{len(textos)} en esta ejecucion")

    resultado = objetivo.copy()
    resultado["texto_es"] = traducciones
    columnas = ["idx_muestra", "user_id", "product_id", "rating", "fecha", "texto", "texto_es"]
    escribir_encabezado = not RUTA_SALIDA.exists()
    resultado[columnas].to_csv(
        RUTA_SALIDA, mode="a", header=escribir_encabezado, index=False
    )

    total_listo = len(ya_traducidas())
    print(f"Guardado incremental en {RUTA_SALIDA}")
    print(f"Progreso total: {total_listo}/{N_MUESTRA}")
    if total_listo < N_MUESTRA:
        print(f"Faltan {N_MUESTRA - total_listo}: vuelve a ejecutar este script.")
    else:
        print("Traduccion completa.")
        verificacion = pd.read_csv(RUTA_SALIDA)
        print("\nEjemplos:")
        for i in range(3):
            print(f"\n  EN: {verificacion.loc[i, 'texto'][:110]}")
            print(f"  ES: {verificacion.loc[i, 'texto_es'][:110]}")


if __name__ == "__main__":
    main()
