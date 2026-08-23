"""Rutas compartidas del proyecto."""
from pathlib import Path

RUTA_SRC = Path(__file__).resolve().parent
RUTA_RAIZ = RUTA_SRC.parent
RUTA_DATOS = RUTA_RAIZ / "data"
RUTA_MODELOS = RUTA_RAIZ / "models"
