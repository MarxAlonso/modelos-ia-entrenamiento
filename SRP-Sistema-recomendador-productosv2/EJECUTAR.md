# Ejecutar el backend

## Opción 1 — doble clic

Doble clic en **`iniciar_demo.cmd`** (en esta carpeta). Levanta el backend y abre el navegador solo.

## Opción 2 — a mano

Abre PowerShell y ejecuta:

```powershell
cd "C:\developer-marx\Proyectos UTP\modelos-ia-entrenamiento\SRP-Sistema-recomendador-productosv2\backend"
..\.venv\Scripts\python -m uvicorn app.main:app --port 8000
```

Espera a que aparezca `Uvicorn running on http://127.0.0.1:8000` y deja esa ventana abierta.

## Abrir en el navegador

| Dirección | Qué es |
|---|---|
| **http://localhost:8000** | La interfaz del recomendador |
| http://localhost:8000/docs | La API (Swagger) |
| http://localhost:8000/salud | Comprobar que el backend responde |

## Detener

Cierra la ventana del backend (o `Ctrl + C` en ella).

## Notas

- La **primera búsqueda** con texto tarda ~15-20 s mientras carga Laya; después es inmediata.
  La primera vez que se ejecuta en una PC necesita internet para descargarlo.
- Si dice que el **puerto 8000 está en uso**, el backend ya está corriendo: abre directamente http://localhost:8000.
- Si falta `models\servir\`, ejecútalo una vez desde esta carpeta:
  `.venv\Scripts\python src\publicar.py --sasrec v003_20261003-1048`
