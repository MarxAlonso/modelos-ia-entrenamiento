"""AUTOENTRENAMIENTO: el sistema se gestiona y reentrena solo.

Flujo MLOps:
  1. Huella (hash) de data/interacciones.csv y data/productos.csv
  2. Si cambian -> ejecuta cadena completa de entrenamiento
  3. Evalua el nuevo HIBRIDO V4 en el mismo split
  4. PROMOCION AUTOMATICA solo si P@10 >= modelo vigente - tolerancia
  5. Versiona modelos en models/versiones/ y registra historia en
     models/estado_entrenamiento.json

Uso:
    venv\\Scripts\\python.exe src/autoentrenamiento.py --verificar   # solo revisar
    venv\\Scripts\\python.exe src/autoentrenamiento.py --entrenar   # forzar ciclo
    venv\\Scripts\\python.exe src\\autoentrenamiento.py --servicio [min]  # vigiar continuo
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

RUTA_BASE = Path(__file__).resolve().parent.parent
RUTA_DATOS = RUTA_BASE / "data"
RUTA_MODELOS = RUTA_BASE / "models"
RUTA_VERSIONES = RUTA_MODELOS / "versiones"
ESTADO = RUTA_MODELOS / "estado_entrenamiento.json"
TOLERANCIA = 0.0005  # promueve si P@10 nuevo >= vigente - 0.0005

ARCHIVOS_VIGILADOS = ["interacciones.csv", "productos.csv", "usuarios.csv"]
CADENA = [
    ("nivel34_torres_gnn.py", "Two-Tower + LightGCN"),
    ("pipeline_final.py", "Hibrido V4"),
    ("recomendaciones_app.py", "Recomendaciones del app"),
]


def huella() -> str:
    h = hashlib.sha256()
    for nombre in ARCHIVOS_VIGILADOS:
        ruta = RUTA_DATOS / nombre
        if ruta.exists():
            h.update(ruta.read_bytes())
            h.update(str(ruta.stat().st_mtime_ns).encode())
    return h.hexdigest()[:16]


def cargar_estado() -> dict:
    if ESTADO.exists():
        return json.loads(ESTADO.read_text(encoding="utf-8"))
    return {"huella": None, "ultima_entrenada": None, "p10_vigente": None,
            "ciclos": [], "promociones": 0, "rechazos": 0}


def guardar_estado(e: dict):
    ESTADO.write_text(json.dumps(e, ensure_ascii=False, indent=2), encoding="utf-8")


def p10_actual() -> float | None:
    ruta = RUTA_MODELOS / "nivel34_resultados.json"
    if not ruta.exists():
        return None
    res = json.loads(ruta.read_text(encoding="utf-8"))
    v4 = res.get("HIBRIDO V4") or {}
    return v4.get("precision_producto")


def ejecutar_ciclo(forzado: bool) -> None:
    estado = cargar_estado()
    nueva = huella()
    sin_cambios = nueva == estado["huella"]

    if sin_cambios and not forzado:
        print(f"[{time.strftime('%H:%M:%S')}] Datos sin cambios ({nueva}). Nada que hacer.")
        return

    motivo = "forzado" if forzado else f"datos cambiaron {estado['huella']} -> {nueva}"
    print(f"=== CICLO DE AUTOENTRENAMIENTO ({motivo}) ===")
    inicio = time.time()

    p10_antes = estado.get("p10_vigente") or p10_actual()

    # respaldo del modelo vigente antes de reentrenar
    RUTA_VERSIONES.mkdir(parents=True, exist_ok=True)
    marca = time.strftime("%Y%m%d-%H%M%S")
    version = RUTA_VERSIONES / marca
    version.mkdir(parents=True, exist_ok=True)
    for artefacto in ["hibrido_v4_final.pkl", "puntajes_hibrido_v4.npz",
                      "torre_doble.keras", "torre_doble_vectores.npz",
                      "lightgcn_embeddings.npz"]:
        origen = RUTA_MODELOS / artefacto
        if origen.exists():
            shutil.copy2(origen, version / artefacto)

    fallo = None
    for script, descripcion in CADENA:
        print(f"--> Entrenando: {descripcion} ({script})")
        r = subprocess.run([sys.executable, str(RUTA_BASE / "src" / script)],
                           capture_output=True, text=True)
        cola = (r.stdout or "").strip().splitlines()
        if cola:
            print("    " + cola[-1])
        if r.returncode != 0:
            fallo = f"{script}: {(r.stderr or '')[-300:]}"
            break

    if fallo:
        print(f"CICLO FALLIDO: {fallo}")
        estado["ciclos"].append({"marca": marca, "estado": "fallo", "error": fallo,
                                 "duracion_s": round(time.time() - inicio, 1)})
        guardar_estado(estado)
        return

    # regenerar datos del app (JSONs)
    subprocess.run([sys.executable, str(RUTA_BASE / "ver_grafos" / "generar_grafos_3d.py"), "--json"],
                   capture_output=True, text=True)

    p10_nuevo = p10_actual()
    promocionado = p10_nuevo is not None and (
        p10_antes is None or p10_nuevo >= p10_antes - TOLERANCIA)

    if promocionado:
        estado["promociones"] += 1
        resultado = f"PROMOCIONADO (P@10 {p10_antes} -> {p10_nuevo})"
    else:
        estado["rechazos"] += 1
        resultado = f"RECHAZADO (P@10 {p10_antes} -> {p10_nuevo}); restaurando version {marca}"
        for artefacto in ["hibrido_v4_final.pkl", "puntajes_hibrido_v4.npz",
                          "torre_doble.keras", "torre_doble_vectores.npz",
                          "lightgcn_embeddings.npz"]:
            respaldo = version / artefacto
            if respaldo.exists():
                shutil.copy2(respaldo, RUTA_MODELOS / artefacto)

    estado["huella"] = nueva
    estado["ultima_entrenada"] = time.strftime("%Y-%m-%d %H:%M:%S")
    estado["p10_vigente"] = p10_nuevo if promocionado else p10_antes
    estado["ciclos"].append({
        "marca": marca, "estado": "ok" if promocionado else "rechazado",
        "p10_antes": p10_antes, "p10_nuevo": p10_nuevo,
        "duracion_s": round(time.time() - inicio, 1),
    })
    estado["ciclos"] = estado["ciclos"][-30:]
    guardar_estado(estado)
    print(f"=== FIN DEL CICLO: {resultado} | {round(time.time()-inicio,1)}s ===")


def main():
    parser = argparse.ArgumentParser(description="Autoentrenamiento del recomendador")
    parser.add_argument("--verificar", action="store_true")
    parser.add_argument("--entrenar", action="store_true")
    parser.add_argument("--servicio", nargs="?", type=float, const=15.0, metavar="MIN")
    args = parser.parse_args()

    estado = cargar_estado()
    print(f"Estado: huella={estado['huella']} | P@10 vigente={estado['p10_vigente']} "
          f"| promociones={estado['promociones']} rechazos={estado['rechazos']}")

    if args.verificar:
        nueva = huella()
        print(f"Huella actual de datos: {nueva}")
        print("Cambios pendientes:" , "NO" if nueva == estado["huella"] else "SI -> ejecute --entrenar")
        return

    if args.entrenar:
        ejecutar_ciclo(forzado=True)
        return

    if args.servicio:
        minutos = args.servicio
        print(f"Modo servicio: vigilando cada {minutos} min (Ctrl+C para salir)")
        while True:
            try:
                ejecutar_ciclo(forzado=False)
            except Exception as e:
                print(f"ERROR en ciclo: {e}")
            time.sleep(minutos * 60)


if __name__ == "__main__":
    main()
