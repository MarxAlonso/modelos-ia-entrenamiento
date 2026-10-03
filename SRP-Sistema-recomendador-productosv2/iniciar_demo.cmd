@echo off
REM ============================================================
REM  SRP v2 - Iniciar la demo (doble clic en este archivo)
REM  1. Verifica que existan el entorno y los modelos
REM  2. Levanta el backend en http://localhost:8000
REM  3. Precarga Laya (asi la primera busqueda no tarda en la expo)
REM  4. Abre el navegador en la demo
REM  Para detener: cerrar la ventana "SRP backend"
REM ============================================================
cd /d "%~dp0"
title SRP - iniciar demo

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] No existe el entorno .venv. Crealo con los pasos del README ^(seccion Pasos, paso 0^).
  pause & exit /b 1
)
if not exist "models\servir\manifiesto.json" (
  echo [ERROR] No hay modelos publicados en models\servir\. Corre primero: entrenar_todo.cmd
  pause & exit /b 1
)

REM Si ya hay algo escuchando en el puerto 8000, se reutiliza
netstat -ano | findstr /R /C:":8000 .*LISTENING" >nul
if %errorlevel%==0 (
  echo El backend ya estaba corriendo en el puerto 8000.
) else (
  echo Iniciando el backend...
  if not exist logs mkdir logs
  start "SRP backend" /D "%~dp0backend" cmd /k "..\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000"
)

echo Esperando a que el backend responda...
set /a intentos=0
:esperar
set /a intentos+=1
if %intentos% gtr 90 (
  echo [ERROR] El backend no respondio en 3 minutos. Revisa la ventana "SRP backend".
  pause & exit /b 1
)
powershell -NoProfile -Command "try { Invoke-RestMethod http://localhost:8000/salud -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 ( timeout /t 2 /nobreak >nul & goto esperar )
echo Backend listo.

echo Precargando Laya (la primera vez puede tardar 20-60 s)...
powershell -NoProfile -Command "$b = [Text.Encoding]::UTF8.GetBytes('{\"texto\":\"busco un reloj\",\"k\":1}'); try { Invoke-RestMethod http://localhost:8000/recomendar/consulta -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 300 | Out-Null; Write-Host 'Laya listo.' } catch { Write-Host 'Aviso: Laya no respondio; la demo funciona igual, sin sus filtros.' }"

start "" "http://localhost:8000/app/#como"
echo.
echo  Demo abierta en http://localhost:8000/app
echo  Para detenerla, cierra la ventana "SRP backend".
timeout /t 8 >nul
