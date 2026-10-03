@echo off
REM Pipeline completo de entrenamiento: two-tower -> SASRec -> publicar en models\servir\
REM Uso: entrenar_todo.cmd   (los logs quedan en logs\)
cd /d "%~dp0"
if not exist logs mkdir logs
set PY=.venv\Scripts\python.exe -u
if not exist data\splits\catalogo.parquet %PY% src\datos.py > logs\datos.log 2>&1 || goto :error
%PY% src\two_tower.py --max-resenas 60000 > logs\two_tower.log 2>&1 || goto :error
%PY% src\sasrec.py > logs\sasrec.log 2>&1 || goto :error
%PY% src\publicar.py > logs\publicar.log 2>&1 || goto :error
echo OK> logs\FIN.txt
exit /b 0
:error
echo ERROR (revisar logs\)> logs\FIN.txt
exit /b 1
