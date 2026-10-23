@echo off
REM Pipeline completo de entrenamiento: datos -> two-tower -> SASRec -> re-ranker -> publicar
REM Uso: entrenar_todo.cmd   (los logs quedan en logs\; tarda ~4 h en la GTX 1650)
REM SASRec usa la configuracion de la version publicada v003 (solo contenido, --p-drop-id 1.0).
cd /d "%~dp0"
if not exist logs mkdir logs
set PY=.venv\Scripts\python.exe -u
if not exist data\splits\catalogo.parquet %PY% src\datos.py > logs\datos.log 2>&1 || goto :error
REM 1) Two-tower con negativos duros (3 por par): separa el producto correcto de los
REM    productos que de verdad se le parecen. Epoca unica: en la GTX 1650 tarda ~4 h.
%PY% src\two_tower.py --max-resenas 120000 --negativos-duros 2 --lote 64 --max-len-producto 128 > logs\two_tower.log 2>&1 || goto :error
%PY% src\sasrec.py --epocas 30 --lr 3e-4 --p-drop-id 1.0 --paciencia 4 > logs\sasrec.log 2>&1 || goto :error
REM 2) Re-ranker: MLP (listwise) + GBDT lambdarank (ganador, se publica)
%PY% src\reranker_mlp.py --epocas 40 --pliegues 5 > logs\reranker_mlp.log 2>&1 || goto :error
%PY% src\reranker_gbdt.py > logs\reranker_gbdt.log 2>&1 || goto :error
REM 3) Modelo de calidad de la resena (RMSE y "recomendaria"): accuracy binaria > 95 %
%PY% src\calidad.py --epocas 3 --max-datos 90000 > logs\calidad.log 2>&1 || goto :error
REM 4) Evaluacion centralizada (todas las metricas del informe en un solo lugar)
%PY% src\evaluacion.py > logs\evaluacion.log 2>&1 || goto :error
REM 5) Publicar (GBDT sobre SASRec v003, que es el que entrenaron los re-rankers)
%PY% src\publicar.py --sasrec v003_20261003-1048 --reranker v005_20261021-2227 > logs\publicar.log 2>&1 || goto :error
echo OK> logs\FIN.txt
exit /b 0
:error
echo ERROR (revisar logs\)> logs\FIN.txt
exit /b 1
