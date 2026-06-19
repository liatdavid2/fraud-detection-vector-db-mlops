@echo off
setlocal
cd /d %~dp0\..
set PYTHONPATH=%CD%\src;%PYTHONPATH%
call .venv\Scripts\activate
python -m fraud_vector_db_mlops.data --make-sample
if errorlevel 1 exit /b %errorlevel%
python -m fraud_vector_db_mlops.validation
if errorlevel 1 exit /b %errorlevel%
python -m fraud_vector_db_mlops.train
if errorlevel 1 exit /b %errorlevel%
endlocal