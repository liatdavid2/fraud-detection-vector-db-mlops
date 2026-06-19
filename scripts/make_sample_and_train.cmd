@echo off
setlocal
cd /d %~dp0\..
set PYTHONPATH=%CD%\src;%PYTHONPATH%
call .venv\Scripts\activate
REM For the real project run, train on the real BAF dataset, not the synthetic smoke-test sample.
python -m fraud_vector_db_mlops.data --download-baf
if errorlevel 1 exit /b %errorlevel%
python -m fraud_vector_db_mlops.validation
if errorlevel 1 exit /b %errorlevel%
python -m fraud_vector_db_mlops.train
if errorlevel 1 exit /b %errorlevel%
endlocal