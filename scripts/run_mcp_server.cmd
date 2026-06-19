@echo off
setlocal

cd /d %~dp0\..

set PYTHONPATH=%CD%\src;%PYTHONPATH%

call .venv\Scripts\activate

python -m fraud_vector_db_mlops.mcp_server

endlocal