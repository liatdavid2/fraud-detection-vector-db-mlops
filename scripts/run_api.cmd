@echo off
call .venv\Scripts\activate
uvicorn fraud_vector_db_mlops.api:app --reload --host 0.0.0.0 --port 8000
