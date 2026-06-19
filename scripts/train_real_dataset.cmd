@echo off
call .venv\Scripts\activate
python -m fraud_vector_db_mlops.data --download
python -m fraud_vector_db_mlops.validation
python -m fraud_vector_db_mlops.train
