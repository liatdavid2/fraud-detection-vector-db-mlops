@echo off
call .venv\Scripts\activate
python -m fraud_vector_db_mlops.data --make-sample
python -m fraud_vector_db_mlops.validation
python -m fraud_vector_db_mlops.train --skip-milvus
