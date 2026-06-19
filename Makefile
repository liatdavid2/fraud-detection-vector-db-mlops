.PHONY: install services sample train validate drift api dashboard test lint

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

services:
	docker compose up -d

sample:
	python -m fraud_vector_db_mlops.data --make-sample

validate:
	python -m fraud_vector_db_mlops.validation

train:
	python -m fraud_vector_db_mlops.train

train-no-milvus:
	python -m fraud_vector_db_mlops.train --skip-milvus

drift:
	python -m fraud_vector_db_mlops.drift

api:
	uvicorn fraud_vector_db_mlops.api:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run src/fraud_vector_db_mlops/dashboard.py

test:
	pytest -q

lint:
	ruff check src tests
