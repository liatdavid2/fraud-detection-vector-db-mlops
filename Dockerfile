FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

COPY requirements.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY models ./models
COPY reports ./reports

EXPOSE 8000

CMD ["uvicorn", "fraud_vector_db_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
